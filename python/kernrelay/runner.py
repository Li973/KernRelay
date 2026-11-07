"""Execute generated schedules through the KernRelay ``nhp_exec``."""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .runtime import EXEC_BIN, KERNRELAY_ROOT, VENDOR_LIB
from .schedule import dump_schedule, validate_schedule
from .stats import summarize

ENV_ERROR_MARKS = ("set device failed", "set device for", "HcclCommInitAll",
                   "init hccl", "fewer devices", "ACL runtime exposes",
                   "acl=207004", "acl=107001")
DETERMINISTIC_MARKS = ("schedule contains wait cycle", "missing plan",
                       "missing event", "unknown node", "exceed",
                       "duplicate engine", "wait on unpublished",
                       "cannot open")


def classify_failure(stderr: str) -> str:
    for m in ENV_ERROR_MARKS:
        if m in stderr:
            return "env"
    for m in DETERMINISTIC_MARKS:
        if m in stderr:
            return "deterministic"
    return "transient"


@dataclass
class RunConfig:
    devices: str = "0,1"
    nranks: int = 2
    samples: int = 5
    process_isolation: bool = False
    warmup: int = 2
    iters_per_process: int = 3
    min_processes: int = 3
    max_attempts: int = 12
    gap_s: float = 2.0
    retry_sleep_s: float = 6.0
    timeout_s: int = 1200
    fail_fast: bool = True
    env: dict = field(default_factory=dict)

    def exec_env(self) -> dict:
        env = dict(os.environ)
        env["ASCEND_RT_VISIBLE_DEVICES"] = self.devices
        libs = [str(VENDOR_LIB), str(KERNRELAY_ROOT / "bin")]
        env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
        env["NHP_PROC_PER_ROUND"] = "1" if self.process_isolation else "0"
        env.update({str(k): str(v) for k, v in self.env.items()})
        return env


def _slug(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _spawn(sched: dict, cfg: RunConfig, workdir: Path, tag: str,
           per_run: int, warmup: int):
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    slug = _slug(tag)
    sched_path = workdir / ("%s.sched.json" % slug)
    out_path = workdir / ("%s.out.json" % slug)
    log_path = workdir / ("%s.log" % slug)
    dump_schedule(sched, sched_path)
    if out_path.exists():
        out_path.unlink()
    cmd = [str(EXEC_BIN), str(sched_path), str(per_run), str(warmup),
           str(out_path)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=cfg.exec_env(), capture_output=True,
                              text=True, timeout=cfg.timeout_s,
                              cwd=str(KERNRELAY_ROOT))
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as ex:
        exit_code, stdout, stderr = 124, ex.stdout or "", "timeout"
    wall_s = time.time() - t0
    log_path.write_text(
        "cmd: %s\nwall_s: %.3f\nexit_code: %s\n"
        "--- stdout ---\n%s\n--- stderr ---\n%s\n"
        % (" ".join(cmd), wall_s, exit_code, stdout, stderr),
        encoding="utf-8")
    result = None
    if out_path.exists():
        try:
            result = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = None
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr,
            "result": result, "wall_s": wall_s, "log": log_path,
            "sched": sched_path, "out": out_path}


def _hint(kind: str) -> str:
    if kind == "env":
        return ("设备可见性/占用问题：核对 npu-smi 与 --devices 映射；"
                "换空闲卡对后重跑")
    if kind == "deterministic":
        return "调度确定性缺陷：修 schedule/执行器后重跑，重试无效"
    return "瞬时失败：已计入 retries，正式数据要求 retries=0"


def run_schedule(sched: dict, cfg: RunConfig, tag: str, workdir) -> dict:
    errors = validate_schedule(sched)
    if errors:
        raise ValueError("%s 结构非法: %s" % (tag, "; ".join(errors)))
    if not EXEC_BIN.exists():
        raise FileNotFoundError("缺少 %s" % EXEC_BIN)
    workdir = Path(workdir).resolve()
    if cfg.process_isolation:
        per_run, warm_eff = 1, max(1, cfg.warmup)
    else:
        per_run = max(1, cfg.iters_per_process)
        warm_eff = max(2, cfg.warmup)
    samples, checksums, attempts = [], [], []
    proc_mins, proc_medians = [], []
    gate = 0
    retries = 0
    teardown_warns = 0
    last_err = None
    target = max(cfg.samples, per_run * cfg.min_processes)
    for attempt in range(cfg.max_attempts):
        run_tag = "%s_attempt%02d" % (tag, attempt)
        rec = _spawn(sched, cfg, workdir, run_tag, per_run, warm_eff)
        accepted = False
        reason = "no result"
        result = rec["result"]
        if result is not None and result.get("checksum") is not None:
            accepted = True
            reason = "ok" if rec["exit_code"] == 0 else "teardown"
            times = result.get("times_us", [result.get("p50_us")])
            samples.extend(times)
            proc_mins.append(min(times))
            proc_medians.append(summarize(times)["p50_us"])
            checksums.append(result["checksum"])
            gate += int(result.get("gate_timeouts", 0))
            if rec["exit_code"] != 0:
                teardown_warns += 1
        else:
            kind = classify_failure(rec["stderr"])
            reason = kind
            last_err = ("%s exit=%s: %s" % (tag, rec["exit_code"],
                                            rec["stderr"].strip()[-800:]))
            if kind in ("env", "deterministic") and cfg.fail_fast:
                raise RuntimeError("%s\n%s" % (last_err, _hint(kind)))
            retries += 1
        attempts.append({"attempt": attempt, "exit_code": rec["exit_code"],
                         "accepted": accepted, "reason": reason,
                         "checksum": result.get("checksum") if result else None,
                         "times_us": result.get("times_us", []) if result else [],
                         "gate_timeouts": (result.get("gate_timeouts", 0)
                                           if result else None),
                         "log": str(rec["log"].relative_to(workdir)),
                         "sched": str(rec["sched"].relative_to(workdir)),
                         "out": str(rec["out"].relative_to(workdir)),
                         "wall_s": rec["wall_s"]})
        if len(samples) >= target and len(proc_mins) >= cfg.min_processes:
            break
        time.sleep(cfg.gap_s if accepted else cfg.retry_sleep_s)
    if not samples:
        raise RuntimeError(last_err or "%s: 没有有效样本" % tag)
    stat = summarize(samples)
    best = stat["min_us"]
    clean_cut = max(3.0 * best, best + 100.0)
    clean_mins = [m for m in proc_mins if m <= clean_cut]
    dirty_mins = [m for m in proc_mins if m > clean_cut]
    clean_typical = summarize(clean_mins)["p50_us"] if clean_mins else None
    return {"tag": tag, "nranks": cfg.nranks, "devices": cfg.devices,
            "process_isolation": cfg.process_isolation,
            "per_process_iters": per_run, "warmup": warm_eff,
            "samples_us": samples, "checksums": checksums,
            "proc_mins_us": proc_mins, "proc_medians_us": proc_medians,
            "clean_cut_us": clean_cut,
            "clean_processes": len(clean_mins),
            "contaminated_processes": len(dirty_mins),
            "gate_timeouts": gate, "retries": retries,
            "teardown_warns": teardown_warns,
            "p50_us": stat["p50_us"], "p90_us": stat["p90_us"],
            "min_us": stat["min_us"], "max_us": stat["max_us"],
            "best_us": best,
            "typical_us": clean_typical if clean_typical is not None
            else summarize(proc_mins)["p50_us"],
            "typical_all_us": summarize(proc_mins)["p50_us"],
            "n_samples": stat["n"], "n_processes": len(proc_mins),
            "stalls": stat["stalls"],
            "ok": gate == 0 and retries == 0,
            "attempts": attempts}


def run_expect_failure(sched: dict, cfg: RunConfig, tag: str, workdir) -> dict:
    workdir = Path(workdir).resolve()
    rec = _spawn(sched, cfg, workdir, tag, 1, 0)
    return {"tag": tag, "exit_code": rec["exit_code"],
            "stderr_tail": rec["stderr"].strip()[-1200:],
            "has_result": rec["result"] is not None,
            "log": str(rec["log"].relative_to(workdir))}
