"""稳定 C ABI 的 Python 封装：外部项目只通过 libkernrelay.so 调 KernRelay。"""
import ctypes
import json
import os
import tempfile
from pathlib import Path

from .runtime import KERNRELAY_ROOT

_LIB = None


def lib_path():
    env = os.environ.get("KERNRELAY_LIB")
    if env:
        return Path(env)
    return KERNRELAY_ROOT / "bin" / "libkernrelay.so"


def lib():
    global _LIB
    if _LIB is None:
        p = lib_path()
        if not p.exists():
            raise FileNotFoundError(
                "%s 不存在，先运行 scripts/build_lib.sh" % p)
        _LIB = ctypes.CDLL(str(p))
        _LIB.kr_version.restype = ctypes.c_char_p
        _LIB.kr_qwen_run.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        _LIB.kr_qwen_run.restype = ctypes.c_int
        _LIB.kr_plan_run.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_char_p]
        _LIB.kr_plan_run.restype = ctypes.c_int
        _LIB.kr_qwen_pp_run.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_size_t]
        _LIB.kr_qwen_pp_run.restype = ctypes.c_int
        _LIB.kr_qwen_tp_run.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_size_t]
        _LIB.kr_qwen_tp_run.restype = ctypes.c_int
        _LIB.kr_qwen_profile_many.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_size_t]
        _LIB.kr_qwen_profile_many.restype = ctypes.c_int
    return _LIB


def version():
    return lib().kr_version().decode()


def qwen_run(model_dir, mode="serial_graph", seq=2048, device=6,
             warmup=1, iters=3):
    buf = ctypes.create_string_buffer(1 << 20)
    rc = lib().kr_qwen_run(
        str(model_dir).encode(), mode.encode(), int(seq), int(device),
        int(warmup), int(iters), buf, len(buf))
    try:
        payload = json.loads(buf.value.decode())
    except Exception:
        payload = {"error": buf.value.decode(errors="replace")}
    payload["_rc"] = rc
    return payload


def qwen_pp_run(model_dir, mode="pp", seq=2048, dev0=6, dev1=7,
                requests=2, warmup=1, iters=3):
    """双卡 PP=2：mode='pp' 流水重叠，'pp_serial' 逐请求串行基线。"""
    buf = ctypes.create_string_buffer(1 << 20)
    rc = lib().kr_qwen_pp_run(
        str(model_dir).encode(), mode.encode(), int(seq), int(dev0),
        int(dev1), int(requests), int(warmup), int(iters), buf, len(buf))
    try:
        payload = json.loads(buf.value.decode())
    except Exception:
        payload = {"error": buf.value.decode(errors="replace")}
    payload["_rc"] = rc
    return payload


def qwen_tp_run(model_dir, mode="tp2", seq=2048, dev0=1, dev1=2, tiles=2,
                warmup=1, iters=3):
    """双卡 TP-MLP：attention 复制，MLP 各半；tp2 / tp2_split。"""
    buf = ctypes.create_string_buffer(1 << 20)
    rc = lib().kr_qwen_tp_run(
        str(model_dir).encode(), mode.encode(), int(seq), int(dev0), int(dev1),
        int(tiles), int(warmup), int(iters), buf, len(buf))
    try:
        payload = json.loads(buf.value.decode())
    except Exception:
        payload = {"error": buf.value.decode(errors="replace")}
    payload["_rc"] = rc
    return payload


def plan_run(schedule_json, devices="6,7", mode="direct", iters=3, warmup=1,
             out_json=None):
    out = Path(out_json) if out_json else Path(
        tempfile.mkstemp(prefix="kr_plan_", suffix=".json")[1])
    rc = lib().kr_plan_run(
        str(schedule_json).encode(), int(iters), int(warmup),
        str(devices).encode(), mode.encode(), str(out).encode())
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
    except Exception:
        payload = {"error": "no result", "rc": rc, "out": str(out)}
    payload["_rc"] = rc
    return payload


def qwen_profile_many(model_dir, cases, seq=2048, device=6):
    """cases: [{"op":..., "k":...}] -> dict; weights loaded once."""
    import tempfile
    fd, path = tempfile.mkstemp(prefix="kr_profile_", suffix=".json")
    os.close(fd)
    Path(path).write_text(json.dumps({"cases": cases}), encoding="utf-8")
    buf = ctypes.create_string_buffer(4 << 20)
    rc = lib().kr_qwen_profile_many(
        str(model_dir).encode(), path.encode(), int(seq), int(device),
        buf, len(buf))
    try:
        payload = json.loads(buf.value.decode())
    except Exception:
        payload = {"error": buf.value.decode(errors="replace")}
    payload["_rc"] = rc
    return payload
