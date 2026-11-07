"""Sample statistics and checksum tolerance shared by all experiments."""

CHECKSUM_RTOL = 1e-3


def summarize(values) -> dict:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {"n": 0, "min_us": None, "p50_us": None, "p90_us": None,
                "max_us": None, "stalls": 0}
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    p90 = vals[min(n - 1, int(round(0.90 * (n - 1))))]
    floor = max(vals[0], 1e-6)
    return {"n": n, "min_us": vals[0], "p50_us": med, "p90_us": p90,
            "max_us": vals[-1],
            "stalls": sum(1 for v in vals if v > 3.0 * floor)}


def checksum_consistent(a, b, rtol: float = CHECKSUM_RTOL) -> bool:
    if a is None or b is None:
        return False
    scale = max(abs(float(a)), abs(float(b)), 1e-6)
    return abs(float(a) - float(b)) / scale < rtol


def checksum_spread(checksums) -> float:
    vals = [float(c) for c in checksums if c is not None]
    if len(vals) < 2:
        return 0.0
    lo, hi = min(vals), max(vals)
    return abs(hi - lo) / max(abs(lo), abs(hi), 1e-6)


def speedup(base_us, cand_us):
    if not base_us or not cand_us:
        return None
    return base_us / cand_us
