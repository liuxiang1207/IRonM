"""统计工具：t 分布分位数（不引入 scipy；用正则化不完全贝塔函数求 t 分布函数，再二分求逆）、
只对已知值求平均。"""

import math
from collections.abc import Iterable

from ironm.contracts.unknown import Unknown, UnknownReason


def _beta_cf(a: float, b: float, x: float) -> float:
    """不完全贝塔函数的连分式（修正 Lentz 法）。"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ArithmeticError(f"不完全贝塔函数未收敛：a={a}, b={b}, x={x}")


def _beta_inc(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    tail = 0.5 * _beta_inc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: int) -> float:
    """自由度 df 的 t 分布的 p 分位数。"""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p 必须在 (0, 1) 内：{p}")
    if df < 1:
        raise ValueError(f"自由度必须 ≥ 1：{df}")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean_known(values: Iterable[float | Unknown]) -> float | Unknown:
    """只对已知值求平均；“未知”不计入。全部未知时结果为“未知（数据不足）”，不算作 0。"""
    known = [v for v in values if not isinstance(v, Unknown)]
    if not known:
        return Unknown(unknown=UnknownReason.INSUFFICIENT_DATA)
    return sum(known) / len(known)
