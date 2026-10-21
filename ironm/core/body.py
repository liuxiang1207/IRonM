"""身体测量的通用换算。"""

from decimal import ROUND_HALF_UP, Decimal


def bmi(height_cm: float, weight_kg: float) -> float:
    """BMI，按一位小数四舍五入（用十进制计算，避免二进制小数在界值处出错）。"""
    metres = Decimal(str(height_cm)) / 100
    value = Decimal(str(weight_kg)) / (metres * metres)
    return float(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
