
from typing import Union

Number = Union[int, float]

def clamp(value, min_value, max_value):
    """Clamp a value between min_value and max_value."""
    return max(min(value, max_value), min_value)


class UnitFloat(float):
    def __new__(cls, value):
        if not 0.0 <= value <= 1.0:
            value = clamp(value, 0.0, 1.0)
        return super().__new__(cls, value)