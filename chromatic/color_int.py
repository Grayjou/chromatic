from __future__ import annotations
from abc import ABC
from typing import Any, ClassVar, Tuple, Union
from .conversions import convert
Scalar = int
Vector = Tuple[int, ...]
RawColor = Union[Scalar, Vector, 'ColorInt']

class ColorInt(ABC):
    # subclass must override these three:
    mode:     ClassVar[str]
    type_:    ClassVar[type]
    maxima:   ClassVar[Union[int, Tuple[int, ...]]]
    null_value: ClassVar[Union[int, Tuple[int, ...]]] = 0
    
    
    def __init__(self, value: RawColor) -> None:
        # 1) if they passed in another ColorInt, auto-convert:
        if isinstance(value, ColorInt):
            if value.mode == self.mode:
                value = value.value
            else:
                conv = getattr(value, f"to_{self.mode.lower()}", None)
                if not conv:
                    raise TypeError(f"Can't convert {value.mode} → {self.mode}")
                value = conv().value   # call it and grab its `.value`
        # 2) clamp against maxima
        elif isinstance(value, tuple):
            if not isinstance(self.maxima, tuple) or len(value) != len(self.maxima):
                raise ValueError(f"{self.mode} expects {self.maxima!r}-shaped tuple")
            value = tuple(max(0, min(v, m)) for v, m in zip(value, self.maxima))
        elif isinstance(value, int):
            # scalar
            if not isinstance(self.maxima, int):
                raise ValueError(f"{self.mode} expects a {len(self.maxima)}-tuple, not a scalar")
            value = max(0, min(value, self.maxima))

        # 3) final type check
        if not isinstance(value, self.type_):
            raise TypeError(f"{self.mode} got {type(value).__name__}, expected {self.type_.__name__}")

        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and other.value == self.value
    
    def __hash__(self) -> int:
        return hash((self.__class__, self.value))
    
    def __len__(self) -> int:
        """Number of channels."""
        if self.type_ is tuple:
            return len(self.value)
        return 1
    
    @property
    def values(self) -> Tuple[int, ...]:
        if self.type_ is tuple:
            return self.value
        return (self.value,)
    @property
    def unit_values(self) -> Tuple[float, ...]:
        """Normalized [0.0–1.0] per channel."""
        if isinstance(self.maxima, tuple):
            return tuple(v / m for v, m in zip(self.values, self.maxima))
        return (self.value / self.maxima,)
    def convert(self, to_mode: str, output_type: str = 'int') -> tuple[int, ...]:
        """Convert to another color mode."""
        if to_mode.lower() == self.mode.lower():
            return self
        conv = convert(color = self.value,
                       from_space= self.mode.lower(),
                          to_space=   to_mode.lower(),
                        to_mode=output_type)
        return conv

class WithAlpha(ABC):
    """Mixin for a 3-channel ColorInt subclass to add an opaque alpha channel."""
    alpha_index: ClassVar[int]  = -1  # last element in the values tuple
    alpha_max:   ClassVar[int]  = 255

    @property
    def alpha(self) -> int:
        return self.values[self.alpha_index]

    @classmethod
    def _validate_alpha_shape(cls, vals: Tuple[int, ...]) -> None:
        if len(vals) != len(cls.maxima):
            raise ValueError(f"{cls.mode} expects {cls.maxima!r}-tuple")

    def with_alpha(self, alpha: int) -> ColorInt:
        """Force a new instance with a specific alpha."""
        base_vals = self.values[:-1]
        return self.__class__(tuple(base_vals) + (alpha,))