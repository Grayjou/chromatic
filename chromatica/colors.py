from __future__ import annotations
from .color_int import ColorInt, WithAlpha
from .conversions import convert
from typing import Any, ClassVar, Tuple, Union, Type
class ColorRGB(ColorInt):
    """
    RGB color mode with 8-bit per channel.
    """
    mode = "RGB"
    type_ = tuple
    maxima = (255, 255, 255)
    null_value = (0, 0, 0)  # Default RGB null value
    def to_hsv(self) -> ColorHSV:
        """Convert RGB to HSV."""
        h,s,v = self.convert(to_mode='hsv', output_type='int')
        return ColorHSV((h, s, v))

class ColorHSV(ColorInt):
    """
    HSV color mode with 8-bit per channel.
    """
    mode = "HSV"
    type_ = tuple
    maxima = (360, 100, 100)
    null_value = (0, 0, 0)  # Default HSV null value

    def to_rgb(self) -> ColorRGB:
        """Convert HSV to RGB."""
        r,g,b = self.convert(to_mode='rgb', output_type='int')
        return ColorRGB((r, g, b))
    def to_pil_format(self) -> PILColorHSV:
        """Convert HSV to PIL-compatible format."""
        hue, base_sat, base_val = self.value
        # PIL uses 0-255 for saturation and value, and hue in degrees
        sat = int(base_sat * 255 / 100)  # convert to 0-255
        val = int(base_val * 255 / 100)  # convert to 0-255
        return PILColorHSV((hue, sat, val))

class PILColorHSV(ColorHSV):
    """
    PIL-compatible HSV color mode with 8-bit per channel.
    """
    mode = "HSV"
    type_ = tuple
    maxima = (360, 255, 255)  # PIL uses 0-255 for saturation and value
    null_value = (0, 0, 0)  # Default PIL HSV null value

    def to_rgb(self) -> ColorRGB:
        """Convert PIL HSV to RGB."""
        r, g, b = convert(color=self.value,
                          from_space=self.mode.lower(),
                          to_space='rgb',
                          input_type='pilint',
                          output_type='int')
        return ColorRGB((r, g, b))

class ColorRGBA(ColorRGB, WithAlpha):
    """
    RGBA color mode with 8-bit per channel and an alpha channel.
    """
    mode = "RGBA"
    type_ = tuple
    maxima = (255, 255, 255, 255)
    alpha_index = -1  # last element in the values tuple
    alpha_max = 255
    null_value = (0, 0, 0, 0)  # Default RGBA null value with full transparency

class ColorL(ColorInt):
    """
    Grayscale mode with 8-bit per channel.
    """
    mode = "L"
    type_ = int
    maxima = (255,)
    null_value = 0  # Default L null value

class ColorCMYK(ColorInt):
    """
    CMYK color mode with 8-bit per channel.
    """
    mode = "CMYK"
    type_ = tuple
    maxima = (100, 100, 100, 100)
    null_value = (0, 0, 0, 0)  # Default CMYK null value


class ColorHSL(ColorInt):
    """
    HSL color mode with 8-bit per channel.
    """
    mode = "HSL"
    type_ = tuple
    maxima = (360, 100, 100)
    null_value = (0, 0, 0)  # Default HSL null value

    def to_rgb(self) -> ColorRGB:
        """Convert HSL to RGB."""
        r, g, b = convert(color=self.value,
                          from_space=self.mode.lower(),
                          to_space='rgb',
                          input_type='int',
                          output_type='int')
        return ColorRGB((r, g, b))
    
class ColorLA(ColorL, WithAlpha):
    """
    Grayscale mode with an alpha channel.
    """
    mode = "LA"
    type_ = tuple
    maxima = (255, 255)
    alpha_index = -1  # last element in the values tuple
    alpha_max = 255
    null_value = (0, 0)  # Default LA null value with full transparency

class ColorP(ColorInt):
    """
    PIL-compatible color mode with 8-bit per channel.
    """
    mode = "P"
    type_ = tuple
    maxima = (255,)  # Palette index in PIL is 0-255
    null_value = (0,)  # Default P null value, typically the first palette index

class ColorPA(ColorP, WithAlpha):
    """
    PIL-compatible color mode with an alpha channel.
    """
    mode = "PA"
    type_ = tuple
    maxima = (255, 255)  # Palette index + alpha in PIL is 0-255, 0-255
    alpha_index = -1  # last element in the values tuple
    alpha_max = 255
    null_value = (0, 0)  # Default PA null value with full transparency

color_classes: dict[str, Type[ColorInt]] = {
    "RGB": ColorRGB,
    "HSV": ColorHSV,
    "HSL": ColorHSL,
    "CMYK": ColorCMYK,
    "L": ColorL,
    "LA": ColorLA,
    "P": ColorP,
    "PA": ColorPA,
    "RGBA": ColorRGBA,
    "PILHSV": PILColorHSV,
}

ColorMode = Union[ColorRGB, ColorHSV, ColorHSL, ColorCMYK, ColorL, ColorLA, ColorP, ColorPA, ColorRGBA, PILColorHSV]



def color(mode: Union[str, Type[ColorInt]], value: Union[int, tuple[int, ...], ColorInt]) -> ColorInt:
    """
    Create a color object of the specified mode with the given value.
    
    Args:
        mode (str): Color mode (e.g., 'RGB', 'HSV', 'HSL', etc.).
        value (int | tuple[int, ...]): Color value(s) in the specified mode.
    
    Returns:
        ColorInt: An instance of the corresponding color class.
    """
    if isinstance(mode, str):
        cls = color_classes.get(mode.upper())
    elif issubclass(mode, ColorInt):
        cls = mode
    if isinstance(value, ColorInt):
        if value.mode.upper() == mode.upper():
            return value
        else:
            return cls(value.convert(to_mode=mode.upper()))

    if cls is None:
        raise ValueError(f"Unsupported color mode: {mode}")
    return cls(value)  # type: ignore

def get_color_mode(mode: Union[str, Type[ColorInt]]) -> Type[ColorInt]:
    """
    Get the color class for the specified mode.
    
    Args:
        mode (str | Type[ColorInt]): Color mode as a string or a ColorInt subclass.
    
    Returns:
        Type[ColorInt]: The corresponding color class.
    """

    if isinstance(mode, str):
        mode = mode.upper()
        return color_classes.get(mode, ColorInt)
    elif issubclass(mode, ColorInt):
        return mode
    else:
        raise ValueError(f"Unsupported color mode: {mode}")