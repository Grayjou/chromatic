from .to_hsv import hsl_to_hsv, unit_rgb_to_hsv, np_hsl_to_hsv, np_rgb_to_hsv
from .to_hsl import hsv_to_hsl, unit_rgb_to_hsl, np_rgb_to_hsl, np_hsv_to_hsl
from .to_rgb import hsv_to_unit_rgb, hsl_to_unit_rgb, np_hsv_to_rgb, np_hsl_to_rgb
import numpy as np
from typing import Literal


def _normalize_rgb(rgb: tuple[int, int, int], input_type: str) -> tuple[float, float, float]:
    if input_type in ('int', 'pilint'):
        return tuple(c / 255.0 for c in rgb)
    return rgb


def _normalize_hsv(hsv: tuple[int, int, int], input_type: str) -> tuple[float, float, float]:
    h, s, v = hsv
    h = float(h)
    if input_type == 'pilint':
        return h, s / 255.0, v / 255.0
    elif input_type == 'int':
        return h, s / 100.0, v / 100.0
    return hsv


def _normalize_hsl(hsl: tuple[int, int, int], input_type: str) -> tuple[float, float, float]:
    h, s, l = hsl
    h = float(h)
    if input_type in ('int', 'pilint'):
        return h, s / 100.0, l / 100.0
    return hsl


def _format_output(
    color: tuple[float, ...],
    to_space: str,
    output_type: Literal['int', 'float', 'pilint']
) -> tuple[int, int, int] | tuple[float, float, float]:
    """
    Format output color tuple depending on target color space and desired output type.
    """
    if to_space == 'rgb':
        if output_type in ('int', 'pilint'):
            return tuple(round(c * 255) for c in color)
        return color

    elif to_space in ('hsv', 'hsl'):
        h, s, v = color
        if output_type == 'pilint':
            return (round(h), round(s * 255), round(v * 255))  # PIL format
        elif output_type == 'int':
            if to_space == 'hsv':
                return (round(h), round(s * 100), round(v * 100))
            else:  # HSL
                return (round(h), round(s * 100), round(v * 100))
        return color

    return color


def convert(
    color: tuple[int, int, int] | tuple[float, float, float],
    from_space: Literal['rgb', 'hsv', 'hsl'],
    to_space: Literal['rgb', 'hsv', 'hsl'],
    input_type: Literal['int', 'float', 'pil_int'] = 'int',
    output_type: Literal['int', 'float', 'pil_int'] = 'int'
) -> tuple:
    """
    Convert color values between RGB, HSV, and HSL color spaces.
    Supports various input/output formats including 'pil_int'.

    Args:
        color (tuple[int, int, int] | tuple[float, float, float]): The color to convert.
        from_space (str): The input color space ('rgb', 'hsv', 'hsl').
        to_space (str): The output color space ('rgb', 'hsv', 'hsl').
        input_type (str): Input value format: 'int', 'float', or 'pil_int'.
        output_type (str): Output value format: 'int', 'float', or 'pil_int'.

    Returns:
        tuple: The converted color in the desired space and format.
    """
    from_space = from_space.lower()
    to_space = to_space.lower()
    input_type = input_type.lower().replace('_', '')
    output_type = output_type.lower().replace('_', '')
    # Step 1: Normalize input to floats
    if from_space == 'rgb':
        color_f = _normalize_rgb(color, input_type)
    elif from_space == 'hsv':
        color_f = _normalize_hsv(color, input_type)
    elif from_space == 'hsl':
        color_f = _normalize_hsl(color, input_type)
    else:
        raise ValueError(f"Unknown from_space: {from_space}")

    # Step 2: Perform the color space conversion
    if from_space == 'rgb' and to_space == 'hsv':
        converted = unit_rgb_to_hsv(*color_f)
    elif from_space == 'hsv' and to_space == 'rgb':
        converted = hsv_to_unit_rgb(*color_f)
    elif from_space == 'hsl' and to_space == 'rgb':
        converted = hsl_to_unit_rgb(*color_f)
    elif from_space == 'rgb' and to_space == 'hsl':
        converted = unit_rgb_to_hsl(*color_f)
    elif from_space == 'hsv' and to_space == 'hsl':
        converted = hsv_to_hsl(*color_f)
    elif from_space == 'hsl' and to_space == 'hsv':
        converted = hsl_to_hsv(*color_f)
    elif from_space == to_space:
        converted = color_f
    else:
        raise ValueError(f"Conversion from {from_space} to {to_space} is not supported.")

    # Step 3: Format output as requested
    return _format_output(converted, to_space, output_type)


def np_convert(
    color: np.ndarray,
    from_space: Literal['rgb', 'hsv', 'hsl'],
    to_space: Literal['rgb', 'hsv', 'hsl'],
    input_type: Literal['int', 'float', 'pil_int'] = 'int',
    output_type: Literal['int', 'float', 'pil_int'] = 'int'
) -> np.ndarray:
    """
    Vectorized color space conversion for numpy arrays, respecting np_* names.

    Args:
        color: np.ndarray of shape (..., 3)
        from_space: 'rgb', 'hsv', 'hsl'
        to_space: 'rgb', 'hsv', 'hsl'
        input_type: 'int', 'float', 'pil_int'
        output_type: 'int', 'float', 'pil_int'

    Returns:
        np.ndarray of shape (..., 3)
    """
    from_space = from_space.lower()
    to_space = to_space.lower()
    input_type = input_type.lower().replace('_', '')
    output_type = output_type.lower().replace('_', '')

    color = np.asarray(color)

    if color.shape[-1] != 3:
        raise ValueError(f"Input color array must have shape (...,3), got {color.shape}")

    # Step 1: normalize input
    if from_space == 'rgb':
        if input_type in ('int', 'pilint'):
            color_f = color / 255.0
        else:
            color_f = color
    elif from_space == 'hsv':
        h, s, v = color[...,0], color[...,1], color[...,2]
        if input_type == 'pilint':
            s /= 255.0
            v /= 255.0
            h = h.astype(float)
        elif input_type == 'int':
            s /= 100.0
            v /= 100.0
            h = h.astype(float)
        color_f = np.stack([h, s, v], axis=-1)
    elif from_space == 'hsl':
        h, s, l = color[...,0], color[...,1], color[...,2]
        if input_type in ('int', 'pilint'):
            s /= 100.0
            l /= 100.0
            h = h.astype(float)
        color_f = np.stack([h, s, l], axis=-1)
    else:
        raise ValueError(f"Unknown from_space: {from_space}")

    # Step 2: convert
    if from_space == 'rgb' and to_space == 'hsv':
        converted = np_rgb_to_hsv(color_f[...,0], color_f[...,1], color_f[...,2])
    elif from_space == 'hsv' and to_space == 'rgb':
        converted = np_hsv_to_rgb(color_f[...,0], color_f[...,1], color_f[...,2])
    elif from_space == 'hsl' and to_space == 'rgb':
        converted = np_hsl_to_rgb(color_f[...,0], color_f[...,1], color_f[...,2])
    elif from_space == 'rgb' and to_space == 'hsl':
        converted = np_rgb_to_hsl(color_f[...,0], color_f[...,1], color_f[...,2])
    elif from_space == 'hsv' and to_space == 'hsl':
        converted = np_hsv_to_hsl(color_f[...,0], color_f[...,1], color_f[...,2])
    elif from_space == 'hsl' and to_space == 'hsv':
        converted = np_hsl_to_hsv(color_f[...,0], color_f[...,1], color_f[...,2])
    elif from_space == to_space:
        converted = color_f
    else:
        raise ValueError(f"Conversion from {from_space} to {to_space} is not supported.")

    # Step 3: format output
    if to_space == 'rgb':
        if output_type in ('int', 'pilint'):
            converted = np.round(converted * 255).astype(int)
    elif to_space in ('hsv', 'hsl'):
        h, s, v = converted[...,0], converted[...,1], converted[...,2]
        if output_type == 'pilint':
            s = np.round(s * 255).astype(int)
            v = np.round(v * 255).astype(int)
            h = np.round(h).astype(int)
        elif output_type == 'int':
            s = np.round(s * 100).astype(int)
            v = np.round(v * 100).astype(int)
            h = np.round(h).astype(int)
        converted = np.stack([h, s, v], axis=-1)

    return converted