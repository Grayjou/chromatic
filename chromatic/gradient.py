from __future__ import annotations
from .color_int import ColorInt
from .colors import color_classes, color, ColorMode
import numpy as np
from numpy import ndarray as NDArray
from typing import Any, ClassVar, Tuple, Union, Callable, Optional, Type, List
from .color_arr import Color1DArr, ColorFormat, Color2DArr

class Gradient1D(Color1DArr):
    """
    Represents a 1D gradient of colors.
    """
    # Inherits repeat method from Color1DArr
    # Inherits __array__, convert method and several ndarray properties from ColorArr
    # Inherits channels and channel_num properties from ColorArr

    def __init__(self, colors: NDArray, color_mode: Union[ColorMode, str] = 'RGB', color_format: ColorFormat = ColorFormat.INT) -> None:
        """
        Initialize a 1D gradient with a list of colors.

        :param colors: A numpy array of shape (n, channels) where n is the number of colors.
        :param color_mode: The color mode to use (e.g., 'RGB', 'HSV').
        :param color_format: The format of the colors (e.g., INT, PIL_INT, UNIT_FLOAT).
        """
        super().__init__(colors, color_mode=color_mode, color_format=color_format)

    @classmethod
    def from_colors(
        cls,
        color1 : Union[ColorInt, Tuple[int, ...], int],
        color2 : Union[ColorInt, Tuple[int, ...], int],
        steps: int,
        color_mode: Optional[Union[ColorMode, str]] = None,
        unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
        color_format: ColorFormat = ColorFormat.INT
    ) -> 'Gradient1D':
        """
        Create a 1D gradient from two colors.

        :param color1: The first color (ColorInt or color-like).
        :param color2: The second color (ColorInt or color-like).
        :param steps: Number of steps in the gradient.
        :param mode: Optional color mode (if not provided, inferred).
        :param unit_transform: Optional function to transform the interpolation parameter.
        :param format: The format of the resulting colors.
        :return: A Gradient1D object.
        """
        # Infer color_mode if possible
        inferred_mode = color_mode
        if isinstance(color1, ColorInt) and isinstance(color2, ColorInt):
            if color1.mode != color2.mode:
                raise ValueError(f"Colors must be in the same mode, got {color1.mode} and {color2.mode}")
            inferred_mode = inferred_mode or color1.mode
        elif isinstance(color1, ColorInt):
            inferred_mode = inferred_mode or color1.mode
        elif isinstance(color2, ColorInt):
            inferred_mode = inferred_mode or color2.mode
        else:
            inferred_mode = inferred_mode or 'RGB'

        if inferred_mode not in color_classes:
            raise ValueError(f"Unsupported color mode: {inferred_mode}. Must be one of {list(color_classes.keys())}")

        # Convert to ColorInt instances in the same mode
        color1 = color(inferred_mode, color1)
        color2 = color(inferred_mode, color2)

        # Get numpy arrays of the color values
        start_color = np.array(color1.value, dtype=float)
        end_color = np.array(color2.value, dtype=float)

        # Interpolation parameter
        unit_array = np.linspace(0, 1, steps, dtype=float)[:, np.newaxis]
        if unit_transform is not None:
            unit_array = unit_transform(unit_array)

        # Linear interpolation
        colors = start_color * (1 - unit_array) + end_color * unit_array

        if format == ColorFormat.INT:
            colors = colors.astype(int)

        return cls(colors, color_mode=inferred_mode, color_format=color_format)
    def wrap_around(
            self,
            width: int,
            height: int,
            center: Union[tuple[int, int], List[int]] = (0, 0),
            angle_start: float = 0.0,
            angle_end: float = 2 * np.pi,
            unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
            outside_fill: Optional[Union[ColorInt, tuple[int, ...], int]] = None,
            radius_offset: float = 0.0,
            base: Optional[NDArray] = None
    ):
        return super().wrap_around(
            width=width,
            height=height,
            center=center,
            angle_start=angle_start,
            angle_end=angle_end,
            unit_transform=unit_transform,
            outside_fill=outside_fill,
            radius_offset=radius_offset,
            base=base
        )

class Gradient2D(Color2DArr):
    """
    Represents a 2D gradient of colors.
    
    This class allows you to create a gradient from four corner colors and access
    individual rows as Gradient1D objects.
    """
    
    # Inherits convert, resize, and repeat methods from Color2DArr
    # Inherits __array__ method and several properties from ArrMixin

    def __init__(self, colors: NDArray, color_mode: Union[ColorMode, str] = 'RGB', color_format: ColorFormat = ColorFormat.INT) -> None:
        """
        Initialize a 2D gradient with a numpy array of colors.

        :param colors: A numpy array of shape (rows, columns, channels).
        :param color_mode: The color mode to use (e.g., 'RGB', 'HSV').
        :param color_format: The format of the colors (e.g., INT, PIL_INT, UNIT_FLOAT).
        """
        super().__init__(colors, color_format=color_format, color_mode = color_mode)

    @classmethod
    def from_colors(
        cls,
        color_tl: Union[ColorInt, tuple[int, ...], int],
        color_tr: Union[ColorInt, tuple[int, ...], int],
        color_bl: Union[ColorInt, tuple[int, ...], int],
        color_br: Union[ColorInt, tuple[int, ...], int],
        width: int,
        height: int,
        color_mode: Union[ColorMode, str] = "RGB",
        color_format: ColorFormat = ColorFormat.INT,
        unit_transform_x: Optional[Callable[[NDArray], NDArray]] = None,
        unit_transform_y: Optional[Callable[[NDArray], NDArray]] = None
    ) -> Gradient2D:
        """
        Create a 2D gradient from four corner colors, with optional unit transforms
        on the x and y axes.

        :param color_tl: Top-left color
        :param color_tr: Top-right color
        :param color_bl: Bottom-left color
        :param color_br: Bottom-right color
        :param width: number of columns
        :param height: number of rows
        :param mode: color mode (optional, default: inferred or RGB)
        :param unit_transform_x: optional transformation of x interpolation factors
        :param unit_transform_y: optional transformation of y interpolation factors
        :return: Gradient2D object
        """

        # Infer color_mode if possible
        inferred_mode = color_mode
        if inferred_mode is None:
            inferred_modes = {
                getattr(color, 'mode', None) for color in (color_tl, color_tr, color_bl, color_br) if isinstance(color, ColorInt)
            }
            if not inferred_modes:
                inferred_mode = 'RGB'
            elif len(inferred_modes) > 1:
                raise ValueError(f"Colors must be in the same mode, got {inferred_modes}")

        color_tl = color(inferred_mode, color_tl)
        color_tr = color(inferred_mode, color_tr)
        color_bl = color(inferred_mode, color_bl)
        color_br = color(inferred_mode, color_br)

        tl = np.array(color_tl.value, dtype=float)
        tr = np.array(color_tr.value, dtype=float)
        bl = np.array(color_bl.value, dtype=float)
        br = np.array(color_br.value, dtype=float)

        # Create interpolation factors
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)

        if unit_transform_x is not None:
            x = unit_transform_x(x)
        if unit_transform_y is not None:
            y = unit_transform_y(y)

        xx, yy = np.meshgrid(x, y)

        # Compute each pixel value as bilinear interpolation
        colors = (
            (1 - xx)[:, :, None] * (1 - yy)[:, :, None] * tl +
            xx[:, :, None] * (1 - yy)[:, :, None] * tr +
            (1 - xx)[:, :, None] * yy[:, :, None] * bl +
            xx[:, :, None] * yy[:, :, None] * br
        )

        colors = np.round(colors).astype(int)

        return cls(colors, color_mode=color_mode, color_format=color_format)

def radial_gradient(
        color1: Union['ColorInt', Tuple[int, ...], int],
        color2: Union['ColorInt', Tuple[int, ...], int],
        height: int,
        width: int,
        center: Union[tuple[int, int], List[int]] = (0, 0),
        radius: float = 1.0,
        color_mode: Union['ColorMode', str] = 'RGB',
        unit_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        outside_fill: Optional['ColorInt'] = None,
        start: float = 0.0,
        end: float = 1.0,
        offset: float = 0.0,
        base: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Create a radial gradient.
    
    If `base` is provided and `outside_fill` is None, the gradient overwrites `base` only 
    within the gradient area, leaving the rest of `base` untouched.
    If `outside_fill` is provided, it fills areas outside the gradient.
    """
    outside_fill_color = color(color_mode, outside_fill).value if outside_fill else None

    y, x = np.indices((height, width), dtype=float)
    cx, cy = center
    dx = x - cx
    dy = y - cy
    distance = np.sqrt(dx**2 + dy**2)

    # Normalize distances to [0, 1]
    unit_array = (distance / radius) - offset

    if unit_transform is not None:
        unit_array = np.where(
            (unit_array > 1.0) | (unit_array < 0.0),
            unit_array,
            unit_transform(unit_array)
        )

    # Clip to [0, 1] for interpolation
    unit_array_clipped = np.clip(unit_array, 0.0, 1.0)

    # Interpolate colors
    col1 = color(color_mode, color1).value
    col2 = color(color_mode, color2).value

    gradient = (
        col1 * (1 - unit_array_clipped[..., None]) +
        col2 * unit_array_clipped[..., None]
    )

    # Initialize result array
    if base is not None:
        if base.shape != gradient.shape:
            raise ValueError(f"`base` shape {base.shape} does not match gradient shape {gradient.shape}")
        result = base.copy()
    else:
        result = gradient.copy()

    # Compute masks
    mask_inside = (
        (unit_array >= start) & (unit_array <= end) & (unit_array >= 0.0) & (unit_array <= 1.0)
    )
    mask_outside = ~mask_inside

    if outside_fill_color is not None:
        # Fill outside gradient with outside_fill_color
        result[mask_outside] = outside_fill_color
    else:
        if base is not None:
            # Only overwrite inside the gradient, keep base outside
            result[mask_inside] = gradient[mask_inside]
        else:
            # No base, no outside_fill → just use gradient everywhere
            result = gradient

    return result

    
def example(output_path=None):
    from PIL import Image
    grad2d = Gradient2D.from_colors(
    color_tl=(255, 0, 255),       # top-left: pink
    color_tr=(255, 255, 0),       # top-right: yellow
    color_bl=(255, 0, 128),       # bottom-left: deep pink
    color_br=(255, 128, 0),       # bottom-right: orange
    width=500,
    height=500
    )
    img = Image.fromarray(grad2d.colors.astype(np.uint8), mode='RGB')
    if output_path:
        img.save(output_path)
    img.show()


def example_2d_gradient(output_path=None):
    from PIL import Image
    grad2d = Gradient2D.from_colors(
        color_tl=(255, 0, 255),
        color_tr=(255, 255, 0),
        color_bl=(255, 0, 128),
        color_br=(255, 128, 0),
        width=500,
        height=500,
        unit_transform_x=lambda x: (1 - np.cos(4*x * np.pi)) / 2,  # sinusoidal easing horizontally
        unit_transform_y=lambda y: (1 - np.cos(4*y * np.pi)) / 2  # quartic vertically
    )
    img = Image.fromarray(grad2d.colors.astype(np.uint8), mode='RGB')
    if output_path:
        img.save(output_path)
    img.show()

def example_radial_gradient(output_path=None):
    from PIL import Image
    def extreme_ease_in(x: float) -> float:
        func = lambda x: 1 - np.sqrt(np.abs(1 - x**2))
        return func(func(x))
    gradient_base = np.full((500, 500, 4), (150, 255, 255, 255), dtype=np.uint8)  # Base color with alpha
    radial_arr = radial_gradient(
        color1=(0,0,0, 255),
        color2=(255, 0, 180, 0),  # Light pink with transparency
        height=500,
        width=500,
        center=(250, 250),  # Center of the image
        radius=125,  # Radius of the gradient
        unit_transform=extreme_ease_in,  # Apply extreme ease-in transformation
        outside_fill=None,
        color_mode='RGBA',  # Use RGBA for transparency
        start=0.0,  # Start of the gradient
        end=1.0,  # End of the gradient
        offset=1.0,  # Offset to ensure the gradient starts at the edge
        base=gradient_base  # Use the base color for areas outside the gradient
    )
    radial_arr = radial_gradient(
        color1 = (0, 0, 255, 0),  # Blue with full transparency
        color2 = (0, 0, 0, 255),  # Black with full opacity
        height=500,
        width=500,
        center=(250, 250),  # Center of the image
        radius=125,  # Radius of the gradient
        unit_transform=None,  # Apply extreme ease-in transformation
        outside_fill=None,  # No fill outside the gradient
        color_mode='RGBA',  # Use RGBA for transparency
        start=0.0,  # Start of the gradient
        end=1.0,  # End of the gradient
        offset=0.0,
        base=radial_arr  # Use the previous gradient as base
    )
    base = Image.new('RGBA', (500, 500), (150, 255, 255, 255))
    img = Image.fromarray(radial_arr.astype(np.uint8), mode='RGBA')
    base.paste(img, (0, 0), img)  # Paste with transparency
    if output_path:
        base.save(output_path)
    base.show()

def example_gradient_rotate(output_path=None):
    from PIL import Image
    grad1d = Gradient1D.from_colors(
        color1=(255, 0, 0),  # Red
        color2=(0, 0, 255),  # Blue
        steps=125,
        color_mode='RGB'
    )
    rotated_grad = grad1d.wrap_around(
        width=500,
        height=500,
        center=(250, 250),
        angle_start=0.0,
        angle_end=2 * np.pi,
        unit_transform=lambda x: (1 - np.cos(8*x * np.pi)) / 2,  # Quadratic easing
        outside_fill=(255, 255, 255),  # White outside the gradient
        radius_offset=0
    )
    img = Image.fromarray(rotated_grad.astype(np.uint8), mode='RGB')
    if output_path:
        img.save(output_path)
    img.show()

def example_arr_rotate(output_path=None):
    from PIL import Image
    arr_1d = Color1DArr(
        np.random.randint(125, 256, (125, 3), dtype=np.uint8),
        color_mode='RGB',
    )
    rotated_grad = arr_1d.wrap_around(
        width=500,
        height=500,
        center=(250, 250),
        angle_start=0.0,
        angle_end=2 * np.pi,
        unit_transform=lambda x: (1 - np.cos(8*x * np.pi)) / 2,  # Quadratic easing
        outside_fill=(255, 255, 255),  # White outside the gradient
        radius_offset=0
    )
    img = Image.fromarray(rotated_grad.astype(np.uint8), mode='RGB')
    if output_path:
        img.save(output_path)
    img.show()