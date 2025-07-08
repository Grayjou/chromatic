from numpy import ndarray as NDArray
from .colors import ColorMode, get_color_mode, color
import numpy as np
from enum import Enum
from .conversions import np_convert
from typing import Type, List, Union, Optional, Callable, Tuple
from .color_int import ColorInt



class ColorFormat(Enum):
    """Enum for color formats used in Color1DArr and Color2DArr.
    This enum defines the formats in which colors can be represented.
    - INT: Integer format (0-255 for each channel).
    - PIL_INT: PIL-compatible integer format (0-255 for each channel but Hue).
    - UNIT_FLOAT: Normalized float format (0.0-1.0 for each channel but Hue).
    """
    INT = "int"
    PIL_INT = "pilint"
    UNIT_FLOAT = "float"

class ColorArr:
    def __init__(self, colors: NDArray, color_mode: Union[ColorMode, str], color_format: ColorFormat = ColorFormat.INT) -> None:
        self.colors: NDArray = colors
        self.color_mode: Type[ColorMode] = get_color_mode(color_mode)
        self.mode: str = self.color_mode.mode.upper()
        self.format: ColorFormat = color_format
    def __array__(self) -> NDArray:
        """
        Convert the color array to a numpy array.
        
        :return: A numpy array of shape (n, channels) or (h, w, channels).
        """
        return self.colors
    def convert(self, to_mode: str, output_type: str = 'int') -> NDArray:
        """
        Convert the color array to another color mode.

        Args:
            to_mode (str): Target color mode (e.g., 'RGB', 'HSV', etc.).
            output_type (str): Output type ('int', 'pilint', 'unitfloat', 'hueunitfloat').

        Returns:
            NDArray: Converted color array.
        """
        return np_convert(
            self.colors,
            from_space=self.mode.lower(),
            to_space=to_mode.lower(),
            input_type=self.format.value,
            output_type=output_type
        )
    @property
    def channel_num(self) -> int:
        return len(self.color_mode.maxima)

    @property
    def channels(self) -> List[NDArray]:
        if self.channel_num == 1 or self.colors.ndim < 3:
            return [self.colors]
        return [self.colors[..., i] for i in range(self.channel_num)]
    
    def unit_array(self, normalize_hue: bool = False) -> NDArray:

        channels = self.channels
        maxima = self.color_mode.maxima
        
        if len(channels) == 1:
            return self.colors/maxima[0]
        else:
            do_normalize_hue = normalize_hue and self.mode.upper().startswith('H')
            if not do_normalize_hue:
                hue = channels[0]
                unit_values = [c/max for c, max in zip(channels[1:], maxima[1:])]
                unit_values.insert(0, hue)  # keep hue as is
                return np.stack(unit_values, axis=-1)
            else:

                unit_values = [c/max for c, max in zip(channels, maxima)]
                return np.stack(unit_values, axis=-1)
_ndarray_props = ['shape', 'dtype', 'ndim', 'size', 'itemsize', 'nbytes']
for _prop in _ndarray_props:
    setattr(
        ColorArr,
        _prop,
        property(lambda self, p=_prop: getattr(self.colors, p))
    )

class Color1DArr(ColorArr):
    def __init__(self, colors: NDArray, color_mode: Union[ColorMode, str], color_format: ColorFormat = ColorFormat.INT):
        super().__init__(colors, color_mode, color_format)

        assert isinstance(self.colors, NDArray), "colors must be a numpy array"

        if len(self.color_mode.maxima) > 1:
            # Multi-channel: expect shape (N, C)
            assert self.colors.ndim >= 2, "multi-channel colors must be at least 2D"
            assert self.colors.shape[-1] == len(self.color_mode.maxima), \
                f"expected last dimension to be {len(self.color_mode.maxima)} channels, got {self.colors.shape[-1]}"
        else:
            # Single-channel: expect shape (N,)
            assert self.colors.ndim == 1, "single-channel colors must be 1D"

    def repeat(self, horizontally: float = 1.0, vertically: int = 1) -> NDArray:
        """
        Repeat the 1D color array horizontally and vertically.

        Horizontally can be a float — repeats the array fully `int(horizontally)` times
        and appends a proportional fraction of the array.

        Args:
            horizontally (float): How many times to repeat horizontally.
            vertically (int): How many times to stack vertically.

        Returns:
            NDArray: Repeated 2D color array.
        """
        if horizontally <= 0:
            raise ValueError("horizontally must be > 0")
        if vertically <= 0:
            raise ValueError("vertically must be > 0")

        n = self.colors.shape[0]
        full_repeats = int(horizontally)
        partial_fraction = horizontally - full_repeats
        partial_count = int(round(partial_fraction * n))

        # Build one row
        row = np.concatenate(
            [self.colors] * full_repeats + ([self.colors[:partial_count]] if partial_count > 0 else [])
        )

        # Stack vertically
        result = np.tile(row, (vertically, 1))
        return result
    # --------------------------------------------------------------------- #
    # 2-D “angular” projection – wrap this 1-D gradient around a point
    # --------------------------------------------------------------------- #
    def wrap_around(
        self,
        width: int,
        height: int,
        center: Union[tuple[int, int], List[int]] = (0, 0),
        *,
        angle_start: float = 0.0,
        angle_end: float = 2 * np.pi,
        unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
        outside_fill: Optional[Union[ColorInt, tuple[int, ...], int]] = None,
        radius_offset: float = 0.0,
        base: Optional[NDArray] = None
    ) -> NDArray:
        """
        Wrap the 1-D gradient around a centre point, producing a 2-D
        (height, width, channels) array whose colour varies with *angle*.

        Args:
            width, height : size of the output image.
            center        : (cx, cy) – origin in pixel coordinates.
            angle_start   : angle (rad) mapped to the first entry of the
                            1-D gradient.
            angle_end     : angle (rad) mapped to the last entry of the
                            1-D gradient.  If angle_end-angle_start == 2π
                            the gradient completes a full circle.
            unit_transform: optional easing function applied to the
                            normalised angle before colour lookup.
            outside_fill  : colour used for pixels whose angle lies
                            outside [angle_start, angle_end).  If None
                            and *base* is given, those pixels are left
                            unchanged.
            radius_offset : minimum distance from *center* that must be
                            exceeded for a pixel to be considered
                            “inside” the gradient.  (Useful when layering
                            several angular gradients.)
            base          : optional image to paint *onto*; must have the
                            same shape and dtype as the output.

        Returns:
            (H, W, C) NumPy array in the same dtype as `self.colors`.
        """

        # ------------------------------------------------------------------
        # set-up          ---------------------------------------------------
        # ------------------------------------------------------------------
        n_steps, n_channels = self.colors.shape

        # output container – start from *base* or zero-array
        if base is not None:
            if base.shape != (height, width, n_channels):
                raise ValueError(
                    f"`base` shape {base.shape} does not match "
                    f"expected {(height, width, n_channels)}")
            out = base.copy()
        else:
            out = np.zeros((height, width, n_channels),
                           dtype=self.colors.dtype)

        # fallback colour for “outside” pixels
        if outside_fill is not None:
            outside_colour = np.asarray(
                color(self.color_mode, outside_fill).value,
                dtype=self.colors.dtype)
        else:
            outside_colour = None

        # ------------------------------------------------------------------
        # compute angle of each pixel relative to centre                    -
        # ------------------------------------------------------------------
        yy, xx = np.indices((height, width), dtype=float)
        cx, cy = center
        dx, dy = xx - cx, yy - cy
        distance = np.hypot(dx, dy)

        angle = np.arctan2(dy, dx)              # [-π, π]
        angle = (angle + 2 * np.pi) % (2 * np.pi)  # → [0, 2π)

        # map angle into the gradient’s [0, 1] domain
        span = (angle_end - angle_start) % (2 * np.pi)
        # if span == 0 we treat it as a full turn
        span = 2 * np.pi if span == 0 else span
        unit = ((angle - angle_start) % (2 * np.pi)) / span

        # mask: pixels whose angle lies inside the active arc *and*
        # whose radius is beyond `radius_offset`
        inside_mask = (unit <= 1.0) & (distance >= radius_offset)

        # optional easing / warping
        if unit_transform is not None:
            unit = np.where(inside_mask, unit_transform(unit), unit)

        # ------------------------------------------------------------------
        # look-up / interpolate colour for every pixel inside_mask          -
        # ------------------------------------------------------------------
        # continuous index in 1-D gradient
        idx_f = unit * (n_steps - 1)
        idx0 = np.floor(idx_f).astype(int).clip(0, n_steps - 1)
        idx1 = np.clip(idx0 + 1, 0, n_steps - 1)
        t = (idx_f - idx0)[..., None]                       # shape (..., 1)

        # linear interpolation between neighbouring colour steps
        col = (1 - t) * self.colors[idx0] + t * self.colors[idx1]

        # ------------------------------------------------------------------
        # write colours into the output                                     -
        # ------------------------------------------------------------------
        out[inside_mask] = col[inside_mask]

        # pixels not inside_mask
        outside_mask = ~inside_mask
        if outside_colour is not None:
            out[outside_mask] = outside_colour
        # else: if base was supplied we already kept its pixels
        #       if base was None they remain zeros → caller can ignore
        return out
    # ------------------------------------------------------------------ #
    # 2-D radial projection – cast this 1-D gradient outward from centre #
    # ------------------------------------------------------------------ #
    def rotate_around(
        self,
        width: int,
        height: int,
        center: Union[tuple[int, int], List[int]] = (0, 0),
        *,
        angle_start: float = 0.0,              # still honoured – lets you
        angle_end: float = 2 * np.pi,          # restrict the radial segment
        unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
        outside_fill: Optional[Union[ColorInt, tuple[int, ...], int]] = None,
        radius_offset: float = 0.0,            # inner “hole” radius
        base: Optional[NDArray] = None
    ) -> NDArray:
        """
        Cast the 1-D gradient radially: colour now varies with *radius* from
        `center`.  Angles may still be limited via `angle_start/angle_end`.

        Parameters are intentionally identical to `wrap_around` so you can
        swap the two calls with no code changes.
        """
        n_steps, n_channels = self.colors.shape

        # ---------- prepare output ------------------------------------------------
        if base is not None:
            if base.shape != (height, width, n_channels):
                raise ValueError(
                    f"`base` shape {base.shape} != required {(height, width, n_channels)}")
            out = base.copy()
        else:
            out = np.zeros((height, width, n_channels), dtype=self.colors.dtype)

        if outside_fill is not None:
            outside_colour = np.asarray(
                color(self.color_mode, outside_fill).value, dtype=self.colors.dtype)
        else:
            outside_colour = None

        # ---------- geometry ------------------------------------------------------
        yy, xx = np.indices((height, width), dtype=float)
        cx, cy = center
        dx, dy = xx - cx, yy - cy
        distance = np.hypot(dx, dy)                       # radius of each pixel

        angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)  # 0‥2π
        span = (angle_end - angle_start) % (2 * np.pi)
        span = 2 * np.pi if span == 0 else span
        angle_mask = ((angle - angle_start) % (2 * np.pi)) <= span

        # ---------- normalise radius to [0,1] -------------------------------------
        max_radius = np.max(distance) if base is None else np.hypot(
            max(cx, width - cx), max(cy, height - cy))

        unit = (distance - radius_offset) / max(max_radius - radius_offset, 1e-9)
        unit = np.clip(unit, 0.0, 1.0)

        if unit_transform is not None:
            unit = unit_transform(unit)

        # ---------- gradient lookup (linear interp) -------------------------------
        idx_f = unit * (n_steps - 1)                     # float index
        idx0 = np.floor(idx_f).astype(int)
        idx1 = np.clip(idx0 + 1, 0, n_steps - 1)
        t = (idx_f - idx0)[..., None]                    # (...,1)

        col = (1 - t) * self.colors[idx0] + t * self.colors[idx1]

        # ---------- compose result ------------------------------------------------
        inside_mask = (distance >= radius_offset) & angle_mask
        out[inside_mask] = col[inside_mask]

        outside_mask = ~inside_mask
        if outside_colour is not None:
            out[outside_mask] = outside_colour

        return out




class Color2DArr(ColorArr):
    def __init__(self, colors: NDArray, color_mode: Union[ColorMode, str], color_format: ColorFormat = ColorFormat.INT):
        self.colors: NDArray = colors
        self.color_mode: Type[ColorMode] = get_color_mode(color_mode)
        self.mode: str = self.color_mode.mode.upper()
        self.format: ColorFormat = color_format

        assert isinstance(self.colors, NDArray), "colors must be a numpy array"

        if len(self.color_mode.maxima) > 1:
            # Multi-channel: expect shape (H, W, C)
            assert self.colors.ndim >= 3, "multi-channel colors must be at least 3D"
            assert self.colors.shape[-1] == len(self.color_mode.maxima), \
                f"expected last dimension to be {len(self.color_mode.maxima)} channels, got {self.colors.shape[-1]}"
        else:
            # Single-channel: expect shape (H, W)
            assert self.colors.ndim == 2, "single-channel colors must be 2D"

    def repeat(self, horizontally: int = 1, vertically: int = 1) -> NDArray:
        """
        Repeat the 2D color array horizontally and vertically.

        Both factors must be positive integers.

        Args:
            horizontally (int): How many times to repeat horizontally.
            vertically (int): How many times to repeat vertically.

        Returns:
            NDArray: Repeated 2D color array.
        """
        if horizontally <= 0:
            raise ValueError("horizontally must be > 0")
        if vertically <= 0:
            raise ValueError("vertically must be > 0")

        # Repeat each axis
        reps = (vertically, horizontally) + (1,) * (self.colors.ndim - 2)
        result = np.tile(self.colors, reps)
        return result


    def resize(self, new_shape: tuple[int, int]) -> NDArray:
        """
        Resize the 2D color array to a new shape.

        Args:
            new_shape (tuple[int, int]): New shape (height, width).

        Returns:
            NDArray: Resized color array.
        """
        if len(new_shape) != 2:
            raise ValueError("new_shape must be a tuple of (height, width)")
        
        from skimage.transform import resize
        resized_colors = resize(self.colors, new_shape, anti_aliasing=True, mode='reflect')
        return resized_colors
