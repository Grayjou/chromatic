# Chromatic - Advanced Color Manipulation Library

Chromatic is a powerful Python library for advanced color manipulation, gradient generation, and color space conversions. Designed for graphics programming, data visualization, and image processing, Chromatic provides intuitive tools for working with colors in various formats and creating stunning gradients with mathematical precision.

## Key Features

- **Multi-format Color Support**: Work with RGB, HSV, HSL, CMYK, RGBA, and more
- **Advanced Gradient Generation**: Create linear, radial, and angular gradients
- **Color Space Conversions**: Convert between color spaces with precision
- **Vectorized Operations**: Optimized numpy-based operations for performance
- **PIL/Pillow Integration**: Seamlessly work with images
- **Mathematical Precision**: Unit-tested color transformations

## Installation

```bash
pip install chromatic
```

## Quick Start

### Color Conversion

```python
from chromatic import convert, ColorRGB, ColorHSV

# Simple color conversion
rgb = (255, 0, 0)  # Red
hsv = convert(rgb, from_space='rgb', to_space='hsv')
print(f"Red in HSV: {hsv}")  # (0, 100, 100)

# Using color classes
red_rgb = ColorRGB((255, 0, 0))
red_hsv = red_rgb.to_hsv()
print(f"Red as HSV object: {red_hsv}")  # ColorHSV((0, 100, 100))
```

### Gradient Generation

```python
from chromatic import Gradient1D, Gradient2D

# Create a 1D gradient from red to blue
gradient_1d = Gradient1D.from_colors(
    color1=(255, 0, 0),
    color2=(0, 0, 255),
    steps=100
)

# Create a 2D gradient from four corner colors
gradient_2d = Gradient2D.from_colors(
    color_tl=(255, 0, 255),   # Top-left: pink
    color_tr=(255, 255, 0),   # Top-right: yellow
    color_bl=(255, 0, 128),   # Bottom-left: deep pink
    color_br=(255, 128, 0),   # Bottom-right: orange
    width=500,
    height=500
)

# Save as image
from PIL import Image
Image.fromarray(gradient_2d.colors.astype(np.uint8), mode='RGB').save('gradient.png')
```

### Radial Gradient

```python
from chromatic import radial_gradient
import numpy as np

# Create radial gradient
gradient = radial_gradient(
    color1=(0, 0, 255, 0),    # Blue with full transparency
    color2=(0, 0, 0, 255),    # Black with full opacity
    height=500,
    width=500,
    center=(250, 250),
    radius=125,
    color_mode='RGBA'
)

# Display
Image.fromarray(gradient.astype(np.uint8), mode='RGBA').show()
```

## Documentation

### Core Components

1. **Color Classes**:
   - `ColorRGB`, `ColorHSV`, `ColorHSL`, `ColorRGBA`, etc.
   - Convert between color spaces
   - Normalize values and clamp ranges

2. **Gradient Generators**:
   - `Gradient1D`: Linear color gradients
   - `Gradient2D`: 2D color fields from corner colors
   - `radial_gradient`: Radial color transitions

3. **Color Space Conversions**:
   - Convert between RGB, HSV, HSL
   - Support for integer, float, and PIL-specific formats
   - Vectorized numpy operations

### Advanced Usage

```python
# Create gradient with custom easing function
import numpy as np

gradient = Gradient1D.from_colors(
    color1=(255, 0, 0),
    color2=(0, 0, 255),
    steps=100,
    unit_transform=lambda x: (1 - np.cos(x * np.pi)) / 2  # Smooth easing
)

# Angular gradient wrapping
rotated = gradient.wrap_around(
    width=500,
    height=500,
    center=(250, 250),
    angle_start=0,
    angle_end=2 * np.pi
)
```

## Examples (WIP Not Avaliable Yet)
Check out the examples in the [examples directory](https://github.com/Grayjou/chromatic/tree/main/examples) to see Chromatic in action:

1. [Basic Gradient](examples/basic_gradient.py)
2. [Radial Transparency](examples/radial_transparency.py)
3. [Color Space Conversions](examples/color_conversions.py)
4. [Animated Gradients](examples/animated_gradients.py) *(requires matplotlib)*

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) for details.

## License

Chromatic is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**Project by Grayjou**  
[GitHub](https://github.com/Grayjou) | [Email](mailto:cgrayjou@gmail.com)  
*Inspired by the beauty of color and light*
