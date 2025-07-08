# Contributing to Chromatic

Thank you for considering contributing to Chromatic! We welcome all contributions that help improve this library for everyone. Whether you're fixing bugs, adding new features, improving documentation, or suggesting ideas, your input is valuable.

## How to Contribute

### 1. Reporting Issues
- **Bug Reports**: Please include:
  - Chromatic version
  - Python version
  - Steps to reproduce
  - Expected vs. actual behavior
  - Relevant code snippets
- **Feature Requests**: Describe:
  - The problem you're trying to solve
  - Why this feature would be valuable
  - How you envision it working

### 2. Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/chromatic.git
   cd chromatic
   ```
3. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```

### 3. Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes following the coding standards below
3. Add tests for your changes (if applicable)
4. Verify all tests pass:
   ```bash
   pytest
   ```
5. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add: New feature for radial gradients"
   ```

### 4. Submitting a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Create a pull request against the main branch
3. Describe your changes and reference any related issues
4. We'll review your PR and provide feedback

## Coding Standards

### General Guidelines
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Keep functions focused (single responsibility principle)
- Document public methods with docstrings
- Use descriptive variable names

### Documentation Standards
```python
def radial_gradient(
    color1: Union['ColorInt', Tuple[int, ...], int],
    color2: Union['ColorInt', Tuple[int, ...], int],
    height: int,
    width: int,
    center: Union[Tuple[int, int], List[int]] = (0, 0),
    radius: float = 1.0,
    color_mode: Union['ColorMode', str] = 'RGB'
) -> np.ndarray:
    """
    Create a radial gradient between two colors.

    Args:
        color1: Starting color (inner color)
        color2: Ending color (outer color)
        height: Output image height
        width: Output image width
        center: Center point of the gradient (x, y)
        radius: Radius of the gradient effect
        color_mode: Color mode for output (RGB, HSV, etc.)

    Returns:
        NumPy array of shape (height, width, channels) representing the gradient
    """
    # Implementation here
```

### Testing Standards
- Tests should be in the `tests/` directory
- Use descriptive test names (test_function_name_scenario)
- Cover both typical and edge cases
- Tests should be independent and self-contained

```python
def test_radial_gradient_creates_correct_size():
    """Test radial gradient creates correct output dimensions"""
    result = radial_gradient(
        color1=(255, 0, 0),
        color2=(0, 0, 255),
        height=100,
        width=200
    )
    assert result.shape == (100, 200, 3)
```

## Development Practices

### Branching Strategy
- `main`: Stable production-ready code
- `develop`: Integration branch for features
- Feature branches: `feature/your-feature-name`
- Bugfix branches: `fix/issue-description`

### Release Process
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag (vX.Y.Z)
4. Build and publish package

## Community Guidelines

- Be respectful and inclusive
- Assume positive intent
- Give constructive feedback
- Help maintain documentation
- Welcome new contributors

## Questions?

Reach out to the maintainers:
- Grayjou (cgrayjou@gmail.com)
- GitHub Issues: https://github.com/Grayjou/chromatic/issues

Thank you for helping make Chromatic better!
