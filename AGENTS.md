# AGENTS.md

This file provides guidance for agentic coding assistants working with the FoundationalDetector codebase.

## Project Overview

FoundationalDetector is a Python desktop application for interactive visualization of patch-level features from whole-slide images (WSI) in digital pathology. It uses PySide6 (Qt 6) for the GUI framework and follows an MVC pattern with Qt signal/slot communication.

## Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
python main.py
```

### Testing
```bash
# Run all tests (if test files exist)
pytest

# Run specific test file
pytest test_module.py

# Run with verbose output
pytest -v

# Run with coverage (if pytest-cov installed)
pytest --cov=.
```

### Code Quality
No dedicated linting or formatting commands are currently configured. Recommended additions:
```bash
# Format code (if black installed)
black .

# Lint code (if flake8 installed)
flake8 .

# Type checking (if mypy installed)
mypy .
```

## Code Style Guidelines

### Import Organization
- **Standard library imports first**: `import os`, `import sys`
- **Third-party imports next**: `import numpy as np`, `from PySide6.QtWidgets import...`
- **Local imports last**: `import data_loader`, `from utils import...`
- **Use `from __future__ import annotations`** at top of all Python files for forward type references
- **Group PySide6 imports** by module (QtCore, QtGui, QtWidgets) with line breaks for readability
- **Use aliases**: `import numpy as np`, `import data_loader` (not `from . import data_loader`)

### Type Hints
- **Use type hints consistently** for all function parameters and return values
- **Import from typing**: `from typing import Dict, List, Optional, Tuple`
- **Use Optional[T]** for parameters that can be None
- **Use Union[T, U]** when multiple types are acceptable
- **Return None explicitly** for functions that don't return values

### Naming Conventions
- **Classes**: PascalCase (`MainWindow`, `SlideGraphicsView`)
- **Functions/Methods**: snake_case (`load_features`, `update_scatter_view`)
- **Variables**: snake_case (`cluster_colors`, `patch_coordinates`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_PATCH_SIZE`, `MAX_CLUSTERS`)
- **Private methods**: prefix with underscore (`_update_scatter_for_cluster`)
- **Signal names**: descriptive snake_case (`cluster_selected`, `patch_hovered`)

### Error Handling
- **Use specific exceptions**: `except FileNotFoundError:` instead of `except Exception:`
- **Log errors appropriately** using print statements or logging module
- **Provide user-friendly error messages** in GUI dialogs
- **Handle missing dependencies gracefully** with fallback behavior
- **Validate file paths and data formats** before processing

### Qt/PySide6 Specific Guidelines
- **Use signals/slots for communication** between components
- **Follow Qt naming conventions** for signal/slot methods
- **Use QPropertyAnimation** for smooth visual effects
- **Implement proper event handlers** for mouse interactions
- **Use QTimer for delayed operations** and performance optimization
- **Manage memory carefully** with QGraphicsItem objects

### Documentation Style
- **Use module-level docstrings** with triple quotes and descriptive headers
- **Include Usage sections** for public functions with code examples
- **Document platform-specific behavior** (Windows OpenSlide DLL handling)
- **Use reStructuredText format** for docstrings (`Parameters`, `Returns`, `Raises`)
- **Keep docstrings concise but informative**

### File Organization
- **Single responsibility principle**: Each module has a clear purpose
- **Keep imports at top** after module docstring and `__future__` import
- **Group related functionality** within files
- **Avoid circular imports** by using forward references or local imports
- **Use dataclasses** for simple data structures (`@dataclass`)

### Performance Considerations
- **Use numpy arrays** for numerical computations instead of Python lists
- **Implement caching** for expensive operations (thumbnail generation, PCA)
- **Use MiniBatchKMeans** for large datasets to improve clustering performance
- **Limit QGraphicsItem creation** for very large numbers of patches/points
- **Use threading** for I/O operations (file loading, image processing)

### Testing Guidelines
- **Write unit tests for utility functions** in `utils.py` (no Qt dependencies)
- **Test data loading logic** independently of GUI components
- **Mock Qt objects** when testing GUI components
- **Test clustering algorithms** with known inputs/outputs
- **Validate coordinate transformations** and geometric calculations

### Platform-Specific Code
- **Use `sys.platform` checks** for Windows-specific behavior
- **Handle OpenSlide DLL loading** on Windows with fallback to openslide-bin
- **Use os.path functions** for cross-platform path handling
- **Test on multiple platforms** when adding platform-specific features

## Dependencies

Core dependencies (see requirements.txt):
- **PySide6**: GUI framework (Qt 6)
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning (K-means, PCA)
- **h5py**: HDF5 file reading
- **openslide-python**: Whole-slide image reading
- **Pillow**: Image processing
- **shapely**: Geometry operations for GeoJSON export
- **pandas**: Data manipulation
- **pytest**: Testing framework (optional)

## Data Directory Structure

Expected layout for feature files:
```
root_dir/
  slideA.svs                    # WSI files at root
  features/
    modelX/                     # Model name
      20x/                      # Magnification
        256px/                  # Patch size
          slideA.h5             # HDF5 with features and coords
```

## Report File Location

**All markdown reports should be placed in the `Reports/` directory**, except for:
- `AGENTS.md` (this file)
- `CLAUDE.md` 

When generating analysis reports, documentation, or other markdown files, use the path `Reports/filename.md`.

## Key Architecture Patterns

- **MVC Pattern**: gui.py (Controller), data_loader.py (Model), slide_view.py/scatter_view.py (Views)
- **Qt Signal/Slot Communication**: Decoupled interaction between components
- **Factory Pattern**: For creating different view types and graphics items
- **Observer Pattern**: Qt signals for state change notifications
- **Strategy Pattern**: Different clustering algorithms and rendering modes