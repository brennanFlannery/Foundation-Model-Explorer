"""
main.py
=======

Entry point for running the FoundationDetector GUI application.

This module initializes the Qt application framework and launches the main
window. It handles platform-specific configuration for OpenSlide library
loading, particularly on Windows systems where DLL paths must be explicitly
configured.

The application expects a specific directory structure containing whole-slide
images (.svs, .tif, .tiff, .ndpi) and associated feature files in HDF5 format.
See the README.md or data_loader.py for details on the expected directory layout.

Usage::

    python main.py

Platform-Specific Notes
-----------------------

Windows:
    On Windows, this script attempts to add a custom OpenSlide DLL directory
    to the system path. The default path is ``C:\\Users\\Brennan\\OpenSlide\\win32-x86-64``.
    If you have installed OpenSlide manually or in a different location, modify
    the ``openslide_dll_dir`` variable in this file to point to your installation.

    If you installed ``openslide-bin`` via pip, this manual configuration is
    usually not necessary as the package handles DLL loading automatically.

Linux/macOS:
    No special configuration is required. OpenSlide libraries are typically
    found via system library paths.

Dependencies
------------

The following packages must be installed:
    - PySide6: GUI framework
    - numpy: Numerical computations
    - Pillow: Image processing
    - scikit-learn: Machine learning (K-means clustering)
    - h5py: HDF5 file reading
    - openslide-python: OpenSlide Python bindings
    - openslide-bin: OpenSlide binary libraries (Windows)

If any dependency is missing, the application will report an error when the
corresponding feature is accessed.

Functions
---------

main()
    Entry point function that creates the Qt application, instantiates the
    main window, and starts the event loop. This function blocks until the
    application is closed by the user.
"""

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import os
import sys

# Debug: Print which Python interpreter is being used
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Attempt to add a DLL search directory for OpenSlide on Windows.
# On non-Windows platforms this call is unnecessary and will be skipped.
# Note: If you've installed openslide-bin via pip, this is usually not needed.
# This is kept as a fallback for manual OpenSlide installations.
# We only call `add_dll_directory` when running on Windows and the directory exists.
if sys.platform == 'win32':
    openslide_dll_dir = r"C:\Users\Brennan\OpenSlide\win32-x86-64"
    if os.path.exists(openslide_dll_dir):
        os.add_dll_directory(openslide_dll_dir)
from gui import MainWindow


def main() -> None:
    """Create and run the FoundationDetector application.
    
    This function initializes the Qt application framework, creates the main
    window instance, displays it, and starts the event loop. The function
    blocks until the user closes the application window.
    
    Returns
    -------
    None
        The function does not return a value. The application exit code is
        returned by QApplication.exec(), but it is not captured here since
        this script is typically run as a standalone application.
    """
    app = QApplication([])
    # Enable high DPI support for better scaling on modern displays
    # Note: AA_UseHighDpiPixmaps deprecated in Qt6, handled automatically
    window = MainWindow()
    window.show()
    # Start the Qt event loop.  Note: app.exec() returns an integer exit
    # code when the event loop finishes, but we don't capture it here
    # because this script is typically run as a standalone application.
    app.exec()


if __name__ == "__main__":
    main()