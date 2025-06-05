@echo off
echo Building HIA Application...
pyinstaller --clean hia_app.spec
echo Build complete. Executable is in dist/HIA folder.
pause 