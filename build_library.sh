#!/bin/bash

# Script para construir a biblioteca drone_people_detector
# Gera um arquivo .whl que pode ser instalado com pip

echo "=========================================="
echo "  Building drone_people_detector library"
echo "=========================================="
echo ""

# Limpar builds anteriores
echo "Cleaning previous builds..."
rm -rf build/ dist/ src/drone_people_detector.egg-info/

# Construir o pacote
echo "Building package..."
python -m build

# Verificar se foi bem-sucedido
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Build successful!"
    echo "=========================================="
    echo ""
    echo "Package created in dist/ directory:"
    ls -lh dist/
    echo ""
    echo "To install locally:"
    echo "  pip install dist/drone_people_detector-1.0.0-py3-none-any.whl"
    echo ""
    echo "To install in another project:"
    echo "  pip install /path/to/drone_people_detector-1.0.0-py3-none-any.whl"
else
    echo ""
    echo "=========================================="
    echo "✗ Build failed!"
    echo "=========================================="
    exit 1
fi
