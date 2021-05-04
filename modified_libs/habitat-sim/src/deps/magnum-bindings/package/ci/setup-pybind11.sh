set -e

wget --no-clobber https://github.com/pybind/pybind11/archive/v2.3.0.tar.gz && tar -xzf v2.3.0.tar.gz

cd pybind11-2.3.0

mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$HOME/pybind11 \
    -DPYBIND11_PYTHON_VERSION=3.6 \
    -DPYBIND11_TEST=OFF \
    -G Ninja
ninja install
