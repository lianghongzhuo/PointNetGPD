#!/bin/sh
mkdir build
cd build
cmake ..
make
cp meshpy/meshrender.so ../meshpy
cd ..
rm -rf build
