#!/usr/bin/env bash

cmake -B build
cmake --build build

mkdir -p output

build/FinalProject monteCarlo testcases/final.txt output/final.bmp 100