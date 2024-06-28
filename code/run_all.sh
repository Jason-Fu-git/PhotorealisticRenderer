#!/usr/bin/env bash

cmake -B build
cmake --build build

# Run all testcases. 
# You can comment some lines to disable the run of specific examples.
mkdir -p output
build/FinalProject whitted testcases/scene01_basic.txt output/scene01.bmp
build/FinalProject whitted testcases/scene02_cube.txt output/scene02.bmp
build/FinalProject whitted testcases/scene03_sphere.txt output/scene03.bmp
build/FinalProject whitted testcases/scene04_axes.txt output/scene04.bmp
build/FinalProject whitted testcases/scene05_bunny_200.txt output/scene05.bmp
build/FinalProject whitted testcases/scene06_bunny_1k.txt output/scene06.bmp
build/FinalProject whitted testcases/scene07_shine.txt output/scene07.bmp
build/FinalProject whitted testcases/scene08_altered_smallpt.txt output/scene08.bmp
build/FinalProject monteCarlo testcases/scene09_smallpt.txt output/scene09.bmp 100
build/FinalProject whitted testcases/scene10_earth.txt output/scene10.bmp
build/FinalProject monteCarlo testcases/scene11_transparentbunny.txt output/scene11.bmp 16
build/FinalProject whitted testcases/scene12_vase.txt output/scene12.bmp
build/FinalProject monteCarlo testcases/scene13_fox.txt output/scene13.bmp 16
build/FinalProject monteCarlo testcases/scene14_nee.txt output/scene14.bmp 16
build/FinalProject monteCarlo testcases/scene15_depth.txt output/scene15.bmp 1000
build/FinalProject monteCarlo testcases/scene16_glossy.txt output/scene16.bmp 1000
build/FinalProject whitted testcases/scene17_interpolation.txt output/scene17.bmp
