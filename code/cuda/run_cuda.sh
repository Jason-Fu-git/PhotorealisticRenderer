#!/usr/bin/env bash

cmake -B build
cmake --build build

# Run all testcases.
# You can comment some lines to disable the run of specific examples.
mkdir -p output

build/FinalProjectCUDA 20000
