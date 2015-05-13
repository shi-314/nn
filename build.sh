#!/bin/sh

# rm -rf build/*
# clear
cmake -Bbuild -H.
make -C build/
echo
./build/nn
