#! /bin/bash

set -eou pipefail

start=$(pwd)

echo "Updating submodules"
git submodule update --init --recursive --remote
cd mxnet-newop

echo "Updating submodules"
git submodule update --init --recursive --remote

echo "Building mxnet"
nice -n20 make -j$(nproc)

echo "Installing python bindings"
cd python && pip install --user -e . 

cd $start