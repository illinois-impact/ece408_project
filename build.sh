#! /bin/bash

# Cause the script to fail on errors
set -eou pipefail -x

# Where the sources for the new layer are
SRC_DIR="./ece408_src"
SRCS=`find $SRC_DIR -type f`

# Where MXNet source lives 
MXNET_SRC_ROOT="./2017fa_ece408_mxnet_skeleton"

# Copy our files to the custom operator directory
for src in $SRCS; do
    cp -v "$src" "$MXNET_SRC_ROOT/src/operator/custom/."
done

# Build MXNet
nice -n20 make -j`nproc` -C 2017fa_ece408_mxnet_skeleton

# Install python bindings
pip install --user -e $MXNET_SRC_ROOT/python