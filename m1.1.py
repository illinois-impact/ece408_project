#!/usr/bin/env python

import mxnet as mx
import logging
from reader import load_mnist

# Log to stdout for MXNet
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

print "Loading fashion-mnist data...",
test_images, test_labels = load_mnist(
    path="/fashion-mnist", rows=72, cols=72, kind="t10k-72")
print "done"

# Do everything in a single batch
batch_size = len(test_images)

# Get iterators that cover the dataset
test_iter = mx.io.NDArrayIter(
    test_images, test_labels, batch_size)

# Evaluate the network
print "Loading model...",
lenet_model = mx.mod.Module.load(
    prefix='/models/baseline', epoch=2, context=mx.cpu())
lenet_model.bind(data_shapes=test_iter.provide_data,
                 label_shapes=test_iter.provide_label)
print "done"

print "New Inference"
acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)
