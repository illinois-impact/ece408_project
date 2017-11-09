#!/usr/bin/env python

import mxnet as mx
import logging
from reader import load_mnist

# Log to stdout for MXNet
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

print "New Inference"
print "Loading fashion-mnist data...",
test_images, test_labels = load_mnist(path="/fashion-mnist", kind="t10k")
# Reshape the data to the format expected by MXNet's default convolutional layers
test_images = test_images.reshape((10000, 1, 28, 28))
test_labels = test_labels.reshape(10000)
# You can reduce the size of the train or test datasets by uncommenting the following lines
# test_images = test_images[:1000]
# test_labels = test_labels[:1000]
print "done"

# Do everything in a single batch
batch_size = len(test_images)

# Get iterators that cover the dataset
test_iter = mx.io.NDArrayIter(
    test_images, test_labels, batch_size)

# Evaluate the network
print "Loading model...",
lenet_model = mx.mod.Module.load(
    prefix='/models/baseline', epoch=1, context=mx.gpu())
lenet_model.bind(data_shapes=test_iter.provide_data,
                 label_shapes=test_iter.provide_label)
print "done"

acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)
