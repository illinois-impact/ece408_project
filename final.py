#!/usr/bin/env python

import mxnet as mx
import logging
from reader import load_mnist

MODEL_DIR = "/models"
MODEL_PREFIX = "ece408-high"

# Log to stdout for MXNet
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

print "Loading fashion-mnist data...",
test_images, test_labels = load_mnist(path="/fashion-mnist", kind="t10k")
test_images = test_images.reshape((10000, 1, 28, 28))
test_labels = test_labels.reshape(10000)
print "done"

# Do everything in a single batch
batch_size = len(test_images)

# Get iterators that cover the dataset
test_iter = mx.io.NDArrayIter(
    test_images, test_labels, batch_size)

# Evaluate the network
print "Loading model...",
lenet_model = mx.mod.Module.load(
    prefix=MODEL_DIR + "/" + MODEL_PREFIX, epoch=1, context=mx.cpu())
lenet_model.bind(data_shapes=test_iter.provide_data,
                 label_shapes=test_iter.provide_label)
print "done"

acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print "Correctness:", acc.get()[1], "Batch Size:", batch_size, "Model:", MODEL_PREFIX
