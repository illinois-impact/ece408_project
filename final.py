#!/usr/bin/env python

import mxnet as mx
import logging
import sys
from reader import load_mnist

MODEL_DIR = "/models"
model_prefix = "ece408-high"
batch_size = float("inf")

if len(sys.argv) > 1:
    batch_size = sys.argv[1]
if len(sys.argv) > 2:
    model_prefix = sys.argv[2]
if len(sys.argv) > 3:
    print "Usage:", sys.argv[0], "<batch_size> <model_name>"
    print "    <model_name> = [ece408-high, ece408-low]"
    sys.exit(-1)

# Log to stdout for MXNet
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

print "Loading fashion-mnist data...",
test_images, test_labels = load_mnist(path="/fashion-mnist", kind="t10k")
test_images = test_images.reshape((10000, 1, 28, 28))
test_labels = test_labels.reshape(10000)
print "done"

# Cap batch size at the size of our training data
batch_size = min(len(test_images), batch_size)

# Get iterators that cover the dataset
test_iter = mx.io.NDArrayIter(
    test_images, test_labels, batch_size)

# Evaluate the network
print "Loading model...",
lenet_model = mx.mod.Module.load(
    prefix=MODEL_DIR + "/" + model_prefix, epoch=1, context=mx.cpu())
lenet_model.bind(data_shapes=test_iter.provide_data,
                 label_shapes=test_iter.provide_label)
print "done"

acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print "Correctness:", acc.get()[1], "Batch Size:", batch_size, "Model:", model_prefix
