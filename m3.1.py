#!/usr/bin/env python

import mxnet as mx
import logging
import sys
from reader import load_mnist

MODEL_DIR = "/models"
model_prefix = "ece408-high"
dataset_size = float("inf")

if len(sys.argv) > 1:
    model_prefix = sys.argv[1]
if len(sys.argv) > 2:
    dataset_size = int(sys.argv[2])
if len(sys.argv) > 3:
    print "Usage:", sys.argv[0], "<model_name>", "<dataset size>"
    print "    <model_name>   = [ece408-high, ece408-low]"
    print "    <dataset_size> = [0 - 10000]"
    sys.exit(-1)

# Log to stdout for MXNet
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

print "New Inference"
print "Loading fashion-mnist data...",
test_images, test_labels = load_mnist(path="/fashion-mnist", kind="t10k")
test_images = test_images.reshape((10000, 1, 28, 28))
test_labels = test_labels.reshape(10000)
print "done"

# Reduce the size of the dataset, if desired
dataset_size = max(0, min(dataset_size, 10000))
test_images = test_images[:dataset_size]
test_labels = test_labels[:dataset_size]

# Cap batch size at the size of our training data
# If you wish to tune the batch size, do it within your CUDA code,
# as you will not be able to modify the python script used in final submission
batch_size = len(test_images)

# Get iterators that cover the dataset
test_iter = mx.io.NDArrayIter(
    test_images, test_labels, batch_size)

# Evaluate the network
print "Loading model...",
lenet_model = mx.mod.Module.load(
    prefix=MODEL_DIR + "/" + model_prefix, epoch=1, context=mx.gpu())
lenet_model.bind(data_shapes=test_iter.provide_data,
                 label_shapes=test_iter.provide_label)
print "done"

acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print "Correctness:", acc.get()[1], "Model:", model_prefix
