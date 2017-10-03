#!/usr/bin/env python

import mxnet as mx
import numpy as np
import logging
from reader import load_mnist

# Log to stdout for MXNet
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

print "Loading fashion-mnist data...",
train_images, train_labels = load_mnist(path="fashion-mnist", kind="train")
test_images, test_labels = load_mnist(path="fashion-mnist", kind="t10k")

# Reshape the data to the format expected by MXNet's default convolutional layers
train_images = train_images.reshape((60000, 1, 28, 28))
train_labels = train_labels.reshape(60000)
test_images = test_images.reshape((10000, 1, 28, 28))
test_labels = test_labels.reshape(10000)

# You can reduce the size of the train or test datasets by uncommenting the following lines
# train_images = train_images[:1000]
# train_labels = train_labels[:1000]
# test_images = test_images[:1000]
# test_labels = test_labels[:1000]
print "done"

# Batch size of 100
batch_size = 100

# Get iterators that cover the datasets.
train_iter = mx.io.NDArrayIter(
    train_images, train_labels, batch_size, shuffle=True)
test_iter = mx.io.NDArrayIter(
    test_images, test_labels, batch_size)

# The rest of the file defines a convolutional neural network.
# You will be writing a new implementation of the mx.sym.Convolution operator.
# The MXNet convolution supports many additional parameters,
# like bias, stride, dilation, padding, and so on.
# ECE408 only requires a standard 2D convolution with square kernels.
# the 'kernel' argument defines the kernel size
# the 'num_filter' argument is the number of output channels
# the 'no-bias' argument tells MXNet not to include a bias term in its convolution.
# You do not need a corresponding command in your layer, since your layer does not support bias.
# This makes  your implementation directly comparable to MXNet's.

data = mx.sym.var('data')

# First Convolution Layer
conv1 = mx.sym.Convolution(data=data, kernel=(
    5, 5), num_filter=20, no_bias=True)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max",
                       kernel=(2, 2), stride=(2, 2))
# Second Convolution Layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(
    5, 5), num_filter=50, no_bias=True)

tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max",
                       kernel=(2, 2), stride=(2, 2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)  # 1
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

# create a trainable module on the CPU
# adjust to context=mx.gpu() to run on the GPU
lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())

# Train the network
lenet_model.fit(train_iter,
                eval_data=test_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                eval_metric='acc',
                batch_end_callback=mx.callback.Speedometer(
                    batch_size, 10),
                epoch_end_callback=mx.callback.module_checkpoint(
                    lenet_model, prefix='baseline'),
                num_epoch=1)
print "training done"

# Evaluate the network
acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)
