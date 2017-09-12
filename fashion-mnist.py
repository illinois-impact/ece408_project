#!/usr/bin/env python

import quilt
import mxnet as mx
import numpy as np

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

# Update to the lastest data
quilt.install("pearson/ece408", force=True)

from quilt.data.pearson import ece408

print "Checking fashion-mnist data dimensions are as expected...",
train_images = ece408.fashion_mnist.train_images()
test_images = ece408.fashion_mnist.test_images()
train_labels = ece408.fashion_mnist.train_labels()
test_labels = ece408.fashion_mnist.test_labels()
label_strings = ece408.fashion_mnist.label_strings()

assert train_labels.shape == (60000, 1)
assert test_labels.shape == (10000, 1)
assert train_images.shape == (60000, 784)
assert test_images.shape == (10000, 784)
assert label_strings.shape == (10, 1)
print "yes"

print "Sanity check mxnet bindings...",
a = mx.nd.ones((2, 3))
b = a * 2 + 1
assert np.array_equal(b.asnumpy(), np.array(
    [[3.,  3.,  3.], [3.,  3.,  3.]], dtype=np.float32))
print "yes"

train_images = train_images.values.reshape((60000, 1, 28, 28))
train_labels = train_labels.values.reshape(60000)
test_images = test_images.values.reshape((10000, 1, 28, 28))
test_labels = test_labels.values.reshape(10000)

# train_images = train_images[:10]
# train_labels = train_labels[:10]

batch_size = 100
train_iter = mx.io.NDArrayIter(
    train_images, train_labels, batch_size, shuffle=True)
test_iter = mx.io.NDArrayIter(
    test_images, test_labels, batch_size)

data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)  # 20
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max",
                       kernel=(2, 2), stride=(2, 2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)  # 50
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

# create a trainable module on GPU 0
lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
# train with the same
lenet_model.fit(train_iter,
                eval_data=test_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                eval_metric='acc',
                batch_end_callback=mx.callback.Speedometer(
                    batch_size, 10),
                num_epoch=1)

print "training done"

acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)
