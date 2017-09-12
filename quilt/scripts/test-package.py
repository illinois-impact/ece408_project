import pandas as pd
import quilt

quilt.install("pearson/ece408", force=True)

from quilt.data.pearson import ece408

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
