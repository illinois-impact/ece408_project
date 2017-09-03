import pandas as pd

def load_mnist(path, kind='train'):
    import os
    import struct
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '../data/%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '../data/%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

images, labels = load_mnist(".")
df = pd.DataFrame(data=images)
df['label']  = labels
df.to_csv("train.data")

images, labels = load_mnist(".", kind="t10k")
df = pd.DataFrame(data=images)
df['label']  = labels
df.to_csv("test.data")