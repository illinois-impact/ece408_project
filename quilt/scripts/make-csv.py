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
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


images, labels = load_mnist(".", kind="train")
train_data_df = pd.DataFrame(data=images)
train_labels_df = pd.DataFrame(data=labels)
train_data_df.to_csv("train_images.data", index=False, header=True)
train_labels_df.to_csv("train_labels.data", index=False, header=['label'])

images, labels = load_mnist(".", kind="t10k")
test_data_df = pd.DataFrame(data=images)
test_labels_df = pd.DataFrame(data=labels)
test_data_df.to_csv("test_images.data", index=False, header=True)
test_labels_df.to_csv("test_labels.data", index=False, header=['label'])

labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"]

df = pd.DataFrame(data=labels)
df.to_csv("label_strings.data", index=False, header=True)
