def load_mnist(path, rows, cols, kind):
    import os
    import gzip
    import numpy as np

    filters = 1

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
        labels.reshape(len(labels))

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), filters, rows, cols)

    return images, labels


def store_mnist(path, images, labels, kind):
    import os
    import gzip
    import numpy as np
    import struct

    """Store data to `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'wb') as lbpath:
        lbpath.write(struct.pack("i", 0))  # magic
        lbpath.write(struct.pack("i", labels.size))  # number of items (32b)
        lbpath.write(labels.tobytes())

    with gzip.open(images_path, 'wb') as imgpath:
        imgpath.write(struct.pack("i", 0))  # magic number
        # number of images (32b)
        imgpath.write(struct.pack("i", images.shape[0]))
        # number of rows (32b)
        imgpath.write(struct.pack("i", images.shape[1]))
        # number of cols (32b)
        imgpath.write(struct.pack("i", images.shape[2]))
        imgpath.write(images.tobytes())

    return images, labels
