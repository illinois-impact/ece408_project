# ECE408

Holds the Fashion MNIST dataset for University of Illinois ECE408/CS483

Install quilt and pandas

    pip install --user quilt pandas

Then, an example python script:

    import pandas as pd
    import quilt

    quilt.install("pearson/ece408", force=True)

    from quilt.data.pearson import ece408

    train_df = ece408.fashion_mnist.train()
    test_df = ece408.fashion_mnist.test()

    assert train_df.shape == (60000, 786)
    assert test_df.shape == (10000, 786)