#! /bin/bash

# CIFAR-10 test
curl http://data.mxnet.io/data/cifar10/cifar10_train.rec -o cifar10_train.rec
curl http://data.mxnet.io/data/cifar10/cifar10_val.rec -o cifar10_val.rec

# MNIST
MNIST="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"
for fn in $MNIST; do
    curl http://yann.lecun.com/exdb/mnist/$fn.gz | gunzip - > $fn
done
