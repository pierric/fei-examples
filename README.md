# fei-examples

+ MNIST
+ CIFAR10 + Resnet / ResNext
+ mxnet custom operator
+ Faster RCNN
    + `LD_LIBRARY_PATH=<path-to-mxnet> stack run faster-rcnn -- --backbone RESNET50FPN --strides [4,8,16,32] --pretrained params/resnet50_v2 --base <path-to-coco> --img-size 512 --img-pixel-means [0.5,0.5,0.5] --train-epochs 20 --train-iter-per-epoch 300 --batch-size=2 --rcnn-batch-rois=256 +RTS -N6`
