# fei-examples

+ MNIST
+ CIFAR10 + Resnet / ResNext
+ mxnet custom operator
+ Faster RCNN
    + `LD_LIBRARY_PATH=<path-to-mxnet> stack run rcnn -- --backbone RESNET50 --pretrained params/resnet50_v2 --base <path-to-coco> --img-size 768 --img-pixel-means 0.5,0.5,0.5 --train-epochs=20 --train-iter-per-epoch 300 --rpn-pre-nms-topk 2000 --rpn-post-nms-topk 1000 --rcnn-batch-size=2 --rcnn-batch-rois=256 +RTS -N4`
