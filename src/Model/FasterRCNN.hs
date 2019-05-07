module Model.FasterRCNN where

import MXNet.Base
import MXNet.NN.Layer
import qualified Model.VGG as VGG

symbolTrain :: [Float] -> [Float] -> Int -> IO (Symbol Float)
symbolTrain anchor_scales anchor_ratios feature_stride = do
    let numAnchors = length anchor_scales * length anchor_ratios
    dat <- variable "data"
    imInfo <- variable "im_info"
    gtBoxes <- variable "gt_boxes"
    rpnLabel <- variable "label"
    rpnBoxTarget <- variable "bbox_target"
    rpnBoxWeight <- variable "bbox_weight"

    convFeat <- VGG.getFeature dat [2, 2, 3, 3, 3] [64, 128, 256, 512, 512] False
    
    rpnConv <- convolution "rpn_conv_3x3" (#data := convFeat .& #kernel := [3,3] .& #pad := [1,1] .& #num_filter := 512 .& Nil)
    rpnRelu <- activation "rpn_relu" (#data := rpnConv .& #act_type := #relu .& Nil)

    rpnClsScore <- convolution "rpn_cls_score" (#data := rpnRelu .& #kernel := [1,1] .& #pad := [0,0] .& #num_filter := 2 * numAnchors .& Nil)
    rpnClsScoreReshape <- reshape "rpn_cls_score_reshape" (#data := rpnClsScore .& #shape := [0, 2, -1, 0] .& Nil)
    rpnClsProb <- softmaxoutput "rpn_cls_prob" (#data := rpnClsScoreReshape .& #label := rpnLabel .& #multi_output := True .&
                                               #normalization := #valid .& #use_ignore := True .& #ignore_label := -1 .& Nil)

    return $ Symbol rpnClsProb

