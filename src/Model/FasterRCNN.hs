module Model.FasterRCNN where

import MXNet.Base
import MXNet.Base.Operators.Symbol (elemwise_mul, elemwise_sub, smooth_l1, softmax, _MakeLoss, _contrib_MultiProposal)
import MXNet.NN.Layer
import qualified Model.VGG as VGG

symbolTrain :: [Float] -> [Float] -> Int -> Int -> Int -> Int -> Float -> Int -> IO (Symbol Float)
symbolTrain anchor_scales anchor_ratios rpn_feature_stride rpn_batch_rois rpn_pre_topk rpn_post_topk rpn_nms_thresh rpn_min_size = do
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
    rpnClsProb <- softmaxoutput "rpn_cls_prob" (#data := rpnClsScoreReshape .& #label := rpnLabel .& #multi_output := True
                                             .& #normalization := #valid .& #use_ignore := True .& #ignore_label := -1 .& Nil)
    rpnClsAct <- softmax "rpn_cls_act" (#data := rpnClsScoreReshape .& #axis := 1 .& Nil)
    rpnClsActReshape <- reshape "rpn_cls_act_reshape" (#data := rpnClsAct .& #shape := [0, 2 * numAnchors, -1, 0] .& Nil)

    rpnBBoxPred <- convolution "rpn_bbox_pred" (#data := rpnRelu .& #kernel := [1,1] .& #pad := [0,0] .& #num_filter := 4 * numAnchors .& Nil)
    rpnBBoxPredReg <- elemwise_sub "rpn_bbox_pred_reg" (#lhs := rpnBBoxPred .& #rhs := rpnBoxTarget .& Nil)
    rpnBBoxPredRegSmooth <- smooth_l1 "rpn_bbox_pred_reg_smooth" (#data := rpnBBoxPredReg .& #scalar := 3.0 .& Nil)
    rpnBBoxLoss_ <- elemwise_mul "rpn_bbox_loss_" (#lhs := rpnBoxWeight .& #rhs := rpnBBoxPredRegSmooth .& Nil)

    rpnBBoxLoss <- _MakeLoss "rpn_bbox_loss" (#data := rpnBBoxLoss_ .& #grad_scale := 1.0 / fromIntegral rpn_batch_rois .& Nil)

    rois <- _contrib_MultiProposal "rois" (#cls_prob := rpnClsActReshape .& #bbox_pred := rpnBBoxPred .& #im_info := imInfo
                                        .& #feature_stride := rpn_feature_stride .& #scales := anchor_scales .& #ratios := anchor_ratios
                                        .& #rpn_pre_nms_top_n := rpn_pre_topk .& #rpn_post_nms_top_n := rpn_post_topk
                                        .& #threshold := rpn_nms_thresh .& #rpn_min_size := rpn_min_size .& Nil)

    return $ Symbol rois

