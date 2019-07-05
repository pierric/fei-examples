module Model.FasterRCNN where

import MXNet.Base
import MXNet.Base.Operators.Symbol (elemwise_mul, elemwise_sub, smooth_l1, softmax, _MakeLoss, _contrib_MultiProposal)
import MXNet.NN.Layer
import qualified Model.VGG as VGG

symbolTrain :: [Float] -> [Float] -> Int -> Int -> Int -> Int -> Int -> Float -> Int -> IO (Symbol Float)
symbolTrain anchor_scales anchor_ratios num_classes rpn_feature_stride rpn_batch_rois rpn_pre_topk rpn_post_topk rpn_nms_thresh rpn_min_size = do
    let numAnchors = length anchor_scales * length anchor_ratios
    -- dat:
    dat <- variable "data"
    -- imInfo:
    imInfo <- variable "im_info"
    -- gtBoxes:
    gtBoxes <- variable "gt_boxes"
    -- rpnLabel: (batch_size, 1, numAnchors * feat_height, feat_width))
    rpnLabel <- variable "label"
    -- rpnBoxTarget: (batch_size, 4 * numAnchors, feat_height, feat_width)
    rpnBoxTarget <- variable "bbox_target"
    -- rpnBoxWeight: (batch_size, 4 * numAnchors, feat_height, feat_width)
    rpnBoxWeight <- variable "bbox_weight"

    convFeat <- VGG.getFeature dat [2, 2, 3, 3, 3] [64, 128, 256, 512, 512] False

    rpnConv <- convolution "rpn_conv_3x3" (#data := convFeat .& #kernel := [3,3] .& #pad := [1,1] .& #num_filter := 512 .& Nil)
    rpnRelu <- activation "rpn_relu" (#data := rpnConv .& #act_type := #relu .& Nil)

    ---------------------------
    -- rpn_clas_prob part
    --
    -- per pixel: fore/back-ground classification
    rpnClsScore <- convolution "rpn_cls_score" (#data := rpnRelu .& #kernel := [1,1] .& #pad := [0,0] .& #num_filter := 2 * numAnchors .& Nil)
    rpnClsScoreReshape <- reshape "rpn_cls_score_reshape" (#data := rpnClsScore .& #shape := [0, 2, -1, 0] .& Nil)
    -- rpnClsProb output shape: (batch_size, [Pr(foreground), Pr(background)], numAnchors * feat_height, feat_width)
    rpnClsProb <- softmaxoutput "rpn_cls_prob" (#data := rpnClsScoreReshape .& #label := rpnLabel .& #multi_output := True
                                             .& #normalization := #valid .& #use_ignore := True .& #ignore_label := -1 .& Nil)

    ---------------------------
    -- rpn_bbox part
    rpnBBoxPred <- convolution "rpn_bbox_pred" (#data := rpnRelu .& #kernel := [1,1] .& #pad := [0,0] .& #num_filter := 4 * numAnchors .& Nil)
    rpnBBoxPredReg <- elemwise_sub "rpn_bbox_pred_reg" (#lhs := rpnBBoxPred .& #rhs := rpnBoxTarget .& Nil)
    rpnBBoxPredRegSmooth <- smooth_l1 "rpn_bbox_pred_reg_smooth" (#data := rpnBBoxPredReg .& #scalar := 3.0 .& Nil)
    rpnBBoxLoss_ <- elemwise_mul "rpn_bbox_loss_" (#lhs := rpnBoxWeight .& #rhs := rpnBBoxPredRegSmooth .& Nil)
    rpnBBoxLoss <- _MakeLoss "rpn_bbox_loss" (#data := rpnBBoxLoss_ .& #grad_scale := 1.0 / fromIntegral rpn_batch_rois .& Nil)

    ---------------------------
    rpnClsAct <- softmax "rpn_cls_act" (#data := rpnClsScoreReshape .& #axis := 1 .& Nil)
    rpnClsActReshape <- reshape "rpn_cls_act_reshape" (#data := rpnClsAct .& #shape := [0, 2 * numAnchors, -1, 0] .& Nil)
    rois <- _contrib_MultiProposal "rois" (#cls_prob := rpnClsActReshape .& #bbox_pred := rpnBBoxPred .& #im_info := imInfo
                                        .& #feature_stride := rpn_feature_stride .& #scales := anchor_scales .& #ratios := anchor_ratios
                                        .& #rpn_pre_nms_top_n := rpn_pre_topk .& #rpn_post_nms_top_n := rpn_post_topk
                                        .& #threshold := rpn_nms_thresh .& #rpn_min_size := rpn_min_size .& Nil)

    group <- _Custom "group" (#data := [rois, gt_boxes] .& #op_type := "proposal_target" .& Nil)
    [rois, label, bboxTarget, bboxWeight] <- mapM (at group) [0..3]

    ---------------------------
    -- cls_prob part
    --
    roiPool <- _ROIPooling "roi_pool" (#data := convFeat .& #rois := rois
                                    .& #pooled_size := rcnn_pooled_size
                                    .& spatial_scale := 1.0 / fromIntegral (rcnn_feature_stride) .& Nil)
    topFeat <- VGG.getTopFeature roiPool
    clsScore <- fullyConnected "cls_score" (#data := topFeat .& #num_hidden := num_classes .& Nil)
    clsProb <- softmaxOutput "cls_prob" (#data := clsScore .& #label := label .& #normalization := #batch .& Nil)

    ---------------------------
    -- bbox_loss part
    --
    bboxPred <- fullyConnected "bbox_pred" (#data := topFeat .& #num_hidden := 4 * num_classes .& Nil)
    bboxPredReg <- elemwise_sub "bbox_pred_reg" (#lhs := bboxPred .& #rhs := bboxTarget .& Nil)
    bboxPredRegSmooth <- smooth_l1 "bbox_pred_reg_smooth" (#data := bboxPredReg .& #scalar := 1.0 .& Nil)
    bboxLoss_ <- elemwise_mul "bbox_loss_" (#lhs := bboxPredRegSmooth .& #rhs := bboxWeight .& Nil)
    bboxLoss  <- _MakeLoss "bbox_loss" (#data := bboxLoss_ .& #grad_scale := 1.0 / fromIntegral rpn_batch_rois .& Nil)

    return $ Symbol rois

