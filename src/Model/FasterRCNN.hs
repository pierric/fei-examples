{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
module Model.FasterRCNN where

import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Unboxed as UV
import qualified Data.HashMap.Strict as M
import Data.IORef
import Data.Array.Repa.Index
import Data.Array.Repa.Shape
import qualified Data.Array.Repa as Repa
import Control.Monad.IO.Class (liftIO)
import Text.Printf (printf)

import MXNet.Base
import MXNet.Base.Operators.NDArray (argmax, argmax_channel)
import MXNet.Base.Operators.Symbol (
    elemwise_mul, elemwise_sub, smooth_l1, softmax, _SoftmaxOutput, _ROIPooling,
    _MakeLoss, _contrib_MultiProposal, _BlockGrad, _Custom)
import qualified MXNet.Base.NDArray as A
import qualified MXNet.NN.NDArray as A
import MXNet.NN.Layer
import MXNet.NN.EvalMetric
import qualified Model.VGG as VGG
import Model.ProposalTarget 


data RcnnConfiguration = RcnnConfiguration {
    rpn_anchor_scales :: [Int],
    rpn_anchor_ratios :: [Float],
    rpn_feature_stride :: Int,
    rpn_batch_rois :: Int,
    rpn_pre_topk :: Int,
    rpn_post_topk :: Int,
    rpn_nms_thresh :: Float,
    rpn_min_size :: Int,
    rpn_fg_fraction :: Float,
    rpn_fg_overlap :: Float,
    rpn_bg_overlap :: Float,
    rpn_allowd_border :: Int,
    rcnn_num_classes   :: Int,
    rcnn_feature_stride :: Int,
    rcnn_pooled_size :: [Int],
    rcnn_batch_rois  :: Int,
    rcnn_batch_size  :: Int,
    rcnn_fg_fraction :: Float,
    rcnn_fg_overlap  :: Float,
    rcnn_bbox_stds   :: [Float],
    pretrained_weights :: String
} deriving Show

symbolTrain :: RcnnConfiguration -> IO (Symbol Float)
symbolTrain RcnnConfiguration{..} =  do
    let numAnchors = length rpn_anchor_scales * length rpn_anchor_ratios
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

    -- VGG-15 without the last pooling layer
    convFeat <- VGG.getFeature dat [2, 2, 3, 3, 3] [64, 128, 256, 512, 512] False False

    rpnConv <- convolution "rpn_conv_3x3" (#data := convFeat .& #kernel := [3,3] .& #pad := [1,1] .& #num_filter := 512 .& Nil)
    rpnRelu <- activation "rpn_relu" (#data := rpnConv .& #act_type := #relu .& Nil)

    ---------------------------
    -- rpn_clas_prob part
    --
    -- per pixel: fore/back-ground classification
    rpnClsScore <- convolution "rpn_cls_score" (#data := rpnRelu .& #kernel := [1,1] .& #pad := [0,0] .& #num_filter := 2 * numAnchors .& Nil)
    rpnClsScoreReshape <- reshape "rpn_cls_score_reshape" (#data := rpnClsScore .& #shape := [0, 2, -1, 0] .& Nil)
    -- rpnClsProb output shape: (batch_size, [Pr(foreground), Pr(background)], numAnchors * feat_height, feat_width)
    rpnClsProb <- _SoftmaxOutput "rpn_cls_prob" (#data := rpnClsScoreReshape .& #label := rpnLabel .& #multi_output := True
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
                                        .& #feature_stride := rpn_feature_stride .& #scales := map fromIntegral rpn_anchor_scales .& #ratios := rpn_anchor_ratios
                                        .& #rpn_pre_nms_top_n := rpn_pre_topk .& #rpn_post_nms_top_n := rpn_post_topk
                                        .& #threshold := rpn_nms_thresh .& #rpn_min_size := rpn_min_size .& Nil)

    proposal <- _Custom "proposal" (#data := [rois, gtBoxes]
                                 .& #op_type     := "proposal_target"
                                 .& #num_classes :≅ rcnn_num_classes
                                 .& #batch_images:≅ rcnn_batch_size
                                 .& #batch_rois  :≅ rcnn_batch_rois
                                 .& #fg_fraction :≅ rcnn_fg_fraction
                                 .& #fg_overlap  :≅ rcnn_fg_overlap
                                 .& #box_stds    :≅ rcnn_bbox_stds
                                 .& Nil)
    [rois, label, bboxTarget, bboxWeight] <- mapM (at proposal) [0..3]

    ---------------------------
    -- cls_prob part
    --
    roiPool <- _ROIPooling "roi_pool" (#data := convFeat .& #rois := rois
                                    .& #pooled_size := rcnn_pooled_size
                                    .& #spatial_scale := 1.0 / fromIntegral rcnn_feature_stride .& Nil)
    topFeat <- VGG.getTopFeature Nothing roiPool
    clsScore <- fullyConnected "cls_score" (#data := topFeat .& #num_hidden := rcnn_num_classes .& Nil)
    clsProb <- _SoftmaxOutput "cls_prob" (#data := clsScore .& #label := label .& #normalization := #batch .& Nil)

    ---------------------------
    -- bbox_loss part
    --
    bboxPred <- fullyConnected "bbox_pred" (#data := topFeat .& #num_hidden := 4 * rcnn_num_classes .& Nil)
    bboxPredReg <- elemwise_sub "bbox_pred_reg" (#lhs := bboxPred .& #rhs := bboxTarget .& Nil)
    bboxPredRegSmooth <- smooth_l1 "bbox_pred_reg_smooth" (#data := bboxPredReg .& #scalar := 1.0 .& Nil)
    bboxLoss_ <- elemwise_mul "bbox_loss_" (#lhs := bboxPredRegSmooth .& #rhs := bboxWeight .& Nil)
    bboxLoss  <- _MakeLoss "bbox_loss" (#data := bboxLoss_ .& #grad_scale := 1.0 / fromIntegral rcnn_batch_rois .& Nil)

    labelReshape    <- reshape "label_reshape"     (#data := label    .& #shape := [rcnn_batch_size, -1] .& Nil)
    clsProbReshape  <- reshape "cls_prob_reshape"  (#data := clsProb  .& #shape := [rcnn_batch_size, -1, rcnn_num_classes] .& Nil)
    bboxLossReshape <- reshape "bbox_loss_reshape" (#data := bboxLoss .& #shape := [rcnn_batch_size, -1, 4 * rcnn_num_classes] .& Nil)
    labelSG <- _BlockGrad "label_sg" (#data := labelReshape .& Nil)

    -- include topFeatures and clsScores for debug
    topFeatuSG <- _BlockGrad "topfeatu_sg" (#data := topFeat .& Nil)
    clsScoreSG <- _BlockGrad "clsscore_sg" (#data := clsScore .& Nil)
    -- roisSG     <- _BlockGrad "rois_sg"     (#data := rois.& Nil)
    -- bboxTSG    <- _BlockGrad "bboxT_sg"    (#data := bboxTarget .& Nil)
    -- bboxWSG    <- _BlockGrad "bboxW_sg"    (#data := bboxWeight .& Nil)

    Symbol <$> group [rpnClsProb, rpnBBoxLoss, clsProbReshape, bboxLossReshape, labelSG, topFeatuSG, clsScoreSG]


symbolInfer :: RcnnConfiguration -> IO (Symbol Float)
symbolInfer RcnnConfiguration{..} = do
    let numAnchors = length rpn_anchor_scales * length rpn_anchor_ratios
    -- dat:
    dat <- variable "data"
    -- imInfo:
    imInfo <- variable "im_info"

    -- VGG-15 without the last pooling layer
    convFeat <- VGG.getFeature dat [2, 2, 3, 3, 3] [64, 128, 256, 512, 512] False False

    rpnConv <- convolution "rpn_conv_3x3" (#data := convFeat .& #kernel := [3,3] .& #pad := [1,1] .& #num_filter := 512 .& Nil)
    rpnRelu <- activation "rpn_relu" (#data := rpnConv .& #act_type := #relu .& Nil)

    ---------------------------
    -- rpn_clas_prob part
    --
    -- per pixel: fore/back-ground classification
    rpnClsScore <- convolution "rpn_cls_score" (#data := rpnRelu .& #kernel := [1,1] .& #pad := [0,0] .& #num_filter := 2 * numAnchors .& Nil)
    rpnClsScoreReshape <- reshape "rpn_cls_score_reshape" (#data := rpnClsScore .& #shape := [0, 2, -1, 0] .& Nil)
    rpnClsAct <- softmax "rpn_cls_act" (#data := rpnClsScoreReshape .& #axis := 1 .& Nil)
    rpnClsActReshape <- reshape "rpn_cls_act_reshape" (#data := rpnClsAct .& #shape := [0, 2 * numAnchors, -1, 0] .& Nil)

    ---------------------------
    -- rpn_bbox part
    rpnBBoxPred <- convolution "rpn_bbox_pred" (#data := rpnRelu .& #kernel := [1,1] .& #pad := [0,0] .& #num_filter := 4 * numAnchors .& Nil)

    ---------------------------
    rois <- _contrib_MultiProposal "rois" (#cls_prob := rpnClsActReshape .& #bbox_pred := rpnBBoxPred .& #im_info := imInfo
                                        .& #feature_stride := rpn_feature_stride .& #scales := map fromIntegral rpn_anchor_scales .& #ratios := rpn_anchor_ratios
                                        .& #rpn_pre_nms_top_n := rpn_pre_topk .& #rpn_post_nms_top_n := rpn_post_topk
                                        .& #threshold := rpn_nms_thresh .& #rpn_min_size := rpn_min_size .& Nil)

    ---------------------------
    -- cls_prob part
    --
    roiPool <- _ROIPooling "roi_pool" (#data := convFeat .& #rois := rois
                                    .& #pooled_size := rcnn_pooled_size
                                    .& #spatial_scale := 1.0 / fromIntegral rcnn_feature_stride .& Nil)
    topFeat <- VGG.getTopFeature Nothing roiPool
    clsScore <- fullyConnected "cls_score" (#data := topFeat .& #num_hidden := rcnn_num_classes .& Nil)
    clsProb <- softmax "cls_prob" (#data := clsScore .& Nil)

    ---------------------------
    -- bbox_loss part
    --
    bboxPred <- fullyConnected "bbox_pred" (#data := topFeat .& #num_hidden := 4 * rcnn_num_classes .& Nil)

    Symbol <$> group [rois, clsProb, bboxPred]

--------------------------------
data RPNAccMetric a = RPNAccMetric Int String

instance EvalMetricMethod RPNAccMetric where
    data MetricData RPNAccMetric a = RPNAccMetricData String Int String (IORef Int) (IORef Int)
    newMetric phase (RPNAccMetric oindex label) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNAccMetricData phase oindex label a b

    format (RPNAccMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ printf "<RPNAcc: %0.2f>" (100 * fromIntegral s / fromIntegral n :: Float)

    evaluate (RPNAccMetricData phase oindex lname cntRef sumRef) bindings outputs = liftIO $  do
        let label = bindings M.! lname
            pred  = outputs !! oindex

        pred <- A.makeNDArrayLike pred contextCPU >>= A.copy pred
        [pred_label] <- argmax_channel (#data := unNDArray pred .& Nil)
        pred_label <- V.convert <$> toVector (NDArray pred_label)
        label <- V.convert <$> toVector label

        let pairs = V.filter ((/= -1) . fst) $ V.zip label pred_label
            equal = V.filter (uncurry (==)) pairs

        modifyIORef' sumRef (+ length equal)
        modifyIORef' cntRef (+ length pairs)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = fromIntegral s / fromIntegral n
        return $ M.singleton (phase ++ "_acc") acc


data RCNNAccMetric a = RCNNAccMetric Int Int

instance EvalMetricMethod RCNNAccMetric where
    data MetricData RCNNAccMetric a = RCNNAccMetricData String Int Int (IORef Int) (IORef Int)
    newMetric phase (RCNNAccMetric cindex lindex) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RCNNAccMetricData phase cindex lindex a b

    format (RCNNAccMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ printf "<RCNNAcc: %0.2f>" (100 * fromIntegral s / fromIntegral n :: Float)

    evaluate (RCNNAccMetricData phase cindex lindex cntRef sumRef) bindings outputs = liftIO $  do
        -- cls_prob: (batch_size, #num_anchors*feat_w*feat_h, #num_classes)
        -- label:    (batch_size, #num_anchors*feat_w*feat_h)
        let cls_prob = outputs !! cindex
            label    = outputs !! lindex

        cls_prob <- A.makeNDArrayLike cls_prob contextCPU >>= A.copy cls_prob
        [pred_class] <- argmax (#data := unNDArray cls_prob .& #axis := Just 2 .& Nil)

        -- debug only
        s1 <- ndshape (NDArray pred_class :: NDArray Float)
        v1 <- toVector (NDArray pred_class :: NDArray Float)
        print (s1, SV.map floor v1 :: SV.Vector Int)

        s1 <- ndshape label
        v1 <- toVector label
        print (s1, SV.map floor v1 :: SV.Vector Int)

        pred_class <- toRepa @DIM2 (NDArray pred_class)
        label <- toRepa @DIM2 label

        let pairs = UV.zip (Repa.toUnboxed label) (Repa.toUnboxed pred_class)
            equal = UV.filter (uncurry (==)) pairs

        modifyIORef' sumRef (+ UV.length equal)
        modifyIORef' cntRef (+ UV.length pairs)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = fromIntegral s / fromIntegral n
        return $ M.singleton (phase ++ "_acc") acc

data RPNLogLossMetric a = RPNLogLossMetric Int String

instance EvalMetricMethod RPNLogLossMetric where
    data MetricData RPNLogLossMetric a = RPNLogLossMetricData String Int String (IORef Int) (IORef Double)
    newMetric phase (RPNLogLossMetric cindex lname) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNLogLossMetricData phase cindex lname a b

    format (RPNLogLossMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ printf "<RPNLogLoss: %0.4f>" (realToFrac s / fromIntegral n :: Float)

    evaluate (RPNLogLossMetricData phase cindex lname cntRef sumRef) bindings outputs = liftIO $  do
        let cls_prob = outputs !! cindex
            label    = bindings M.! lname

        -- (batch_size, #num_anchors*feat_w*feat_h) to (batch_size*#num_anchors*feat_w*feat_h,)
        label <- A.reshape label [-1]
        label <- toRepa @DIM1 label
        let Z :. size = Repa.extent label

        -- (batch_size, #channel, #num_anchors*feat_w, feat_h) to (batch_size, #channel, #num_anchors*feat_w*feat_h)
        -- to (batch_size, #num_anchors*feat_w*feat_h, #channel) to (batch_size*#num_anchors*feat_w*feat_h, #channel)
        cls_prob <- A.makeNDArrayLike cls_prob contextCPU >>= A.copy cls_prob
        pred  <- A.reshape cls_prob [0, 0, -1] >>= flip A.transpose [0, 2, 1] >>= flip A.reshape [size, -1]
        pred  <- toRepa @DIM2 pred

        -- mark out labels where value -1
        let mask = Repa.computeUnboxedS $ Repa.map (/= -1) label

        pred  <- Repa.selectP (mask #!) (\i -> pred  Repa.! (Z :. i :. (floor $ label #! i))) size
        -- traceShowM pred
        label <- Repa.selectP (mask #!) (label #!) size

        let pred_with_ep = Repa.map ((0 -) . log)  (pred Repa.+^ constant (Z :. size) 1e-14)
        cls_loss <- Repa.foldP (+) 0 pred_with_ep

        let cls_loss_val = realToFrac (cls_loss #! 0)
        modifyIORef' sumRef (+ cls_loss_val)
        modifyIORef' cntRef (+ size)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase ++ "_acc") acc

data RCNNLogLossMetric a = RCNNLogLossMetric Int Int

instance EvalMetricMethod RCNNLogLossMetric where
    data MetricData RCNNLogLossMetric a = RCNNLogLossMetricData String Int Int (IORef Int) (IORef Double)
    newMetric phase (RCNNLogLossMetric cindex lindex) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RCNNLogLossMetricData phase cindex lindex a b

    format (RCNNLogLossMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ printf "<RCNNLogLoss: %0.4f>" (realToFrac s / fromIntegral n :: Float)

    evaluate (RCNNLogLossMetricData phase cindex lindex cntRef sumRef) bindings outputs = liftIO $  do
        let cls_prob = outputs !! cindex
            label    = outputs !! lindex

        cls_prob <- toRepa @DIM3 cls_prob
        label    <- toRepa @DIM2 label

        let lbl_shp@(Z :. _ :. size) = Repa.extent label
            cls = Repa.fromFunction lbl_shp (\ pos@(Z :. bi :. ai) -> cls_prob Repa.! (Z :. bi :. ai :. (floor $ label Repa.! pos)))

        cls_loss_val <- Repa.sumAllP $ Repa.map (\v -> - log(1e-14 + v)) cls
        -- traceShowM cls_loss_val
        modifyIORef' sumRef (+ realToFrac cls_loss_val)
        modifyIORef' cntRef (+ size)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase ++ "_acc") acc

data RPNL1LossMetric a = RPNL1LossMetric Int String

instance EvalMetricMethod RPNL1LossMetric where
    data MetricData RPNL1LossMetric a = RPNL1LossMetricData String Int String (IORef Int) (IORef Double)
    newMetric phase (RPNL1LossMetric bindex blabel) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNL1LossMetricData phase bindex blabel a b

    format (RPNL1LossMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ printf "<RPNL1Loss: %0.3f>" (realToFrac s / fromIntegral n :: Float)

    evaluate (RPNL1LossMetricData phase bindex blabel cntRef sumRef) bindings outputs = liftIO $  do
        let bbox_loss   = outputs !! bindex
            bbox_weight = bindings M.! blabel

        bbox_loss   <- toRepa @DIM4 bbox_loss
        all_loss    <- Repa.sumAllP bbox_loss

        bbox_weight <- toRepa @DIM4 bbox_weight
        all_pos_weight <- Repa.sumAllP $ Repa.map (\w -> if w > 0 then 1 else 0) bbox_weight

        modifyIORef' sumRef (+ realToFrac all_loss)
        modifyIORef' cntRef (+ (all_pos_weight `div` 4))

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase ++ "_acc") acc

data RCNNL1LossMetric a = RCNNL1LossMetric Int Int

instance EvalMetricMethod RCNNL1LossMetric where
    data MetricData RCNNL1LossMetric a = RCNNL1LossMetricData String Int Int (IORef Int) (IORef Double)
    newMetric phase (RCNNL1LossMetric bindex lindex) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RCNNL1LossMetricData phase bindex lindex a b

    format (RCNNL1LossMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ printf "<RCNNL1Loss: %0.3f>" (realToFrac s / fromIntegral n :: Float)

    evaluate (RCNNL1LossMetricData phase bindex lindex cntRef sumRef) bindings outputs = liftIO $ do
        let bbox_loss = outputs !! bindex
            label     = outputs !! lindex

        bbox_loss <- toRepa @DIM3 bbox_loss
        all_loss  <- Repa.sumAllP bbox_loss

        label     <- toRepa @DIM2 label
        all_pos   <- Repa.sumAllP $ Repa.map (\w -> if w > 0 then 1 else 0) label

        modifyIORef' sumRef (+ realToFrac all_loss)
        modifyIORef' cntRef (+ all_pos)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase ++ "_acc") acc

constant :: (Shape sh, UV.Unbox a) => sh -> a -> Repa.Array Repa.U sh a
constant shp val = Repa.fromListUnboxed shp (replicate (size shp) val)
