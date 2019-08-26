{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
module Model.FasterRCNN where

import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as UVM
import qualified Data.HashMap.Strict as M
import Data.IORef
import Data.Array.Repa.Index
import Data.Array.Repa.Shape
import Data.Array.Repa.Slice
import qualified Data.Array.Repa as Repa
import Data.Random (shuffle, runRVar, StdRandom(..))
import Data.Random.Vector (randomElement)
import Control.Exception.Base(assert)
import Control.Lens ((^.), makeLenses)
import Control.Monad (replicateM, forM_, join)
import Control.Monad.IO.Class (liftIO)
import Text.Printf (printf)

import MXNet.Base
import MXNet.Base.Operators.NDArray (_set_value_upd, argmax, argmax_channel)
import MXNet.Base.Operators.Symbol (
    elemwise_mul, elemwise_sub, smooth_l1, softmax, _SoftmaxOutput, _ROIPooling,
    _MakeLoss, _contrib_MultiProposal, _BlockGrad, _Custom)
import qualified MXNet.Base.NDArray as A
import qualified MXNet.NN.NDArray as A
import MXNet.NN.Layer
import MXNet.NN.EvalMetric
import qualified Model.VGG as VGG

import Debug.Trace

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
    topFeat <- VGG.getTopFeature (Just "rcnn_") roiPool
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

    Symbol <$> group [rpnClsProb, rpnBBoxLoss, clsProbReshape, bboxLossReshape, labelSG]

--------------------------------

data ProposalTargetProp = ProposalTargetProp {
    _num_classes :: Int,
    _batch_images :: Int,
    _batch_rois :: Int,
    _fg_fraction :: Float,
    _fg_overlap :: Float,
    _box_stds :: [Float]
}
makeLenses ''ProposalTargetProp

instance CustomOperationProp ProposalTargetProp where
    prop_list_arguments _        = ["rois", "gt_boxes"]
    prop_list_outputs _          = ["rois_output", "label", "bbox_target", "bbox_weight"]
    prop_list_auxiliary_states _ = []
    prop_infer_shape prop [rpn_rois_shape, gt_boxes_shape] =
        let prop_batch_size   = prop ^. batch_rois
            prop_num_classes  = prop ^. num_classes
            output_rois_shape = [prop_batch_size, 5]
            label_shape       = [prop_batch_size]
            bbox_target_shape = [prop_batch_size, prop_num_classes * 4]
            bbox_weight_shape = [prop_batch_size, prop_num_classes * 4]
        in ([rpn_rois_shape, gt_boxes_shape],
            [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape],
            [])
    prop_declare_backward_dependency prop grad_out data_in data_out = []

    data Operation ProposalTargetProp = ProposalTarget ProposalTargetProp
    prop_create_operator prop _ _ = return (ProposalTarget prop)

instance CustomOperation (Operation ProposalTargetProp) where
    forward (ProposalTarget prop) [ReqWrite, ReqWrite, ReqWrite, ReqWrite] inputs outputs aux is_train = do
        -- :param: rois, shape of (N*nms_top_n, 5), [image_index_in_batch, bbox0, bbox1, bbox2, bbox3]
        -- :param: gt_boxes, shape of (N, M, 5), M varies per image. [bbox0, bbox1, bbox2, bbox3, class]
        let [rois, gt_boxes] = inputs
            [rois_output, label_output, bbox_target_output, bbox_weight_output] = outputs
            batch_size = prop ^. batch_images

        -- convert NDArray to Vector of Repa array.
        r_rois   <- toRepa @DIM2 (NDArray rois)     >>= return . toRows2
        r_gt     <- toRepa @DIM3 (NDArray gt_boxes) >>= return . toRows3

        assert (batch_size == length r_gt) (return ())

        (rois, labels, bbox_targets, bbox_weights) <- V.unzip4 <$> V.mapM (sample_batch r_rois r_gt) (V.enumFromN (0 :: Int) batch_size)
        let rois'   = vstack $ V.map (Repa.reshape (Z :. 1 :. 5)) $ join rois
            labels' = join labels
            bbox_targets' = vstack bbox_targets
            bbox_weights' = vstack bbox_weights

            rois_output_nd        = NDArray rois_output        :: NDArray Float
            bbox_target_output_nd = NDArray bbox_target_output :: NDArray Float
            bbox_weight_output_nd = NDArray bbox_weight_output :: NDArray Float
            label_output_nd       = NDArray label_output       :: NDArray Float

        ndsize rois_output_nd >>= \s -> assert (s == Repa.size (Repa.extent rois'))         (return ())
        ndsize bbox_target_output_nd >>= \s -> assert (s == Repa.size (Repa.extent bbox_targets')) (return ())
        ndsize bbox_weight_output_nd >>= \s -> assert (s == Repa.size (Repa.extent bbox_weights')) (return ())

        copyFromRepa rois_output_nd rois'
        copyFromRepa bbox_target_output_nd bbox_targets'
        copyFromRepa bbox_weight_output_nd bbox_weights'
        copyFromVector label_output_nd $ V.convert labels'

      where
        toRows2 arr = let Z :. rows :._ = Repa.extent arr
                          range = V.enumFromN (0 :: Int) rows
                      in V.map (\i -> Repa.slice arr (Z :. i :. All)) range

        toRows3 arr = let Z :. rows :. _ :. _ = Repa.extent arr
                          range = V.enumFromN (0 :: Int) rows
                      in V.map (\i -> Repa.slice arr (Z :. i :. All :. All)) range

        sample_batch :: V.Vector (Repa.Array _ DIM1 Float) -> V.Vector (Repa.Array _ DIM2 Float) -> Int -> IO (_, _, _, _)
        sample_batch r_rois r_gt index = do
            let rois_this_image   = V.filter (\roi -> floor (roi #! 0) == index) r_rois
                all_gt_this_image = toRows2 $ r_gt %! index
                gt_this_image     = V.filter (\gt  -> gt  #! 4 > 0) all_gt_this_image

            let num_rois_per_image = (prop ^. batch_rois) `div` (prop ^. batch_images)
                fg_rois_per_image = round (prop ^. fg_fraction * fromIntegral num_rois_per_image)

            -- WHY?
            -- append gt boxes to rois
            let prepend_index = (Repa.fromListUnboxed (Z :. 1) [fromIntegral index] Repa.++)
                gt_boxes_as_rois = V.map (\gt -> prepend_index $ Repa.extract (Z :. 0) (Z :. 4) gt) gt_this_image
                rois_this_image' = rois_this_image V.++ gt_boxes_as_rois

            sample_rois rois_this_image' gt_this_image
                (prop ^. num_classes) num_rois_per_image fg_rois_per_image (prop ^. fg_overlap) (prop ^. box_stds)

    backward _ [ReqWrite, ReqWrite] _ _ [in_grad_0, in_grad_1] _ _ = do
        _set_value_upd [in_grad_0] (#src := 0 .& Nil)
        _set_value_upd [in_grad_1] (#src := 0 .& Nil)


sample_rois :: V.Vector (Repa.Array _ DIM1 Float) -> V.Vector (Repa.Array _ DIM1 Float) -> Int -> Int -> Int -> Float -> [Float]
            -> IO (V.Vector (Repa.Array Repa.D Repa.DIM1 Float),
                   V.Vector Float,
                   Repa.Array _ Repa.DIM2 Float,
                   Repa.Array _ Repa.DIM2 Float)
sample_rois rois gt num_classes rois_per_image fg_rois_per_image fg_overlap box_stds = do
    -- :param rois: [num_rois, 5] (batch_index, x1, y1, x2, y2)
    -- :param gt: [num_rois, 5] (x1, y1, x2, y2, cls)
    --
    -- :returns: sampled (rois, labels, regression, weight)
    let num_rois = V.length rois
    -- print(num_rois, V.length gt_boxes)
    -- assert (num_rois == V.length gt_boxes) (return ())
    let aoi_boxes = V.map (Repa.extract (Z:.1) (Z:.4)) rois
        gt_boxes  = V.map (Repa.extract (Z:.0) (Z:.4)) gt
        overlaps  = Repa.computeUnboxedS $ overlapMatrix aoi_boxes gt_boxes

    let maxIndices = argMax overlaps
        gt_chosen  = V.map (gt %!) maxIndices

    -- a uniform sampling w/o replacement from the fg boxes if there are too many
    fg_indexes <- let fg_indexes = V.filter (\(i, j) -> Repa.index overlaps (Z :. i :. j) >= fg_overlap) (V.indexed maxIndices)
                  in if length fg_indexes > fg_rois_per_image then
                        V.fromList . take fg_rois_per_image <$> runRVar' (shuffle $ V.toList fg_indexes)
                     else
                        return fg_indexes

    -- slightly different from the orignal implemetation:
    -- a uniform sampling w/ replacement if not enough bg boxes
    let bg_rois_this_image = rois_per_image - length fg_indexes
    bg_indexes <- let bg_indexes = V.filter (\(i, j) -> Repa.index overlaps (Z :. i :. j) <  fg_overlap) (V.indexed maxIndices)
                      num_bg_indexes = length bg_indexes
                  in case compare num_bg_indexes bg_rois_this_image of
                        GT -> V.fromList . take bg_rois_this_image <$> runRVar' (shuffle $ V.toList bg_indexes)
                        LT -> V.fromList <$> runRVar' (replicateM bg_rois_this_image (randomElement bg_indexes))
                        EQ -> return bg_indexes

    let keep_indexes = V.map fst $ fg_indexes V.++ bg_indexes

        rois_keep    = V.map (rois %!) keep_indexes
        roi_box_keep = V.map (asTuple . Repa.extract (Z:.1) (Z:.4)) rois_keep

        gt_keep      = V.map (gt_chosen  %!) keep_indexes
        gt_box_keep  = V.map (asTuple . Repa.extract (Z:.0) (Z:.4)) gt_keep
        labels_keep  = V.take (length fg_indexes) (V.map (#! 4) gt_keep) V.++ V.replicate bg_rois_this_image 0

        targets = V.zipWith (bboxTransform box_stds) roi_box_keep gt_box_keep

    -- regression is indexed by class.
    bbox_target <- UVM.replicate (rois_per_image * 4 * num_classes) (0 :: Float)
    bbox_weight <- UVM.replicate (rois_per_image * 4 * num_classes) (0 :: Float)

    -- only assign regression and weights for the foreground boxes.
    forM_ [0..length fg_indexes-1] $ \i -> do
        let lbl = floor (labels_keep %! i)
            (tgt0, tgt1, tgt2, tgt3) = targets %! i :: Box
        assert (lbl >= 0 && lbl < num_classes) (return ())
        let tgt_dst = UVM.slice (i * 4 * num_classes + 4 * lbl) 4 bbox_target
        UVM.write tgt_dst 0 tgt0
        UVM.write tgt_dst 1 tgt1
        UVM.write tgt_dst 2 tgt2
        UVM.write tgt_dst 3 tgt3
        let wgh_dst = UVM.slice (i * 4 * num_classes + 4 * lbl) 4 bbox_weight
        UVM.set wgh_dst 1

    let shape = Z :. rois_per_image :. 4 * num_classes
    bbox_target <- Repa.fromUnboxed shape <$> UV.freeze bbox_target
    bbox_weight <- Repa.fromUnboxed shape <$> UV.freeze bbox_weight
    return (rois_keep, labels_keep, bbox_target, bbox_weight)

  where
    runRVar' = flip runRVar StdRandom

overlapMatrix :: V.Vector (Repa.Array Repa.D Repa.DIM1 Float) -> V.Vector (Repa.Array Repa.D Repa.DIM1 Float) -> Repa.Array Repa.D Repa.DIM2 Float
overlapMatrix rois gt = Repa.fromFunction (Z :. width :. height) calcOvp
  where
    width  = length rois
    height = length gt

    calcArea box = (box #! 2 - box #! 0 + 1) * (box #! 3 - box #! 1 + 1)
    area1 = V.map calcArea rois
    area2 = V.map calcArea gt

    calcOvp (Z :. ind_rois :. ind_gt) =
        let b1 = rois %! ind_rois
            b2 = gt   %! ind_gt
            iw = min (b1 #! 2) (b2 #! 2) - max (b1 #! 0) (b2 #! 0) + 1
            ih = min (b1 #! 3) (b2 #! 3) - max (b1 #! 1) (b2 #! 1) + 1
            areaI = iw * ih
            areaU = area1 %! ind_rois + area2 %! ind_gt - areaI
        in if iw > 0 && ih > 0 then areaI / areaU else 0

argMax overlaps =
    let Z :. m :. n = Repa.extent overlaps
        findMax row = UV.maxIndex $ Repa.toUnboxed $ Repa.computeS $ Repa.slice overlaps (Z :. row :. All)
    in V.map findMax $ V.enumFromN (0 :: Int) m

type Box = (Float, Float, Float, Float)
whctr :: Box -> Box
whctr (x0, y0, x1, y1) = (w, h, x, y)
  where
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    x = x0 + 0.5 * (w - 1)
    y = y0 + 0.5 * (h - 1)

asTuple :: Repa.Array Repa.D Repa.DIM1 Float -> (Float, Float, Float, Float)
asTuple box = (box #! 0, box #! 1, box #! 2, box #! 3)

bboxTransform :: [Float] -> Box -> Box -> Box
bboxTransform [std0, std1, std2, std3] box1 box2 =
    let (w1, h1, cx1, cy1) = whctr box1
        (w2, h2, cx2, cy2) = whctr box2
        dx = (cx2 - cx1) / (w1 + 1e-14) / std0
        dy = (cy2 - cy1) / (h1 + 1e-14) / std1
        dw = log (w2 / w1) / std2
        dh = log (h2 / h1) / std3
    in (dx, dy, dw, dh)

(#!) :: (Shape sh, Repa.Source r e) => Repa.Array r sh e -> Int -> e 
(#!) = Repa.linearIndex
(%!) = (V.!)

vstack :: Repa.Source r Float => V.Vector (Repa.Array r Repa.DIM2 Float) -> Repa.Array Repa.D Repa.DIM2 Float
vstack = Repa.transpose . V.foldl1 (Repa.++) . V.map Repa.transpose


test_sample_rois = let
        v1 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 0.8, 0.8, 2.2, 2.2]
        v2 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 2.2, 2.2, 4.5, 4.5]
        v3 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 4.2, 1, 6.5, 2.8]
        v4 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 6, 3, 7, 4]
        rois = V.fromList $ map Repa.delay [v1, v2, v3, v4]
        g1 = Repa.fromListUnboxed (Z:.5::DIM1) [1,1,2,2,1]
        g2 = Repa.fromListUnboxed (Z:.5::DIM1) [2,3,3,4,1]
        g3 = Repa.fromListUnboxed (Z:.5::DIM1) [4,1,6,3,2]
        gt_boxes = V.fromList $ map Repa.delay [g1, g2, g3]
      in sample_rois rois gt_boxes 3 6 2 0.5 [0.1, 0.1, 0.1, 0.1]


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
        let cls_prob = outputs !! cindex
            label    = outputs !! lindex

        cls_prob <- A.makeNDArrayLike cls_prob contextCPU >>= A.copy cls_prob
        [pred_class] <- argmax (#data := unNDArray cls_prob .& #axis := Just 1 .& Nil)
        pred_class <- V.convert <$> toVector (NDArray pred_class)
        label <- V.convert <$> toVector label

        let pairs = V.zip label pred_class
            equal = V.filter (uncurry (==)) pairs

        modifyIORef' sumRef (+ length equal)
        modifyIORef' cntRef (+ length pairs)

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
        return $ printf "<RPNLogLoss: %0.3f>" (realToFrac s / fromIntegral n :: Float)

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
        let mask = Repa.map (/= -1) label :: Repa.Array _ _ Bool

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
        return $ printf "<RCNNLogLoss: %0.3f>" (realToFrac s / fromIntegral n :: Float)

    evaluate (RCNNLogLossMetricData phase cindex lindex cntRef sumRef) bindings outputs = liftIO $  do
        let cls_prob = outputs !! cindex
            label    = outputs !! lindex

        cls_prob <- toRepa @DIM2 cls_prob
        label    <- toRepa @DIM1 label
        
        let Z:.size = Repa.extent label 
            cls = Repa.fromFunction (Z :. size) (\ (Z :. i) -> cls_prob Repa.! (Z :. i :. (floor $ label #! i)))

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

        bbox_loss <- toRepa @DIM2 bbox_loss
        all_loss  <- Repa.sumAllP bbox_loss

        label     <- toRepa @DIM1 label
        all_pos   <- Repa.sumAllP $ Repa.map (\w -> if w > 0 then 1 else 0) label

        modifyIORef' sumRef (+ realToFrac all_loss)
        modifyIORef' cntRef (+ all_pos)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase ++ "_acc") acc

constant :: (Shape sh, UV.Unbox a) => sh -> a -> Repa.Array Repa.U sh a
constant shp val = Repa.fromListUnboxed shp (replicate (size shp) val)
