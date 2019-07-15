{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PartialTypeSignatures #-}
module Model.FasterRCNN where

import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as UVM
import Data.Array.Repa.Index
import Data.Array.Repa.Shape
import Data.Array.Repa.Slice
import qualified Data.Array.Repa as Repa
import Data.Random (shuffle, runRVar, StdRandom(..))
import Data.Random.Vector (randomElement)
import Control.Exception.Base(assert)
import Control.Lens ((^.), makeLenses)
import Control.Monad (replicateM, forM_, join)

import MXNet.Base
import MXNet.Base.Operators.NDArray (_set_value_upd)
import MXNet.Base.Operators.Symbol (
    elemwise_mul, elemwise_sub, smooth_l1, softmax, _SoftmaxOutput, _ROIPooling,
    _MakeLoss, _contrib_MultiProposal, _BlockGrad, _Custom)
import MXNet.NN.Layer
import qualified Model.VGG as VGG

symbolTrain :: [Float] -> [Float] -> Int -> Int -> Int -> Int -> Int -> Float -> Int -> Int -> [Int] -> IO (Symbol Float)
symbolTrain anchor_scales anchor_ratios num_classes
            rpn_feature_stride rpn_batch_rois rpn_pre_topk rpn_post_topk rpn_nms_thresh rpn_min_size
            rcnn_feature_stride rcnn_pooled_size= do
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
                                        .& #feature_stride := rpn_feature_stride .& #scales := anchor_scales .& #ratios := anchor_ratios
                                        .& #rpn_pre_nms_top_n := rpn_pre_topk .& #rpn_post_nms_top_n := rpn_post_topk
                                        .& #threshold := rpn_nms_thresh .& #rpn_min_size := rpn_min_size .& Nil)

    proposal <- _Custom "proposal" (#data := [rois, gtBoxes] .& #op_type := "proposal_target" .& Nil)
    [rois, label, bboxTarget, bboxWeight] <- mapM (at proposal) [0..3]

    ---------------------------
    -- cls_prob part
    --
    roiPool <- _ROIPooling "roi_pool" (#data := convFeat .& #rois := rois
                                    .& #pooled_size := rcnn_pooled_size
                                    .& #spatial_scale := 1.0 / fromIntegral rcnn_feature_stride .& Nil)
    topFeat <- VGG.getTopFeature roiPool
    clsScore <- fullyConnected "cls_score" (#data := topFeat .& #num_hidden := num_classes .& Nil)
    clsProb <- _SoftmaxOutput "cls_prob" (#data := clsScore .& #label := label .& #normalization := #batch .& Nil)

    ---------------------------
    -- bbox_loss part
    --
    bboxPred <- fullyConnected "bbox_pred" (#data := topFeat .& #num_hidden := 4 * num_classes .& Nil)
    bboxPredReg <- elemwise_sub "bbox_pred_reg" (#lhs := bboxPred .& #rhs := bboxTarget .& Nil)
    bboxPredRegSmooth <- smooth_l1 "bbox_pred_reg_smooth" (#data := bboxPredReg .& #scalar := 1.0 .& Nil)
    bboxLoss_ <- elemwise_mul "bbox_loss_" (#lhs := bboxPredRegSmooth .& #rhs := bboxWeight .& Nil)
    bboxLoss  <- _MakeLoss "bbox_loss" (#data := bboxLoss_ .& #grad_scale := 1.0 / fromIntegral rpn_batch_rois .& Nil)

    labelSG <- _BlockGrad "label_sg" (#data := label .& Nil)

    Symbol <$> group [rpnClsProb, rpnBBoxLoss, clsProb, bboxLoss, labelSG]

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

    data Operation ProposalTargetProp = ProposalTarget ProposalTargetProp
    prop_create_operator prop _ _ = return (ProposalTarget prop)

instance CustomOperation (Operation ProposalTargetProp) where
    forward (ProposalTarget prop) [ReqWrite] inputs outputs aux is_train = do
        -- :param: rois, shape of (N*nms_top_n, 5), [image_index_in_batch, bbox0, bbox1, bbox2, bbox3]
        -- :param: gt_boxes, shape of (N, M, 5), M varies per image. [bbox0, bbox1, bbox2, bbox3, class]
        let [rois, gt_boxes] = inputs
            [rois_output, label_output, bbox_target_output, bbox_weight_output] = outputs
            batch_size = prop ^. batch_images

        r_rois   <- arrayToRepa rois     >>= return . toRows2
        r_gt     <- arrayToRepa gt_boxes >>= return . toRows3

        assert (batch_size == length r_gt) (return ())

        (rois, labels, bbox_targets, bbox_weights) <- V.unzip4 <$> V.mapM (sample_batch r_rois r_gt) (V.enumFromN (0 :: Int) batch_size)
        let rois'   = Repa.computeUnboxedS $ vstack $ V.map (Repa.reshape (Z :. 1 :. 5)) $ join rois
            labels' = join labels 
            bbox_targets' = Repa.computeUnboxedS $ vstack bbox_targets
            bbox_weights' = Repa.computeUnboxedS $ vstack bbox_weights

        ndsize (NDArray rois_output)        >>= \s -> assert (s == Repa.size (Repa.extent rois'))         (return ())
        ndsize (NDArray bbox_target_output) >>= \s -> assert (s == Repa.size (Repa.extent bbox_targets')) (return ())
        ndsize (NDArray bbox_weight_output) >>= \s -> assert (s == Repa.size (Repa.extent bbox_weights')) (return ())

        copyFromVector (NDArray rois_output        :: NDArray Float) $ UV.convert $ Repa.toUnboxed rois'
        copyFromVector (NDArray label_output       :: NDArray Float) $ V.convert labels'
        copyFromVector (NDArray bbox_target_output :: NDArray Float) $ UV.convert $ Repa.toUnboxed bbox_targets'
        copyFromVector (NDArray bbox_weight_output :: NDArray Float) $ UV.convert $ Repa.toUnboxed bbox_weights'

      where
        arrayToRepa :: Repa.Shape sh => NDArrayHandle -> IO (Repa.Array Repa.U sh Float)
        arrayToRepa hdl = do
            let arr = NDArray hdl
            shape <- shapeOfList <$> ndshape arr
            vec   <- toVector arr
            return $ Repa.fromUnboxed shape $ UV.convert vec
        
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

    backward _ [ReqWrite] _ _ [in_grad_0, in_grad_1] _ _ = do
        _set_value_upd [in_grad_0] (#src := 0 .& Nil)
        _set_value_upd [in_grad_1] (#src := 0 .& Nil)


sample_rois :: V.Vector (Repa.Array _ DIM1 Float) -> V.Vector (Repa.Array _ DIM1 Float) -> Int -> Int -> Int -> Float -> [Float]
            -> IO (V.Vector (Repa.Array Repa.D Repa.DIM1 Float), 
                   V.Vector Float, 
                   Repa.Array _ Repa.DIM2 Float, 
                   Repa.Array _ Repa.DIM2 Float)
sample_rois rois gt_boxes num_classes rois_per_image fg_rois_per_image fg_overlap box_stds = do
    -- :param rois: [num_rois, 5] (batch_index, x1, y1, x2, y2)
    -- :param gt_boxes: [num_rois, 5] (x1, y1, x2, y2, cls)
    --
    -- :returns: sampled (rois, labels, regression, weight)
    let num_rois = V.length rois
    assert (num_rois == V.length gt_boxes) (return ())
    overlaps <- Repa.computeUnboxedP $ overlapMatrix rois gt_boxes

    let maxIndices = argMax overlaps
        gt_chosen  = V.map (gt_boxes %!) maxIndices

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
    forM_ [0..num_rois-1] $ \i -> do
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
            iw = min (b1 #! 2) (b2 #! 2) - max (b1 #! 0) (b2 #! 0)
            ih = min (b1 #! 3) (b2 #! 3) - max (b1 #! 1) (b2 #! 1)
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

(#!) = Repa.linearIndex
(%!) = (V.!)

vstack :: Repa.Source r Float => V.Vector (Repa.Array r Repa.DIM2 Float) -> Repa.Array Repa.D Repa.DIM2 Float
vstack = Repa.transpose . V.foldl1 (Repa.++) . V.map Repa.transpose
