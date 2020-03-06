{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
module Model.ProposalTarget where

import Control.Lens ((^.), makeLenses)
import Control.Monad (replicateM, forM_, join)
import Control.Exception.Base(assert)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as UVM
import qualified Data.Array.Repa as Repa
import Data.Array.Repa.Index
import Data.Array.Repa.Shape
import Data.Array.Repa.Slice
import Data.Random (shuffle, runRVar, StdRandom(..))
import Data.Random.Vector (randomElement)

import MXNet.Base
import MXNet.Base.Operators.NDArray (_set_value_upd)

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
                      in V.map (\i -> Repa.computeUnboxedS $ Repa.slice arr (Z :. i :. All)) range

        toRows3 arr = let Z :. rows :. _ :. _ = Repa.extent arr
                          range = V.enumFromN (0 :: Int) rows
                      in V.map (\i -> Repa.computeUnboxedS $ Repa.slice arr (Z :. i :. All :. All)) range

        sample_batch :: V.Vector (Repa.Array Repa.U DIM1 Float) -> V.Vector (Repa.Array _ DIM2 Float) -> Int -> IO (_, _, _, _)
        sample_batch r_rois r_gt index = do
            let rois_this_image   = V.filter (\roi -> floor (roi #! 0) == index) r_rois
                all_gt_this_image = toRows2 $ r_gt %! index
                gt_this_image     = V.filter (\gt  -> gt  #! 4 > 0) all_gt_this_image

            let num_rois_per_image = (prop ^. batch_rois) `div` (prop ^. batch_images)
                fg_rois_per_image = round (prop ^. fg_fraction * fromIntegral num_rois_per_image)

            -- WHY?
            -- append gt boxes to rois
            let prepend_index = Repa.computeUnboxedS . (Repa.fromListUnboxed (Z :. 1) [fromIntegral index] Repa.++)
                gt_boxes_as_rois = V.map (\gt -> prepend_index $ Repa.extract (Z :. 0) (Z :. 4) gt) gt_this_image
                rois_this_image' = rois_this_image V.++ gt_boxes_as_rois

            sample_rois rois_this_image' gt_this_image
                (prop ^. num_classes) num_rois_per_image fg_rois_per_image (prop ^. fg_overlap) (prop ^. box_stds)

    backward _ [ReqWrite, ReqWrite] _ _ [in_grad_0, in_grad_1] _ _ = do
        _set_value_upd [in_grad_0] (#src := 0 .& Nil)
        _set_value_upd [in_grad_1] (#src := 0 .& Nil)


sample_rois :: V.Vector (Repa.Array Repa.U DIM1 Float) -> V.Vector (Repa.Array Repa.U DIM1 Float) -> Int -> Int -> Int -> Float -> [Float]
            -> IO (V.Vector (Repa.Array Repa.U Repa.DIM1 Float),
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
    let aoi_boxes = V.map (Repa.computeUnboxedS . Repa.extract (Z:.1) (Z:.4)) rois
        gt_boxes  = V.map (Repa.computeUnboxedS . Repa.extract (Z:.0) (Z:.4)) gt
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
        roi_box_keep = V.map (asTuple . Repa.computeUnboxedS . Repa.extract (Z:.1) (Z:.4)) rois_keep

        gt_keep      = V.map (gt_chosen  %!) keep_indexes
        gt_box_keep  = V.map (asTuple . Repa.computeUnboxedS . Repa.extract (Z:.0) (Z:.4)) gt_keep
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

overlapMatrix :: V.Vector (Repa.Array Repa.U Repa.DIM1 Float) -> V.Vector (Repa.Array Repa.U Repa.DIM1 Float) -> Repa.Array Repa.D Repa.DIM2 Float
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

asTuple :: Repa.Array Repa.U Repa.DIM1 Float -> (Float, Float, Float, Float)
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

(#!) :: (Shape sh, UV.Unbox e) => Repa.Array Repa.U sh e -> Int -> e
(#!) = Repa.linearIndex
(%!) = (V.!)

vstack :: Repa.Source r Float => V.Vector (Repa.Array r Repa.DIM2 Float) -> Repa.Array Repa.D Repa.DIM2 Float
vstack = Repa.transpose . V.foldl1 (Repa.++) . V.map Repa.transpose


test_sample_rois = let
        v1 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 0.8, 0.8, 2.2, 2.2]
        v2 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 2.2, 2.2, 4.5, 4.5]
        v3 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 4.2, 1, 6.5, 2.8]
        v4 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 6, 3, 7, 4]
        rois = V.fromList [v1, v2, v3, v4]
        g1 = Repa.fromListUnboxed (Z:.5::DIM1) [1,1,2,2,1]
        g2 = Repa.fromListUnboxed (Z:.5::DIM1) [2,3,3,4,1]
        g3 = Repa.fromListUnboxed (Z:.5::DIM1) [4,1,6,3,2]
        gt_boxes = V.fromList [g1, g2, g3]
      in sample_rois rois gt_boxes 3 6 2 0.5 [0.1, 0.1, 0.1, 0.1]



