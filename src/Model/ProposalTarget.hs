{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ExplicitForAll #-}
module Model.ProposalTarget where

import Control.Lens ((^.), makeLenses)
import Control.Monad (replicateM, forM_, join)
import Control.Exception.Base(assert, throw, Exception)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as UVM
import qualified Data.Array.Repa as Repa
import Data.Array.Repa (Array, U, D)
import Data.Array.Repa.Index
import Data.Array.Repa.Shape
import Data.Array.Repa.Slice
import Data.Random (shuffle, runRVar, StdRandom(..))
import Data.Random.Vector (randomElement)
import Text.PrettyPrint.ANSI.Leijen (putDoc, Pretty(..), (<+>), text)

import MXNet.Base
import MXNet.Base.Operators.NDArray (_set_value_upd)

instance (Pretty e, UV.Unbox e, Shape d) => Pretty (Array U d e) where
    pretty arr = text (Repa.showShape $ Repa.extent arr) <+> pretty (UV.toList $ Repa.toUnboxed arr)

data ProposalTargetProp = ProposalTargetProp {
    _pt_num_classes :: Int,
    _pt_batch_images :: Int,
    _pt_batch_rois :: Int,
    _pt_fg_fraction :: Float,
    _pt_fg_overlap :: Float,
    _pt_box_stds :: [Float]
}
makeLenses ''ProposalTargetProp

instance CustomOperationProp ProposalTargetProp where
    prop_list_arguments _        = ["rois", "gt_boxes"]
    prop_list_outputs _          = ["rois_output", "label", "bbox_target", "bbox_weight"]
    prop_list_auxiliary_states _ = []
    prop_infer_shape prop [rpn_rois_shape, gt_boxes_shape] =
        let prop_batch_size   = prop ^. pt_batch_rois
            prop_num_classes  = prop ^. pt_num_classes
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
            batch_size = prop ^. pt_batch_images

        -- convert NDArray to Vector of Repa array.
        r_rois   <- toRepa @DIM2 (NDArray rois)     >>= return . toRows2
        r_gt     <- toRepa @DIM3 (NDArray gt_boxes) >>= return . toRows3

        assert (batch_size == length r_gt) (return ())

        (rois, labels, bbox_targets, bbox_weights) <- V.unzip4 <$> V.mapM (sample_batch r_rois r_gt) (V.enumFromN (0 :: Int) batch_size)
        let rois'   = vstack $ V.map (Repa.computeUnboxedS . Repa.reshape (Z :. 1 :. 5 :: DIM2)) $ join rois
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

        sample_batch r_rois r_gt index = do
            let rois_this_image   = V.filter (\roi -> floor (roi #! 0) == index) r_rois
                all_gt_this_image = toRows2 $ r_gt %! index
                gt_this_image     = V.filter (\gt  -> gt  #! 4 > 0) all_gt_this_image

            -- WHY?
            -- append gt boxes to rois
            let prepend_index = Repa.computeUnboxedS . (Repa.fromListUnboxed (Z :. 1) [fromIntegral index] Repa.++)
                gt_boxes_as_rois = V.map (\gt -> prepend_index $ Repa.extract (Z :. 0) (Z :. 4) gt) gt_this_image
                rois_this_image' = rois_this_image V.++ gt_boxes_as_rois

            sample_rois rois_this_image' gt_this_image prop

    backward _ [ReqWrite, ReqWrite] _ _ [in_grad_0, in_grad_1] _ _ = do
        _set_value_upd [in_grad_0] (#src := 0 .& Nil)
        _set_value_upd [in_grad_1] (#src := 0 .& Nil)


sample_rois :: V.Vector (Array U DIM1 Float) -> V.Vector (Array U DIM1 Float)
            -> ProposalTargetProp
            -> IO (V.Vector (Array U DIM1 Float),
                   V.Vector Float,
                   Array U DIM2 Float,
                   Array U DIM2 Float)
sample_rois rois gt prop = do
    -- :param rois: [num_rois, 5] (batch_index, x1, y1, x2, y2)
    -- :param gt: [num_rois, 5] (x1, y1, x2, y2, cls)
    --
    -- :returns: sampled (rois, labels, regression, weight)
    -- let num_rois = V.length rois
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
        roi_box_keep = V.map (Repa.computeUnboxedS . Repa.extract (Z:.1) (Z:.4)) rois_keep

        gt_keep      = V.map (gt_chosen  %!) keep_indexes
        gt_box_keep  = V.map (Repa.computeUnboxedS . Repa.extract (Z:.0) (Z:.4)) gt_keep
        labels_keep  = V.take (length fg_indexes) (V.map (#! 4) gt_keep) V.++ V.replicate bg_rois_this_image 0

        targets = V.zipWith (bboxTransform box_stds) roi_box_keep gt_box_keep

    -- regression is indexed by class.
    bbox_target <- UVM.replicate (rois_per_image * 4 * num_classes) (0 :: Float)
    bbox_weight <- UVM.replicate (rois_per_image * 4 * num_classes) (0 :: Float)

    -- only assign regression and weights for the foreground boxes.
    forM_ [0..length fg_indexes-1] $ \i -> do
        let lbl = floor (labels_keep %! i)
            tgt = targets %! i :: RBox
        assert (lbl >= 0 && lbl < num_classes) (return ())
        let tgt_dst = UVM.slice (i * 4 * num_classes + 4 * lbl) 4 bbox_target
        UVM.write tgt_dst 0 (tgt #! 0)
        UVM.write tgt_dst 1 (tgt #! 1)
        UVM.write tgt_dst 2 (tgt #! 2)
        UVM.write tgt_dst 3 (tgt #! 3)
        let wgh_dst = UVM.slice (i * 4 * num_classes + 4 * lbl) 4 bbox_weight
        UVM.set wgh_dst 1

    let shape = Z :. rois_per_image :. 4 * num_classes
    bbox_target <- Repa.fromUnboxed shape <$> UV.freeze bbox_target
    bbox_weight <- Repa.fromUnboxed shape <$> UV.freeze bbox_weight
    return (rois_keep, labels_keep, bbox_target, bbox_weight)

  where
    runRVar' = flip runRVar StdRandom
    num_classes = prop ^. pt_num_classes
    rois_per_image = (prop ^. pt_batch_rois) `div` (prop ^. pt_batch_images)
    fg_rois_per_image = round (prop ^. pt_fg_fraction * fromIntegral rois_per_image)
    fg_overlap = prop ^. pt_fg_overlap
    box_stds = Repa.fromListUnboxed (Z:.4) (prop ^. pt_box_stds)

overlapMatrix :: V.Vector (Array U DIM1 Float) -> V.Vector (Array U DIM1 Float) -> Array D DIM2 Float
overlapMatrix rois gt = Repa.fromFunction (Z :. width :. height) calcOvp
  where
    width  = length rois
    height = length gt

    area1 = V.map bboxArea rois
    area2 = V.map bboxArea gt

    calcOvp (Z :. ind_rois :. ind_gt) =
        case bboxIntersect (rois %! ind_rois) (gt   %! ind_gt) of
           Nothing -> 0
           Just boxI -> let areaI = bboxArea boxI
                            areaU = area1 %! ind_rois + area2 %! ind_gt - areaI
                        in areaI / areaU

bboxArea :: RBox -> Float
bboxArea box = (box #! 2 - box #! 0 + 1) * (box #! 3 - box #! 1 + 1)

bboxIntersect :: RBox -> RBox -> Maybe RBox
bboxIntersect box1 box2 | not valid = Nothing
                        | otherwise = Just $ Repa.fromListUnboxed (Z:.4) [x1, y1, x2, y2]
  where
    valid = x2 - x1 > 0 && y2 - y1 > 0
    x1 = max (box1 #! 0) (box2 #! 0)
    x2 = min (box1 #! 2) (box2 #! 2)
    y1 = max (box1 #! 1) (box2 #! 1)
    y2 = min (box1 #! 3) (box2 #! 3)

bboxIOU :: RBox -> RBox -> Float
bboxIOU box1 box2 = case bboxIntersect box1 box2 of
                      Nothing -> 0
                      Just boxI -> let areaI = bboxArea boxI
                                       areaU = bboxArea box1 + bboxArea box2 - areaI
                                   in areaI / areaU

whctr :: RBox -> RBox
whctr box1 = Repa.fromListUnboxed (Z:.4) [w, h, x, y]
  where
    [x0, y0, x1, y1] = Repa.toList box1
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    x = x0 + 0.5 * (w - 1)
    y = y0 + 0.5 * (h - 1)

bboxTransform :: RBox -> RBox -> RBox -> RBox
bboxTransform stds box1 box2 =
    let [w1, h1, cx1, cy1] = Repa.toList $ whctr box1
        [w2, h2, cx2, cy2] = Repa.toList $ whctr box2
        dx = (cx2 - cx1) / (w1 + 1e-14)
        dy = (cy2 - cy1) / (h1 + 1e-14)
        dw = log (w2 / w1)
        dh = log (h2 / h1)
    in Repa.computeS $ Repa.fromListUnboxed (Z:.4) [dx, dy, dw, dh] Repa./^ stds


type RBox = Array U DIM1 Float

ctrwh :: RBox -> RBox
ctrwh box1 = Repa.fromListUnboxed (Z:.4) [x0, y0, x1, y1]
  where
    [w, h, cx, cy] = Repa.toList box1
    x0 = cx - 0.5 * (w - 1)
    y0 = cy - 0.5 * (h - 1)
    x1 = w + x0 - 1
    y1 = h + y0 - 1

bboxTransInv :: RBox -> RBox -> RBox -> RBox
bboxTransInv stds box delta =
    let [dx, dy, dw, dh] = Repa.toList $ delta Repa.*^ stds
        [w1, h1, cx1, cy1] = Repa.toList $ whctr box
        w2 = exp dw * w1
        h2 = exp dh * w2
        cx2 = dx * w1 + cx1
        cy2 = dy * h1 + cy1
    in ctrwh $ Repa.fromListUnboxed (Z:.4) [w2, h2, cx2, cy2]


bboxClip :: Float -> Float -> RBox -> RBox
bboxClip height width box = Repa.fromListUnboxed (Z:.4) [x0', y0', x1', y1']
  where
    [x0, y0, x1, y1] = Repa.toList box
    w' = width - 1
    h' = height - 1
    x0' = max 0 (min x0 w')
    y0' = max 0 (min y0 h')
    x1' = max 0 (min x1 w')
    y1' = max 0 (min y1 h')


(#!) :: (Shape sh, UV.Unbox e) => Array U sh e -> Int -> e
(#!) = Repa.linearIndex
(%!) :: V.Vector a -> Int -> a
(%!) = (V.!)

expandDim :: (Shape sh, UV.Unbox e) => Int -> Array U sh e -> Array U (sh :. Int) e
expandDim axis arr | axis >=0 && axis < rank = Repa.computeS $ Repa.reshape shape_new arr
                   | otherwise = error "Bad axis to expand."
  where
    shape = Repa.extent arr
    rank = Repa.rank shape
    (h, t) = splitAt (rank - axis) $ Repa.listOfShape shape
    shape_new = Repa.shapeOfList $ h ++ [1] ++ t


vstack :: (Shape sh, UV.Unbox e) => V.Vector (Array U sh e) -> Array U sh e
-- alternative definition:
-- vstack = Repa.transpose . V.foldl1 (Repa.++) . V.map Repa.transpose
vstack arrs = Repa.fromUnboxed shape_new $ UV.concat $ V.toList $ V.map Repa.toUnboxed arrs
  where
    sumShape sh1 sh2 = let a1:r1 = reverse $ listOfShape sh1
                           a2:r2 = reverse $ listOfShape sh2
                       in if r1 == r2
                          then shapeOfList $ reverse $ (a1+a2):r1
                          else error "Cannot stack array because of incompatible shapes"
    shape_new = V.foldl1' sumShape $ V.map Repa.extent arrs


vunstack :: (Unstackable sh, UV.Unbox e) => Array U sh e -> V.Vector (Array U (PredDIM sh) e)
vunstack arr = V.map (\i -> Repa.computeS $ Repa.slice arr (makeSliceAtAxis0 shape i)) range
  where
    shape = Repa.extent arr
    dim0 = last $ listOfShape shape
    range = V.enumFromN (0::Int) dim0

class (Shape sh,
       Shape (PredDIM sh),
       Slice (SliceAtAxis0 sh),
       FullShape (SliceAtAxis0 sh) ~ sh,
       SliceShape (SliceAtAxis0 sh) ~ PredDIM sh
      ) => Unstackable sh where
    type PredDIM sh
    type SliceAtAxis0 sh
    makeSliceAtAxis0 :: sh -> Int -> SliceAtAxis0 sh

instance Unstackable DIM2 where
    type PredDIM DIM2 = DIM1
    type SliceAtAxis0 DIM2 = Z:.Int:.All
    makeSliceAtAxis0 _ i = Z:.i:.All

instance Unstackable DIM3 where
    type PredDIM DIM3 = DIM2
    type SliceAtAxis0 DIM3 = Z:.Int:.All:.All
    makeSliceAtAxis0 (sh:._) i = makeSliceAtAxis0 sh i :. All


data ReshapeError = ReshapeMismatch (V.Vector Int) (V.Vector Int)
                  | ReshapeTooManyMinusOne (V.Vector Int)
  deriving Show
instance Exception ReshapeError

reshapeEx :: (Shape sh1, Shape sh2, UV.Unbox e) => sh2 -> Array U sh1 e -> Array U sh2 e
reshapeEx shape arr = Repa.computeS $ Repa.reshape (shapeOfList $ V.toList $ V.reverse filled_new_shape) arr
  where
    old_shape = V.reverse $ V.fromList $ listOfShape $ Repa.extent arr
    new_shape = V.reverse $ V.fromList $ listOfShape shape
    shapeMismatch, tooManyN1 :: forall a. a
    shapeMismatch = throw (ReshapeMismatch new_shape old_shape)
    tooManyN1 = throw (ReshapeTooManyMinusOne new_shape)

    sizeEqual sh = V.product old_shape == V.product sh
    replaceZ i v | v == 0 = case old_shape V.!? i of
                              Just v' -> v'
                              Nothing -> shapeMismatch
                  | otherwise = v
    new_shape_nz = V.imap replaceZ new_shape


    minus_n1s = V.elemIndices (-1) new_shape_nz
    filled_new_shape
        | V.null minus_n1s = if sizeEqual new_shape_nz then new_shape_nz else shapeMismatch
        | [s] <- V.toList minus_n1s = let (new_p1, new_p2) = V.splitAt s new_shape_nz
                                      in matchN1 new_p1 (V.tail new_p2) old_shape
        | otherwise = tooManyN1

    matchN1 sh1a sh1b sh2 | r == 0 = sh1a V.++ V.fromList [q] V.++ sh1b
                          | otherwise = shapeMismatch
      where size1 = V.product $ sh1a V.++ sh1b
            size2 = V.product sh2
            (q, r) = size2 `divMod` size1

argMax :: (UVM.Unbox e, Ord e)
       => Array U DIM2 e -> V.Vector Int
--argMax overlaps =
--    let Z :. m :. n = Repa.extent overlaps
--        findMax row = UV.maxIndex $ Repa.toUnboxed $ Repa.computeS $ Repa.slice overlaps (Z :. row :. All)
--    in V.map findMax $ V.enumFromN (0 :: Int) m
argMax arr = V.map (UV.maxIndex . Repa.toUnboxed) (vunstack arr)

-- test_sample_rois = let
--         v1 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 0.8, 0.8, 2.2, 2.2]
--         v2 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 2.2, 2.2, 4.5, 4.5]
--         v3 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 4.2, 1, 6.5, 2.8]
--         v4 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 6, 3, 7, 4]
--         rois = V.fromList [v1, v2, v3, v4]
--         g1 = Repa.fromListUnboxed (Z:.5::DIM1) [1,1,2,2,1]
--         g2 = Repa.fromListUnboxed (Z:.5::DIM1) [2,3,3,4,1]
--         g3 = Repa.fromListUnboxed (Z:.5::DIM1) [4,1,6,3,2]
--         gt_boxes = V.fromList [g1, g2, g3]
--       in sample_rois rois gt_boxes 3 6 2 0.5 [0.1, 0.1, 0.1, 0.1]



