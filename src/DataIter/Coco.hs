{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module DataIter.Coco where

import Data.Maybe (catMaybes, fromMaybe)
import Data.List (unzip6)
import Control.Exception
import Data.Array.Repa (Array, DIM1, DIM3, D, U, (:.)(..), Z (..), Any(..),
    fromListUnboxed, extent, backpermute, extend, (-^), (+^), (*^), (/^))
import qualified Data.Array.Repa as Repa
import Data.Array.Repa.Repr.Unboxed (Unbox)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import Control.Lens ((^.), view, makeLenses)
import Data.Conduit
import qualified Data.Conduit.Combinators as C (yieldMany)
import qualified Data.Conduit.List as C
import Control.Monad.Reader
import Data.Maybe (fromJust)
import qualified Data.Random as RND (shuffleN, runRVar, StdRandom(..))
import Control.Monad.Trans.Resource
import Control.DeepSeq

import MXNet.Base (NDArray(..), Fullfilled, ArgsHMap, ParameterList, Attr(..), (!), (!?), (.&), HMap(..), ArgOf(..), fromVector)
import MXNet.Base.Operators.NDArray (stack)
import MXNet.NN.DataIter.Conduit
import MXNet.NN.DataIter.Coco
import qualified MXNet.NN.DataIter.Anchor as Anchor
import MXNet.Coco.Types

type instance ParameterList "WithAnchors" =
    '[ '("batch_size",     'AttrReq Int),
       -- anchors are generated on feature image with a stride
       '("feature_width",  'AttrReq Int),
       '("feature_height", 'AttrReq Int),
       '("feature_stride", 'AttrOpt Int),

       '("anchor_scales",  'AttrOpt [Int]),
       '("anchor_ratios",  'AttrOpt [Float]),
       '("allowed_border", 'AttrOpt Int),
       '("batch_rois",     'AttrOpt Int),
       '("fg_fraction",    'AttrOpt Float),
       '("fg_overlap",     'AttrOpt Float),
       '("bg_overlap",     'AttrOpt Float),
       '("fixed_num_gt",   'AttrOpt (Maybe Int))]

withAnchors :: (MonadIO m, Fullfilled "WithAnchors" args) =>
    ArgsHMap "WithAnchors" args ->
    ConduitT (String, ImageTensor, ImageInfo, GTBoxes)
             ((NDArray Float, NDArray Float, NDArray Float), (NDArray Float, NDArray Float, NDArray Float))
             m ()
withAnchors args = do
    anchors <- runReaderT (Anchor.anchors featureStride featW featH) anchConf
    C.mapM (assignAnchors anchConf anchors featW featH maxGT)
        .| C.chunksOf batchSize
        .| C.mapM toNDArray
  where
    batchSize = args ! #batch_size
    batchRois = fromMaybe 256 $ args !? #batch_rois
    featW = args ! #feature_width
    featH = args ! #feature_height
    featureStride = fromMaybe 16 $ args !? #feature_stride
    maxGT = fromMaybe Nothing $ args !? #fixed_num_gt
    anchConf = Anchor.Configuration {
        Anchor._conf_anchor_scales  = fromMaybe [8, 16, 32] $ args !? #anchor_scales,
        Anchor._conf_anchor_ratios  = fromMaybe [0.5, 1, 2] $ args !? #anchor_ratios,
        Anchor._conf_allowed_border = fromMaybe 0 $ args !? #allowed_border,
        Anchor._conf_fg_num         = floor $ (fromMaybe 0.5 $ args !? #fg_fraction) * fromIntegral batchRois,
        Anchor._conf_batch_num      = batchRois,
        Anchor._conf_fg_overlap     = fromMaybe 0.7 $ args !? #fg_overlap,
        Anchor._conf_bg_overlap     = fromMaybe 0.3 $ args !? #bg_overlap
    }

assignAnchors :: MonadIO m =>
    Anchor.Configuration ->
    V.Vector (Anchor.Anchor U) ->
    Int -> Int -> Maybe Int ->
    (String, ImageTensor, ImageInfo, GTBoxes) ->
    m (ImageTensor, ImageInfo, GTBoxes, Repa.Array U DIM1 Float, Repa.Array U DIM3 Float, Repa.Array U DIM3 Float)
assignAnchors conf anchors featureWidth featureHeight maxGT (ident, img, info, gt) = do
    let imHeight = floor $ info Anchor.#! 0
        imWidth  = floor $ info Anchor.#! 1
    (lbls, targets, weights) <- runReaderT (Anchor.assign gt imWidth imHeight anchors) conf

    -- reshape and transpose labls   from (feat_h * feat_w * #anch,  ) to (#anch,     feat_h, feat_w)
    -- reshape and transpose targets from (feat_h * feat_w * #anch, 4) to (#anch * 4, feat_h, feat_w)
    -- reshape and transpose weights from (feat_h * feat_w * #anch, 4) to (#anch * 4, feat_h, feat_w)
    let numAnch = length (conf ^. Anchor.conf_anchor_scales) * length (conf ^. Anchor.conf_anchor_ratios)
    lbls    <- return $ Repa.computeS $
                Repa.reshape (Z :. numAnch * featureHeight * featureWidth) $
                Repa.transpose $
                Repa.reshape (Z :. featureHeight * featureWidth :. numAnch) lbls
    targets <- return $ Repa.computeS $
                Repa.reshape (Z :. numAnch * 4 :. featureHeight :. featureWidth) $
                Repa.transpose $
                Repa.reshape (Z :. featureHeight * featureWidth :. numAnch * 4) targets
    weights <- return $ Repa.computeS $
                Repa.reshape (Z :. numAnch * 4 :. featureHeight :. featureWidth) $
                Repa.transpose $
                Repa.reshape (Z :. featureHeight * featureWidth :. numAnch * 4) weights

    -- optionally extend gt to a fixed number (padding with 0s)
    gtRet <- case maxGT of
      Nothing -> return gt
      Just maxGT -> do
        let numGT = V.length gt
            nullGT = fromListUnboxed (Z:.5) [0, 0, 0, 0, 0]
        if numGT <= maxGT then
            return $ gt V.++ V.replicate (maxGT - numGT) nullGT
        else
            return $ V.take maxGT gt

    return $!! (img, info, gtRet, lbls, targets, weights)

toNDArray :: MonadIO m =>
    [((ImageTensor, ImageInfo, GTBoxes, Array U DIM1 Float, Array U DIM3 Float, Array U DIM3 Float))] ->
    m ((NDArray Float, NDArray Float, NDArray Float), (NDArray Float, NDArray Float, NDArray Float))
toNDArray dat = liftIO $ do
    imagesC  <- convertToMX images
    infosC   <- convertToMX infos
    gtboxesC <- mapM (convertToMX . V.toList) gtboxes >>= stackList
    labelsC  <- convertToMX labels
    targetsC <- convertToMX targets
    weightsC <- convertToMX weights
    return $!! ((imagesC, infosC, gtboxesC), (labelsC, targetsC, weightsC))
  where
    (images, infos, gtboxes, labels, targets, weights) = unzip6 dat

    stackList arrs = do
        let hdls = map unNDArray arrs
        NDArray . head <$> stack (#data := hdls .& #num_args := length hdls .& Nil)

    repaToNDArray :: Repa.Shape sh => Array U sh Float -> IO (NDArray Float)
    repaToNDArray arr = do
        let sh = reverse $ Repa.listOfShape $ Repa.extent arr
        fromVector sh $ SV.convert $ Repa.toUnboxed arr

    convertToMX arr = mapM repaToNDArray arr >>= stackList
