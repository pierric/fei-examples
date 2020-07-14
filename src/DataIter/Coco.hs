{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
module DataIter.Coco where

import           Data.Array.Repa          ((:.) (..), Array, DIM1, DIM3, U,
                                           Z (..), fromListUnboxed)
import qualified Data.Array.Repa          as Repa
import           Data.Conduit
import qualified Data.Conduit.List        as C
import           RIO
import qualified RIO.NonEmpty             as RNE
import qualified RIO.Vector.Boxed         as V
import qualified RIO.Vector.Storable      as SV

import           MXNet.Base               (ArgsHMap, Attr (..), Fullfilled,
                                           NDArray (..), ParameterList,
                                           fromVector, (!), (!?))
import qualified MXNet.NN.DataIter.Anchor as Anchor
import           MXNet.NN.DataIter.Coco
import           MXNet.NN.Utils.Repa

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
    scales = fromMaybe [8, 16, 32] $ args !? #anchor_scales
    ratios = fromMaybe [0.5, 1, 2] $ args !? #anchor_ratios
    anchors = Anchor.anchors (featH, featW) featureStride 32 scales ratios
    anchConf = Anchor.Configuration {
        Anchor._conf_anchor_scales  = scales,
        Anchor._conf_anchor_ratios  = ratios,
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
    let imHeight = floor $ info ^#! 0
        imWidth  = floor $ info ^#! 1
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
    imagesC  <- repaToNDArray $ evstack images
    infosC   <- repaToNDArray $ evstack infos
    gtboxesC <- repaToNDArray $ evstack $ V.map evstack gtboxes
    labelsC  <- repaToNDArray $ evstack labels
    targetsC <- repaToNDArray $ evstack targets
    weightsC <- repaToNDArray $ evstack weights
    return $!! ((imagesC, infosC, gtboxesC), (labelsC, targetsC, weightsC))
  where
    (images, infos, gtboxes, labels, targets, weights) = V.unzip6 $ V.fromList dat
    evstack arrs = vstack $ V.map (expandDim 0) arrs

repaToNDArray :: Repa.Shape sh => Array U sh Float -> IO (NDArray Float)
repaToNDArray arr = do
    let Just sh = RNE.nonEmpty $ reverse $ Repa.listOfShape $ Repa.extent arr
    fromVector sh $ SV.convert $ Repa.toUnboxed arr

