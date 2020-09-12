{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE OverloadedLists       #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE TemplateHaskell       #-}
module RCNN where

import           Control.Applicative               (ZipList (..))
import           Control.Lens                      (makePrisms, _1, _2)
import           Control.Monad.Trans.Resource
import qualified Data.Array.Repa                   as Repa
import           Data.Bitraversable                (bitraverse)
import           Formatting                        (sformat, string, (%))
import           Options.Applicative               (Parser, auto, eitherReader,
                                                    help, long, metavar, option,
                                                    showDefault, strOption,
                                                    switch, value)
import           RIO
import           RIO.Directory                     (canonicalizePath,
                                                    doesFileExist)
import qualified RIO.HashSet                       as S
import           RIO.List                          (lastMaybe, unzip, unzip3,
                                                    unzip7)
import           RIO.List.Partial                  (maximum)
import qualified RIO.NonEmpty                      as RNE
import qualified RIO.NonEmpty.Partial              as RNE
import qualified RIO.Text                          as T
import qualified RIO.Vector.Boxed                  as V

import           MXNet.Base
import           MXNet.Base.ParserUtils            (decimal, endOfInput, list,
                                                    parseOnly, rational)
import           MXNet.NN
import qualified MXNet.NN.DataIter.Anchor          as Anchor
import qualified MXNet.NN.DataIter.Coco            as Coco
import qualified MXNet.NN.Initializer              as I
import           MXNet.NN.ModelZoo.RCNN.FasterRCNN

data ProgConfig = ProgConfig
    { ds_base_path            :: String
    , ds_img_size             :: Int
    , ds_img_pixel_means      :: [Float]
    , ds_img_pixel_stds       :: [Float]
    , pg_train_epochs         :: Int
    , pg_train_iter_per_epoch :: Int
    , pg_infer                :: Bool
    , pg_infer_image_id       :: Int
    }
    deriving Show

cmdArgParser :: Parser (RcnnConfiguration, ProgConfig)
cmdArgParser = liftA2 (,) rcnn prog
  where
    floatList = eitherReader $ parseOnly (list rational<* endOfInput) . T.pack
    intList   = eitherReader $ parseOnly (list decimal <* endOfInput) . T.pack

    rcnn = RcnnConfiguration
           <$> option intList   (long "rpn-anchor-scales" <> metavar "SCALES"
                                                          <> showDefault
                                                          <> value [8,16,32]
                                                          <> help "rpn anchor scales")
           <*> option floatList (long "rpn-anchor-ratios" <> metavar "RATIOS"
                                                          <> showDefault
                                                          <> value [0.5,1,2]
                                                          <> help "rpn anchor ratios")
           <*> option auto      (long "rpn-anchor-bsize"  <> metavar "BSIZE"
                                                          <> showDefault
                                                          <> value 16
                                                          <> help "rpn anchor base size")
           <*> option auto      (long "rpn-batch-rois"    <> metavar "BATCH-ROIS"
                                                          <> showDefault
                                                          <> value 256
                                                          <> help "rpn number of rois per batch")
           <*> option auto      (long "rpn-pre-nms-topk"  <> metavar "PRE-NMS-TOPK"
                                                          <> showDefault
                                                          <> value 12000
                                                          <> help "rpn nms pre-top-k")
           <*> option auto      (long "rpn-post-nms-topk" <> metavar "POST-NMS-TOPK"
                                                          <> showDefault
                                                          <> value 2000
                                                          <> help "rpn nms post-top-k")
           <*> option auto      (long "rpn-nms-thresh"    <> metavar "NMS-THRESH"
                                                          <> showDefault
                                                          <> value 0.7
                                                          <> help "rpn nms threshold")
           <*> option auto      (long "rpn-min-size"      <> metavar "MIN-SIZE"
                                                          <> showDefault
                                                          <> value 16
                                                          <> help "rpn min size")
           <*> option auto      (long "rpn-fg-fraction"   <> metavar "FG-FRACTION"
                                                          <> showDefault
                                                          <> value 0.5
                                                          <> help "rpn foreground fraction")
           <*> option auto      (long "rpn-fg-overlap"    <> metavar "FG-OVERLAP"
                                                          <> showDefault
                                                          <> value 0.7
                                                          <> help "rpn foreground iou threshold")
           <*> option auto      (long "rpn-bg-overlap"    <> metavar "BG-OVERLAP"
                                                          <> showDefault
                                                          <> value 0.3
                                                          <> help "rpn background iou threshold")
           <*> option auto      (long "rpn-allowed-border"<> metavar "ALLOWED-BORDER"
                                                          <> showDefault
                                                          <> value 0
                                                          <> help "rpn allowed border")
           <*> option auto      (long "rcnn-num-classes"  <> metavar "NUM-CLASSES"
                                                          <> showDefault
                                                          <> value 81
                                                          <> help "rcnn number of classes")
           <*> option auto      (long "rcnn-pooled-size"  <> metavar "POOLED-SIZE"
                                                          <> showDefault
                                                          <> value 14
                                                          <> help "rcnn pooled size")
           <*> option auto      (long "rcnn-batch-rois"   <> metavar "BATCH_ROIS"
                                                          <> showDefault
                                                          <> value 128
                                                          <> help "rcnn batch rois")
           <*> option auto      (long "rcnn-fg-fraction"  <> metavar "FG-FRACTION"
                                                          <> showDefault
                                                          <> value 0.25
                                                          <> help "rcnn foreground fraction")
           <*> option auto      (long "rcnn-fg-overlap"   <> metavar "FG-OVERLAP"
                                                          <> showDefault
                                                          <> value 0.5
                                                          <> help "rcnn foreground iou threshold")
           <*> option auto      (long "rcnn-max-num-gt"   <> metavar "NUM-GT"
                                                          <> showDefault
                                                          <> value 100
                                                          <> help "rcnn max number of gt")
           <*> option intList   (long "strides"           <> metavar "STRIDE"
                                                          <> showDefault
                                                          <> value [16]
                                                          <> help "feature stride")
           <*> option auto      (long "batch-size"        <> metavar "BATCH-SIZE"
                                                          <> showDefault
                                                          <> value 1
                                                          <> help "batch size")
           <*> strOption        (long "pretrained"        <> metavar "PATH"
                                                          <> value ""
                                                          <> help "path to pretrained model")
           <*> option auto      (long "backbone"          <> metavar "BACKBONE"
                                                          <> value VGG16
                                                          <> help "vgg-16 or resnet-50")

    prog = ProgConfig
           <$> strOption        (long "base"              <> metavar "PATH"
                                                          <> help "path to the dataset")
           <*> option auto      (long "img-size"          <> metavar "SIZE"
                                                          <> showDefault
                                                          <> value 1024
                                                          <> help "long side of image")
           <*> option floatList (long "img-pixel-means"   <> metavar "RGB-MEAN"
                                                          <> showDefault
                                                          <> value [0,0,0]
                                                          <> help "RGB mean of images")
           <*> option floatList (long "img-pixel-stds"    <> metavar "RGB-STDS"
                                                          <> showDefault
                                                          <> value [1,1,1]
                                                          <> help "RGB std-dev of images")
           <*> option auto      (long "train-epochs"      <> metavar "EPOCHS"
                                                          <> value 500
                                                          <> help "number of epochs to train")
           <*> option auto      (long "train-iter-per-epoch" <> metavar "ITER-PER-EPOCH"
                                                          <> value 100
                                                          <> help "number of iter per epoch")
           <*> switch           (long "inference"         <> showDefault
                                                          <> help "do inference")
           <*> option auto      (long "inference-img-id"  <> value 0
                                                          <> help "image id")


toTriple [a, b, c] = (a, b, c)
toTriple x         = error (show x)


default_initializer :: Initializer Float
default_initializer name = case name of
    "features.rpn.rpn_conv_3x3.weight"  -> I.normal 0.01 name
    "features.rpn.rpn_conv_3x3.bias"    -> I.zeros name
    "features.rpn.rpn_cls_score.weight" -> I.normal 0.01 name
    "features.rpn.rpn_cls_score.bias"   -> I.zeros name
    "features.rpn.rpn_bbox_pred.weight" -> I.normal 0.01 name
    "features.rpn.rpn_bbox_pred.bias"   -> I.zeros name
    "features.rcnn.rcnn_cls_score.weight"     -> I.normal 0.01 name
    "features.rcnn.rcnn_cls_score.bias"       -> I.zeros name
    "features.rcnn.rcnn_bbox_feature.weight"  -> I.normal 0.01 name
    "features.rcnn.rcnn_bbox_feature.bias"    -> I.zeros name
    "features.rcnn.rcnn_bbox_pred.weight"     -> I.normal 0.001 name
    "features.rcnn.rcnn_bbox_pred.bias"       -> I.zeros name
    "features.fpn.0.conv1.weight"             -> I.xavier 1 I.XavierUniform I.XavierIn name
    "features.fpn.0.conv2.weight"             -> I.xavier 1 I.XavierUniform I.XavierIn name
    "features.fpn.1.conv1.weight"             -> I.xavier 1 I.XavierUniform I.XavierIn name
    "features.fpn.1.conv2.weight"             -> I.xavier 1 I.XavierUniform I.XavierIn name
    "features.fpn.2.conv1.weight"             -> I.xavier 1 I.XavierUniform I.XavierIn name
    "features.fpn.2.conv2.weight"             -> I.xavier 1 I.XavierUniform I.XavierIn name
    "features.fpn.3.conv1.weight"             -> I.xavier 1 I.XavierUniform I.XavierIn name
    "features.fpn.3.conv2.weight"             -> I.xavier 1 I.XavierUniform I.XavierIn name
    _ | T.isSuffixOf ".running_mean" name -> I.zeros name
      | T.isSuffixOf ".running_var"  name -> I.ones name
      | T.isSuffixOf ".beta"         name -> I.zeros name
      | T.isSuffixOf ".gamma"        name -> I.ones  name
      | otherwise -> I.zeros name

loadWeights :: (MonadIO m, MonadReader env m, HasLogFunc env, HasCallStack)
            => String -> Module t a m ()
loadWeights weights_path = do
    weights_path <- liftIO $ canonicalizePath weights_path
    e <- liftIO $ doesFileExist (weights_path ++ ".params")
    if not e
        then lift . logInfo . display $ sformat ("'" % string % ".params' doesn't exist.") weights_path
        else loadState weights_path ["features.9.gamma",
                                     "features.9.beta",
                                     "features.9.running_var",
                                     "features.9.running_mean",
                                     "output.weight",
                                     "output.bias"
                                    ]

data Stage = TRAIN
    | INFERENCE

fixedParams :: Backbone -> Stage -> SymbolHandle -> IO (HashSet Text)
fixedParams backbone stage symbol = do
    argnames <- listArguments symbol
    return $ case (stage, backbone) of
        (INFERENCE, _)
            -> S.fromList argnames
        (TRAIN, VGG16)
            -> S.fromList [n | n <- argnames
                          -- fix conv_1_1, conv_1_2, conv_2_1, conv_2_2
                          ,  elemM [0, 2, 5, 7] (layer n)]
        (TRAIN, r) | r `elem` ([RESNET50, RESNET50FPN, RESNET101] :: [Backbone])
            -> S.fromList [n | n <- argnames
                          -- fix conv_0, stage_1_*, *_gamma, *_beta
                          , let layer_idx = layer n
                          , elemM [0, 1, 5] layer_idx ||
                            (leqM 9 layer_idx && elemM ["gamma", "beta"] (lastName n))]

  where
    toMaybe = either (const Nothing) Just
    layer param = case T.split (=='.') param of
                    "features":n:_ -> toMaybe $ parseOnly decimal n
                    _              -> Nothing
    lastName = lastMaybe . T.split (=='.')
    elemM :: Eq a => [a] -> Maybe a -> Bool
    elemM b = isJust . (>>= guard) . liftM (`elem` b)
    leqM  n = isJust . (>>= guard) . liftM (<= n)

data App c = App LogFunc c
makePrisms ''App

instance HasLogFunc (App c) where
    logFuncL = _App . _1

instance Coco.HasDatasetConfig (App Coco.CocoConfig) where
    type DatasetTag (App Coco.CocoConfig) = "coco"
    datasetConfig = _App . _2

runApp :: c -> ReaderT (App c) (ResourceT IO) a -> IO a
runApp conf body = do
    logopt <- logOptionsHandle stdout False
    runResourceT $ withLogFunc logopt $ \logfunc ->
        flip runReaderT (App logfunc conf) body

generateTargets :: (SymbolHandle -> Layer (NonEmpty SymbolHandle))
                -> Coco.ImageInfo
                -> NonEmpty Int
                -> Anchor.Configuration
                -> [Anchor.GTBox Repa.U]
                -> IO (NDArray Float, NDArray Float, NDArray Float)
generateTargets feature_net im_info strides anchor_conf gt_boxes = do
    feats  <- runLayerBuilder $ variable "data" >>= feature_net

    -- there should equally number of features and strides, and pair them.
    let feat_stride = RNE.zip feats strides

    layers <- mapM (uncurry make) feat_stride
    let (cls_targets, box_targets, box_masks) = unzip3 $ RNE.toList layers
    cls_targets <- mapM fromRepa cls_targets
    box_targets <- mapM fromRepa box_targets
    box_masks   <- mapM fromRepa box_masks
    cls_targets <- concat_ 0 cls_targets
    box_targets <- concat_ 0 box_targets
    box_masks   <- concat_ 0 box_masks
    return (cls_targets, box_targets, box_masks)

  where
    [img_height, img_width, _] = Repa.toList im_info
    -- we have padded the image to a square
    img_size = floor (max img_height img_width)
    base_size = anchor_conf ^. Anchor.conf_anchor_base_size
    scales    = anchor_conf ^. Anchor.conf_anchor_scales
    ratios    = anchor_conf ^. Anchor.conf_anchor_ratios
    make :: SymbolHandle -> Int -> IO (Anchor.Labels, Anchor.Targets, Anchor.Weights)
    make feat stride = do
        (_, outputs, _, _) <- inferShape feat [("data", STensor [1,3,img_size,img_size])]
        let [(_, STensor [_, _, h, w])] = outputs
            anchors = Anchor.anchors (h, w) stride base_size scales ratios
        runReaderT (Anchor.assign (V.fromList gt_boxes) img_size img_size anchors) anchor_conf

padLength :: DType a => [NDArray a] -> a -> IO [NDArray a]
padLength arrays value = do
    shps <- mapM ndshape arrays
    let max_num = maximum $ map RNE.head shps
    forM (zip arrays shps) $ \(a, n :| shp) ->
        if n == max_num
        then return a
        else do
            padding <- full value ((max_num - n) :| shp)
            concat_ 0 [a, padding]

withRpnTargets :: MonadIO m
               => RcnnConfiguration
               -> (String, Coco.ImageTensor, Coco.ImageInfo, Coco.GTBoxes)
               -> m (String, [NDArray Float])
withRpnTargets RcnnConfiguration{..} dat = liftIO $ do
    (cls_targets, box_targets, box_weights) <-
        generateTargets extract info (RNE.fromList feature_strides) conf (V.toList gt)
    imgA  <- fromRepa img
    infoA <- fromRepa info
    gtA   <- stack 0 . V.toList =<< mapM fromRepa gt
    return (filename, [gtA, imgA, infoA, cls_targets, box_targets, box_weights])
    where
        (filename, img, info, gt) = dat
        conf = Anchor.Configuration
               { Anchor._conf_anchor_scales    = rpn_anchor_scales
               , Anchor._conf_anchor_ratios    = rpn_anchor_ratios
               , Anchor._conf_anchor_base_size = rpn_anchor_base_size
               , Anchor._conf_allowed_border   = rpn_allowd_border
               , Anchor._conf_fg_num           = floor $ (rpn_fg_fraction * fromIntegral rpn_batch_rois)
               , Anchor._conf_batch_num        = rpn_batch_rois
               , Anchor._conf_bg_overlap       = rpn_bg_overlap
               , Anchor._conf_fg_overlap       = rpn_fg_overlap
               }
        extract = sequential "features" . features1 backbone


withRpnTargets'Mask :: MonadIO m
                    => RcnnConfiguration
                    -> (String, Coco.ImageTensor, Coco.ImageInfo, Coco.GTBoxes, Coco.Masks)
                    -> m (String, [NDArray Float])
withRpnTargets'Mask conf dat = do
    let (filename, img, info, gt, msks) = dat
    (_, ret) <- withRpnTargets conf (filename, img, info, gt)
    liftIO $ do
        msksA <- stack 0 . V.toList =<< mapM fromRepa msks
        msksA <- divScalar 255 =<< cast #float32 msksA :: IO (NDArray Float)
        return (filename, msksA : ret)

concatBatch :: MonadIO m => [(String, [NDArray Float])] -> m ([String], [NDArray Float])
concatBatch batch = liftIO $ do
    let (filenames, tensors) = unzip batch
        gt : others = unzipList tensors
    -- gt in the batch may not have the same number
    -- must be padded with -1 before stacking
    gt     <- stack 0 =<< padLength gt (-1)
    -- other tensors can be simply stacked
    others <- mapM (stack 0) others
    return (filenames, gt : others)


concatBatch'Mask :: MonadIO m => [(String, [NDArray Float])] -> m ([String], [NDArray Float])
concatBatch'Mask batch = liftIO $ do
    let (filenames, tensors) = unzip batch
        mask_gt : box_gt : others = unzipList tensors
    mask_gt <- stack 0 =<< padLength mask_gt 0
    box_gt  <- stack 0 =<< padLength box_gt (-1)
    others <- mapM (stack 0) others
    return (filenames, mask_gt : box_gt : others)

unzipList :: [[a]] -> [[a]]
unzipList = getZipList . traverse ZipList


