{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards   #-}
{-# LANGUAGE TemplateHaskell   #-}
module Main where

import           Control.Lens                      (makePrisms, use, (.=), _1,
                                                    _2)
import           Control.Monad.Trans.Resource
import           Data.Array.Repa                   ((:.) (..), Array, DIM1,
                                                    DIM2, DIM3, U, Z (..))
import qualified Data.Array.Repa                   as Repa
import           Data.Conduit                      ((.|))
import qualified Data.Conduit.List                 as C
import qualified Data.Vector.Algorithms.Intro      as VA
import qualified Data.Vector.Mutable               as VM
import           Formatting                        (fixed, formatToString, int,
                                                    left, sformat, stext,
                                                    string, (%))
import           Options.Applicative               (Parser, auto, eitherReader,
                                                    execParser, fullDesc,
                                                    header, help, helper, info,
                                                    long, metavar, option,
                                                    showDefault, strOption,
                                                    switch, value, (<**>))
import           RIO                               hiding (Const)
import           RIO.Char                          (isDigit)
import           RIO.Directory                     (canonicalizePath,
                                                    doesFileExist)
import           RIO.FilePath
import qualified RIO.HashMap                       as M
import qualified RIO.HashSet                       as S
import           RIO.List                          (sort)
import           RIO.List.Partial                  (last)
import           RIO.NonEmpty                      (nonEmpty)
import qualified RIO.Text                          as T
import qualified RIO.Vector.Boxed                  as V
import qualified RIO.Vector.Boxed.Partial          as V (head)
import qualified RIO.Vector.Storable               as VS
import qualified RIO.Vector.Unboxed                as VU
import qualified RIO.Vector.Unboxed.Partial        as VU ((//))
import qualified Text.PrettyPrint.Leijen.Text      as PP

import qualified DataIter.Coco                     as Coco
import           MXNet.Base                        (ArgOf (..), FShape (..),
                                                    HMap (..), NDArray,
                                                    SymbolHandle, contextCPU,
                                                    contextGPU0, fromVector,
                                                    getInternalByName,
                                                    inferOutputShape,
                                                    listArguments,
                                                    mxListAllOpNames,
                                                    mxNotifyShutdown,
                                                    registerCustomOperator,
                                                    toRepa, (.&))
import qualified MXNet.Base.ParserUtils            as P
import           MXNet.Coco.Types                  (images, img_id)
import           MXNet.NN
import qualified MXNet.NN.DataIter.Anchor          as Anchor
import qualified MXNet.NN.DataIter.Coco            as Coco
import           MXNet.NN.DataIter.ConduitAsync
import qualified MXNet.NN.Initializer              as I
import           MXNet.NN.ModelZoo.RCNN.FasterRCNN
import           MXNet.NN.ModelZoo.Utils.Box
import qualified MXNet.NN.NDArray                  as A

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
cmdArgParser = liftA2 (,)
                (RcnnConfiguration
                    <$> option intList   (long "rpn-anchor-scales" <> metavar "SCALES"         <> showDefault <> value [8,16,32] <> help "rpn anchor scales")
                    <*> option floatList (long "rpn-anchor-ratios" <> metavar "RATIOS"         <> showDefault <> value [0.5,1,2] <> help "rpn anchor ratios")
                    -- <*> option auto      (long "rpn-feat-stride"   <> metavar "STRIDE"         <> showDefault <> value 16        <> help "rpn feature stride")
                    <*> option auto      (long "rpn-batch-rois"    <> metavar "BATCH-ROIS"     <> showDefault <> value 256       <> help "rpn number of rois per batch")
                    <*> option auto      (long "rpn-pre-nms-topk"  <> metavar "PRE-NMS-TOPK"   <> showDefault <> value 12000     <> help "rpn nms pre-top-k")
                    <*> option auto      (long "rpn-post-nms-topk" <> metavar "POST-NMS-TOPK"  <> showDefault <> value 2000      <> help "rpn nms post-top-k")
                    <*> option auto      (long "rpn-nms-thresh"    <> metavar "NMS-THRESH"     <> showDefault <> value 0.7       <> help "rpn nms threshold")
                    <*> option auto      (long "rpn-min-size"      <> metavar "MIN-SIZE"       <> showDefault <> value 16        <> help "rpn min size")
                    <*> option auto      (long "rpn-fg-fraction"   <> metavar "FG-FRACTION"    <> showDefault <> value 0.5       <> help "rpn foreground fraction")
                    <*> option auto      (long "rpn-fg-overlap"    <> metavar "FG-OVERLAP"     <> showDefault <> value 0.7       <> help "rpn foreground iou threshold")
                    <*> option auto      (long "rpn-bg-overlap"    <> metavar "BG-OVERLAP"     <> showDefault <> value 0.3       <> help "rpn background iou threshold")
                    <*> option auto      (long "rpn-allowed-border"<> metavar "ALLOWED-BORDER" <> showDefault <> value 0         <> help "rpn allowed border")
                    <*> option auto      (long "rcnn-num-classes"  <> metavar "NUM-CLASSES"    <> showDefault <> value 81        <> help "rcnn number of classes")
                    <*> option auto      (long "rcnn-feat-stride"  <> metavar "FEATURE-STRIDE" <> showDefault <> value [16]        <> help "rcnn feature stride")
                    <*> option intList   (long "rcnn-pooled-size"  <> metavar "POOLED-SIZE"    <> showDefault <> value [7,7]     <> help "rcnn pooled size")
                    <*> option auto      (long "rcnn-batch-rois"   <> metavar "BATCH_ROIS"     <> showDefault <> value 128       <> help "rcnn batch rois")
                    <*> option auto      (long "rcnn-batch-size"   <> metavar "BATCH-SIZE"     <> showDefault <> value 1         <> help "rcnn batch size")
                    <*> option auto      (long "rcnn-fg-fraction"  <> metavar "FG-FRACTION"    <> showDefault <> value 0.25      <> help "rcnn foreground fraction")
                    <*> option auto      (long "rcnn-fg-overlap"   <> metavar "FG-OVERLAP"     <> showDefault <> value 0.5       <> help "rcnn foreground iou threshold")
                    <*> option auto      (long "rcnn-max-num-gt"   <> metavar "NUM-GT"     <> showDefault <> value 100       <> help "rcnn max number of gt")
                    -- <*> option floatList (long "rcnn-bbox-stds"    <> metavar "BBOX-STDDEV"    <> showDefault <> value [0.1, 0.1, 0.2, 0.2] <> help "standard deviation of bbox")
                    <*> strOption        (long "pretrained"        <> metavar "PATH"           <> value "" <> help "path to pretrained model")
                    <*> option auto      (long "backbone"          <> metavar "BACKBONE"       <> value VGG16 <> help "vgg-16 or resnet-50"))
                (ProgConfig
                    <$> strOption        (long "base" <> metavar "PATH" <> help "path to the dataset")
                    <*> option auto      (long "img-size"          <> metavar "SIZE" <> showDefault <> value 1024 <> help "long side of image")
                    <*> option floatList (long "img-pixel-means"   <> metavar "RGB-MEAN" <> showDefault <> value [0,0,0] <> help "RGB mean of images")
                    <*> option floatList (long "img-pixel-stds"    <> metavar "RGB-STDS" <> showDefault <> value [1,1,1] <> help "RGB std-dev of images")
                    <*> option auto      (long "train-epochs"      <> metavar "EPOCHS" <> value 500 <> help "number of epochs to train")
                    <*> option auto      (long "train-iter-per-epoch" <> metavar "ITER-PER-EPOCH" <> value 100 <> help "number of iter per epoch")
                    <*> switch           (long "inference" <> showDefault <> help "do inference")
                    <*> option auto      (long "inference-img-id" <> value 0 <> help "image id"))
  where
    floatList = eitherReader $ P.parseOnly (P.list P.rational<* P.endOfInput) . T.pack
    intList   = eitherReader $ P.parseOnly (P.list P.decimal <* P.endOfInput) . T.pack

-- buildProposalTargetProp params = do
--     let params' = M.fromList params
--     return $ ProposalTargetProp {
--         _pt_num_classes = parseLitR decimal $ params' ^?! ix "num_classes",
--         _pt_batch_images= parseLitR decimal $ params' ^?! ix "batch_images",
--         _pt_batch_rois  = parseLitR decimal $ params' ^?! ix "batch_rois",
--         _pt_fg_fraction = parseLitR rational $ params' ^?! ix "fg_fraction",
--         _pt_fg_overlap  = parseLitR rational $ params' ^?! ix "fg_overlap",
--         _pt_box_stds    = parseLitR floatList $ params' ^?! ix "box_stds"
--     }
--
--   where
--     floatList = char '[' *> sepBy rational (char ',') <* char ']'
--
-- parseLit :: HasCallStack => P.Parser a -> Text -> Either String a
-- parseLit  c t = (parseOnly (c <* endOfInput) t)
--
-- parseLitR :: HasCallStack => P.Parser a -> Text -> a
-- parseLitR c t = parseLit c t ^?! _Right
--
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
    "features.rcnn.cls_score.weight"     -> I.normal 0.01 name
    "features.rcnn.cls_score.bias"       -> I.zeros name
    "features.rcnn.bbox_pred.weight"     -> I.normal 0.001 name
    "features.rcnn.bbox_pred.bias"       -> I.zeros name
    _ | T.isSuffixOf ".running_mean" name -> I.zeros name
      | T.isSuffixOf ".running_var"  name -> I.zeros name
      | otherwise -> I.empty name

loadWeights weights_path = do
    weights_path <- liftIO $ canonicalizePath weights_path
    e <- liftIO $ doesFileExist (weights_path ++ ".params")
    if not e
        then lift . logInfo . display $ sformat ("'" % string % ".params' doesn't exist.") weights_path
        else loadState weights_path ["features.rpn.rpn_conv_3x3.weight",
                                     "features.rpn.rpn_conv_3x3.bias",
                                     "features.rpn.rpn_cls_score.weight",
                                     "features.rpn.rpn_cls_score.bias",
                                     "features.rpn.rpn_bbox_pred.weight",
                                     "features.rpn.rpn_bbox_pred.bias",
                                     "features.rcnn.cls_score.weight",
                                     "features.rcnn.cls_score.bias",
                                     "features.rcnn.bbox_pred.weight",
                                     "features.rcnn.bbox_pred.bias"]

data Stage = TRAIN
    | INFERENCE

fixedParams :: Backbone -> Stage -> SymbolHandle -> IO (HashSet Text)
fixedParams backbone stage symbol = do
    argnames <- listArguments symbol
    return $ case (stage, backbone) of
        (INFERENCE, _)    -> S.fromList argnames
        (TRAIN, VGG16)    -> S.fromList [n | n <- argnames
                                        -- fix conv_1_1, conv_1_2, conv_2_1, conv_2_2
                                        , layer n `elemL` ["0", "2", "5", "7"]]
        (TRAIN, RESNET50) -> S.fromList [n | n <- argnames
                                        -- fix conv_0, stage_1_*, *_gamma, *_beta
                                        , layer n `elemL` ["1", "5"] || name n `elemL` ["gamma", "beta"]]
                                        -- , layer n `elemL` ["1", "5"]]
        (TRAIN, RESNET101)-> S.fromList [n | n <- argnames
                                        -- fix conv_0, stage_1_*, *_gamma, *_beta
                                        , layer n `elemL` ["1", "5"] || name n `elemL` ["gamma", "beta"]]

  where
    layer param = case T.split (=='.') param of
                    "features":n:_ -> n
                    _              -> "<na>"
    name  param = last $ T.split (=='.') param
    elemL :: Eq a => a -> [a] -> Bool
    elemL = elem

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

main :: IO ()
main = do
    _    <- mxListAllOpNames
    registerCustomOperator ("anchor_generator", Anchor.buildAnchorGenerator)
    -- registerCustomOperator ("proposal_target", buildProposalTargetProp)

    (rcnn_conf, pg_conf@ProgConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
    if pg_infer then
        mainInfer rcnn_conf pg_conf
    else
        mainTrain rcnn_conf pg_conf
    mxNotifyShutdown

mainInfer rcnn_conf@RcnnConfiguration{..} ProgConfig{..} = do
    sym <- runLayerBuilder $ symbolInfer rcnn_conf
    fixed_params <- fixedParams backbone INFERENCE sym
    fixed_params <- return $ S.difference fixed_params (S.fromList ["data", "im_info"])

    coco_inst@(Coco.Coco _ _ coco_inst_ _) <- Coco.coco ds_base_path "val2017"
    sess <- newMVar =<< initialize @"faster_rcnn" sym (Config {
                _cfg_data  = M.fromList [("data",    (STensor [3, ds_img_size, ds_img_size])),
                ("im_info", (STensor [3]))],
                _cfg_label = [],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_fixed_params = fixed_params,
                _cfg_context = contextGPU0
            })
    let vimg = V.filter (\img -> img ^. img_id == pg_infer_image_id) $ coco_inst_ ^. images
        img = V.head vimg

    when (V.null vimg) (throwString $ "image_id " ++ show pg_infer_image_id ++ " not found in val2017")

    let coco_conf = Coco.CocoConfig coco_inst ds_img_size (toTriple ds_img_pixel_means) (toTriple ds_img_pixel_stds)
    runApp coco_conf $ withSession sess $ do
        (img_tensor_r, img_info_r) <- Coco.loadImage img
        img_tensor <- liftIO $ Coco.repaToNDArray img_tensor_r
        img_info   <- liftIO $ Coco.repaToNDArray img_info_r
        img_tensor <- liftIO $ A.expandDims img_tensor 0
        img_info   <- liftIO $ A.expandDims img_info   0

        checkpoint <- lastSavedState "checkpoints" "faster_rcnn"
        case checkpoint of
            Nothing -> do
                throwString $ "Checkpoint not found."
            Just filename -> do
                loadState filename []
                let binding = M.fromList [ ("data",    img_tensor)
                                         , ("im_info", img_info)]
                [rois, cls_prob, bbox_pred] <- forwardOnly binding
                liftIO $ do
                    rois     <- toRepa @DIM2 rois       -- [NUM_ROIS, 5]
                    cls_prob <- toRepa @DIM2 cls_prob   -- [NUM_ROIS, NUM_CLASSES]
                    deltas   <- toRepa @DIM2 bbox_pred  -- [NUM_ROIS, NUM_CLASSES*4]

                    let box_stds = Repa.fromListUnboxed (Z :. 4 :: DIM1) [0.1, 0.1, 0.2, 0.2]
                        rois' = Repa.computeS $ Repa.traverse rois
                                    (\(Z :. n :. 5) -> Z :. n :. 4)
                                    (\lk (Z:.i:.j) -> lk (Z:.i:.(j+1)))
                        deltas' = reshapeEx (Z :. 0 :. (-1) :. 4 :: DIM3) deltas
                        pred_boxes = decodeBoxes rois' deltas' box_stds img_info_r
                        (cls_ids, cls_scores) = decodeScores cls_prob 1e-3
                        res = cls_ids Repa.++ cls_scores Repa.++ pred_boxes
                        -- exlcude background class 0
                        -- and transpose from [NUM_ROIS, NUM_CLASSES_FG, 6] to [NUM_CLASSES_FG, NUM_ROIS, 6]
                        res_no_bg = Repa.traverse res
                                        (\(Z:.i:.j:.k) -> Z:.(j-1):.i:.k)
                                        (\lk (Z:.j:.i:.k) -> lk (Z:.i:.(j+1):.k))

                        -- nms the boxes
                        res_out = V.concatMap (nmsBoxes 0.3) $ vunstack $
                                        Repa.computeS res_no_bg

                        -- keep only those with confidence >= 0.7
                        res_good = V.filter ((>= 0.7) . (^#! 1)) $ res_out
                    PP.putDoc . PP.pretty $ V.toList $ V.map PrettyArray res_good
                -- logInfo . display $ length res_good
                -- putStrLn $ prettyShow $ V.toList $ V.concatMap vunstack a
                -- putDoc . pretty $ V.toList $ vunstack cls_prob
                logInfo . display $ sformat ("Done: " % int) (img ^. img_id)

  where
    decodeBoxes rois deltas box_stds im_info =
        -- rois: [N, 4]
        -- deltas: [N, NUM_CLASSES, 4]
        -- box_stds: [4]
        -- return: [N, NUM_CLASSES, 4]
        let [height, width, scale] = Repa.toList im_info :: [Float]
            shape = Repa.extent deltas
            eachClass roi = (Repa.computeS . Repa.map (/ scale)) . bboxClip height width . bboxTransInv box_stds roi
            eachROI roi = V.map (eachClass roi) . vunstack
            pred_boxes = V.zipWith eachROI (vunstack rois) (vunstack deltas)
        in Repa.computeUnboxedS $ Repa.reshape shape $ vstack $ V.concat $ V.toList pred_boxes

    decodeScores :: Array U DIM2 Float -> Float -> (Array U DIM3 Float, Array U DIM3 Float)
    decodeScores cls_prob thr =
        let Z:.num_rois:.num_classes = Repa.extent cls_prob
            cls_id = Repa.fromUnboxed (Z:.1:.num_classes) $ VU.enumFromN 0 num_classes
            -- cls_ids :: [NUM_ROIS, NUM_CLASSES]
            cls_ids = vstack $ V.replicate num_rois cls_id
            -- cls_scores :: [NUM_ROIS, NUM_CLASSES]
            cls_ids_masked = Repa.computeS $ Repa.zipWith (\v1 v2 -> if v2 >= thr then v1 else (-1)) cls_ids cls_prob
            cls_scs_masked = Repa.computeS $ Repa.map (\v -> if v >= thr then v else 0) cls_prob
        in (reshapeEx (Z:.0:.0:.1) cls_ids_masked, reshapeEx (Z:.0:.0:.1) cls_scs_masked)

    nmsBoxes :: Float -> Array U DIM2 Float -> V.Vector (Array U DIM1 Float)
    nmsBoxes threshold boxes = runST $ do
        items <- V.thaw (vunstack boxes)
        go items
        V.filter ((/= -1) . (^#! 0)) <$> V.freeze items
      where
        cmp a b | a ^#! 0 == -1 = GT
                | b ^#! 0 == -1 = LT
                | otherwise = compare (b ^#! 1) (a ^#! 1)
        go :: VM.MVector s (Array U DIM1 Float) -> ST s ()
        go items =
            when (VM.length items > 0) $ do
                VA.sortBy cmp items
                box0 <- VM.read items 0
                when (box0 ^#! 0 == -1) $ do
                    items <- return $ VM.tail items
                    V.forM_ (V.enumFromN 0 (VM.length items)) $ \k -> do
                        boxK <- VM.read items k
                        let b1 = Repa.computeS $ Repa.extract (Z:.2) (Z:.4) box0
                            b2 = Repa.computeS $ Repa.extract (Z:.2) (Z:.4) boxK
                        when (bboxIOU b1 b2 >= threshold) $ do
                            let boxK' = Repa.fromUnboxed (Z:.6 :: DIM1) $ Repa.toUnboxed boxK VU.// [(0, -1)]
                            VM.write items k boxK'
                    go items

mainTrain rcnn_conf@RcnnConfiguration{..} ProgConfig{..} = do
    sym  <- runLayerBuilder $ symbolTrain rcnn_conf
    fixed_params <- fixedParams backbone TRAIN sym

    coco_inst <- Coco.coco ds_base_path "train2017"

    let coco_conf = Coco.CocoConfig coco_inst ds_img_size (toTriple ds_img_pixel_means) (toTriple ds_img_pixel_stds)
        data_iter = asyncConduit (Just rcnn_batch_size) $ Coco.cocoImagesBBoxes True .| C.chunksOf rcnn_batch_size .| C.mapM toNDArray

    sess <- newMVar =<< initialize @"faster_rcnn" sym (Config {
                _cfg_data  = M.fromList [("data",     (STensor [3, ds_img_size, ds_img_size])),
                                         ("im_info",  (STensor [3])),
                                         ("gt_boxes", (STensor [0, 5])),
                                         ("mean",     (STensor [4])),
                                         ("std",      (STensor [4]))],
                _cfg_label = [],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_fixed_params = fixed_params,
                _cfg_context = contextGPU0
            })

    mean <- fromVector [4] [0,0,0,0]
    std  <- fromVector [4] [0.1, 0.1, 0.2, 0.2]

    optimizer <- makeOptimizer SGD'Mom (Const 0.001) (#momentum := 0.9
                                                   .& #wd := 0.0005
                                                   .& #rescale_grad := 1 / (fromIntegral rcnn_batch_size)
                                                   .& #clip_gradient := 5
                                                   .& Nil)

    runApp coco_conf $ do
        checkpoint <- lastSavedState "checkpoints" "faster_rcnn"
        start_epoch <- case checkpoint of
            Nothing -> do
                logInfo . display $ sformat string pretrained_weights
                unless (null pretrained_weights)
                    (withSession sess $ loadWeights pretrained_weights)
                return (1 :: Int)
            Just filename -> do
                withSession sess $ loadState filename []
                let (base, _) = splitExtension filename
                    fn_rev = T.reverse $ T.pack base
                    epoch = P.parseR (P.takeWhile isDigit <* P.takeText) fn_rev
                    epoch_next = P.parseR P.decimal $ T.reverse epoch
                return epoch_next
        logInfo . display $ sformat ("fixed parameters: " % stext) (tshow (sort $ S.toList fixed_params))

        metric <- newMetric "train" (RPNAccMetric 0 "label" :* RCNNAccMetric 2 4 :* RPNLogLossMetric 0 "label" :* RCNNLogLossMetric 2 4 :* RPNL1LossMetric 1 "bbox_weight" :* RCNNL1LossMetric 3 4 :* MNil)

        -- update the internal counting of the iterations
        -- the lr is updated as per to it
        withSession sess $
            untag . mod_statistics . stat_num_upd .= (start_epoch - 1) * pg_train_iter_per_epoch

        forM_ ([start_epoch..pg_train_epochs] :: [Int]) $ \ ei -> do
            logInfo . display $ sformat ("Epoch " % int) ei
            let slice = takeD pg_train_iter_per_epoch data_iter
            void $ forEachD_i slice $ \(i, (x0, x1, x2)) -> withSession sess $ do
                let binding = M.fromList [ ("data",     x0)
                                         , ("im_info",  x1)
                                         , ("gt_boxes", x2)
                                         , ("mean",     mean)
                                         , ("std",      std)]
                fitAndEval optimizer binding metric
                eval <- formatMetric metric
                lr <- use (untag . mod_statistics . stat_last_lr)
                logInfo . display $ sformat (int % " " % stext % " LR: " % fixed 5) i eval lr

            withSession sess $ saveState (ei == 1)
                (formatToString ("checkpoints/faster_rcnn_epoch_" % left 3 '0') ei)


toNDArray :: MonadIO m => [(String, Coco.ImageTensor, Coco.ImageInfo, Coco.GTBoxes)] -> m (NDArray Float, NDArray Float, NDArray Float)
toNDArray dat = liftIO $ do
    imagesC  <- repaToNDArray $ evstack images
    infosC   <- repaToNDArray $ evstack infos
    gtboxesC <- repaToNDArray $ evstack $ V.map evstack gtboxes
    return $!! (imagesC, infosC, gtboxesC)
  where
    (_, images, infos, gtboxes) = V.unzip4 $ V.fromList dat
    evstack arrs = vstack $ V.map (expandDim 0) arrs

repaToNDArray :: Repa.Shape sh => Array U sh Float -> IO (NDArray Float)
repaToNDArray arr = do
    let Just sh = nonEmpty $ reverse $ Repa.listOfShape $ Repa.extent arr
    fromVector sh $ VS.convert $ Repa.toUnboxed arr
