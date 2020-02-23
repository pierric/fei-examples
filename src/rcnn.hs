{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
module Main where

import qualified Data.HashMap.Strict as M
import qualified Data.HashSet as S
import Control.Monad (forM_, void, unless, when)
import Control.Applicative (liftA2)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import Control.Monad.Trans.Resource
import Control.Monad.Reader
import Control.Lens ((.=), use)
import System.IO (hFlush, stdout)
import Options.Applicative (
    Parser, execParser,
    long, value, option, auto, strOption, metavar, showDefault, eitherReader, help,
    info, helper, fullDesc, header, (<**>))
import Data.Attoparsec.Text (sepBy, char, rational, decimal, endOfInput, parseOnly)
import qualified Data.Text as T
import Data.Conduit
import qualified Data.Conduit.List as C
import System.Directory (doesFileExist, canonicalizePath)
import Text.Printf (printf)

import MXNet.Base (
    NDArray(..), toVector, execForward,
    contextCPU, contextGPU0,
    mxListAllOpNames, mxNotifyShutdown, mxNDArraySave,
    registerCustomOperator,
    ndshape,
    listOutputs, internals, inferShape, at', at,
    HMap(..), (.&), ArgOf(..))
import MXNet.NN
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Conduit
import MXNet.NN.Utils (loadState, saveState)
import Model.FasterRCNN
import qualified DataIter.Coco as Coco
import qualified MXNet.NN.DataIter.Coco as Coco

import qualified Data.Array.Repa as Repa
import qualified Data.Vector as V
import Debug.Trace


data DatasetConfig = DatasetConfig {
    ds_base_path       :: String,
    ds_img_size        :: Int,
    ds_img_pixel_means :: [Float],
    ds_img_pixel_stds  :: [Float]
} deriving Show

cmdArgParser :: Parser (RcnnConfiguration, DatasetConfig)
cmdArgParser = liftA2 (,)
                (RcnnConfiguration
                    <$> option intList   (long "rpn-anchor-scales" <> metavar "SCALES"         <> showDefault <> value [8,16,32] <> help "rpn anchor scales")
                    <*> option floatList (long "rpn-anchor-ratios" <> metavar "RATIOS"         <> showDefault <> value [0.5,1,2] <> help "rpn anchor ratios")
                    <*> option auto      (long "rpn-feat-stride"   <> metavar "STRIDE"         <> showDefault <> value 16        <> help "rpn feature stride")
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
                    <*> option auto      (long "rcnn-feat-stride"  <> metavar "FEATURE-STRIDE" <> showDefault <> value 16        <> help "rcnn feature stride")
                    <*> option intList   (long "rcnn-pooled-size"  <> metavar "POOLED-SIZE"    <> showDefault <> value [7,7]     <> help "rcnn pooled size")
                    <*> option auto      (long "rcnn-batch-rois"   <> metavar "BATCH_ROIS"     <> showDefault <> value 128       <> help "rcnn batch rois")
                    <*> option auto      (long "rcnn-batch-size"   <> metavar "BATCH-SIZE"     <> showDefault <> value 1         <> help "rcnn batch size")
                    <*> option auto      (long "rcnn-fg-fraction"  <> metavar "FG-FRACTION"    <> showDefault <> value 0.25      <> help "rcnn foreground fraction")
                    <*> option auto      (long "rcnn-fg-overlap"   <> metavar "FG-OVERLAP"     <> showDefault <> value 0.5       <> help "rcnn foreground iou threshold")
                    <*> option floatList (long "rcnn-bbox-stds"    <> metavar "BBOX-STDDEV"    <> showDefault <> value [0.1, 0.1, 0.2, 0.2] <> help "standard deviation of bbox")
                    <*> strOption        (long "pretrained"        <> metavar "PATH"           <> value "" <> help "path to pretrained model"))
                (DatasetConfig
                    <$> strOption        (long "base" <> metavar "PATH" <> help "path to the dataset")
                    <*> option auto      (long "img-size"          <> metavar "SIZE" <> showDefault <> value 1024 <> help "long side of image")
                    <*> option floatList (long "img-pixel-means"   <> metavar "RGB-MEAN" <> showDefault <> value [0,0,0] <> help "RGB mean of images")
                    <*> option floatList (long "img-pixel-stds"    <> metavar "RGB-STDS" <> showDefault <> value [1,1,1] <> help "RGB std-dev of images"))
  where
    list obj  = parseOnly (sepBy obj (char ',') <* endOfInput) . T.pack
    floatList = eitherReader $ list rational
    intList   = eitherReader $ list decimal

buildProposalTargetProp params = do
    let params' = M.fromList params
    return $ ProposalTargetProp {
        _num_classes = read $ params' M.! "num_classes",
        _batch_images= read $ params' M.! "batch_images",
        _batch_rois  = read $ params' M.! "batch_rois",
        _fg_fraction = read $ params' M.! "fg_fraction",
        _fg_overlap  = read $ params' M.! "fg_overlap",
        _box_stds    = read $ params' M.! "box_stds"
    }

toTriple [a, b, c] = (a, b, c)
toTriple x = error (show x)


default_initializer :: Initializer Float
default_initializer name = trace name $ case name of
    "rpn_conv_3x3_weight"  -> normal 0.01 name
    "rpn_conv_3x3_bias"    -> zeros name
    "rpn_cls_score_weight" -> normal 0.01 name
    "rpn_cls_score_bias"   -> zeros name
    "rpn_bbox_pred_weight" -> normal 0.01 name
    "rpn_bbox_pred_bias"   -> zeros name
    "cls_score_weight"     -> normal 0.01 name
    "cls_score_bias"       -> zeros name
    "bbox_pred_weight"     -> normal 0.001 name
    "bbox_pred_bias"       -> zeros name
    _ -> empty name

loadWeights weights_path = do
    weights_path <- liftIO $ canonicalizePath weights_path
    e <- liftIO $ doesFileExist (weights_path ++ ".params")
    if not e
        then liftIO $ putStrLn $ "'" ++ weights_path ++ ".params' doesn't exist."
        else loadState weights_path ["rpn_conv_3x3_weight",
                                       "rpn_conv_3x3_bias",
                                       "rpn_cls_score_weight",
                                       "rpn_cls_score_bias",
                                       "rpn_bbox_pred_weight",
                                       "rpn_bbox_pred_bias",
                                       "cls_score_weight",
                                       "cls_score_bias",
                                       "bbox_pred_weight",
                                       "bbox_pred_bias"]

main :: IO ()
main = do
    _    <- mxListAllOpNames
    registerCustomOperator ("proposal_target", buildProposalTargetProp)
    (rcnn_conf@RcnnConfiguration{..}, DatasetConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
    sym  <- symbolTrain rcnn_conf

    rpn_cls_score_output <- internals sym >>= flip at' "rpn_cls_score_output"
    -- get the feature (width, height) at the top of feature extraction.
    (_, [(_, [_, _, feat_width, feat_height])], _, _) <- inferShape rpn_cls_score_output [("data", [1, 3, ds_img_size, ds_img_size])]

    coco_inst <- Coco.coco ds_base_path "train2017"

    let fixed_num_gt = if rcnn_batch_size == 1 then Nothing else Just 10

    let coco_conf = Coco.CocoConfig coco_inst ds_img_size (toTuple ds_img_pixel_means) (toTuple ds_img_pixel_stds)
        anchors = Coco.withAnchors (#batch_size     := (rcnn_batch_size :: Int)
                                 .& #feature_width  := feat_width
                                 .& #feature_height := feat_height
                                 .& #feature_stride := rpn_feature_stride
                                 .& #anchor_scales  := rpn_anchor_scales
                                 .& #anchor_ratios  := rpn_anchor_ratios
                                 .& #allowed_border := rpn_allowd_border
                                 .& #batch_rois     := rpn_batch_rois
                                 .& #fg_fraction    := rpn_fg_fraction
                                 .& #fg_overlap     := rpn_fg_overlap
                                 .& #bg_overlap     := rpn_bg_overlap
                                 .& #fixed_num_gt   := fixed_num_gt
                                 .& Nil)
        data_iter = ConduitData (Just rcnn_batch_size) $
                        Coco.cocoImages True .|  C.mapM Coco.loadImageAndBBoxes .| C.catMaybes .| anchors

    sess <- initialize @"fastrcnn" sym $ Config {
        _cfg_data  = M.fromList [("data",        [3, ds_img_size, ds_img_size]),
                                 ("im_info",     [3]),
                                 ("gt_boxes",    [0, 5])],
        _cfg_label = ["label", "bbox_target", "bbox_weight"],
        _cfg_initializers = M.empty,
        _cfg_default_initializer = default_initializer,
        _cfg_fixed_params = S.fromList [],
        _cfg_context = contextCPU
    }
    optimizer <- makeOptimizer SGD'Mom (Const 0.0000001) (#momentum := 0.9
                                                   .& #wd := 0.0005
                                                   .& #rescale_grad := 1 / (fromIntegral rcnn_batch_size)
                                                   .& #clip_gradient := 5
                                                   .& Nil)

    runResourceT $ flip runReaderT coco_conf $ train sess $ do
        -- sess_callbacks .= [Callback DumpLearningRate, Callback (Checkpoint "checkpoints")]
        checkpoint <- lastSavedState "checkpoints"
        liftIO $ print checkpoint
        start_epoch <- case checkpoint of
            Nothing -> do
                liftIO $ print pretrained_weights
                unless (null pretrained_weights) (loadWeights pretrained_weights)
                return (1 :: Int)
            Just filename -> do
                loadState filename []
                return $ read (take 3 $ drop 30 filename) + 1

        metric <- newMetric "train" (RPNAccMetric 0 "label" :* RCNNAccMetric 2 4 :* RPNLogLossMetric 0 "label" :* RCNNLogLossMetric 2 4 :* RPNL1LossMetric 1 "bbox_weight" :* RCNNL1LossMetric 3 4 :* MNil)

        forM_ [start_epoch..1000] $ \ ei -> do
            liftIO $ putStrLn $ "Epoch " ++ show ei
            liftIO $ hFlush stdout
            void $ forEachD_i (liftD $ takeD 50 data_iter) $ \(i, ((x0, x1, x2), (y0, y1, y2))) -> do
                let binding = M.fromList [ ("data",        x0)
                                         , ("im_info",     x1)
                                         , ("gt_boxes",    x2)
                                         , ("label",       y0)
                                         , ("bbox_target", y1)
                                         , ("bbox_weight", y2) ]
                fitAndEval optimizer binding metric
                eval <- format metric
                liftIO $ do
                    putStrLn $ show i ++ " " ++ eval
                    hFlush stdout

            saveState (ei == 1) (printf "checkpoints/faster_rcnn_epoch_%03d" ei)
            liftIO $ putStrLn ""

    mxNotifyShutdown


toTuple [a, b, c] = (a, b, c)
