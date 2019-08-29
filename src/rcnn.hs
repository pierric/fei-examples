{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
module Main where

import qualified Data.HashMap.Strict as M
import Control.Monad (forM_, void, unless)
import Control.Applicative (liftA2)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import System.IO (hFlush, stdout)
import Options.Applicative (
    Parser, execParser, 
    long, value, option, auto, strOption, metavar, showDefault, eitherReader, help,
    info, helper, fullDesc, header, (<**>))
import Data.Attoparsec.Text (sepBy, char, rational, decimal, endOfInput, parseOnly)
import qualified Data.Text as T
import System.Directory (doesFileExist, canonicalizePath)

import MXNet.Base (
    NDArray(..), toVector,
    contextCPU, contextGPU0, 
    mxListAllOpNames, mxNotifyShutdown, mxNDArraySave,
    registerCustomOperator, 
    ndshape,
    listOutputs, internals, inferShape, at',
    HMap(..), (.&), ArgOf(..))
import MXNet.NN
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Conduit
import MXNet.NN.DataIter.Coco as Coco
import MXNet.NN.Utils (loadSession, saveSession)
import Model.FasterRCNN

-- import qualified Data.Vector as V
-- import Control.Lens ((^.), (.=), use)
-- import MXNet.Coco.Types (images, img_id, img_file_name)
-- import qualified MXNet.NN.DataIter.Anchor as Anchor
-- import Data.Array.Repa (Array, DIM1, DIM2, D, U, (:.)(..), Z (..), All(..), (+^), fromListUnboxed)
-- import qualified Data.Array.Repa as Repa
-- import qualified Data.Vector.Unboxed as UV
-- import MXNet.Base ((!), (!?), (.&), fromVector)
-- import Control.Lens ((^.), (%~) , view, makeLenses, _1, _2)
-- import Control.Monad.Reader
-- import Control.Exception
-- import Debug.Trace
-- import MXNet.Base (ndsize)
-- import MXNet.NN.DataIter.Vec

-- import qualified Foreign.CUDA.Driver.Profiler as CUDA

data CocoConfig = CocoConfig {
    coco_base_path       :: String,
    coco_img_short_side  :: Int,
    coco_img_long_side   :: Int,
    coco_img_pixel_means :: [Float],
    coco_img_pixel_stds  :: [Float]
} deriving Show

cmdArgParser :: Parser (RcnnConfiguration, CocoConfig)
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
                (CocoConfig
                    <$> strOption        (long "coco" <> metavar "PATH" <> help "path to the coco dataset")
                    <*> option auto      (long "img-short-side"    <> metavar "SIZE" <> showDefault <> value 600  <> help "short side of image")
                    <*> option auto      (long "img-long-side"     <> metavar "SIZE" <> showDefault <> value 1000 <> help "long side of image")
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
default_initializer name = case name of
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
        else loadSession weights_path ["rpn_conv_3x3_weight",
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
    (rcnn_conf@RcnnConfiguration{..}, CocoConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
    sym  <- symbolTrain rcnn_conf

    -- res  <- inferShape sym [
    --             ("data",        [1,3, 600, 1000]),
    --             ("im_info",     [1, 3]),
    --             ("gt_boxes",    [1, 0, 5]) ]
    -- print res

    rpn_cls_score_output <- internals sym >>= flip at' "rpn_cls_score_output"
    let extr_feature_shape (w, h) = do
            -- get the feature (width, height) at the top of feature extraction.
            (_, [(_, [_, _, feat_width, feat_height])], _, _) <- inferShape rpn_cls_score_output [("data", [1, 3,w, h])]
            return (feat_width, feat_height)

    coco_inst <- coco coco_base_path "train2017"
    let data_iter = cocoImagesWithAnchors coco_inst extr_feature_shape
                        (#anchor_scales := rpn_anchor_scales
                      .& #anchor_ratios := rpn_anchor_ratios
                      .& #batch_rois    := rpn_batch_rois
                      .& #feature_stride:= rpn_feature_stride
                      .& #allowed_border:= rpn_allowd_border
                      .& #fg_fraction   := rpn_fg_fraction
                      .& #fg_overlap    := rpn_fg_overlap
                      .& #bg_overlap    := rpn_bg_overlap
                      .& #short_size    := coco_img_short_side
                      .& #long_size     := coco_img_long_side
                      .& #mean          := toTriple coco_img_pixel_means
                      .& #std           := toTriple coco_img_pixel_stds
                      .& #batch_size    := rcnn_batch_size
                      .& Nil)
    
    sess <- initialize sym $ Config { 
        _cfg_data  = M.fromList [("data",        [3, coco_img_short_side, coco_img_long_side]), 
                                 ("im_info",     [3]),
                                 ("gt_boxes",    [0, 5])],
        _cfg_label = M.empty,
        _cfg_initializers = M.empty,
        _cfg_default_initializer = default_initializer,
        _cfg_context = contextCPU
    }
    optimizer <- makeOptimizer SGD'Mom (Const 0.001) (#momentum := 0.9 
                                                   .& #wd := 0.0005 
                                                   .& #rescale_grad := 1 / (fromIntegral rcnn_batch_size) 
                                                   .& #clip_gradient := 5 
                                                   .& Nil)

    train sess $ do
        sess_callbacks .= [Callback DumpLearningRate, Callback (Checkpoint "checkpoints")]

        unless (null pretrained_weights) (loadWeights pretrained_weights)
    
        liftIO $ putStrLn $ "[Train] "
        metric <- newMetric "train" (RPNAccMetric 0 "label" :* RCNNAccMetric 2 4 :* RPNLogLossMetric 0 "label" :* RCNNLogLossMetric 2 4 :* RPNL1LossMetric 1 "bbox_weight" :* RCNNL1LossMetric 3 4 :* MNil)
        forM_ [5] $ \ ei -> do
            liftIO $ putStrLn $ "Epoch " ++ show ei
            liftIO $ hFlush stdout
            void $ forEachD_i data_iter $ \(i, ((x0, x1, x2), (y0, y1, y2))) -> do
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
            liftIO $ putStrLn ""

    -- CUDA.stop
    mxNotifyShutdown

-- main0 = do
--     _    <- mxListAllOpNames
--     registerCustomOperator ("proposal_target", buildProposalTargetProp)
--     (rcnn_conf@RcnnConfiguration{..}, CocoConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
--     sym  <- symbolTrain rcnn_conf

--     -- res  <- inferShape sym [
--     --             ("data",        [1,3, 600, 1000]),
--     --             ("im_info",     [1, 3]),
--     --             ("gt_boxes",    [1, 0, 5]) ]
--     -- print res

--     rpn_cls_score_output <- internals sym >>= flip at' "rpn_cls_score_output"
--     let extr_feature_shape (w, h) = do
--             -- get the feature (width, height) at the top of feature extraction.
--             (_, [(_, [_, _, feat_width, feat_height])], _, _) <- inferShape rpn_cls_score_output [("data", [1, 3,w, h])]
--             return (feat_width, feat_height)

--     coco_inst@(Coco _ _ _coco_inst) <- coco coco_base_path "val2017"
--     let data_iter = cocoImagesWithAnchors coco_inst extr_feature_shape
--                         (#anchor_scales := rpn_anchor_scales
--                       .& #anchor_ratios := rpn_anchor_ratios
--                       .& #batch_rois    := rpn_batch_rois
--                       .& #feature_stride:= rpn_feature_stride
--                       .& #allowed_border:= rpn_allowd_border
--                       .& #fg_fraction   := rpn_fg_fraction
--                       .& #fg_overlap    := rpn_fg_overlap
--                       .& #bg_overlap    := rpn_bg_overlap
--                       .& #short_size    := coco_img_short_side
--                       .& #long_size     := coco_img_long_side
--                       .& #mean          := toTriple coco_img_pixel_means
--                       .& #std           := toTriple coco_img_pixel_stds
--                       .& #batch_size    := rcnn_batch_size
--                       .& Nil)
    
--     let cnt = 36
--     print (V.map (\img -> (img ^. img_id, img ^. img_file_name)) $ V.take cnt $ _coco_inst ^. images)

--     let anchorMake info = do
--             vinfo <- toVector info            
--             let imHeight = floor $ vinfo SV.! 0
--                 imWidth  = floor $ vinfo SV.! 1
--             (featureWidth, featureHeight) <- extr_feature_shape (imWidth, imHeight)
--             anchors <- runReaderT (Anchor.anchors rpn_feature_stride featureWidth featureHeight) anchConf
--             convertToMX $ V.toList anchors
--           where 
--             anchConf = Anchor.Configuration {
--                             Anchor._conf_anchor_scales  = rpn_anchor_scales,
--                             Anchor._conf_anchor_ratios  = rpn_anchor_ratios,
--                             Anchor._conf_allowed_border = rpn_allowd_border,
--                             Anchor._conf_fg_num         = floor $ rpn_fg_fraction * fromIntegral rpn_batch_rois,
--                             Anchor._conf_batch_num      = rpn_batch_rois,
--                             Anchor._conf_fg_overlap     = rpn_fg_overlap,
--                             Anchor._conf_bg_overlap     = rpn_bg_overlap
--                         }
--             convert :: Repa.Shape sh => [Array U sh Float] -> ([Int], UV.Vector Float)
--             convert xs = assert (not (null xs)) $ (ext, vec)
--               where
--                 vec = UV.concat $ map Repa.toUnboxed xs
--                 sh0 = Repa.extent (head xs)
--                 ext = length xs : reverse (Repa.listOfShape sh0)
                    
--             convertToMX :: Repa.Shape sh => [Array U sh Float] -> IO (NDArray Float)
--             convertToMX   = uncurry fromVector . (_2 %~ UV.convert) . convert

--     ds <- takeD cnt data_iter
--     forM_ (zip[0..] ds) $ \ (i, ((d0, d1, d2), (l1, l2, l3))) -> do
--         anch <- anchorMake d1
--         mxNDArraySave ("image" ++ show i) (zip ["anchors", "image", "info", "gt", "label", "target", "weight"] (map unNDArray [anch, d0, d1, d2, l1, l2, l3]))


--     mxNotifyShutdown
--     return ()

-- main1 = do
--     _    <- mxListAllOpNames
--     registerCustomOperator ("proposal_target", buildProposalTargetProp)
--     (rcnn_conf@RcnnConfiguration{..}, CocoConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
--     sym  <- symbolTrain rcnn_conf
--     res  <- inferShape sym [
--                 ("data",        [1,3, 600, 1000]),
--                 ("im_info",     [1, 3]),
--                 ("gt_boxes",    [1, 0, 5]) ]
--     print res

--     mxNotifyShutdown
--     return ()

-- main2 = do
--     _    <- mxListAllOpNames
--     registerCustomOperator ("proposal_target", buildProposalTargetProp)
--     (rcnn_conf@RcnnConfiguration{..}, CocoConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
--     sym  <- symbolTrain rcnn_conf

--     sess <- initialize sym $ Config { 
--         _cfg_data  = M.fromList [("data",        [3, coco_img_short_side, coco_img_long_side]), 
--                                  ("im_info",     [3]),
--                                  ("gt_boxes",    [0, 5])],
--         _cfg_label = M.empty,
--         _cfg_initializers = M.empty,
--         _cfg_default_initializer = default_initializer,
--         _cfg_context = contextCPU
--     }

--     let params = M.toList (sess ^. sess_param)

--     print (sum $ map count params)

--     totalSize <- mapM calcSize params
--     print (sum totalSize)


--   where
--     calcSize (_, ParameterI a Nothing) = ndsize a
--     calcSize (_, ParameterI a (Just b)) = liftM2 (+) (ndsize a) (ndsize b)
--     calcSize (_, ParameterA a) = ndsize a

--     count (_, ParameterI a Nothing) = 1
--     count (_, ParameterI a (Just b)) = 2
--     count (_, ParameterA a) = 1
