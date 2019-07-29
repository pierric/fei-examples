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
    long, value, option, auto, strOption, metavar, showDefault, maybeReader, help,
    info, helper, fullDesc, header, (<**>))
import Data.Attoparsec.Text (sepBy, char, rational, decimal, parse, maybeResult)
import qualified Data.Text as T

import MXNet.Base (
    NDArray(..), toVector,
    contextCPU, contextGPU0, 
    mxListAllOpNames, mxNotifyShutdown, 
    registerCustomOperator, 
    ndshape,
    listOutputs, internals, inferShape, at',
    HMap(..), (.&), ArgOf(..))
-- import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Conduit
import MXNet.NN.DataIter.Coco as Coco
import Model.FasterRCNN

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
                    <*> option auto      (long "rcnn-num-classes"  <> metavar "NUM-CLASSES"    <> showDefault <> value 21        <> help "rcnn number of classes")
                    <*> option auto      (long "rcnn-feat-stride"  <> metavar "FEATURE-STRIDE" <> showDefault <> value 16        <> help "rcnn feature stride")
                    <*> option intList   (long "rcnn-pooled-size"  <> metavar "POOLED-SIZE"    <> showDefault <> value [14,14]   <> help "rcnn pooled size")
                    <*> option auto      (long "rcnn-batch-rois"   <> metavar "BATCH_ROIS"     <> showDefault <> value 128       <> help "rcnn batch rois")
                    <*> option auto      (long "rcnn-batch-size"   <> metavar "BATCH-SIZE"     <> showDefault <> value 1         <> help "rcnn batch size")
                    <*> option auto      (long "rcnn-fg-fraction"  <> metavar "FG-FRACTION"    <> showDefault <> value 0.25      <> help "rcnn foreground fraction")
                    <*> option auto      (long "rcnn-fg-overlap"   <> metavar "FG-OVERLAP"     <> showDefault <> value 0.5       <> help "rcnn foreground iou threshold")
                    <*> option floatList (long "rcnn-bbox-stds"    <> metavar "BBOX-STDDEV"    <> showDefault <> value [0.1, 0.1, 0.2, 0.2] <> help "standard deviation of bbox"))                    
                (CocoConfig
                    <$> strOption        (long "coco" <> metavar "PATH" <> help "path to the coco dataset")
                    <*> option auto      (long "img-short-side"    <> metavar "SIZE" <> showDefault <> value 600  <> help "short side of image")
                    <*> option auto      (long "img-long-side"     <> metavar "SIZE" <> showDefault <> value 1000 <> help "long side of image")
                    <*> option floatList (long "img-pixel-means"   <> metavar "RGB-MEAN" <> showDefault <> value [0,0,0] <> help "RGB mean of images")
                    <*> option floatList (long "img-pixel-stds"    <> metavar "RGB-STDS" <> showDefault <> value [1,1,1] <> help "RGB std-dev of images"))
  where
    list obj  = maybeResult . parse (sepBy obj (char ',')) . T.pack
    floatList = maybeReader $ list rational
    intList   = maybeReader $ list decimal

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


default_initializer :: Initializer Float
default_initializer name shp = normal 0.1 name shp

main :: IO ()
main = do
    _    <- mxListAllOpNames
    registerCustomOperator ("proposal_target", buildProposalTargetProp)
    (rcnn_conf@RcnnConfiguration{..}, CocoConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
    sym  <- symbolTrain rcnn_conf

    rpn_cls_score_output <- internals sym >>= flip at' "rpn_cls_score_output"
    let extr_feature_shape (w, h) = do
            -- get the feature (width, height) at the top of feature extraction.
            (_, [(_, [_, _, feat_width, feat_height])], _, _) <- inferShape rpn_cls_score_output [("data", [1, 3,w, h])]
            return (feat_width, feat_height)

    coco_inst <- coco coco_base_path "val2017"
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
    optimizer <- makeOptimizer SGD'Mom (Const 0.0002) Nil

    train sess $
        void $ forEachD_i data_iter $ \(i, ((x0, x1, x2), (y0, y1, y2))) -> do
            -- fitDataset trainingData testingData bind optimizer (CrossEntropy "y" :* Accuracy "y" :* MNil) 2
            liftIO $ putStrLn $ "[Train] "
            forM_ [1, 2] $ \ind -> do
                liftIO $ putStrLn $ "iteration " ++ show ind
                void $ forEachD_i data_iter $ \(i, ((x0, x1, x2), (y0, y1, y2))) -> do
                    let binding = M.fromList [ ("data",        x0)
                                             , ("im_info",     x1)
                                             , ("gt_boxes",    x2)
                                             , ("label",       y0)
                                             , ("bbox_target", y1)
                                             , ("bbox_weight", y2) ]
                    metric <- newMetric "train" MNil
                    fitAndEval optimizer binding metric
                    liftIO $ do
                        putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show 10000
                        hFlush stdout
                liftIO $ putStrLn ""

    mxNotifyShutdown

main0 = do
    (rcnn_conf@RcnnConfiguration{..}, CocoConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
    _    <- mxListAllOpNames
    registerCustomOperator ("proposal_target", buildProposalTargetProp)
    sym  <- symbolTrain rcnn_conf
    isym <- internals sym
    fsym <- at' isym "rpn_cls_prob_output"
    res  <- inferShape fsym [("data", [1,3,coco_img_long_side, coco_img_short_side])]
    print res
    -- let arg_ind = scanl (+) 0 $ map length shapes
    --     arg_shp = concat shapes
    -- print (names, arg_ind, arg_shp)
    -- (inp_shp, out_shp, aux_shp, complete) <- mxSymbolInferShape sym names arg_ind arg_shp
    -- -- unless complete $ error "incomplete shapes"
    -- print (inp_shp, out_shp, aux_shp, complete)
    mxNotifyShutdown
    return ()

main1 = do
    _    <- mxListAllOpNames
    registerCustomOperator ("proposal_target", buildProposalTargetProp)
    let conf = RcnnConfiguration [1,2,4]  [0.5,1,2]  16  1  12000  1000  0.8  10  0.7  0.5  0.2  1  6  16  [14, 14] 128 1 0.25 0.5 [0.1, 0.1, 0.2, 0.2]
    sym  <- symbolTrain conf
    res  <- inferShape sym [
                ("data",        [1,3, 600, 1000]),
                ("im_info",     [1, 3]),
                ("gt_boxes",    [1, 0, 5]) ]
    print res

    -- ([  ("data",[1,3,600,1000]),
    --     ("conv1_1-weight",[64,3,3,3]),
    --     ("conv1_1-bias",[64]),
    --     ("conv1_2-weight",[64,64,3,3]),
    --     ("conv1_2-bias",[64]),
    --     ("conv2_1-weight",[128,64,3,3]),
    --     ("conv2_1-bias",[128]),
    --     ("conv2_2-weight",[128,128,3,3]),
    --     ("conv2_2-bias",[128]),
    --     ("conv3_1-weight",[256,128,3,3]),
    --     ("conv3_1-bias",[256]),
    --     ("conv3_2-weight",[256,256,3,3]),
    --     ("conv3_2-bias",[256]),
    --     ("conv3_3-weight",[256,256,3,3]),
    --     ("conv3_3-bias",[256]),
    --     ("conv4_1-weight",[512,256,3,3]),
    --     ("conv4_1-bias",[512]),
    --     ("conv4_2-weight",[512,512,3,3]),
    --     ("conv4_2-bias",[512]),
    --     ("conv4_3-weight",[512,512,3,3]),
    --     ("conv4_3-bias",[512]),
    --     ("conv5_1-weight",[512,512,3,3]),
    --     ("conv5_1-bias",[512]),
    --     ("conv5_2-weight",[512,512,3,3]),
    --     ("conv5_2-bias",[512]),
    --     ("conv5_3-weight",[512,512,3,3]),
    --     ("conv5_3-bias",[512]),
    --     ("rpn_conv_3x3-weight",[512,512,3,3]),
    --     ("rpn_conv_3x3-bias",[512]),
    --     ("rpn_cls_score-weight",[18,512,1,1]),
    --     ("rpn_cls_score-bias",[18]),
    --     ("label",[1,5022]),
    --     ("bbox_weight",[1,36,18,31]),
    --     ("rpn_bbox_pred-weight",[36,512,1,1]),
    --     ("rpn_bbox_pred-bias",[36]),
    --     ("bbox_target",[1,36,18,31]),
    --     ("im_info",[1,3]),
    --     ("gt_boxes",[1,0,5]),
    --     ("fc6-weight",[4096,100352]),
    --     ("fc6-bias",[4096]),
    --     ("fc7-weight",[4096,4096]),
    --     ("fc7-bias",[4096]),
    --     ("cls_score-weight",[6,4096]),
    --     ("cls_score-bias",[6]),
    --     ("bbox_pred-weight",[24,4096]),
    --     ("bbox_pred-bias",[24])],[("rpn_cls_prob_output",[1,2,162,31]),
    --     ("rpn_bbox_loss_output",[1,36,18,31]),
    --     ("cls_prob_output",[128,6]),
    --     ("bbox_loss_output",[128,24]),
    --     ("label_sg_output",[128])],[],True)

    -- let arg_ind = scanl (+) 0 $ map length shapes
    --     arg_shp = concat shapes
    -- print (names, arg_ind, arg_shp)
    -- (inp_shp, out_shp, aux_shp, complete) <- mxSymbolInferShape sym names arg_ind arg_shp
    -- -- unless complete $ error "incomplete shapes"
    -- print (inp_shp, out_shp, aux_shp, complete)
    mxNotifyShutdown
    return ()