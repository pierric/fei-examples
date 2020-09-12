{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE RecordWildCards #-}
module Main where

import           Control.Lens                      (use, (.=))
import           Data.Conduit                      ((.|))
import qualified Data.Conduit.List                 as C
import           Formatting                        (fixed, formatToString, int,
                                                    left, sformat, stext,
                                                    string, (%))
import           Options.Applicative               (execParser, fullDesc,
                                                    header, helper, info,
                                                    (<**>))
import           RIO                               hiding (Const)
import           RIO.Char                          (isDigit)
import           RIO.FilePath
import qualified RIO.HashMap                       as M
import qualified RIO.HashSet                       as S
import           RIO.List                          (sort)
import qualified RIO.Text                          as T

import           MXNet.Base
import qualified MXNet.Base.ParserUtils            as P
import           MXNet.NN
import qualified MXNet.NN.DataIter.Anchor          as Anchor
import qualified MXNet.NN.DataIter.Coco            as Coco
import           MXNet.NN.DataIter.Conduit
import           MXNet.NN.ModelZoo.RCNN.FasterRCNN (RCNNAccMetric (..),
                                                    RCNNL1LossMetric (..),
                                                    RCNNLogLossMetric (..),
                                                    RPNAccMetric (..),
                                                    RPNL1LossMetric (..),
                                                    RPNLogLossMetric (..),
                                                    RcnnConfiguration (..))
import           MXNet.NN.ModelZoo.RCNN.MaskRCNN

import           RCNN


main :: IO ()
main = do
    _    <- mxListAllOpNames
    registerCustomOperator ("anchor_generator", Anchor.buildAnchorGenerator)

    (rcnn_conf, pg_conf@ProgConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Mask-RCNN")
    if pg_infer then
        -- mainInfer rcnn_conf pg_conf
        error "cannot infer"
    else
        mainTrain rcnn_conf pg_conf
    mxNotifyShutdown


mainTrain rcnn_conf@RcnnConfiguration{..} ProgConfig{..} = do
    sym  <- runLayerBuilder $ graphTrain rcnn_conf
    fixed_params <- fixedParams backbone TRAIN sym

    coco_inst <- Coco.coco ds_base_path "train2017"

    let coco_conf = Coco.CocoConfig coco_inst ds_img_size (toTriple ds_img_pixel_means) (toTriple ds_img_pixel_stds)
        -- There is a serious problem with asyncConduit. It made the training loop running
        -- in different threads, which is very bad because the execution of ExecutorForward
        -- has a thread-local state (saving the temporary workspace for cudnn)
        --
        -- data_iter = asyncConduit (Just batch_size) $
        --
        data_iter = ConduitData (Just batch_size) $
                    Coco.cocoImagesBBoxesMasks True        .|
                    C.mapM (withRpnTargets'Mask rcnn_conf) .|
                    C.chunksOf batch_size                  .|
                    C.mapM concatBatch'Mask

    sess <- newMVar =<< initialize @"mask_rcnn" sym (Config {
                _cfg_data  = M.fromList [ ("data",     STensor [batch_size, 3, ds_img_size, ds_img_size])
                                        , ("im_info",  STensor [batch_size, 3])
                                        , ("gt_boxes", STensor [batch_size, 1, 5])
                                        , ("gt_masks", STensor [batch_size, 1, ds_img_size, ds_img_size])
                                        ],
                _cfg_label = ["rpn_cls_targets"
                             ,"rpn_box_targets"
                             ,"rpn_box_masks"
                             ,"box_reg_mean"
                             ,"box_reg_std"
                             ],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_fixed_params = fixed_params,
                _cfg_context = contextGPU0
            })

    mean <- fromVector [4] [0,0,0,0]
    std  <- fromVector [4] [0.1, 0.1, 0.2, 0.2]

    let lr_sched = lrOfFactor (#base := 0.004 .& #factor := 0.5 .& #step := 2000 .& Nil)
    -- optimizer <- makeOptimizer SGD'Mom lr_sched (#momentum := 0.9
    --                                           .& #wd := 0.0001
    --                                           .& #rescale_grad := 1 / (fromIntegral batch_size)
    --                                           .& #clip_gradient := 10
    --                                           .& Nil)
    optimizer <- makeOptimizer ADAM lr_sched (#rescale_grad := 1 / (fromIntegral batch_size)
                                           .& #wd := 0.0001
                                           .& #clip_gradient := 10 .& Nil)

    runApp coco_conf $ do
        checkpoint <- lastSavedState "checkpoints" "mask_rcnn"
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

        metric <- newMetric "train" (RPNAccMetric "rpn_cls_targets" :*
                                     RCNNAccMetric :*
                                     RPNLogLossMetric "rpn_cls_targets" :*
                                     RCNNLogLossMetric :*
                                     RPNL1LossMetric :*
                                     RCNNL1LossMetric :* MNil)

        -- update the internal counting of the iterations
        -- the lr is updated as per to it
        withSession sess $ do
            untag . mod_statistics . stat_num_upd .= (start_epoch - 1) * pg_train_iter_per_epoch

        forM_ ([start_epoch..pg_train_epochs] :: [Int]) $ \ ei -> do
            logInfo . display $ sformat ("Epoch " % int) ei
            let slice = takeD pg_train_iter_per_epoch data_iter
            void $ forEachD_i slice $ \(i, (fn, [x0, x1, x2, x3, y0, y1, y2])) -> withSession sess $ do
                let binding = M.fromList [ ("gt_masks",        x0)
                                         , ("gt_boxes",        x1)
                                         , ("data",            x2)
                                         , ("im_info",         x3)
                                         , ("rpn_cls_targets", y0)
                                         , ("rpn_box_targets", y1)
                                         , ("rpn_box_masks",   y2)
                                         , ("box_reg_mean",    mean)
                                         , ("box_reg_std",     std)
                                         ]
                fitAndEval optimizer binding metric
                eval <- formatMetric metric
                lr <- use (untag . mod_statistics . stat_last_lr)
                logInfo . display $ sformat (int % " " % stext % " LR: " % fixed 5) i eval lr

            withSession sess $ saveState (ei == 1)
                (formatToString ("checkpoints/mask_rcnn_epoch_" % left 3 '0') ei)

