{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE RecordWildCards #-}
module Main where

import           Control.Lens                      (ix, use, (.=), (^?!))
import           Data.Conduit                      ((.|))
import qualified Data.Conduit.List                 as C
import           Data.Random.Source.StdGen         (mkStdGen)
import           Formatting                        (fixed, formatToString, int,
                                                    left, sformat, stext,
                                                    string, (%))
import           Options.Applicative               (command, execParser,
                                                    fullDesc, header, helper,
                                                    hsubparser, info, progDesc,
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
import           MXNet.NN.ModelZoo.RCNN.FasterRCNN (RcnnConfiguration (..))
import           MXNet.NN.ModelZoo.RCNN.MaskRCNN

import           RCNN


main :: IO ()
main = do
    mxRandomSeed 8
    registerCustomOperator ("anchor_generator", Anchor.buildAnchorGenerator)

    let (apRcnnT, apRcnnI) = apRcnn
        apT = liftA3 (,,) apRcnnT apCommon apTrain
        apI = liftA3 (,,) apRcnnI apCommon (pure NoExtraArgs)
        whole = hsubparser
                ( command "train"     (info apT (progDesc "Train"))
               <> command "inference" (info apI (progDesc "Run inference"))
                )
    args <- liftIO $ execParser $ info (whole <**> helper) (fullDesc <> header "Faster-RCNN")
    case args of
      (RcnnConfigurationTrain{}, _, _)     -> mainTrain args

mainTrain (rcnn_conf@RcnnConfigurationTrain{..}, CommonArgs{..}, TrainArgs{..}) = do
    rand_gen   <- liftIO $ newIORef $ mkStdGen 42
    train_inst <- Coco.coco ds_base_path "train2017"
    val_inst   <- Coco.coco ds_base_path "val2017"

    let train_coco_conf = Coco.CocoConfig train_inst ds_img_size (toTriple ds_img_pixel_means) (toTriple ds_img_pixel_stds)

        -- There is a serious problem with asyncConduit. It made the training loop running
        -- in different threads, which is very bad because the execution of ExecutorForward
        -- has a thread-local state (saving the temporary workspace for cudnn)
        --
        -- data_iter = asyncConduit (Just batch_size) $
        --
        train_data_iter = ConduitData (Just batch_size) $
                          Coco.cocoImagesBBoxesMasks rand_gen    .|
                          C.mapM (withRpnTargets'Mask rcnn_conf) .|
                          C.chunksOf batch_size                  .|
                          C.mapM concatBatch'Mask

    runFeiM . WithNept "jiasen/mask-rcnn" $ do
    -- runFeiM . Simple $ do
        (_, sym)     <- runLayerBuilder $ graphT rcnn_conf
        fixed_params <- liftIO $ fixedParams backbone TRAIN sym

        initSession @"mask_rcnn" sym (Config {
            _cfg_data  = M.fromList [ ("data",     STensor [batch_size, 3, ds_img_size, ds_img_size])
                                    , ("im_info",  STensor [batch_size, 3])
                                    , ("gt_boxes", STensor [batch_size, 1, 5])
                                    , ("gt_masks", STensor [batch_size, 1, ds_img_size, ds_img_size])
                                    ],
            _cfg_label = ["rpn_cls_targets"
                         ,"rpn_box_targets"
                         ,"rpn_box_masks"
                         ],
            _cfg_initializers = M.empty,
            _cfg_default_initializer = default_initializer,
            _cfg_fixed_params = fixed_params,
            _cfg_context = contextGPU0 })

        let lr_sched0 = lrOfPoly (#base := 0.01 .& #power := 1 .& #maxnup := 10000 .& Nil)
            lr_sched  = WarmupScheduler 500 lr_sched0
        optm <- makeOptimizer SGD'Mom lr_sched (#momentum := 0.9
                                             .& #wd := 0.0001
                                             .& #clip_gradient := 10
                                             .& Nil)

        checkpoint <- lastSavedState "checkpoints" "mask_rcnn"
        start_epoch <- case checkpoint of
            Nothing -> do
                logInfo . display $ sformat string pretrained_weights
                unless (null pretrained_weights)
                    (askSession $ loadWeights pretrained_weights)
                return (1 :: Int)
            Just filename -> do
                askSession $ loadState filename []
                let (base, _) = splitExtension filename
                    fn_rev = T.reverse $ T.pack base
                    epoch = P.parseR (P.takeWhile isDigit <* P.takeText) fn_rev
                    epoch_next = (P.parseR P.decimal $ T.reverse epoch) + 1
                return epoch_next
        logInfo . display $ sformat ("fixed parameters: " % stext) (tshow (sort $ S.toList fixed_params))

        metric <- newMetric "train" (Accuracy (Just "RPN-acc") (PredByThreshold 0.5) 0
                                        (\_ preds -> preds ^?! ix 0)
                                        (\bindings _ -> bindings ^?! ix "rpn_cls_targets")
                                  :* Accuracy (Just "RCNN-acc") PredByArgmax 1
                                        (\_ preds -> preds ^?! ix 3)
                                        (\_ preds -> preds ^?! ix 5)
                                  :* Norm (Just "RPN-L1") 1
                                        (\_ preds -> preds ^?! ix 2)
                                  :* Norm (Just "RCNN-L1") 1
                                        (\_ preds -> preds ^?! ix 4)
                                  :* Loss (Just "Mask-ce")
                                        (\preds -> preds ^?! ix 6)
                                  :* Loss (Just "RPN-cls-loss")
                                        (\preds -> preds ^?! ix 1)
                                  :* Loss (Just "RPN-box-loss")
                                        (\preds -> preds ^?! ix 2)
                                  :* CrossEntropy (Just "RCNN-cls-loss") True
                                        (\_ preds -> preds ^?! ix 3)
                                        (\_ preds -> preds ^?! ix 5)
                                  :* Loss (Just "RCNN-box-loss")
                                        (\preds -> preds ^?! ix 4)
                                  :* MNil)

        -- update the internal counting of the iterations
        -- the lr is updated as per to it
        askSession $ do
            untag . mod_statistics . stat_num_upd .= (start_epoch - 1) * pg_train_iter_per_epoch

        forM_ ([start_epoch..pg_train_epochs] :: [Int]) $ \ ei -> do
            logInfo . display $ sformat ("Epoch " % int) ei

            let slice = takeD pg_train_iter_per_epoch train_data_iter

            flip runReaderT train_coco_conf $ do
                void $ forEachD_pi 32 slice $ \(i, (fn, input)) -> lift . askSession $ do
                    let [x0, x1, x2, x3, y0, y1, y2] = input
                        binding = M.fromList [ ("gt_masks",        x0)
                                             , ("gt_boxes",        x1)
                                             , ("data",            x2)
                                             , ("im_info",         x3)
                                             , ("rpn_cls_targets", y0)
                                             , ("rpn_box_targets", y1)
                                             , ("rpn_box_masks",   y2)
                                             ]
                    fitAndEval optm binding metric

                    kv <- metricsToList metric
                    lift $ mapM_ (uncurry neptLog) kv

                    when (i `mod` 20 == 0) $ do
                        eval <- metricFormat metric
                        lr <- use (untag . mod_statistics . stat_last_lr)
                        logInfo . display $ sformat (int % " " % stext % " LR: " % fixed 5) i eval lr

            askSession $ saveState (ei == 1)
                (formatToString ("checkpoints/mask_rcnn_epoch_" % left 3 '0') ei)

