{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE RecordWildCards #-}
module Main where

import           Codec.Picture                     (PixelRGBA8 (..), writePng)
import           Control.Lens                      (ix, use, (.=), (^?!))
import           Data.Array.Repa                   (Array, DIM1, DIM2, DIM3,
                                                    DIM4, U)
import qualified Data.Array.Repa                   as Repa
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
import qualified RIO.Vector.Boxed                  as VB
import qualified RIO.Vector.Storable               as VS

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor       as Ops
import qualified MXNet.Base.ParserUtils            as P
import           MXNet.NN
import qualified MXNet.NN.DataIter.Anchor          as Anchor
import qualified MXNet.NN.DataIter.Coco            as Coco
import           MXNet.NN.DataIter.Conduit
import           MXNet.NN.ModelZoo.RCNN.FasterRCNN
import           MXNet.NN.Utils.Render

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
      (RcnnConfigurationInference{}, _, _) -> mainInfer args


-- data Dbg e a = Dbg (e a)
--
-- instance EvalMetricMethod e => EvalMetricMethod (Dbg e) where
--     data MetricData (Dbg e) a = DbgPriv (MetricData e a)
--     newMetric phase (Dbg conf) = do
--         p <- newMetric phase conf
--         return $ DbgPriv p
--     evalMetric (DbgPriv p) bindings outputs = do
--         liftIO $ do
--             a <- toCPU $ bindings ^?! ix "rpn_cls_targets"
--             a <- toVector =<< prim Ops._norm (#ord := 1 .& #data := a .& Nil)
--             b <- toCPU $ outputs ^?! ix 0
--             b <- toVector =<< prim Ops._norm (#ord := 1 .& #data := b .& Nil)
--             traceShowM (a, b)
--         evalMetric p bindings outputs
--     formatMetric (DbgPriv p) = formatMetric p
--
-- data AccDbg a = AccDbg (Accuracy a)
--
-- instance EvalMetricMethod AccDbg where
--     data MetricData AccDbg a = AccDbgPriv (MetricData Accuracy a)
--     newMetric phase (AccDbg conf) = do
--         priv <- newMetric phase conf
--         return $ AccDbgPriv priv
--     evalMetric (AccDbgPriv accpriv) bindings outputs = do
--         liftIO $ do
--             let AccuracyPriv acc _ _ _ = accpriv
--             lbl <- toCPU $ _mtr_acc_get_gt acc bindings outputs
--             -- x <- toCPU $ outputs ^?! ix 6
--             traceShowM =<< toVector lbl
--             -- traceShowM . VS.take 20 =<< toVector x
--         ret <- evalMetric accpriv bindings outputs
--         return ret
--     formatMetric (AccDbgPriv acc) = formatMetric acc
--
--
-- data Dbg a = Dbg (M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a)
--
-- instance EvalMetricMethod Dbg where
--     data MetricData Dbg a = DbgPriv (Dbg a)
--     newMetric phase a = return (DbgPriv a)
--     evalMetric (DbgPriv (Dbg __get)) bindings outputs = liftIO $ do
--         array <- toCPU $ __get bindings outputs
--         shp <- ndshape array
--         val <- toVector array
--         traceShowM (shp, val)
--         return M.empty
--     formatMetric _ = return "<DBG>"


mainTrain (rcnn_conf@RcnnConfigurationTrain{..}, CommonArgs{..}, TrainArgs{..}) = do
    rand_gen  <- liftIO $ newIORef $ mkStdGen 19
    coco_inst <- Coco.coco ds_base_path "train2017"
    let coco_conf = Coco.CocoConfig coco_inst ds_img_size
                        (toTriple ds_img_pixel_means)
                        (toTriple ds_img_pixel_stds)
        -- There is a serious problem with asyncConduit. It made the training loop running
        -- in different threads, which is very bad because the execution of ExecutorForward
        -- has a thread-local state (saving the temporary workspace for cudnn)
        --
        -- data_iter = asyncConduit (Just batch_size) $
        --
        data_iter = ConduitData (Just batch_size) $
                    Coco.cocoImagesBBoxes rand_gen           .|
                    C.mapM (Coco.augmentWithBBoxes rand_gen) .|
                    C.mapM (withRpnTargets rcnn_conf) .|
                    C.chunksOf batch_size             .|
                    C.mapM concatBatch

    runFeiM'nept "jiasen/faster-rcnn" coco_conf $ do
        (_, sym)     <- runLayerBuilder $ graphT rcnn_conf
        fixed_params <- liftIO $ fixedParams backbone TRAIN sym

        initSession @"faster_rcnn" sym (Config {
            _cfg_data  = M.fromList [("data",     (STensor [batch_size, 3, ds_img_size, ds_img_size]))
                                    ,("im_info",  (STensor [batch_size, 3]))
                                    ,("gt_boxes", (STensor [batch_size, 1, 5]))
                                    ],
            _cfg_label = ["rpn_cls_targets"
                         ,"rpn_box_targets"
                         ,"rpn_box_masks"
                         ],
            _cfg_initializers = M.empty,
            _cfg_default_initializer = default_initializer,
            _cfg_fixed_params = fixed_params,
            _cfg_context = contextGPU0 })

        let lr_sched = lrOfFactor (#base := 0.01 .& #factor := 0.5 .& #step := 5000 .& Nil)
        optm <- makeOptimizer SGD'Mom lr_sched (#momentum := 0.9
                                             .& #wd := 0.0001
                                             .& #rescale_grad := 1 / (fromIntegral batch_size)
                                             .& #clip_gradient := 10
                                             .& Nil)
        -- optm <- makeOptimizer ADAMW lr_sched (#rescale_grad := 1 / (fromIntegral batch_size)
        --                                    .& #eta := 0.001
        --                                    .& #wd := 0.0001
        --                                    .& #clip_gradient := 10 .& Nil)

        checkpoint <- lastSavedState "checkpoints" "faster_rcnn"
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
                                  :* CrossEntropy (Just "RPN-ce") False
                                        (\_ preds -> preds ^?! ix 0)
                                        (\bindings _ -> bindings ^?! ix "rpn_cls_targets")
                                  :* CrossEntropy (Just "RCNN-ce") True
                                        (\_ preds -> preds ^?! ix 3)
                                        (\_ preds -> preds ^?! ix 5)
                                  :* Norm (Just "RPN-L1") 1
                                        (\_ preds -> preds ^?! ix 2)
                                  :* Norm (Just "RCNN-L1") 1
                                        (\_ preds -> preds ^?! ix 4)
                                  :* MNil)

        -- update the internal counting of the iterations
        -- the lr is updated as per to it
        askSession $ do
            untag . mod_statistics . stat_num_upd .= (start_epoch - 1) * pg_train_iter_per_epoch

        forM_ ([start_epoch..pg_train_epochs] :: [Int]) $ \ ei -> do
             logInfo . display $ sformat ("Epoch " % int) ei
             let slice = takeD pg_train_iter_per_epoch data_iter
             void $ forEachD_i slice $ \(i, (fn, [x0, x1, x2, y0, y1, y2])) -> askSession $ do
                 let binding = M.fromList [ ("gt_boxes",        x0)
                                          , ("data",            x1)
                                          , ("im_info",         x2)
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
                 (formatToString ("checkpoints/faster_rcnn_epoch_" % left 3 '0') ei)


mainInfer (rcnn_conf@RcnnConfigurationInference{..}, CommonArgs{..}, NoExtraArgs) = do
    coco_inst@(Coco.Coco _ _ coco_inst_ _) <- Coco.coco ds_base_path "val2017"
    rand_gen <- newIORef $ mkStdGen 24

    let coco_conf = Coco.CocoConfig coco_inst ds_img_size
                        (toTriple ds_img_pixel_means)
                        (toTriple ds_img_pixel_stds)
        data_iter = ConduitData (Just batch_size) (
                        Coco.cocoImagesBBoxes rand_gen .|
                        C.mapM toListNDArray           .|
                        C.chunksOf batch_size          .|
                        C.mapM concatBatch
                    ) & takeD 5

    runFeiM coco_conf $ do
        (_, sym)  <- runLayerBuilder $ graphI rcnn_conf
        fixed_params <- liftIO $ fixedParams backbone INFERENCE sym
        fixed_params <- return $ S.difference fixed_params (S.fromList ["data", "im_info"])

        initSession @"faster_rcnn" sym (Config {
                _cfg_data  = M.fromList [("data",    (STensor [batch_size, 3, ds_img_size, ds_img_size])),
                                         ("im_info", (STensor [batch_size, 3]))],
                _cfg_label = [],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_fixed_params = fixed_params,
                _cfg_context = contextGPU0
            })

        askSession $ do
             loadState checkpoint []
             void $ forEachD_i data_iter $ \(i, (fn, [x0, x1, x2])) -> do
                let bindings = M.fromList [ ("data",            x1)
                                          , ("im_info",         x2)
                                          ]
                [cls_ids, scores, boxes] <- forwardOnly bindings

                -- cls_ids: (B, num_fg_classes * rcnn_nms_topk, 1)
                -- scores : (B, num_fg_classes * rcnn_nms_topk, 1)
                -- boxes  : (B, num_fg_classes * rcnn_nms_topk, 4)

                liftIO $ do
                    fn      <- pure $ VB.fromList fn
                    infos   <- vunstack <$> toRepa @DIM2 x2
                    -- gt_boxes<- vunstack <$> toRepa @DIM3 x0
                    cls_ids <- vunstack <$> toRepa @DIM3 cls_ids
                    scores  <- vunstack <$> toRepa @DIM3 scores
                    boxes   <- vunstack <$> toRepa @DIM3 boxes
                    mean    <- fromVector [3] (VS.fromList ds_img_pixel_means)
                    std     <- fromVector [3] (VS.fromList ds_img_pixel_stds)
                    images  <- transpose x1 [0, 2, 3, 1] >>=
                               mulBroadcast std          >>=
                               addBroadcast mean         >>=
                               mulScalar 255
                    images  <- vunstack <$> toRepa @DIM4 images
                    forM_ (VB.zip6 fn images infos cls_ids scores boxes) renderImageBBoxes


renderImageBBoxes :: (String, Array U DIM3 Float, Array U DIM1 Float, Array U DIM2 Float, Array U DIM2 Float, Array U DIM2 Float) -> IO ()
renderImageBBoxes (filename, image, info, cls_ids, scores, boxes) = do
    let [height, width, scale] = Repa.toUnboxed info
        jp_image = imageFromRepa image
    -- TODO scale the image and bboxes back to orignal size
    width  <- pure $ floor width
    height <- pure $ floor height
    writePng filename $ render width height $ do
        drawImage jp_image
        let all_boxes  = vunstack boxes
            all_scores = VB.convert $ Repa.toUnboxed scores
            all_cls    = VB.convert $ Repa.toUnboxed cls_ids
        forM_ (VB.zip3 all_cls all_scores all_boxes) $ \(cls, score, box) -> do
            let [x, y, x', y'] = Repa.toUnboxed box
            when (score >= 0.5) $ do
                traceShowM (score, box)
                drawBox (PixelRGBA8 255 0 0 255) 1.0 x y x' y' Nothing

