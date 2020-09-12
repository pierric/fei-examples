{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE RecordWildCards #-}
module Main where

import           Control.Lens                      (use, (.=))
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
import qualified RIO.Vector.Storable               as VS
import qualified RIO.Vector.Unboxed                as VU
import qualified RIO.Vector.Unboxed.Partial        as VU ((//))
import qualified Text.PrettyPrint.Leijen.Text      as PP

import           MXNet.Base
import qualified MXNet.Base.ParserUtils            as P
import           MXNet.Coco.Types                  (images, img_id)
import           MXNet.NN
import qualified MXNet.NN.DataIter.Anchor          as Anchor
import qualified MXNet.NN.DataIter.Coco            as Coco
import           MXNet.NN.DataIter.Conduit
import           MXNet.NN.ModelZoo.RCNN.FasterRCNN
import           MXNet.NN.ModelZoo.Utils.Box

import           RCNN


main :: IO ()
main = do
    _    <- mxListAllOpNames
    registerCustomOperator ("anchor_generator", Anchor.buildAnchorGenerator)

    (rcnn_conf, pg_conf@ProgConfig{..}) <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "Faster-RCNN")
    if pg_infer then
        -- mainInfer rcnn_conf pg_conf
        error "cannot infer"
    else
        mainTrain rcnn_conf pg_conf
    mxNotifyShutdown

--mainInfer rcnn_conf@RcnnConfiguration{..} ProgConfig{..} = do
--    sym <- runLayerBuilder $ symbolInfer rcnn_conf
--    fixed_params <- fixedParams backbone INFERENCE sym
--    fixed_params <- return $ S.difference fixed_params (S.fromList ["data", "im_info"])
--
--    coco_inst@(Coco.Coco _ _ coco_inst_ _) <- Coco.coco ds_base_path "val2017"
--    sess <- newMVar =<< initialize @"faster_rcnn" sym (Config {
--                _cfg_data  = M.fromList [("data",    (STensor [batch_size, 3, ds_img_size, ds_img_size])),
--                ("im_info", (STensor [batch_size, 3]))],
--                _cfg_label = [],
--                _cfg_initializers = M.empty,
--                _cfg_default_initializer = default_initializer,
--                _cfg_fixed_params = fixed_params,
--                _cfg_context = contextGPU0
--            })
--    let vimg = V.filter (\img -> img ^. img_id == pg_infer_image_id) $ coco_inst_ ^. images
--        img = V.head vimg
--
--    when (V.null vimg) (throwString $ "image_id " ++ show pg_infer_image_id ++ " not found in val2017")
--
--    let coco_conf = Coco.CocoConfig coco_inst ds_img_size (toTriple ds_img_pixel_means) (toTriple ds_img_pixel_stds)
--    runApp coco_conf $ withSession sess $ do
--        (img_tensor_r, img_info_r) <- Coco.loadImage img
--        img_tensor <- liftIO $ repaToNDArray img_tensor_r
--        img_info   <- liftIO $ repaToNDArray img_info_r
--        img_tensor <- liftIO $ expandDims 0 img_tensor
--        img_info   <- liftIO $ expandDims 0 img_info
--
--        checkpoint <- lastSavedState "checkpoints" "faster_rcnn"
--        case checkpoint of
--            Nothing -> do
--                throwString $ "Checkpoint not found."
--            Just filename -> do
--                loadState filename []
--                let binding = M.fromList [ ("data",    img_tensor)
--                                         , ("im_info", img_info)]
--                [rois, cls_prob, bbox_pred] <- forwardOnly binding
--                liftIO $ do
--                    rois     <- toRepa @DIM2 rois       -- [NUM_ROIS, 5]
--                    cls_prob <- toRepa @DIM2 cls_prob   -- [NUM_ROIS, NUM_CLASSES]
--                    deltas   <- toRepa @DIM2 bbox_pred  -- [NUM_ROIS, NUM_CLASSES*4]
--
--                    let box_stds = Repa.fromListUnboxed (Z :. 4 :: DIM1) [0.1, 0.1, 0.2, 0.2]
--                        rois' = Repa.computeS $ Repa.traverse rois
--                                    (\(Z :. n :. 5) -> Z :. n :. 4)
--                                    (\lk (Z:.i:.j) -> lk (Z:.i:.(j+1)))
--                        deltas' = reshapeEx (Z :. 0 :. (-1) :. 4 :: DIM3) deltas
--                        pred_boxes = decodeBoxes rois' deltas' box_stds img_info_r
--                        (cls_ids, cls_scores) = decodeScores cls_prob 1e-3
--                        res = cls_ids Repa.++ cls_scores Repa.++ pred_boxes
--                        -- exlcude background class 0
--                        -- and transpose from [NUM_ROIS, NUM_CLASSES_FG, 6] to [NUM_CLASSES_FG, NUM_ROIS, 6]
--                        res_no_bg = Repa.traverse res
--                                        (\(Z:.i:.j:.k) -> Z:.(j-1):.i:.k)
--                                        (\lk (Z:.j:.i:.k) -> lk (Z:.i:.(j+1):.k))
--
--                        -- nms the boxes
--                        res_out = V.concatMap (nmsBoxes 0.3) $ vunstack $
--                                        Repa.computeS res_no_bg
--
--                        -- keep only those with confidence >= 0.7
--                        res_good = V.filter ((>= 0.7) . (^#! 1)) $ res_out
--                    PP.putDoc . PP.pretty $ V.toList $ V.map PrettyArray res_good
--                -- logInfo . display $ length res_good
--                -- putStrLn $ prettyShow $ V.toList $ V.concatMap vunstack a
--                -- putDoc . pretty $ V.toList $ vunstack cls_prob
--                logInfo . display $ sformat ("Done: " % int) (img ^. img_id)
--
--  where
--    decodeBoxes rois deltas box_stds im_info =
--        -- rois: [N, 4]
--        -- deltas: [N, NUM_CLASSES, 4]
--        -- box_stds: [4]
--        -- return: [N, NUM_CLASSES, 4]
--        let [height, width, scale] = Repa.toList im_info :: [Float]
--            shape = Repa.extent deltas
--            eachClass roi = (Repa.computeS . Repa.map (/ scale)) . bboxClip height width . bboxTransInv box_stds roi
--            eachROI roi = V.map (eachClass roi) . vunstack
--            pred_boxes = V.zipWith eachROI (vunstack rois) (vunstack deltas)
--        in Repa.computeUnboxedS $ Repa.reshape shape $ vstack $ V.concat $ V.toList pred_boxes
--
--    decodeScores :: Array U DIM2 Float -> Float -> (Array U DIM3 Float, Array U DIM3 Float)
--    decodeScores cls_prob thr =
--        let Z:.num_rois:.num_classes = Repa.extent cls_prob
--            cls_id = Repa.fromUnboxed (Z:.1:.num_classes) $ VU.enumFromN 0 num_classes
--            -- cls_ids :: [NUM_ROIS, NUM_CLASSES]
--            cls_ids = vstack $ V.replicate num_rois cls_id
--            -- cls_scores :: [NUM_ROIS, NUM_CLASSES]
--            cls_ids_masked = Repa.computeS $ Repa.zipWith (\v1 v2 -> if v2 >= thr then v1 else (-1)) cls_ids cls_prob
--            cls_scs_masked = Repa.computeS $ Repa.map (\v -> if v >= thr then v else 0) cls_prob
--        in (reshapeEx (Z:.0:.0:.1) cls_ids_masked, reshapeEx (Z:.0:.0:.1) cls_scs_masked)
--
--    nmsBoxes :: Float -> Array U DIM2 Float -> V.Vector (Array U DIM1 Float)
--    nmsBoxes threshold boxes = runST $ do
--        items <- V.thaw (vunstack boxes)
--        go items
--        V.filter ((/= -1) . (^#! 0)) <$> V.freeze items
--      where
--        cmp a b | a ^#! 0 == -1 = GT
--                | b ^#! 0 == -1 = LT
--                | otherwise = compare (b ^#! 1) (a ^#! 1)
--        go :: VM.MVector s (Array U DIM1 Float) -> ST s ()
--        go items =
--            when (VM.length items > 0) $ do
--                VA.sortBy cmp items
--                box0 <- VM.read items 0
--                when (box0 ^#! 0 == -1) $ do
--                    items <- return $ VM.tail items
--                    V.forM_ (V.enumFromN 0 (VM.length items)) $ \k -> do
--                        boxK <- VM.read items k
--                        let b1 = Repa.computeS $ Repa.extract (Z:.2) (Z:.4) box0
--                            b2 = Repa.computeS $ Repa.extract (Z:.2) (Z:.4) boxK
--                        when (bboxIOU b1 b2 >= threshold) $ do
--                            let boxK' = Repa.fromUnboxed (Z:.6 :: DIM1) $ Repa.toUnboxed boxK VU.// [(0, -1)]
--                            VM.write items k boxK'
--                    go items

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
                    Coco.cocoImagesBBoxes True        .|
                    C.mapM (withRpnTargets rcnn_conf) .|
                    C.chunksOf batch_size             .|
                    C.mapM concatBatch

    sess <- newMVar =<< initialize @"faster_rcnn" sym (Config {
                _cfg_data  = M.fromList [("data",     (STensor [batch_size, 3, ds_img_size, ds_img_size]))
                                        ,("im_info",  (STensor [batch_size, 3]))
                                        ,("gt_boxes", (STensor [batch_size, 1, 5]))
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
            void $ forEachD_i slice $ \(i, (fn, [x0, x1, x2, y0, y1, y2])) -> withSession sess $ do
                let binding = M.fromList [ ("gt_boxes",        x0)
                                         , ("data",            x1)
                                         , ("im_info",         x2)
                                         , ("rpn_cls_targets", y0)
                                         , ("rpn_box_targets", y1)
                                         , ("rpn_box_masks",   y2)
                                         , ("box_reg_mean",    mean)
                                         , ("box_reg_std",     std)
                                         ]
                fitAndEval optimizer binding metric
                eval <- formatMetric metric
                lr <- use (untag . mod_statistics . stat_last_lr)

                ------------------------------------------------------
                -- code for debugging
                ------------------------------------------------------
                -- logInfo $ display $ tshow fn
                -- exec <- use (untag . mod_executor)
                -- liftIO $ do
                --     let dir = "dumps"
                --         file = sformat ("dumps/" % int % ".params") i
                --     ex <- doesPathExist dir
                --     when (not ex) (createDirectory dir)
                --     outs  <- execGetOutputs exec
                --     let names = ["data", "im_info", "gt_boxes",
                --                  "rpn_cls_prob", "rpn_cls_loss",
                --                  "rpn_bbox_loss", "cls_prob",
                --                  "bbox_loss", "cls_targets",
                --                  "aa", "pp", "rs", "rr"]
                --     mxNDArraySave file $ zip names $ map unNDArray $ [x0, x1, x2] ++ outs

                -- params <- use (untag . mod_params)
                -- let calcSize (ParameterV a) = ndsize a
                --     calcSize (ParameterF a) = ndsize a
                --     calcSize (ParameterA a) = ndsize a
                --     calcSize (ParameterG a b) = liftM2 (+) (ndsize a) (ndsize b)
                --     arrays (ParameterV a)   = ([a] :: [NDArray Float])
                --     arrays (ParameterF a)   = [a]
                --     arrays (ParameterA a)   = [a]
                --     arrays (ParameterG a b) = [a, b]
                -- let hasnan (ParameterV a)   = hasnan_ a
                --     hasnan (ParameterF a)   = hasnan_ a
                --     hasnan (ParameterA a)   = hasnan_ a
                --     hasnan (ParameterG a b) = hasnan_ a
                --     hasnan_ a = do
                --         v <- toVector a
                --         return $ isJust $ VS.findIndex isNaN v
                --     gradn :: Parameter Float -> IO (Maybe Float)
                --     gradn (ParameterG _ b) = do
                --         v <- prim _norm (#data := b .& Nil)
                --         v <- toVector v
                --         return $ v VS.!? 0
                --     gradn _                = return $ Nothing
                -- pn <- liftIO $ mapM hasnan params
                -- traceShowM $ M.keys $ M.filter id pn
                -- gn <- liftIO $ mapM gradn params
                -- let gn' = M.toList $ M.filter isJust gn
                --     gn'' = sortBy (compare `on` snd) gn'
                -- traceShowM gn''
                -- arrs <- liftIO $ mapM calcSize params
                -- size <- return $ sum arrs
                -- traceShowM ("total params (#float)", size)

                logInfo . display $ sformat (int % " " % stext % " LR: " % fixed 5) i eval lr

            withSession sess $ saveState (ei == 1)
                (formatToString ("checkpoints/faster_rcnn_epoch_" % left 3 '0') ei)

