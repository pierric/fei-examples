import           Data.Conduit                      (runConduit, (.|))
import qualified Data.Conduit.Combinators          as CM
import qualified Data.Conduit.List                 as C
import           Data.Random.Source.StdGen         (mkStdGen)
import           MXNet.Base
import           MXNet.NN
import qualified MXNet.NN.DataIter.Anchor          as Anchor
import           MXNet.NN.DataIter.Class
import qualified MXNet.NN.DataIter.Coco            as Coco
import           MXNet.NN.DataIter.Conduit
import           MXNet.NN.ModelZoo.RCNN.FasterRCNN as FasterRCNN
import           MXNet.NN.ModelZoo.RCNN.MaskRCNN   as MaskRCNN
import           Prelude                           (print, putStrLn)
import           RCNN
import           RIO
import qualified RIO.HashMap                       as M
import qualified RIO.HashSet                       as S
import qualified RIO.Text                          as T
import           System.ProgressBar

c_ds_img_size = 256
c_ds_img_pixel_means = (0.5,0.5,0.5)
c_ds_img_pixel_stds = (0,0,0)
c_rpn_anchor_base_size = 16
c_rpn_anchor_scales = [8, 16, 32]
c_rpn_anchor_ratios = [0.5, 1, 2]
c_feature_strides = [4, 8, 16, 32]
c_batch_size = 2

rcnn_conf = RcnnConfigurationTrain {
    batch_size = c_batch_size,
    backbone = RESNET50FPN,
    feature_strides = c_feature_strides,
    rpn_min_size = 16,
    rpn_nms_thresh = 0.7,
    rpn_batch_rois = 256,
    rpn_fg_fraction = 0.5,
    rpn_pre_topk = 12000,
    rpn_post_topk = 2000,
    rpn_anchor_base_size = c_rpn_anchor_base_size,
    rpn_anchor_ratios = c_rpn_anchor_ratios,
    rpn_anchor_scales = c_rpn_anchor_scales,
    rpn_allowd_border = 0,
    rpn_bg_overlap = 0.3,
    rpn_fg_overlap = 0.7,
    rcnn_fg_fraction = 0.25,
    rcnn_fg_overlap = 0.5,
    rcnn_max_num_gt = 100,
    rcnn_pooled_size = 14,
    rcnn_batch_rois = 128,
    rcnn_num_classes = 81,
    bbox_reg_std = (0.1, 0.1, 0.2, 0.2)
    }

dataloading = do
    rand_gen   <- newIORef $ mkStdGen 42
    train_inst <- Coco.coco "/home/jiasen/playground/datasets/coco/" "train2017"

    let train_coco_conf = Coco.CocoConfig train_inst c_ds_img_size c_ds_img_pixel_means c_ds_img_pixel_stds

    cached_anchors <- forM c_feature_strides $ \stride -> do
                        let (h, w) = (128, 128)
                        anchs <- Anchor.anchors (h, w) stride c_rpn_anchor_base_size c_rpn_anchor_scales c_rpn_anchor_ratios
                        anchs <- reshape [h, w, -1] anchs
                        return (stride, anchs)
    cached_anchors <- pure $ M.fromList cached_anchors

    let di = ConduitData (Just 2) $
              Coco.cocoImagesBBoxesMasks rand_gen
           .| C.mapM (withRpnTargets'Mask rcnn_conf cached_anchors)
           .| C.chunksOf 2
           .| C.mapM concatBatch'Mask
           .| C.mapM (\r -> liftIO waitAll >> return r)

    return $ (rand_gen, train_coco_conf, takeD 500 di)


tick pb = C.mapM (\r -> liftIO (incProgress pb 1) >> return r)

check1 = do
    mxRandomSeed 8
    (rand_gen_ref, conf, di) <- dataloading
    flip runReaderT conf $ do
        forM_ ([0..10]::[Int]) $ \i -> do
            pb <- liftIO $ do
                    rg <- readIORef rand_gen_ref
                    print rg
                    newProgressBar (defStyle{stylePostfix = exact <> percentage})
                            10 (Progress 0 500 ())
            _ <- forEachD_pi 32 di $ \_ -> liftIO (incProgress pb 1)
            return ()
    putStrLn ""

check2 = do
    mxRandomSeed 8
    registerCustomOperator ("anchor_generator", Anchor.buildAnchorGenerator)

    runFeiM . Simple $ do
        (_, sym) <- runLayerBuilder $ MaskRCNN.graphT rcnn_conf

        -- attrs <- listAttrs (sym :: Symbol Float)
        -- let parseShape :: Text -> Maybe [Int]
        --     parseShape = readMaybe . T.unpack
        --     inp = M.fromList [ ("data",     [c_batch_size, 3, c_ds_img_size, c_ds_img_size])
        --                      , ("im_info",  [c_batch_size, 3])
        --                      , ("gt_boxes", [c_batch_size, 1, 5])
        --                      -- , ("rpn_cls_targets", [c_batch_size, -1, 1])
        --                      -- , ("rpn_box_targets", [c_batch_size, -1, 4])
        --                      -- , ("rpn_box_masks",   [c_batch_size, -1, 4])
        --                      ]
        --     shapes = M.mapMaybe (M.lookup "__shape__" >=> parseShape) attrs `M.union` inp
        -- (a,b,c,d) <- inferShape sym  (M.toList shapes)
        -- traceShowM a
        -- traceShowM b
        -- traceShowM c
        -- traceShowM d
        --
        -- (args, outs, auxs, cc) <- inferShape sym
        --                             [ ("data",     [c_batch_size, 3, c_ds_img_size, c_ds_img_size])
        --                             , ("im_info",  [c_batch_size, 3])
        --                             , ("gt_boxes", [c_batch_size, 1, 5])
        --                             , ("rpn_cls_targets", [c_batch_size, -1, 1])
        --                             , ("rpn_box_targets", [c_batch_size, -1, 4])
        --                             , ("rpn_box_masks",   [c_batch_size, -1, 4])
        --                             ]
        -- forM outs $ \(n, s) -> traceShowM (n, s)

        -- Just symX <- getInternalByName sym "features.rcnn.D_output"
        -- (args, outs, auxs, _) <- inferShape symX
        --                          [ ("data",     [c_batch_size, 3, c_ds_img_size, c_ds_img_size])
        --                          -- , ("im_info",  [c_batch_size, 3])
        --                          -- , ("gt_boxes", [c_batch_size, 1, 5])
        --                          ]
        -- traceShowM outs

        fixed_params <- liftIO $ fixedParams (backbone rcnn_conf) TRAIN sym
        initSession @"check" sym (Config {
            _cfg_data  = M.fromList [ ("data",     [c_batch_size, 3, c_ds_img_size, c_ds_img_size])
                                    , ("im_info",  [c_batch_size, 3])
                                    , ("gt_boxes", [c_batch_size, 1, 5])
                                    , ("gt_masks", [c_batch_size, 1, c_ds_img_size, c_ds_img_size])
                                    -- , ("rpn_cls_targets", [c_batch_size, -1, 1])
                                    -- , ("rpn_box_targets", [c_batch_size, -1, 4])
                                    -- , ("rpn_box_masks",   [c_batch_size, -1, 4])
                                    -- , ("gt_masks", [c_batch_size, 1, c_ds_img_size, c_ds_img_size])
                                    ],
            _cfg_label = [],
            _cfg_initializers = M.empty,
            _cfg_default_initializer = default_initializer,
            _cfg_fixed_params = fixed_params,
            _cfg_context = contextGPU0 })

check3 = runFeiM . Simple $ do
    sym <- runLayerBuilder graph
    initSession @"check" sym (Config {
        _cfg_data  = M.fromList [("x", [2, 3])
                                -- , ("y", [2, 3])
                                ],
        _cfg_label = [],
        _cfg_initializers = M.empty,
        _cfg_default_initializer = default_initializer,
        _cfg_fixed_params = S.empty,
        _cfg_context = contextGPU0 })
    where
        graph :: Layer (Symbol Float)
        graph = do
            x <- variable "x"
            y <- variable "y"
            sequential "S" $ do
                z <- geqScalar 0 (y :: Symbol Float)
                z <- castToFloat z
                addNoBroadcast x z

main = check2
