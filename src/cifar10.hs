module Main where

import RIO
import RIO.List.Partial (last)
import qualified RIO.HashMap as M
import qualified RIO.HashSet as S
import qualified RIO.Text as T
import Control.Lens (use)
import Options.Applicative
import Formatting (sformat, formatToString, int, stext, left, float, (%))

import MXNet.Base (
    NDArray(..),
    contextCPU, contextGPU0,
    mxListAllOpNames,
    FShape(..),
    (.&), HMap(..), ArgOf(..),
    listArguments)
import MXNet.NN
import MXNet.NN.DataIter.Streaming
import qualified MXNet.NN.Initializer as I
import qualified MXNet.NN.ModelZoo.Resnet as Resnet
import qualified MXNet.NN.ModelZoo.Resnext as Resnext

type ArrayF = NDArray Float

data Model   = Resnet | Resnext deriving (Show, Read)
data ProgArg = ProgArg Model (Maybe String)
cmdArgParser :: Parser ProgArg
cmdArgParser = ProgArg
                <$> (option auto  $ short 'm' <> metavar "MODEL" <> showDefault <> value Resnet)
                <*> (option maybe $ short 'p' <> metavar "PRETRAINED" <> showDefault <> value Nothing)
  where
    maybe = maybeReader (Just . Just)

default_initializer :: Initializer Float
default_initializer name shp
    | T.isSuffixOf ".bias"  name = I.zeros name shp
    | T.isSuffixOf ".beta"  name = I.zeros name shp
    | T.isSuffixOf ".gamma" name = I.ones  name shp
    | T.isSuffixOf ".running_mean" name = I.zeros name shp
    | T.isSuffixOf ".running_var"  name = I.ones  name shp
    | otherwise = case shp of
                    [_,_] -> I.xavier 2.0 I.XavierGaussian I.XavierIn name shp
                    _ -> I.normal 0.1 name shp

main :: IO ()
main = do
    ProgArg model pretrained <- execParser $ info
        (cmdArgParser <**> helper) (fullDesc <> header "CIFAR-10 solver")

    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    net  <- case model of
              Resnet  -> Resnet.symbol 10 50 32
              Resnext -> Resnext.symbol

    fixed <- case pretrained of
        Nothing -> return S.empty
        Just _  -> fixedParams net model

    sess <- initialize @"cifar10" net $ Config {
                _cfg_data = M.singleton "x" (STensor [3,32,32]),
                _cfg_label = ["y"],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_fixed_params = fixed,
                _cfg_context = contextGPU0
            }

    -- cbTP <- dumpThroughputEpoch
    -- sess <- return $ (sess_callbacks %~ ([cbTP, Callback (Checkpoint "tmp")] ++)) sess

    let lr_scheduler = lrOfMultifactor $ #steps := [100, 200, 300]
                                      .& #base := 0.0001
                                      .& #factor:= 0.75 .& Nil
    optimizer <- makeOptimizer SGD'Mom lr_scheduler Nil
    metric <- newMetric "train" (CrossEntropy "y" :* Accuracy "y" :* MNil)

    runSimpleApp $ train sess $ do

        let trainingData = imageRecordIter (#path_imgrec := "data/cifar10_train.rec"
                                         .& #data_shape  := [3,32,32]
                                         .& #batch_size  := 128 .& Nil)

        case pretrained of
            Just path -> loadState path ["output.weight", "output.bias"]
            Nothing -> return ()

        forM_ ([1..100] :: [Int]) $ \ ei -> do
            logInfo . display $ sformat ("Epoch " % int) ei
            void $ forEachD_i trainingData $ \(i, (x, y)) -> do
                let binding = M.fromList [("x", x), ("y", y)]
                fitAndEval optimizer binding metric
                eval <- formatMetric metric
                lr <- use (untag . mod_statistics . stat_last_lr)
                when (i `mod` 20 == 0) $ do
                    logInfo . display $ sformat (int % " " % stext % " LR: " % float) i eval lr

            saveState (ei == 1)
                (formatToString ("checkpoints/cifar10_resnet50_epoch_" % left 3 '0') ei)

fixedParams symbol _ = do
    argnames <- listArguments symbol
    return $ S.fromList [n | n <- argnames
                        -- fix conv_0, stage_1_*, *_gamma, *_beta
                        , layer n `elemL` ["1", "5"] || name n `elemL` ["gamma", "beta"]]

  where
    layer param = case T.split (=='.') param of
                    "features":n:_ -> n
                    _ -> "<na>"
    name param = last $ T.split (=='.') param
    elemL :: Eq a => a -> [a] -> Bool
    elemL = elem
