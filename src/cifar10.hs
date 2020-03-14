{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where

import qualified Data.HashMap.Strict as M
import qualified Data.HashSet as S
import Control.Monad (forM_, void, when)
import Control.Monad.IO.Class
import Control.Lens ((%~))
import System.IO (hFlush, stdout)
import Options.Applicative
import Data.Semigroup ((<>))
import Control.Lens ((.=), (^.), use)
import Text.Printf
import qualified Data.Text as T

import MXNet.Base (
    NDArray(..),
    contextCPU, contextGPU0,
    mxListAllOpNames,
    (.&), HMap(..), ArgOf(..),
    listArguments)
import MXNet.NN
import MXNet.NN.Utils
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Streaming
import qualified Model.Resnet as Resnet
import qualified Model.Resnext as Resnext

type ArrayF = NDArray Float
type DS = StreamData (Module "cifar10" Float IO) (ArrayF, ArrayF)

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
    | endsWith ".bias"  name = zeros name shp
    | endsWith ".beta"  name = zeros name shp
    | endsWith ".gamma" name = ones  name shp
    | endsWith ".running_mean" name = zeros name shp
    | endsWith ".running_var"  name = ones  name shp
    | otherwise = case shp of
                    [_,_] -> xavier 2.0 XavierGaussian XavierIn name shp
                    _ -> normal 0.1 name shp

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
                _cfg_data = M.singleton "x" [3,32,32],
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

    train sess $ do

        let trainingData = imageRecordIter (#path_imgrec := "data/cifar10_train.rec"
                                         .& #data_shape  := [3,32,32]
                                         .& #batch_size  := 128 .& Nil) :: DS

        case pretrained of
            Just path -> loadState path ["output.weight", "output.bias"]
            Nothing -> return ()

        forM_ [1..100::Int] $ \ ei -> do
            liftIO $ putStrLn $ "Epoch " ++ show ei
            void $ forEachD_i trainingData $ \(i, (x, y)) -> do
                let binding = M.fromList [("x", x), ("y", y)]
                fitAndEval optimizer binding metric
                eval <- format metric
                lr <- use (untag . mod_statistics . stat_last_lr)
                liftIO $ do
                    when (i `mod` 20 == 0) $ do
                        putStrLn $ show i ++ " " ++ eval ++ " LR: " ++ show lr
                    hFlush stdout

            saveState (ei == 1) (printf "checkpoints/cifar10_resnet50_epoch_%03d" ei)
            liftIO $ putStrLn ""

fixedParams symbol _ = do
    argnames <- listArguments symbol
    return $ S.fromList [n | n <- argnames
                        -- fix conv_0, stage_1_*, *_gamma, *_beta
                        , layer n `elem` ["1", "5"] || name n `elem` ["gamma", "beta"]]

  where
    layer param = case T.splitOn "." $ T.pack param of
                    "features":n:_ -> n
                    _ -> "<na>"
    name  param = last $ T.splitOn "." $ T.pack param
