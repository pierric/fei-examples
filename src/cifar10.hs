{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import qualified Data.HashMap.Strict as M
import Control.Monad (forM_, void)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import Control.Lens ((%~))
import System.IO (hFlush, stdout)
import Options.Applicative (Parser, execParser, header, info, fullDesc, helper, value, option, auto, metavar, short, showDefault, (<**>))
import Data.Semigroup ((<>))

import MXNet.Base (NDArray(..), contextCPU, contextGPU0, mxListAllOpNames, toVector, (.&), HMap(..), ArgOf(..))
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN
import MXNet.NN.Utils
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Streaming
import qualified Model.Resnet as Resnet
import qualified Model.Resnext as Resnext

type ArrayF = NDArray Float
type DS = StreamData (TrainM Float IO) (ArrayF, ArrayF)

data Model   = Resnet | Resnext deriving (Show, Read)
data ProgArg = ProgArg Model
cmdArgParser :: Parser ProgArg
cmdArgParser = ProgArg <$> (option auto $ short 'm' <> metavar "MODEL" <> showDefault <> value Resnet)

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: Initializer Float
default_initializer name shp
    | endsWith "-bias"  name = zeros name shp
    | endsWith "-beta"  name = zeros name shp
    | endsWith "-gamma" name = ones  name shp
    | endsWith "-moving-mean" name = zeros name shp
    | endsWith "-moving-var"  name = ones  name shp
    | otherwise = case shp of 
                    [_,_] -> xavier 2.0 XavierGaussian XavierIn name shp
                    _ -> normal 0.1 name shp

main :: IO ()
main = do
    ProgArg model <- execParser $ info (cmdArgParser <**> helper) (fullDesc <> header "CIFAR-10 solver")
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    net  <- case model of 
              Resnet  -> Resnet.symbol 10 34 [3,32,32]
              Resnext -> Resnext.symbol
    sess <- initialize net $ Config { 
                _cfg_data = ("x", [3,32,32]),
                _cfg_label = ("y", [1]),
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_context = contextGPU0
            } 

    cbTP <- dumpThroughputEpoch
    sess <- return $ (sess_callbacks %~ ([Callback DumpLearningRate, cbTP, Callback (Checkpoint "tmp")] ++)) sess
    
    optimizer <- makeOptimizer ADAM (lrOfPoly $ #maxnup := 10000 .& #base := 0.05 .& #power := 1 .& Nil) Nil

    train sess $ do 

        let trainingData = imageRecordIter (#path_imgrec := "dataiter/test/data/cifar10_train.rec" .&
                                            #data_shape  := [3,32,32] .&
                                            #batch_size  := 128 .& Nil)
        let testingData  = imageRecordIter (#path_imgrec := "dataiter/test/data/cifar10_val.rec" .&
                                            #data_shape  := [3,32,32] .&
                                            #batch_size  := 32 .& Nil)
        fitDataset optimizer trainingData testingData (CrossEntropy :+ Accuracy :+ MNil) 18
