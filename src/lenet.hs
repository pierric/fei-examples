{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import qualified Data.HashMap.Strict as M
import Control.Monad (forM_, void)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import System.IO (hFlush, stdout)

import MXNet.Base (NDArray(..), contextCPU, contextGPU0, mxListAllOpNames, toVector, (.&), HMap(..), ArgOf(..), waitAll)
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Conduit
import qualified Model.Lenet as Model

type ArrayF = NDArray Float
type DS = ConduitData (Module "lenet" Float) (ArrayF, ArrayF)

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: Initializer Float
default_initializer name shp@[_]   = zeros name shp
default_initializer name shp@[_,_] = xavier 2.0 XavierGaussian XavierIn name shp
default_initializer name shp = normal 0.1 name shp

main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    net  <- Model.symbol
    sess <- initialize net $ Config {
                _cfg_data = M.singleton "x" [1,28,28],
                _cfg_label = ["y"],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_context = contextGPU0
            }
    optimizer <- makeOptimizer SGD'Mom (Const 0.0002) Nil

    train sess $ do

        let trainingData = mnistIter (#image := "data/train-images-idx3-ubyte"
                                   .& #label := "data/train-labels-idx1-ubyte"
                                   .& #batch_size := 128 .& Nil) :: DS
        let testingData  = mnistIter (#image := "data/t10k-images-idx3-ubyte"
                                   .& #label := "data/t10k-labels-idx1-ubyte"
                                   .& #batch_size := 16  .& Nil) ::DS

        total1 <- sizeD trainingData
        total2 <- sizeD testingData

        liftIO $ putStrLn $ "[Train] "
        forM_ (range 1) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            -- metric <- newMetric "train" (CrossEntropy "y")
            metric <- newMetric "train" MNil
            void $ forEachD_i trainingData $ \(i, (x, y)) -> do
                -- liftIO $ putStrLn "A"
                fitAndEval optimizer (M.fromList [("x", x), ("y", y)]) metric
                -- liftIO $ putStrLn "B"
                eval <- format metric
                liftIO $ do
                    putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total1 ++ " " ++ eval
                    hFlush stdout
                    -- putStrLn "C"
                    waitAll
            liftIO $ putStrLn "D"

            metric <- newMetric "val" (Accuracy "y")
            result <- forEachD_i testingData $ \(i, (x, y)) -> do
                pred <- forwardOnly (M.singleton "x" x)
                evaluate metric (M.singleton "y" y) pred
                eval <- format metric
                liftIO $ do
                    putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total2 ++ " " ++ eval
                    hFlush stdout
            liftIO $ putStrLn ""

            -- let (ls,ps) = unzip result
            --     ls_unbatched = mconcat ls
            --     ps_unbatched = mconcat ps
            --     total_test_items = SV.length ls_unbatched
            --     correct = SV.length $ SV.filter id $ SV.zipWith (==) ls_unbatched ps_unbatched
            -- liftIO $ putStrLn $ "Accuracy: " ++ show correct ++ "/" ++ show total_test_items

--   where
--     argmax :: ArrayF -> IO ArrayF
--     argmax (NDArray ys) = NDArray . head <$> A.argmax (#data := ys .& #axis := Just 1 .& Nil)
