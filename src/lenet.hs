module Main where

import           Control.Lens              (ix, use, (^?!))
import           Formatting                (int, sformat, stext, (%))
import           RIO                       hiding (Const)
import qualified RIO.HashMap               as M
import qualified RIO.HashSet               as S
import           RIO.List                  (unzip)
import qualified RIO.Vector.Boxed          as V
import qualified RIO.Vector.Storable       as SV

import           MXNet.Base
import           MXNet.NN
import           MXNet.NN.DataIter.Conduit
import qualified MXNet.NN.Initializer      as I
import qualified MXNet.NN.ModelZoo.Lenet   as Model

batch_size = 128

range :: Int -> Vector Int
range = V.enumFromTo 1

default_initializer :: Initializer Float
default_initializer name shp =
    case length shp of
        1 -> I.zeros name shp
        2 -> I.xavier 2.0 I.XavierGaussian I.XavierIn name shp
        _ -> I.normal 0.1 name shp

main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    net  <- runLayerBuilder Model.symbol
    sess <- newMVar =<< initialize @"lenet" net (Config {
            _cfg_data = M.singleton "x" (STensor [batch_size , 1,28,28]),
            _cfg_label = ["y"],
            _cfg_initializers = M.empty,
            _cfg_default_initializer = default_initializer,
            _cfg_fixed_params = S.fromList [],
            _cfg_context = contextGPU0
            })
    optimizer <- makeOptimizer SGD'Mom (Const 0.0002) Nil

    runSimpleApp $ do
        let trainingData = mnistIter (#image := "data/train-images-idx3-ubyte"
                                   .& #label := "data/train-labels-idx1-ubyte"
                                   .& #batch_size := batch_size  .& Nil)
        let testingData  = mnistIter (#image := "data/t10k-images-idx3-ubyte"
                                   .& #label := "data/t10k-labels-idx1-ubyte"
                                   .& #batch_size := 16  .& Nil)

        total1 <- sizeD trainingData
        total2 <- sizeD testingData

        logInfo . display $ sformat "[Train] "

        let acc_metric = Accuracy "ACC" PredByArgmax
                            (\_ p -> p ^?! ix 0)
                            (\b _ -> b ^?! ix "y")
            ce_metric  = CrossEntropy "CE" True
                            (\_ p -> p ^?! ix 0)
                            (\b _ -> b ^?! ix "y")

        forM_ (range 5) $ \ind -> do
            logInfo .display $ sformat ("iteration " % int) ind
            metric <- newMetric "train" (ce_metric :* acc_metric :* MNil)
            void $ forEachD_i trainingData $ \(i, (x, y)) -> withSession sess $ do
                fitAndEval optimizer (M.fromList [("x", x), ("y", y)]) metric
                eval <- formatMetric metric
                logInfo . display $ sformat (int % "/" % int % " " % stext) i total1 eval

            metric <- newMetric "val" acc_metric
            forEachD_i testingData $ \(i, (x, y)) -> withSession sess $ do
                pred <- forwardOnly (M.singleton "x" x)
                void $ evalMetric metric (M.singleton "y" y) pred
                eval <- formatMetric metric
                logInfo . display $ sformat (int % "/" % int % " " % stext) i total2 eval
