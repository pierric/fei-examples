module Main where

import RIO hiding (Const)
import RIO.List (unzip)
import qualified RIO.HashMap as M
import qualified RIO.HashSet as S
import qualified RIO.Vector.Boxed as V
import qualified RIO.Vector.Storable as SV
import Formatting (sformat, (%), stext, int)

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN
import MXNet.NN.DataIter.Conduit
import qualified MXNet.NN.Initializer as I
import qualified MXNet.NN.ModelZoo.Lenet as Model

type ArrayF = NDArray Float

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
    net  <- Model.symbol
    sess <- initialize @"lenet" net $ Config {
                _cfg_data = M.singleton "x" (STensor [1,28,28]),
                _cfg_label = ["y"],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_fixed_params = S.fromList [],
                _cfg_context = contextGPU0
            }
    optimizer <- makeOptimizer SGD'Mom (Const 0.0002) Nil

    runSimpleApp $ train sess $ do
        let trainingData = mnistIter (#image := "data/train-images-idx3-ubyte"
                                   .& #label := "data/train-labels-idx1-ubyte"
                                   .& #batch_size := 128 .& Nil)
        let testingData  = mnistIter (#image := "data/t10k-images-idx3-ubyte"
                                   .& #label := "data/t10k-labels-idx1-ubyte"
                                   .& #batch_size := 16  .& Nil)

        total1 <- sizeD trainingData
        total2 <- sizeD testingData

        logInfo . display $ sformat "[Train] "
        forM_ (range 1) $ \ind -> do
            logInfo .display $ sformat ("iteration " % int) ind
            metric <- newMetric "train" (CrossEntropy "y")
            void $ forEachD_i trainingData $ \(i, (x, y)) -> do
                fitAndEval optimizer (M.fromList [("x", x), ("y", y)]) metric
                eval <- formatMetric metric
                logInfo . display $ sformat ("\r\ESC[K" % int % "/" % int % " " % stext) i total1 eval

            metric <- newMetric "val" (Accuracy "y")
            result <- forEachD_i testingData $ \(i, (x, y)) -> do
                pred <- forwardOnly (M.singleton "x" x)
                evalMetric metric (M.singleton "y" y) pred
                eval <- formatMetric metric
                logInfo . display $ sformat ("\r\ESC[K" % int % "/" % int % " " % stext) i total2 eval
                let [y'] = pred
                ind1 <- liftIO $ toVector y
                ind2 <- liftIO $ argmax y' >>= toVector
                return (ind1, ind2)

            let (ls,ps) = unzip result
                ls_unbatched = mconcat ls
                ps_unbatched = mconcat ps
                total_test_items = SV.length ls_unbatched
                correct = SV.length $ SV.filter id $ SV.zipWith (==) ls_unbatched ps_unbatched
            logInfo . display $ sformat ("Accuracy: " % int % "/" % int) correct total_test_items

   where
     argmax (NDArray ys) = NDArray <$> sing A.argmax (#data := ys .& #axis := Just 1 .& Nil)
