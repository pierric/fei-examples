module Main where

import           Control.Lens                (ix, (^?!))
import           Formatting
import           RIO                         hiding (Const)
import qualified RIO.HashMap                 as M
import qualified RIO.HashSet                 as S
import           RIO.List                    (unzip)
import qualified RIO.NonEmpty                as RNE
import qualified RIO.Text                    as T
import qualified RIO.Vector.Boxed            as V
import qualified RIO.Vector.Storable         as SV

import           MXNet.Base
import           MXNet.Base.Operators.Tensor
import           MXNet.NN
import           MXNet.NN.DataIter.Class
import           MXNet.NN.DataIter.Streaming
import qualified MXNet.NN.Initializer        as I
import           MXNet.NN.Layer

batch_size  = 128

data SoftmaxProp = SoftmaxProp

instance CustomOperationProp SoftmaxProp where
    prop_list_arguments _        = ["data", "label"]
    prop_list_outputs _          = ["output"]
    prop_list_auxiliary_states _ = []
    prop_infer_shape _ [data_shape, _] =
        -- data: [batch_size, N]
        -- label: [batch_size]
        -- output: [batch_size, N]
        -- loss: [batch_size]
        let batch_size : _ = data_shape
            out_shape = batch_size : []
        in ([data_shape, out_shape], [data_shape], [])
    prop_declare_backward_dependency _ grad_out data_in data_out = data_in ++ data_out

    data Operation SoftmaxProp = Softmax
    prop_create_operator _ _ _ = return Softmax

instance CustomOperation (Operation SoftmaxProp) where
    forward _ [ReqWrite] [in_data, label] [out] aux is_train = do
        label <- prim _one_hot (#indices := label .& #depth := 10 .& Nil)
        r <- prim _softmax (#data := in_data .& Nil)
        void $ copy r out

    backward _ [ReqWrite] [_, label] [out] [in_grad, _] _ aux = do
        label <- prim _one_hot (#indices := label .& #depth := 10 .& Nil)
        result <- prim _elemwise_sub (#lhs := out .& #rhs := label .& Nil)
        void $ copy result in_grad


symbol :: Layer (Symbol Float)
symbol = do
    x  <- variable "x"
    y  <- variable "y"

    sequential "custom-op" $ do
        v1 <- convolution (#data := x  .& #kernel := [5,5] .& #num_filter := 20 .& Nil)
        a1 <- activation  (#data := v1 .& #act_type := #tanh .& Nil)
        p1 <- pooling     (#data := a1 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

        v2 <- convolution (#data := p1 .& #kernel := [5,5] .& #num_filter := 50 .& Nil)
        a2 <- activation  (#data := v2 .& #act_type := #tanh .& Nil)
        p2 <- pooling     (#data := a2 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

        fl <- flatten     p2

        v3 <- fullyConnected (#data := fl .& #num_hidden := 500 .& Nil)
        a3 <- activation     (#data := v3 .& #act_type := #tanh .& Nil)

        v4 <- fullyConnected (#data := a3 .& #num_hidden := 10  .& Nil)
        named "softmax" $ prim _Custom (#data := [v4, y] .& #op_type := "softmax_custom" .& Nil)

default_initializer :: Initializer Float
default_initializer name shp
    | T.isSuffixOf "-bias" name = I.zeros name shp
    | otherwise = I.normal 0.1 name shp

main :: IO ()
main = runFeiM . Simple $ do
    liftIO $ registerCustomOperator ("softmax_custom", \_ -> return SoftmaxProp)
    net  <- runLayerBuilder symbol
    initSession @"lenet" net (Config {
        _cfg_data = M.singleton "x" [batch_size, 1,28,28],
        _cfg_label = ["y"],
        _cfg_initializers = M.empty,
        _cfg_default_initializer = default_initializer,
        _cfg_fixed_params = S.fromList [],
        _cfg_context = contextGPU0 })
    optimizer <- makeOptimizer SGD'Mom (Const 0.0002) Nil

    let ce  = CrossEntropy Nothing True
                  (\_ p -> p ^?! ix 0)
                  (\b _ -> b ^?! ix "y")
        acc = Accuracy Nothing PredByArgmax 0
                  (\_ p -> p ^?! ix 0)
                  (\b _ -> b ^?! ix "y")

        trainingData = mnistIter (#image := "data/train-images-idx3-ubyte"
                               .& #label := "data/train-labels-idx1-ubyte"
                               .& #batch_size := batch_size .& Nil)
        testingData  = mnistIter (#image := "data/t10k-images-idx3-ubyte"
                               .& #label := "data/t10k-labels-idx1-ubyte"
                               .& #batch_size := 16  .& Nil)

    total <- sizeD trainingData
    logInfo . display $ sformat "[Train] "
    forM_ (V.enumFromTo 1 10) $ \ind -> do
        logInfo . display $ sformat ("iteration " % int) ind
        metric <- newMetric "train" (ce :* acc :* MNil)
        void $ forEachD_i trainingData $ \(i, (x, y)) -> askSession $ do
            fitAndEval optimizer (M.fromList [("x", x), ("y", y)]) metric
            eval <- metricFormat metric
            when (i `mod` 100 == 1) $
                logInfo . display $ sformat (int % "/" % int % ":" % stext) i total eval

    metric <- newMetric "val" (acc :* MNil)
    forEachD_i testingData $ \(i, (x, y)) -> askSession $ do
        pred <- forwardOnly (M.singleton "x" x)
        void $ metricUpdate metric (M.singleton "y" y) pred
    eval <- metricFormat metric
    logInfo . display $ sformat ("Validation: " % stext) eval
