module Main where

import RIO hiding (Const)
import RIO.List (unzip)
import qualified RIO.NonEmpty as RNE
import qualified RIO.Text as T
import qualified RIO.HashSet as S
import qualified RIO.HashMap as M
import qualified RIO.Vector.Boxed as V
import qualified RIO.Vector.Storable as SV
import Formatting

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.Base.Operators.Symbol (_Custom)
import MXNet.NN
import qualified MXNet.NN.Initializer as I
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Streaming

type ArrayF = NDArray Float

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
        let STensor (batch_size :| _) = data_shape
            out_shape = STensor (batch_size :| [])
        in ([data_shape, out_shape], [data_shape], [])
    prop_declare_backward_dependency _ grad_out data_in data_out = data_in ++ data_out

    data Operation SoftmaxProp = Softmax
    prop_create_operator _ _ _ = return Softmax

instance CustomOperation (Operation SoftmaxProp) where
    forward _ [ReqWrite] [in_data, label] [out] aux is_train = do
        label <- sing A.one_hot (#indices := label .& #depth := 10 .& Nil)
        r <- sing A.softmax (#data := in_data .& Nil)
        A._copyto_upd [out] (#data := r .& Nil)

    backward _ [ReqWrite] [_, label] [out] [in_grad, _] _ aux = do
        label <- sing A.one_hot (#indices := label .& #depth := 10 .& Nil)
        result <- sing A.elemwise_sub (#lhs := out .& #rhs := label .& Nil)
        A._copyto_upd [in_grad] (#data := result .& Nil)


symbol :: DType a => IO (Symbol a)
symbol = do
    x  <- variable "x"
    y  <- variable "y"

    v1 <- convolution "conv1"   (#data := x  .& #kernel := [5,5] .& #num_filter := 20 .& Nil)
    a1 <- activation "conv1-a"  (#data := v1 .& #act_type := #tanh .& Nil)
    p1 <- pooling "conv1-p"     (#data := a1 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

    v2 <- convolution "conv2"   (#data := p1 .& #kernel := [5,5] .& #num_filter := 50 .& Nil)
    a2 <- activation "conv2-a"  (#data := v2 .& #act_type := #tanh .& Nil)
    p2 <- pooling "conv2-p"     (#data := a2 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

    fl <- flatten "flatten"     (#data := p2 .& Nil)

    v3 <- fullyConnected "fc1"  (#data := fl .& #num_hidden := 500 .& Nil)
    a3 <- activation "fc1-a"    (#data := v3 .& #act_type := #tanh .& Nil)

    v4 <- fullyConnected "fc2"  (#data := a3 .& #num_hidden := 10  .& Nil)
    a4 <- _Custom "softmax" (#data := [v4, y] .& #op_type := "softmax_custom" .& Nil)
    return $ Symbol a4

default_initializer :: Initializer Float
default_initializer name shp
    | T.isSuffixOf "-bias" name = I.zeros name shp
    | otherwise = I.normal 0.1 name shp

main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    registerCustomOperator ("softmax_custom", \_ -> return SoftmaxProp)
    net  <- symbol

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
        logInfo . display $ sformat "[Train] "
        forM_ (V.enumFromTo 1 20) $ \ind -> do
            logInfo . display $ sformat ("iteration " % int) ind
            metric <- newMetric "train" (CrossEntropy "y" :* Accuracy "y" :* MNil)
            void $ forEachD_i trainingData $ \(i, (x, y)) -> do
                fitAndEval optimizer (M.fromList [("x", x), ("y", y)]) metric
                eval <- formatMetric metric
                logInfo . display $ sformat ("\r\ESC[K" % int % "/" % int % ":" % stext) i total1 eval

        logInfo . display $ sformat "[Test] "

        total2 <- sizeD testingData
        result <- forEachD_i testingData $ \(i, (x, y)) -> do
            logInfo . display $ sformat ("\r\ESC[K" % int % "/" % int) i total2
            ~[y'] <- forwardOnly (M.singleton "x" x)
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
    argmax :: ArrayF -> IO ArrayF
    argmax (NDArray ys) = NDArray <$> sing A.argmax (#data := ys .& #axis := Just 1 .& Nil)
