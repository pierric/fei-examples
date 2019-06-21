module Main where

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A
import qualified MXNet.Base.Operators.Symbol as S
import qualified MXNet.NN as NN
import qualified MXNet.NN.Utils as NN
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Streaming
import qualified Data.HashMap.Strict as M
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import Control.Monad (forM_, void)
import System.IO (hFlush, stdout)
import qualified Numeric.LinearAlgebra as L

type ArrayF = NDArray Float

data SoftmaxProp = SoftmaxProp

instance CustomOperationProp SoftmaxProp where
    prop_list_arguments _        = ["data", "label"]
    prop_list_outputs _          = ["output"]
    prop_list_auxiliary_states _ = []
    prop_infer_shape _ [data_shape, _] =
        let output_shape = data_shape
        in ([data_shape, [head data_shape]], [output_shape], [])
    prop_declare_backward_dependency _ grad_out data_in data_out = data_in ++ data_out

    data Operation SoftmaxProp = Softmax
    prop_create_operator _ _ _ = return Softmax

instance CustomOperation (Operation SoftmaxProp) where
    forward _ [ReqWrite] [in_data,_] [out_data] aux is_train = do
        -- let in_data_ = (NDArray in_data :: ArrayF)
        -- [_, num_classes] <- ndshape in_data_
        -- vec <- toVector in_data_
        -- let batch_exp = L.toRows $ exp $ L.reshape num_classes vec :: [L.Vector Float]
        --     norm1 = map (realToFrac . L.sumElements) $ batch_exp :: [L.Vector Float]
        --     output = L.fromRows $ zipWith (/) batch_exp norm1
        -- copyFromVector (NDArray out_data :: ArrayF) vec 

        [result] <- A.softmax (#data := in_data .& #axis := 1 .& Nil)
        A._copyto_upd [out_data] (#data := result .& Nil)

    backward _ [ReqWrite] [_, label] [out_data] [in_grad, _] _ aux = do
        -- let out_data_ = NDArray out_data :: ArrayF
        --     label_    = NDArray label :: ArrayF
        -- out_shp@[_, num_classes] <- ndshape out_data_
        -- vec_lbl <- toVector label_
        -- vec_out <- toVector out_data_
        -- let rows = L.toRows $ L.reshape num_classes vec_out :: [L.Vector Float]
        --     upd :: L.Vector Float -> Float -> L.Vector Float
        --     upd row n = let n_ = floor n
        --                 in row SV.// [(n_, row SV.! n_ - 1)]
        --     result = L.fromRows $ zipWith upd rows (L.toList vec_lbl) :: L.Matrix Float
        -- copyFromVector (NDArray in_grad :: ArrayF) (L.flatten result)

        [label_onehot] <- A.one_hot (#indices := label .& #depth := num_classes .& Nil)
        [result] <- A.elemwise_sub (#lhs := out_data .& #rhs := label_onehot .& Nil)
        A._copyto_upd [in_grad] (#data := result .& Nil)


symbol :: DType a => IO (Symbol a)
symbol = do
    x  <- NN.variable "x"
    y  <- NN.variable "y"

    v1 <- NN.convolution "conv1"   (#data := x  .& #kernel := [5,5] .& #num_filter := 20 .& Nil)
    a1 <- NN.activation "conv1-a"  (#data := v1 .& #act_type := #tanh .& Nil)
    p1 <- NN.pooling "conv1-p"     (#data := a1 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

    v2 <- NN.convolution "conv2"   (#data := p1 .& #kernel := [5,5] .& #num_filter := 50 .& Nil)
    a2 <- NN.activation "conv2-a"  (#data := v2 .& #act_type := #tanh .& Nil)
    p2 <- NN.pooling "conv2-p"     (#data := a2 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

    fl <- NN.flatten "flatten"     (#data := p2 .& Nil)

    v3 <- NN.fullyConnected "fc1"  (#data := fl .& #num_hidden := 500 .& Nil)
    a3 <- NN.activation "fc1-a"    (#data := v3 .& #act_type := #tanh .& Nil)

    v4 <- NN.fullyConnected "fc2"  (#data := a3 .& #num_hidden := 10  .& Nil)
    a4 <- S._Custom "softmax" (#data := [v4, y] .& #op_type := "softmax_custom" .& Nil)
    return $ Symbol a4

default_initializer :: NN.Initializer Float
default_initializer name shp
    | NN.endsWith "-bias" name = NN.zeros name shp
    | otherwise = NN.normal 0.1 name shp

main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    registerCustomOperator ("softmax_custom", \_ -> return SoftmaxProp)
    net  <- symbol

    sess <- NN.initialize net $ NN.Config {
                NN._cfg_data = ("x", [1,28,28]),
                NN._cfg_label = ("y", [1]),
                NN._cfg_initializers = M.empty,
                NN._cfg_default_initializer = default_initializer,
                NN._cfg_context = contextCPU
            }
    optimizer <- NN.makeOptimizer NN.SGD'Mom (NN.Const 0.0002) Nil

    NN.train sess $ do

        let trainingData = mnistIter (#image := "data/train-images-idx3-ubyte" .&
                                      #label := "data/train-labels-idx1-ubyte" .&
                                      #batch_size := 128 .& Nil)
        let testingData  = mnistIter (#image := "data/t10k-images-idx3-ubyte" .&
                                      #label := "data/t10k-labels-idx1-ubyte" .&
                                      #batch_size := 16  .& Nil)

        total1 <- sizeD trainingData
        liftIO $ putStrLn $ "[Train] "
        forM_ (enumFromTo 1 1 :: [Int]) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            metric <- NN.newMetric "train" NN.CrossEntropy
            void $ forEachD_i trainingData $ \(i, (x, y)) -> do
                NN.fitAndEval optimizer (M.fromList [("x", x), ("y", y)]) metric
                eval <- NN.format metric
                liftIO $ do
                   putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total1 ++ " " ++ eval
                   hFlush stdout
            liftIO $ putStrLn ""

        liftIO $ putStrLn $ "[Test] "

        total2 <- sizeD testingData
        result <- forEachD_i testingData $ \(i, (x, y)) -> do
            liftIO $ do
                putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total2
                hFlush stdout
            [y'] <- NN.forwardOnly (M.fromList [("x", Just x), ("y", Nothing)])
            ind1 <- liftIO $ toVector y
            ind2 <- liftIO $ argmax y' >>= toVector
            return (ind1, ind2)
        liftIO $ putStr "\r\ESC[K"

        let (ls,ps) = unzip result
            ls_unbatched = mconcat ls
            ps_unbatched = mconcat ps
            total_test_items = SV.length ls_unbatched
            correct = SV.length $ SV.filter id $ SV.zipWith (==) ls_unbatched ps_unbatched
        liftIO $ putStrLn $ "Accuracy: " ++ show correct ++ "/" ++ show total_test_items

  where
    argmax :: ArrayF -> IO ArrayF
    argmax (NDArray ys) = NDArray . head <$> A.argmax (#data := ys .& #axis := Just 1 .& Nil)
