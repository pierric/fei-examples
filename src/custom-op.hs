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

type ArrayF = NDArray Float

data SoftmaxProp = SoftmaxProp

instance CustomOperationProp SoftmaxProp where
    prop_list_arguments _        = ["data", "label"]
    prop_list_outputs _          = ["output"]
    prop_list_auxiliary_states _ = []
    prop_infer_shape _ in_shape =
        let data_shape = head in_shape
            label_shape = [head data_shape]
            output_shape = data_shape
        in ([data_shape, label_shape], [output_shape], [])
    prop_declare_backward_dependency _ grad_out data_in data_out = data_in ++ data_out

    data Operation SoftmaxProp = Softmax
    prop_create_operator _ _ _ = return Softmax

instance CustomOperation (Operation SoftmaxProp) where
    forward _ reqs in_data out_data aux is_train = do
        print "Forward"
        forM_ in_data $ \x -> do
            sh <- ndshape (NDArray x)
            print sh
        putStrLn "--"
        forM_ out_data $ \x -> do
            sh <- ndshape (NDArray x)
            print sh

    backward _ reqs out_grad in_data out_data in_grad aux = do
        print "Backward"
        forM_ out_grad $ \x -> do
            sh <- ndshape (NDArray x)
            print sh
        putStrLn "--"
        forM_ in_data $ \x -> do
            sh <- ndshape (NDArray x)
            print sh
        putStrLn "--"
        forM_ out_data $ \x -> do
            sh <- ndshape (NDArray x)
            print sh
        putStrLn "--"
        forM_ in_grad $ \x -> do
            sh <- ndshape (NDArray x)
            print sh
        putStrLn "--"


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

    let inferShape :: DType a => Symbol a -> M.HashMap String (NDArray a) -> IO (M.HashMap String [Int], M.HashMap String [Int])
        inferShape (Symbol sym) known = do
            let (names, vals) = unzip $ M.toList known
            shapes <- mapM ndshape vals
            let arg_ind = scanl (+) 0 $ map length shapes
                arg_shp = concat shapes
            print (names, arg_ind, arg_shp)
            (inp_shp, _, aux_shp, complete) <- mxSymbolInferShape sym names arg_ind arg_shp
            -- if (not complete) then throwM InferredShapeInComplete else return ()
            inps <- mxSymbolListArguments sym
            auxs <- mxSymbolListAuxiliaryStates sym
            return (M.fromList $ zip inps inp_shp, M.fromList $ zip auxs aux_shp)

    dummyX <- makeEmptyNDArray [1,1,28,28] contextCPU
    dummyY <- makeEmptyNDArray [1,1] contextCPU
    let placeholders = M.fromList [("x", dummyX), ("y", dummyY)]
    print "call infer"
    (inp_with_shp, aux_with_shp) <- inferShape net placeholders
    print (inp_with_shp, aux_with_shp)


    print "init"
    sess <- NN.initialize net $ NN.Config {
                NN._cfg_data = ("x", [1,28,28]),
                NN._cfg_label = ("y", [1]),
                NN._cfg_initializers = M.empty,
                NN._cfg_default_initializer = default_initializer,
                NN._cfg_context = contextCPU
            }
    optimizer <- NN.makeOptimizer NN.SGD'Mom (NN.Const 0.0002) Nil

    print "train"

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
