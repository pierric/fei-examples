module Model.VGG where

import Text.Printf (printf)
import Control.Monad (foldM)

import MXNet.Base
import MXNet.NN.Layer

getFeature :: SymbolHandle -> [Int] -> [Int] -> Bool -> IO SymbolHandle
getFeature internalLayer layers filters withBatchNorm =
    foldM build1 internalLayer $ zip3 [1::Int ..] layers filters
  where
    build1 sym (idx, num, filter) = do 
        sym <- foldM (build2 idx) sym $ zip [1::Int ..] (replicate num filter)
        pooling (printf "pool%d" idx) (#data := sym .& #pool_type := #max .& #kernel := [2,2] .& #stride := [2,2] .& Nil)

    build2 idx1 sym (idx2, filter) = do
        let ident = printf "%d_%d" idx1 idx2
        sym <- convolution ("conv" ++ ident) (#data := sym .& #kernel := [3,3] .& #pad := [1,1] .& #num_filter := filter .& Nil)
        sym <- if withBatchNorm then batchnorm ("bn" ++ ident) (#data := sym .& Nil) else return sym
        activation ("relu" ++ ident) (#data := sym .& #act_type := #relu .& Nil)


getClassifier :: SymbolHandle -> Int -> IO SymbolHandle
getClassifier input_data num_classes = do
    sym <- flatten "flatten" (#data := input_data .& Nil)
    sym <- fullyConnected "fc6" (#data := sym .& #num_hidden := 4096 .& Nil)
    sym <- activation "relu6" (#data := sym .& #act_type := #relu .& Nil)
    sym <- dropout "drop6" (#data := sym .& #p := 0.5 .& Nil)
    sym <- fullyConnected "fc7" (#data := sym .& #num_hidden := 4096 .& Nil)
    sym <- activation "relu7" (#data := sym .& #act_type := #relu .& Nil)
    sym <- dropout "drop7" (#data := sym .& #p := 0.5 .& Nil)
    sym <- fullyConnected "fc8" (#data := sym .& #num_hidden := num_classes .& Nil)
    return sym

symbol :: Int -> Int -> Bool -> IO (Symbol Float)
symbol num_classes num_layers withBatchNorm = do
    sym <- variable "data"
    sym <- getFeature sym layers filters withBatchNorm
    sym <- getClassifier sym num_classes
    sym <- softmaxoutput "softmax" (#data := sym .& Nil)
    return (Symbol sym)

  where  
    (layers, filters) = case num_layers of
                            11 -> ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512])
                            13 -> ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512])
                            16 -> ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
                            19 -> ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])

