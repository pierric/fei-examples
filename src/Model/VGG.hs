module Model.VGG where

import Text.Printf (printf)
import Control.Monad (foldM)
import Data.Maybe (fromMaybe)

import MXNet.Base
import MXNet.NN.Layer

getFeature :: SymbolHandle -> [Int] -> [Int] -> Bool -> Bool -> IO SymbolHandle
getFeature internalLayer layers filters with_batch_norm with_last_pooling= do
    sym <- foldM build1 internalLayer specs
    -- inlining the build1 below, and omit pooling depending on the with_last_pooling
    case last_group of
        (idx, num, filter) -> do
            sym <- foldM (build2 idx) sym $ zip [1::Int ..] (replicate num filter)
            if not with_last_pooling
                then return sym
                else pooling (printf "pool%d" idx) (#data := sym .& #pool_type := #max .& #kernel := [2,2] .& #stride := [2,2] .& Nil)

  where
    last_group:groups = reverse $ zip3 [1::Int ..] layers filters
    specs = reverse groups

    build1 sym (idx, num, filter) = do 
        sym <- foldM (build2 idx) sym $ zip [1::Int ..] (replicate num filter)
        pooling (printf "pool%d" idx) (#data := sym .& #pool_type := #max .& #kernel := [2,2] .& #stride := [2,2] .& Nil)

    build2 idx1 sym (idx2, filter) = do
        let ident = printf "%d_%d" idx1 idx2
        sym <- convolution ("conv" ++ ident) (#data := sym .& #kernel := [3,3] .& #pad := [1,1] .& #num_filter := filter .& Nil)
        sym <- if with_batch_norm then batchnorm ("bn" ++ ident) (#data := sym .& Nil) else return sym
        activation ("relu" ++ ident) (#data := sym .& #act_type := #relu .& Nil)

getTopFeature :: Maybe String -> SymbolHandle -> IO SymbolHandle
getTopFeature prefix input_data = do
    let addPrefix = (fromMaybe "" prefix ++)
    sym <- flatten (addPrefix "flatten") (#data := input_data .& Nil)
    sym <- fullyConnected (addPrefix "fc6") (#data := sym .& #num_hidden := 4096 .& Nil)
    sym <- activation (addPrefix "relu6") (#data := sym .& #act_type := #relu .& Nil)
    sym <- dropout (addPrefix "drop6") (#data := sym .& #p := 0.5 .& Nil)
    sym <- fullyConnected (addPrefix "fc7") (#data := sym .& #num_hidden := 4096 .& Nil)
    sym <- activation (addPrefix "relu7") (#data := sym .& #act_type := #relu .& Nil)
    dropout (addPrefix "drop7") (#data := sym .& #p := 0.5 .& Nil)

getClassifier :: Maybe String -> SymbolHandle -> Int -> IO SymbolHandle
getClassifier prefix input_data num_classes = do
    let addPrefix = (fromMaybe "" prefix ++)
    sym <- getTopFeature prefix input_data
    fullyConnected (addPrefix "fc8") (#data := sym .& #num_hidden := num_classes .& Nil)

symbol :: Int -> Int -> Bool -> IO (Symbol Float)
symbol num_classes num_layers with_batch_norm = do
    sym <- variable "data"
    sym <- getFeature sym layers filters with_batch_norm True
    sym <- getClassifier Nothing sym num_classes
    sym <- softmaxoutput "softmax" (#data := sym .& Nil)
    return (Symbol sym)

  where  
    (layers, filters) = case num_layers of
                            11 -> ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512])
                            13 -> ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512])
                            16 -> ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
                            19 -> ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])

