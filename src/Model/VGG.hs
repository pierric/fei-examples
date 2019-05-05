module Model.VGG where

import Text.Printf (printf)
import Control.Monad (foldM)

import MXNet.Base
import MXNet.NN.Layer

getFeature :: DType a => Symbol a -> [Int] -> [Int] -> Bool -> IO (Symbol a)
getFeature internalLayer layers filters withBatchNorm = do
    sym <- foldM build1 (unSymbol internalLayer) $ zip3 [1::Int ..] layers filters
    return (Symbol sym)
  where
    build1 sym (idx, num, filter) = do 
        sym <- foldM (build2 idx) sym $ zip [1::Int ..] (replicate num filter)
        pooling (printf "pool%d" idx) (#data := sym .& #pool_type := #max .& #kernel := [2,2] .& #stride := [2,2] .& Nil)

    build2 idx1 sym (idx2, filter) = do
        let ident = printf "%d_%d" idx1 idx2
        sym <- convolution ("conv" ++ ident) (#data := sym .& #kernel := [3,3] .& #pad := [1,1] .& #num_filter := filter .& Nil)
        sym <- if withBatchNorm then batchnorm ("bn" ++ ident) (#data := sym .& Nil) else return sym
        activation ("relu" ++ ident) (#data := sym .& #act_type := #relu .& Nil)


