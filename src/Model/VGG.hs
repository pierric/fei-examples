{-# LANGUAGE OverloadedStrings #-}
module Model.VGG where

import Text.Printf (printf)
import Control.Monad (foldM)
import qualified Control.Monad.State as ST
import Data.Maybe (fromMaybe)
import qualified Data.Text as T
import qualified Data.Text.Read as T
import Data.Either (either)
import Data.Maybe (catMaybes)
import Data.List (sort)

import MXNet.Base
import MXNet.NN.Layer

opWithID op args = do
    n <- ST.get
    ST.put (n+1)
    ST.lift $ op (printf "features.%d" n) args

lastID sym = do
    names <- listArguments sym
    let ids = catMaybes [ignore $ T.decimal $ tn !! 1 | n <- names, let tn = T.splitOn "." (T.pack n), length tn > 2]
    return $ last $ sort ids
  where
    ignore :: Either String (Int, T.Text) -> Maybe Int
    ignore = either (const Nothing) (Just . fst)

getFeature :: SymbolHandle -> [Int] -> [Int] -> Bool -> Bool -> IO SymbolHandle
getFeature internalLayer layers filters with_batch_norm with_last_pooling = flip ST.evalStateT (0::Int) $ do
    sym <- foldM build1 internalLayer specs
    -- inlining the build1 below, and omit pooling depending on the with_last_pooling
    case last_group of
        (idx, num, filter) -> do
            sym <- foldM (build2 idx) sym $ zip [1::Int ..] (replicate num filter)
            if not with_last_pooling
                then return sym
                else opWithID pooling (#data := sym .& #pool_type := #max .& #kernel := [2,2] .& #stride := [2,2] .& Nil)

  where
    last_group:groups = reverse $ zip3 [1::Int ..] layers filters
    specs = reverse groups

    build1 sym (idx, num, filter) = do
        sym <- foldM (build2 idx) sym $ zip [1::Int ..] (replicate num filter)
        opWithID pooling (#data := sym .& #pool_type := #max .& #kernel := [2,2] .& #stride := [2,2] .& Nil)

    build2 idx1 sym (idx2, filter) = do
        sym <- opWithID convolution (#data := sym .& #kernel := [3,3] .& #pad := [1,1] .& #num_filter := filter .& #workspace := 2048 .& Nil)
        sym <- if with_batch_norm then opWithID batchnorm (#data := sym .& Nil) else return sym
        opWithID activation (#data := sym .& #act_type := #relu .& Nil)

getTopFeature :: SymbolHandle -> IO SymbolHandle
getTopFeature input = do
    id <- lastID input
    flip ST.evalStateT (id + 1) $ do
        sym <- opWithID flatten (#data := input .& Nil)
        sym <- opWithID fullyConnected (#data := sym .& #num_hidden := 4096 .& Nil)
        sym <- opWithID activation (#data := sym .& #act_type := #relu .& Nil)
        sym <- opWithID dropout (#data := sym .& #p := 0.5 .& Nil)
        sym <- opWithID fullyConnected (#data := sym .& #num_hidden := 4096 .& Nil)
        sym <- opWithID activation (#data := sym .& #act_type := #relu .& Nil)
        opWithID dropout (#data := sym .& #p := 0.5 .& Nil)

getClassifier :: SymbolHandle -> Int -> IO SymbolHandle
getClassifier input num_classes = do
    sym <- getTopFeature input
    fullyConnected "output" (#data := sym .& #num_hidden := num_classes .& Nil)

symbol :: Int -> Int -> Bool -> IO (Symbol Float)
symbol num_classes num_layers with_batch_norm = do
    sym <- variable "data"
    sym <- getFeature sym layers filters with_batch_norm True
    sym <- getClassifier sym num_classes
    sym <- softmaxoutput "softmax" (#data := sym .& Nil)
    return (Symbol sym)

  where
    (layers, filters) = case num_layers of
                            11 -> ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512])
                            13 -> ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512])
                            16 -> ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
                            19 -> ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])

