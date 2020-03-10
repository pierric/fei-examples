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

{-
VGG(
  (features): HybridSequential(
    (0): Conv2D(3 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Activation(relu)
    (2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Activation(relu)
    (4): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (5): Conv2D(64 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): Activation(relu)
    (7): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): Activation(relu)
    (9): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (10): Conv2D(128 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): Activation(relu)
    (12): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): Activation(relu)
    (14): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): Activation(relu)
    (16): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (17): Conv2D(256 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): Activation(relu)
    (19): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): Activation(relu)
    (21): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): Activation(relu)
    (23): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (24): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): Activation(relu)
    (26): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): Activation(relu)
    (28): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): Activation(relu)
    ** (30): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (31): Dense(25088 -> 4096, Activation(relu))
    (32): Dropout(p = 0.5, axes=())
    (33): Dense(4096 -> 4096, Activation(relu))
    (34): Dropout(p = 0.5, axes=())
  )
  (output): Dense(4096 -> 1000, linear)
)

** It appears only if `with_last_pooling` is True.
 -}


getFeature :: SymbolHandle -> [Int] -> [Int] -> Bool -> Bool -> IO (SymbolHandle, SymbolHandle -> IO SymbolHandle)
getFeature internalLayer layers filters with_batch_norm with_last_pooling = flip ST.evalStateT (0::Int) $ do
    sym <- foldM build1 internalLayer specs
    -- inlining the build1 below, and omit pooling depending on the with_last_pooling
    case last_group of
        (idx, num, filter) -> do
            sym <- foldM (build2 idx) sym $ zip [1::Int ..] (replicate num filter)
            id  <- ST.get
            sym <- if not with_last_pooling
                then return sym
                else opWithID pooling (#data := sym .& #pool_type := #max .& #kernel := [2,2] .& #stride := [2,2] .& Nil)
            return (sym, getTopFeature (id + 1))

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

getTopFeature :: Int -> SymbolHandle -> IO SymbolHandle
getTopFeature id input = do
    flip ST.evalStateT id $ do
        xid <- ST.get
        sym <- ST.lift $ flatten (printf "features.%d.flatten" xid) (#data := input .& Nil)
        sym <- opWithID fullyConnected (#data := sym .& #num_hidden := 4096 .& Nil)
        sym <- ST.lift $ activation (printf "features.%d.activation" xid) (#data := sym .& #act_type := #relu .& Nil)
        sym <- opWithID dropout (#data := sym .& #p := 0.5 .& Nil)
        xid <- ST.get
        sym <- opWithID fullyConnected (#data := sym .& #num_hidden := 4096 .& Nil)
        sym <- ST.lift $ activation (printf "features.%d.activation" xid) (#data := sym .& #act_type := #relu .& Nil)
        opWithID dropout (#data := sym .& #p := 0.5 .& Nil)

symbol :: Int -> Int -> Bool -> IO (Symbol Float)
symbol num_classes num_layers with_batch_norm = do
    sym <- variable "data"
    (sym, makeTop) <- getFeature sym layers filters with_batch_norm True
    sym <- makeTop sym 
    fullyConnected "output" (#data := sym .& #num_hidden := num_classes .& Nil)
    sym <- softmaxoutput "softmax" (#data := sym .& Nil)
    return (Symbol sym)

  where
    (layers, filters) = case num_layers of
                            11 -> ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512])
                            13 -> ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512])
                            16 -> ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
                            19 -> ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])

