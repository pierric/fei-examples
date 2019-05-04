{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Model.Resnext where

import Control.Monad (foldM, when, void)
import Data.Maybe (fromMaybe)

import MXNet.Base
import MXNet.NN.Layer

-- ResNet
-- #layer: 164
-- #stage: 3
-- #layer per stage: 18
-- #filter of stage 1: 64
-- #filter of stage 2: 128
-- #filter of stage 3: 256

symbol :: DType a => IO (Symbol a)
symbol = do
    x  <- variable "x"
    y  <- variable "y"

    xcp <- identity "id" (
            #data := x .& Nil)

    bnx <- batchnorm "bn-x" (
            #data := xcp .& 
            #eps := eps .& 
            #momentum := bn_mom .& 
            #fix_gamma := True .& Nil)

    cvx <- convolution "conv-bn-x" (
            #data := bnx .& 
            #kernel := [3,3] .& 
            #num_filter := 16 .& 
            #stride := [1,1] .& 
            #pad := [1,1] .& 
            #workspace := conv_workspace .& 
            #no_bias := True .& Nil)

    bdy <- foldM (\layer (num_filter, stride, dim_match, name) -> 
                    residual name (#data       := layer .&
                                   #num_filter := num_filter .&
                                   #stride     := stride .&
                                   #dim_match  := dim_match .& resargs)) 
                 cvx 
                 residual'parms
    
    pool1 <- pooling "pool1" (
              #data := bdy .&
              #kernel := [7,7] .&
              #pool_type := #avg .&
              #global_pool := True .& Nil)
    flat  <- flatten "flat-1" (
              #data := pool1 .& Nil)
    fc1   <- fullyConnected "fc-1" (
              #data := flat .&
              #num_hidden := 10 .& Nil)
    Symbol <$> softmaxoutput "softmax" (
              #data := fc1 .& 
              #label := y .& Nil)
  where
    bn_mom = 0.9 :: Float
    conv_workspace = 256 :: Int
    eps = 2e-5 :: Double
    residual'parms =  [ (64,  [1,1], False, "stage1-unit1") ] ++ map (\i -> (64,  [1,1], True, "stage1-unit" ++ show i)) [2..18 :: Int]
                   ++ [ (128, [2,2], False, "stage2-unit1") ] ++ map (\i -> (128, [1,1], True, "stage2-unit" ++ show i)) [2..18 :: Int]
                   ++ [ (256, [2,2], False, "stage3-unit1") ] ++ map (\i -> (256, [1,1], True, "stage3-unit" ++ show i)) [2..18 :: Int]
    resargs = #bottle_neck := True .& #workspace := conv_workspace .& #memonger := False .& Nil

type instance ParameterList "_residual_layer(resnext)" = 
  '[ '("data"       , 'AttrReq SymbolHandle)
   , '("num_filter" , 'AttrReq Int)
   , '("stride"     , 'AttrReq [Int])
   , '("dim_match"  , 'AttrReq Bool)
   , '("bottle_neck", 'AttrOpt Bool)
   , '("num_group"  , 'AttrOpt Int)
   , '("bn_mom"     , 'AttrOpt Float)
   , '("workspace"  , 'AttrOpt Int)
   , '("memonger"   , 'AttrOpt Bool) ]
residual :: (Fullfilled "_residual_layer(resnext)" args) 
         => String -> ArgsHMap "_residual_layer(resnext)" args -> IO SymbolHandle
residual name args = do
    let dat        = args ! #data
        num_filter = args ! #num_filter
        stride     = args ! #stride
        dim_match  = args ! #dim_match
        bottle_neck= fromMaybe True $ args !? #bottle_neck
        num_group  = fromMaybe 32   $ args !? #num_group
        bn_mom     = fromMaybe 0.9  $ args !? #bn_mom
        workspace  = fromMaybe 256  $ args !? #workspace
        memonger   = fromMaybe False$ args !? #memonger
        eps = 2e-5 :: Double
    if bottle_neck
      then do
        conv1 <- convolution (name ++ "-conv1") (
                    #data      := dat .&
                    #kernel    := [1,1] .&
                    #num_filter:= num_filter `div` 2 .&
                    #stride    := [1,1] .&
                    #pad       := [0,0] .&
                    #workspace := workspace .&
                    #no_bias   := True .& Nil)
        bn1 <- batchnorm (name ++ "-bn1") (
                    #data      := conv1 .&
                    #eps       := eps .&
                    #momentum  := bn_mom .&
                    #fix_gamma := False .& Nil)
        act1 <- activation (name ++ "-relu1") (
                    #data      := bn1 .&
                    #act_type  := #relu .& Nil)
        conv2 <- convolution (name ++ "-conv2") (
                    #data      := act1 .&
                    #kernel    := [3,3] .&
                    #num_filter:= num_filter `div` 2 .&
                    #stride    := stride .&
                    #pad       := [1,1] .&
                    #num_group := num_group .&
                    #workspace := workspace .&
                    #no_bias   := True .& Nil)
        bn2 <- batchnorm (name ++ "-bn2") (
                    #data      := conv2 .&
                    #eps       := eps .&
                    #momentum  := bn_mom .&
                    #fix_gamma := False .& Nil)
        act2 <- activation (name ++ "-relu2") (
                    #data      := bn2 .&
                    #act_type  := #relu .& Nil)
        conv3 <- convolution (name ++ "-conv3") (
                    #data      := act2 .&
                    #kernel    := [1,1] .&
                    #num_filter:= num_filter .&
                    #stride    := [1,1] .&
                    #pad       := [0,0] .&
                    #workspace := workspace .&
                    #no_bias   := True .& Nil)
        bn3 <- batchnorm (name ++ "-bn3") (
                    #data      := conv3 .&
                    #eps       := eps .&
                    #momentum  := bn_mom .&
                    #fix_gamma := False .& Nil)
        shortcut <- if dim_match
                    then return dat
                    else do
                        shortcut_conv <- convolution (name ++ "-sc") (
                                #data        := dat .&
                                #kernel      := [1,1] .&
                                #num_filter  := num_filter .&
                                #stride      := stride .&
                                #workspace   := workspace .&
                                #no_bias     := True .& Nil)
                        batchnorm (name ++ "-sc-bn") (
                                #data        := shortcut_conv .&
                                #eps         := eps .&
                                #momentum    := bn_mom .&
                                #fix_gamma   := False .& Nil)
        when memonger $ 
          void $ mxSymbolSetAttr shortcut "mirror_stage" "true"
        eltwise <- plus name (
                    #lhs := bn3 .& 
                    #rhs := shortcut .& Nil)
        activation (name ++ "-relu") (
                    #data     := eltwise .&
                    #act_type := #relu .& Nil)
      else do
        conv1 <- convolution (name ++ "-conv1") (
                    #data        := dat .&
                    #kernel      := [3,3] .&
                    #num_filter  := num_filter .&
                    #stride      := stride .&
                    #pad         := [1,1] .&
                    #workspace   := workspace .&
                    #no_bias     := True .& Nil)
        bn1 <- batchnorm (name ++ "-bn1") ( 
                    #data        := conv1 .&
                    #eps         := eps .&
                    #momentum    := bn_mom .&
                    #fix_gamma   := False .& Nil)
        act1 <- activation (name ++ "-relu1") (
                    #data        := bn1 .&
                    #act_type    := #relu .& Nil)
        conv2 <- convolution (name ++ "-conv2") (
                    #data        := act1 .&
                    #kernel      := [3,3] .&
                    #num_filter  := num_filter .&
                    #stride      := [1,1] .&
                    #pad         := [1,1] .&
                    #workspace   := workspace .&
                    #no_bias     := True .& Nil)
        bn2 <- batchnorm (name ++ "-bn2") (
                    #data        := conv2 .&
                    #eps         := eps .&
                    #momentum    := bn_mom .&
                    #fix_gamma   := False .& Nil)
        shortcut <- if dim_match
                    then return dat
                    else do
                        shortcut_conv <- convolution (name ++ "-sc") (
                                #data        := act1 .&
                                #kernel      := [1,1] .&
                                #num_filter  := num_filter .&
                                #stride      := stride .&
                                #workspace   := workspace .&
                                #no_bias     := True .& Nil)
                        batchnorm (name ++ "-sc-bn") (
                                #data        := shortcut_conv.&
                                #eps         := eps .&
                                #momentum    := bn_mom .&
                                #fix_gamma   := False .& Nil)
        when memonger $ 
          void $ mxSymbolSetAttr shortcut "mirror_stage" "true"
        eltwise <- plus name (
                    #lhs := bn2 .&
                    #rhs := shortcut .& Nil)
        activation (name ++ "-relu") (
                    #data     := eltwise .&
                    #act_type := #relu .& Nil)