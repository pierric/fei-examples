{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}

module Model.Resnet (symbol, getFeature, getTopFeature) where

import Control.Monad (foldM, when, void)
import Control.Exception.Base (Exception, throw, throwIO)
import Data.Maybe (fromMaybe)
import Data.Typeable (Typeable)
import Text.Printf

import MXNet.Base
import MXNet.NN.Layer

data NoKnownExperiment = NoKnownExperiment Int
    deriving (Typeable, Show)
instance Exception NoKnownExperiment

-------------------------------------------------------------------------------
-- ResNet

symbol :: DType a => Int -> Int -> Int -> IO (Symbol a)
symbol num_classes num_layers image_size = do
    let args = if image_size <= 28 then args_small_image else args_large_image

    x <- variable "x"
    y <- variable "y"

    (u, makeTop) <- getFeature x args
    u <- makeTop u

    flt <- flatten "flt" (#data := u .& Nil)
    fc1 <- fullyConnected "output" (#data := flt .& #num_hidden := num_classes .& Nil)

    ret <- softmaxoutput "softmax" (#data := fc1 .& #label := y .& Nil)
    return $ Symbol ret

  where
    args_common = #workspace := 256 .& Nil
    args_small_image
        | (num_layers - 2) `mod` 9 == 0 && num_layers >= 164 = #num_stages := 3
                                                           .& #filter_list := [64, 64, 128, 256]
                                                           .& #units := replicate 3 ((num_layers - 2) `div` 9)
                                                           .& #bottle_neck := True
                                                           .& args_common
        | (num_layers - 2) `mod` 6 == 0 && num_layers < 164 = #num_stages := 3
                                                          .& #filter_list := [64, 64, 32, 64]
                                                          .& #units := replicate 3 ((num_layers - 2) `div` 6)
                                                          .& #bottle_neck := False
                                                          .& args_common

    args_large_image
        | num_layers == 18  = #num_stages := 4
                          .& #filter_list := [64, 64, 128, 256, 512]
                          .& #units := [2,2,2,2]
                          .& #bottle_neck := False
                          .& args_common
        | num_layers == 34  = #num_stages := 4
                          .& #filter_list := [64, 64, 128, 256, 512]
                          .& #units := [3,4,6,3]
                          .& #bottle_neck := False
                          .& args_common
        | num_layers == 50  = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,4,6,3]
                          .& #bottle_neck := True
                          .& args_common
        | num_layers == 101 = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,4,23,3]
                          .& #bottle_neck := True
                          .& args_common
        | num_layers == 152 = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,8,36,3]
                          .& #bottle_neck := True
                          .& args_common
        | num_layers == 200 = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,24,36,3]
                          .& #bottle_neck := True
                          .& args_common
        | num_layers == 269 = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,30,48,8]
                          .& #bottle_neck := True
                          .& args_common

eps :: Double
eps = 2e-5

bn_mom :: Float
bn_mom = 0.9

type instance ParameterList "resnet" =
  '[ '("num_stages" , 'AttrReq Int)
   , '("filter_list", 'AttrReq [Int])
   , '("units"      , 'AttrReq [Int])
   , '("bottle_neck", 'AttrReq Bool)
   , '("workspace"  , 'AttrReq Int)]

getFeature :: (Fullfilled "resnet" args) => SymbolHandle -> ArgsHMap "resnet" args -> IO (SymbolHandle, SymbolHandle -> IO SymbolHandle)
getFeature inp args = do
    bnx <- batchnorm "features.0" (#data := inp
                          .& #eps := eps
                          .& #momentum := bn_mom
                          .& #fix_gamma := True
                          .& Nil)

    bdy <- convolution "features.1" (#data      := bnx
                                 .& #kernel    := [7,7]
                                 .& #num_filter:= filter0
                                 .& #stride    := [2,2]
                                 .& #pad       := [3,3]
                                 .& #workspace := conv_workspace
                                 .& #no_bias   := True
                                 .& Nil)
    bdy <- batchnorm "features.2" (#data      := bdy
                                .& #fix_gamma := False
                                .& #eps       := eps
                                .& #momentum  := bn_mom
                                .& Nil)
    bdy <- activation "features.3" (#data      := bdy
                                 .& #act_type  := #relu
                                 .& Nil)
    bdy <- pooling "features.4" (#data      := bdy
                              .& #kernel    := [3,3]
                              .& #stride    := [2,2]
                              .& #pad       := [1,1]
                              .& #pool_type := #max
                              .& Nil)

    bdy <- foldM (buildLayer bottle_neck conv_workspace) bdy (zip3 [0::Int ..2] filter_list units)

    return (bdy, flip getTopFeature args)

  where
    filter0 : filter_list = args ! #filter_list
    units = args ! #units
    bottle_neck = args ! #bottle_neck
    conv_workspace = args ! #workspace

getTopFeature :: (Fullfilled "resnet" args) => SymbolHandle -> ArgsHMap "resnet" args -> IO SymbolHandle
getTopFeature inp args = do
    bdy <- buildLayer bottle_neck conv_workspace inp (3, filter, unit)
    bn1 <- batchnorm "features.9" (#data := bdy
                          .& #eps := eps
                          .& #momentum := bn_mom
                          .& #fix_gamma := False
                          .& Nil)
    ac1 <- activation "features.10" (#data := bn1
                             .& #act_type := #relu
                             .& Nil)
    pl1 <- pooling "features.11" (#data := ac1
                          .& #kernel := [7,7]
                          .& #pool_type := #avg
                          .& #global_pool := True
                          .& Nil)

    return pl1
  where
    filter = last $ args ! #filter_list
    unit = last $ args ! #units
    bottle_neck = args ! #bottle_neck
    conv_workspace = args ! #workspace

buildLayer :: Bool -> Int -> SymbolHandle -> (Int, Int, Int) -> IO SymbolHandle
buildLayer bottle_neck workspace bdy (stage_id, filter_size, unit) = do
    bdy <- residual (name 0) (#data := bdy .& #num_filter := filter_size .& #stride := stride0 .& #dim_match := False .& resargs)
    foldM (\bdy unit_id ->
            residual (name unit_id) (#data := bdy .& #num_filter := filter_size .& #stride := [1,1] .& #dim_match := True .& resargs))
          bdy [1..unit-1]
  where
    stride0 = if stage_id == 0 then [1,1] else [2,2]
    name unit_id = printf "features.%d.%d" (stage_id+5) unit_id
    resargs = #bottle_neck := bottle_neck .& #workspace := workspace .& #memonger := False .& Nil

type instance ParameterList "_residual_layer(resnet)" =
  '[ '("data"       , 'AttrReq SymbolHandle)
   , '("num_filter" , 'AttrReq Int)
   , '("stride"     , 'AttrReq [Int])
   , '("dim_match"  , 'AttrReq Bool)
   , '("bottle_neck", 'AttrOpt Bool)
   , '("bn_mom"     , 'AttrOpt Float)
   , '("workspace"  , 'AttrOpt Int)
   , '("memonger"   , 'AttrOpt Bool) ]
residual :: (Fullfilled "_residual_layer(resnet)" args)
         => String -> ArgsHMap "_residual_layer(resnet)" args -> IO SymbolHandle
residual name args = do
    let dat        = args ! #data
        num_filter = args ! #num_filter
        stride     = args ! #stride
        dim_match  = args ! #dim_match
        bottle_neck= fromMaybe True $ args !? #bottle_neck
        bn_mom     = fromMaybe 0.9  $ args !? #bn_mom
        workspace  = fromMaybe 256  $ args !? #workspace
        memonger   = fromMaybe False$ args !? #memonger
    if bottle_neck
      then do
        bn1 <- batchnorm (name ++ ".bn1") (#data := dat
                                        .& #eps  := eps
                                        .& #momentum  := bn_mom
                                        .& #fix_gamma := False .& Nil)
        act1 <- activation (name ++ ".relu1") (#data := bn1 .& #act_type := #relu .& Nil)
        conv1 <- convolution (name ++ ".conv1") (#data := act1
                                              .& #kernel := [1,1]
                                              .& #num_filter := num_filter `div` 4
                                              .& #stride := [1,1]
                                              .& #pad := [0,0]
                                              .& #workspace := workspace
                                              .& #no_bias   := True
                                              .& Nil)
        bn2 <- batchnorm (name ++ ".bn2") (#data := conv1
                                        .& #eps  := eps
                                        .& #momentum  := bn_mom
                                        .& #fix_gamma := False
                                        .& Nil)
        act2 <- activation (name ++ ".relu2") (#data := bn2
                                            .& #act_type := #relu
                                            .& Nil)
        conv2 <- convolution (name ++ ".conv2") (#data := act2
                                              .& #kernel := [3,3]
                                              .& #num_filter := (num_filter `div` 4)
                                              .& #stride    := stride
                                              .& #pad       := [1,1]
                                              .& #workspace := workspace
                                              .& #no_bias   := True
                                              .& Nil)
        bn3 <- batchnorm (name ++ ".bn3") (#data      := conv2
                                        .& #eps       := eps
                                        .& #momentum  := bn_mom
                                        .& #fix_gamma := False
                                        .& Nil)
        act3 <- activation (name ++ ".relu3") (#data := bn3
                                            .& #act_type := #relu
                                            .& Nil)
        conv3 <- convolution (name ++ ".conv3") (#data := act3
                                              .& #kernel := [1,1]
                                              .& #num_filter := num_filter
                                              .& #stride    := [1,1]
                                              .& #pad       := [0,0]
                                              .& #workspace := workspace
                                              .& #no_bias   := True
                                              .& Nil)
        shortcut <- if dim_match
                    then return dat
                    else convolution (name ++ ".downsample") (#data       := act1
                                                   .& #kernel     := [1,1]
                                                   .& #num_filter := num_filter
                                                   .& #stride     := stride
                                                   .& #workspace  := workspace
                                                   .& #no_bias    := True
                                                   .& Nil)
        when memonger $
          void $ mxSymbolSetAttr shortcut "mirror_stage" "true"
        plus name (#lhs := conv3 .& #rhs := shortcut .& Nil)
      else do
        bn1 <- batchnorm (name ++ ".bn1") (#data      := dat
                                        .& #eps       := eps
                                        .& #momentum  := bn_mom
                                        .& #fix_gamma := False
                                        .& Nil)
        act1 <- activation (name ++ ".relu1") (#data      := bn1
                                            .& #act_type  := #relu .& Nil)
        conv1 <- convolution (name ++ ".conv1") (#data      := act1
                                              .& #kernel    := [3,3]
                                              .& #num_filter:= num_filter
                                              .& #stride    := stride
                                              .& #pad       := [1,1]
                                              .& #workspace := workspace
                                              .& #no_bias   := True
                                              .& Nil)
        bn2 <- batchnorm (name ++ ".bn2") (#data      := conv1
                                        .& #eps       := eps
                                        .& #momentum  := bn_mom
                                        .& #fix_gamma := False
                                        .& Nil)
        act2 <- activation (name ++ ".relu2") (#data      := bn2
                                            .& #act_type  := #relu
                                            .& Nil)
        conv2 <- convolution (name ++ ".conv2") (#data      := act2
                                              .& #kernel    := [3,3]
                                              .& #num_filter:= num_filter
                                              .& #stride    := [1,1]
                                              .& #pad       := [1,1]
                                              .& #workspace := workspace
                                              .& #no_bias   := True
                                              .& Nil)
        shortcut <- if dim_match
                    then return dat
                    else convolution (name ++ ".downsample") (#data      := act1
                                                   .& #kernel    := [1,1]
                                                   .& #num_filter:= num_filter
                                                   .& #stride    := stride
                                                   .& #workspace := workspace
                                                   .& #no_bias   := True .& Nil)
        when memonger $
          void $ mxSymbolSetAttr shortcut "mirror_stage" "true"
        plus name (#lhs := conv2 .& #rhs := shortcut .& Nil)
