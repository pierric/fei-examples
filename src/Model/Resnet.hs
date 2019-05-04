{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}

module Model.Resnet (symbol) where

import Control.Monad (foldM, when, void)
import Control.Exception.Base (Exception, throw, throwIO)
import Data.Maybe (fromMaybe)
import Data.Typeable (Typeable)

import MXNet.Base
import MXNet.NN.Layer

data NoKnownExperiment = NoKnownExperiment Int
    deriving (Typeable, Show)
instance Exception NoKnownExperiment

-------------------------------------------------------------------------------
-- ResNet

symbol :: DType a => Int -> Int -> [Int] -> IO (Symbol a)
symbol num_classes num_layers image_shape@[_, height, _] =
    if height <= 28 then do
        handle <- if (num_layers - 2) `mod` 9 == 0 && num_layers >= 164 then
                      resnet $ 
                        #image_shape := image_shape .& 
                        #num_classes := num_classes .& 
                        #num_stages := 3 .& 
                        #filter_list := [64, 64, 128, 256] .& 
                        #units := replicate 3 ((num_layers - 2) `div` 9) .& 
                        #bottle_neck := True .& 
                        #workspace := 256 .& Nil
                  else if (num_layers - 2) `mod` 6 == 0 && num_layers < 164 then
                      resnet $
                        #image_shape := image_shape .& 
                        #num_classes := num_classes .& 
                        #num_stages := 3 .& 
                        #filter_list := [64, 64, 32, 64] .& 
                        #units := replicate 3 ((num_layers - 2) `div` 6) .& 
                        #bottle_neck := False .& 
                        #workspace := 256 .& Nil
                  else
                      throwIO $ NoKnownExperiment num_layers
        return $ Symbol handle
    else do
        handle <- resnet $ #image_shape := image_shape .& #num_classes := num_classes .& #num_stages := 4 .& case num_layers of
          18  -> #filter_list := [64, 64, 128, 256, 512] .& #units := [2,2,2,2] .& #bottle_neck := False .& #workspace := 256 .& Nil
          34  -> #filter_list := [64, 64, 128, 256, 512] .& #units := [3,4,6,3] .& #bottle_neck := False .& #workspace := 256 .& Nil
          50  -> #filter_list := [64, 256, 512, 1024, 2048] .& #units := [3,4,6,3]   .& #bottle_neck := True .& #workspace := 256 .& Nil
          101 -> #filter_list := [64, 256, 512, 1024, 2048] .& #units := [3,4,23,3]  .& #bottle_neck := True .& #workspace := 256 .& Nil
          152 -> #filter_list := [64, 256, 512, 1024, 2048] .& #units := [3,8,36,3]  .& #bottle_neck := True .& #workspace := 256 .& Nil
          200 -> #filter_list := [64, 256, 512, 1024, 2048] .& #units := [3,24,36,3] .& #bottle_neck := True .& #workspace := 256 .& Nil
          269 -> #filter_list := [64, 256, 512, 1024, 2048] .& #units := [3,30,48,8] .& #bottle_neck := True .& #workspace := 256 .& Nil
          _   -> throw $ NoKnownExperiment num_layers
        return $ Symbol handle

type instance ParameterList "resnet" = 
  '[ '("num_classes", 'AttrReq Int)
   , '("num_stages" , 'AttrReq Int)
   , '("filter_list", 'AttrReq [Int])
   , '("units"      , 'AttrReq [Int])
   , '("bottle_neck", 'AttrReq Bool)
   , '("workspace"  , 'AttrReq Int) 
   , '("image_shape", 'AttrReq [Int])]
resnet :: (Fullfilled "resnet" args) => ArgsHMap "resnet" args -> IO SymbolHandle
resnet args = do
    x  <- variable "x"
    y  <- variable "y"

    xcp <- identity "id" (
            #data := x .& Nil)

    bnx <- batchnorm "bn-x" (
            #data := xcp .& 
            #eps := eps .& 
            #momentum := bn_mom .& 
            #fix_gamma := True .& Nil)

    let [_, height, _] = args ! #image_shape
        filter0 : filter_list = args ! #filter_list
    bdy <- if height <= 32 
             then
                convolution "conv-bn-x" (
                          #data      := bnx .& 
                          #kernel    := [3,3] .& 
                          #num_filter:= filter0 .& 
                          #stride    := [1,1] .& 
                          #pad       := [1,1] .& 
                          #workspace := conv_workspace .& 
                          #no_bias   := True .& Nil)
             else do
                bdy <- convolution "conv-bn-x" (
                          #data      := bnx .& 
                          #kernel    := [7,7] .& 
                          #num_filter:= filter0 .& 
                          #stride    := [2,2] .& 
                          #pad       := [3,3] .& 
                          #workspace := conv_workspace .& 
                          #no_bias   := True .& Nil)
                bdy <- batchnorm "bn-0" (
                          #data      := bdy .&
                          #fix_gamma := False .&
                          #eps       := eps .&
                          #momentum  := bn_mom .& Nil)
                bdy <- activation "relu0" (
                          #data      := bdy .&
                          #act_type  := #relu .& Nil)
                pooling "max" (
                          #data      := bdy .&
                          #kernel    := [3,3] .&
                          #stride    := [2,2] .&
                          #pad       := [1,1] .&
                          #pool_type := #max .& Nil)
    
    bdy <- foldM build_layer bdy (zip3 [0::Int ..] filter_list (args ! #units))
    
    bn1 <- batchnorm "bn-1" (
            #data := bdy .& 
            #eps := eps .& 
            #momentum := bn_mom .& 
            #fix_gamma := False .& Nil)
    ac1 <- activation "relu-1" (
            #data := bn1 .& 
            #act_type := #relu .& Nil)
    pl1 <- pooling "pool-1" (
            #data := ac1 .&
            #kernel := [7,7] .& 
            #pool_type := #avg .& 
            #global_pool := True .& Nil)
    
    flt <- flatten "flt-1" (
            #data := pl1 .& Nil)
    fc1 <- fullyConnected "fc-1" (
            #data := flt .& 
            #num_hidden := args ! #num_classes .& Nil)
    
    softmaxoutput "softmax" (
            #data := fc1 .& 
            #label := y .& Nil)
  where
    bn_mom = 0.9 :: Float
    conv_workspace = 256 :: Int
    eps = 2e-5 :: Double

    build_layer bdy (stage_id, filter_size, unit) = do
        let stride0 = if stage_id == 0 then [1,1] else [2,2]
            name unit_id = "stage" ++ show stage_id ++ "_unit" ++ show unit_id
            resargs = #bottle_neck := False .& #workspace := conv_workspace .& #memonger := False .& Nil
        bdy <- residual (name 0) (#data := bdy .& #num_filter := filter_size .& #stride := stride0 .& #dim_match := False .& resargs)
        foldM (\bdy unit_id -> 
                residual (name unit_id) (#data := bdy .& #num_filter := filter_size .& #stride := [1,1] .& #dim_match := True .& resargs))
              bdy [1..unit]

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
        eps = 2e-5 :: Double
    if bottle_neck
      then do
        bn1 <- batchnorm (name ++ "-bn1") (
                    #data := dat .&
                    #eps  := eps .&
                    #momentum  := bn_mom .&
                    #fix_gamma := False .& Nil)
        act1 <- activation (name ++ "-relu1") (
                    #data := bn1 .&
                    #act_type := #relu .& Nil)
        conv1 <- convolution (name ++ "-conv1") (
                    #data := act1 .&
                    #kernel := [1,1] .&
                    #num_filter := num_filter `div` 4 .&
                    #stride := [1,1] .&
                    #pad := [0,0] .&
                    #workspace := workspace .&
                    #no_bias   := True .& Nil)
        bn2 <- batchnorm (name ++ "-bn2") (
                    #data := conv1 .&
                    #eps  := eps   .&
                    #momentum  := bn_mom .&
                    #fix_gamma := False .& Nil)
        act2 <- activation (name ++ "-relu2") (
                    #data := bn2 .&
                    #act_type := #relu .& Nil)
        conv2 <- convolution (name ++ "-conv2") (
                    #data := act2 .&
                    #kernel := [3,3] .&
                    #num_filter := (num_filter `div` 4) .&
                    #stride    := stride .&
                    #pad       := [1,1] .&
                    #workspace := workspace .&
                    #no_bias   := True .& Nil)
        bn3 <- batchnorm (name ++ "-bn3") (
                    #data      := conv2 .&
                    #eps       := eps .&
                    #momentum  := bn_mom .&
                    #fix_gamma := False .& Nil)
        act3 <- activation (name ++ "-relu3") (
                    #data := bn3 .&
                    #act_type := #relu .& Nil)
        conv3 <- convolution (name ++ "-conv3") (
                    #data := act3 .&
                    #kernel := [1,1] .&
                    #num_filter := num_filter .&
                    #stride    := [1,1] .&
                    #pad       := [0,0] .&
                    #workspace := workspace .&
                    #no_bias   := True .& Nil)
        shortcut <- if dim_match
                    then return dat
                    else convolution (name ++ "-sc") (
                            #data       := act1 .&
                            #kernel     := [1,1] .&
                            #num_filter := num_filter .&
                            #stride     := stride .&
                            #workspace  := workspace .&
                            #no_bias    := True .& Nil)
        when memonger $ 
          void $ mxSymbolSetAttr shortcut "mirror_stage" "true"
        plus name (#lhs := conv3 .& #rhs := shortcut .& Nil)
      else do
        bn1 <- batchnorm (name ++ "-bn1") (
                    #data      := dat  .&
                    #eps       := eps .&
                    #momentum  := bn_mom .&
                    #fix_gamma := False .& Nil)
        act1 <- activation (name ++ "-relu1") (
                    #data      := bn1 .&
                    #act_type  := #relu .& Nil)
        conv1 <- convolution (name ++ "-conv1") (
                    #data      := act1  .&
                    #kernel    := [3,3]  .&
                    #num_filter:= num_filter  .&
                    #stride    := stride .&
                    #pad       := [1,1] .&
                    #workspace := workspace .&
                    #no_bias   := True .& Nil)
        bn2 <- batchnorm (name ++ "-bn2") (
                    #data      := conv1 .&
                    #eps       := eps .&
                    #momentum  := bn_mom .&
                    #fix_gamma := False .& Nil)
        act2 <- activation (name ++ "-relu2") (
                    #data      := bn2 .&
                    #act_type  := #relu .& Nil)
        conv2 <- convolution (name ++ "-conv2") (
                    #data      := act2  .&
                    #kernel    := [3,3]  .&
                    #num_filter:= num_filter  .&
                    #stride    := [1,1] .&
                    #pad       := [1,1] .&
                    #workspace := workspace .&
                    #no_bias   := True .& Nil)
        shortcut <- if dim_match
                    then return dat
                    else convolution (name ++ "-sc") (
                            #data      := act1 .&
                            #kernel    := [1,1] .&
                            #num_filter:= num_filter .&
                            #stride    := stride .&
                            #workspace := workspace.&
                            #no_bias   := True .& Nil)
        when memonger $
          void $ mxSymbolSetAttr shortcut "mirror_stage" "true"
        plus name (#lhs := conv2 .& #rhs := shortcut .& Nil)