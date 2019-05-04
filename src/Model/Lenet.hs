{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Model.Lenet (symbol) where

import MXNet.Base
import MXNet.NN.Layer

-- # first conv
-- conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
-- tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
-- pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
-- # second conv
-- conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
-- tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
-- pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
-- # first fullc
-- flatten = mx.symbol.Flatten(data=pool2)
-- fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
-- tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
-- # second fullc
-- fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
-- # loss
-- lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

symbol :: DType a => IO (Symbol a)
symbol = do
    x  <- variable "x"
    y  <- variable "y"

    v1 <- convolution "conv1"   (#data := x  .& #kernel := [5,5] .& #num_filter := 20 .& Nil)
    a1 <- activation "conv1-a"  (#data := v1 .& #act_type := #tanh .& Nil)
    p1 <- pooling "conv1-p"     (#data := a1 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

    v2 <- convolution "conv2"   (#data := p1 .& #kernel := [5,5] .& #num_filter := 50 .& Nil)
    a2 <- activation "conv2-a"  (#data := v2 .& #act_type := #tanh .& Nil)
    p2 <- pooling "conv2-p"     (#data := a2 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

    fl <- flatten "flatten"     (#data := p2 .& Nil)

    v3 <- fullyConnected "fc1"  (#data := fl .& #num_hidden := 500 .& Nil)
    a3 <- activation "fc1-a"    (#data := v3 .& #act_type := #tanh .& Nil)

    v4 <- fullyConnected "fc2"  (#data := a3 .& #num_hidden := 10  .& Nil)
    a4 <- softmaxoutput "softmax" (#data := v4 .& #label := y .& Nil)
    return $ Symbol a4