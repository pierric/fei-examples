{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TemplateHaskell       #-}
module Common where

import           Control.Lens
import           Control.Monad.Trans.Resource
import           MXNet.Base
import           MXNet.NN
import           RIO                          hiding (view)
import           RIO.Process

data FeiApp t n x = FeiApp
    { _fa_log_func        :: !LogFunc
    , _fa_process_context :: !ProcessContext
    , _fa_session         :: MVar (TaggedModuleState t n)
    , _fa_extra           :: x
    }
makeLenses ''FeiApp

instance HasLogFunc (FeiApp t n x) where
    logFuncL = fa_log_func

instance HasSessionRef (FeiApp t n x) (TaggedModuleState t n) where
    sessionRefL = fa_session

type FeiM t n x a = ReaderT (FeiApp t n x) (ResourceT IO) a


data SessionAlreadyExist = SessionAlreadyExist
    deriving (Typeable, Show)
instance Exception SessionAlreadyExist


runFeiM :: x -> FeiM n t x a -> IO a
runFeiM x body = do
    void mxListAllOpNames
    logopt  <- logOptionsHandle stdout False
    context <- mkDefaultProcessContext
    session <- newEmptyMVar
    runResourceT $ withLogFunc logopt $ \logfunc ->
        flip runReaderT (FeiApp logfunc context session x) body


initSession :: forall n t x. FloatDType t => SymbolHandle -> Config t -> FeiM t n x ()
initSession sym cfg = do
    sess_ref <- view $ fa_session
    liftIO $ do
        sess <- initialize sym cfg
        succ <- tryPutMVar sess_ref sess
        when (not succ) $ throwM SessionAlreadyExist
