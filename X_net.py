#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import scipy.io
import os
import sys
import time
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from PIL import ImageOps
from PIL import Image
from skimage.filters import threshold_otsu
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
from skimage.measure import block_reduce
from scipy.ndimage import zoom
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
import argparse
import math
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian
from pdb import set_trace as bp


DATASET = 1
PRE_TRAIN = 0
TRAIN = 1
NAME_DATASET = ["Texas", "California"]
USE_PATCHES = 0  # DATASET

SEN12_OPT_PATH="Data/Sen12/Optical/"
SEN12_SAR_PATH="Data/Sen12/Sar/"


PATCH_SIZE = 100


PATCH_SIZE_AFF = 20
PATCH_STRIDE_AFF = PATCH_SIZE_AFF // 4
BATCH_SIZE_AFF = 100
ZERO_PAD = 0

W_REG = 0.001
W_TRAN = 3.0
W_CYCLE = 2.0
MAX_GRAD_NORM = 1.0
DROP_PROB = 0.2
ALPHA_LEAKY = 0.3

CHANNELS_SAR=1
CHANNELS_OPT=3

np.random.seed(41148)

nf1 = 100
nf2 = 50
nf3 = 20
nf4 = 10

fs1 = 3
fs2 = 3
fs3 = 3
fs4 = 3

if DATASET == 1:
    nc1 = 11
    nc2 = 3
elif DATASET == 0:
    nc1 = 7
    nc2 = 10
else:
    print("Wrong dataset")
    exit()
specs_X_to_Y = [
    [nc1, nf1, fs1, 1],
    [nf1, nf2, fs2, 1],
    [nf2, nf3, fs3, 1],
    [nf3, nc2, fs4, 1],
]

specs_Y_to_X = [
    [nc2, nf1, fs1, 1],
    [nf1, nf2, fs2, 1],
    [nf2, nf3, fs3, 1],
    [nf3, nc1, fs4, 1],
]
sen12_specs_X_to_Y = [
    [1, nf1, fs1, 1],
    [nf1, nf2, fs2, 1],
    [nf2, nf3, fs3, 1],
    [nf3, 3, fs4, 1],
]

sen12_specs_Y_to_X = [
    [3, nf1, fs1, 1],
    [nf1, nf2, fs2, 1],
    [nf2, nf3, fs3, 1],
    [nf3, 1, fs4, 1],
]


# In[2]:


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    file_names=list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            names,files=getListOfFiles(fullPath)
            allFiles = allFiles + files
            file_names=file_names+names
        else:
            allFiles.append(fullPath)
            file_names.append(entry)
                
    return file_names,allFiles


# In[3]:


def findnth(string, substring, n):
    parts = string.split(substring, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(string) - len(parts[-1]) - len(substring)


# In[4]:


class Sen12DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_IDs,
                 optpath,
                 sarpath,
                 batch_size,
                 dim,
                 n_channels_x,
                 n_channels_y,
                 to_fit=True,
                 shuffle=True):
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels_x = n_channels_x
        self.n_channels_y = n_channels_y
        self.optpath=optpath
        self.sarpath=sarpath
        self.shuffle = shuffle
        self.to_fit=to_fit
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels_x))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels_y))

        # Generate data
        for i, ID_x in enumerate(list_IDs_temp):
            
            ID_y=ID_x.replace("_s1_","_s2_")
            
            ind1=findnth(ID_x,"_",1)+1
            ind2=findnth(ID_x,"_",3)
            
            x_dir=ID_x[ind1:ind2]
            y_dir=ID_y[ind1:ind2]
            
            
            tmp_x=imread(os.path.join(self.sarpath,x_dir,ID_x))
            tmp_x=resize(tmp_x,(PATCH_SIZE,PATCH_SIZE))
            tmp_x=np.reshape(tmp_x,(PATCH_SIZE,PATCH_SIZE,self.n_channels_x))
            
            
            tmp_y=imread(os.path.join(self.optpath,y_dir,ID_y))
            tmp_y=resize(tmp_y,(PATCH_SIZE,PATCH_SIZE))
            tmp_y=np.reshape(tmp_y,(PATCH_SIZE,PATCH_SIZE,self.n_channels_y))
            
            X[i,] = tmp_x
            Y[i,] = tmp_y

        return X,Y


# In[5]:


def network(specs,name):

    net_input=tf.keras.Input(shape=(PATCH_SIZE,PATCH_SIZE,specs[0][0]),name=name+"_input")
    out=net_input
    for i,l in enumerate(specs):
        out=tf.keras.layers.Conv2D(specs[i][1],specs[i][2],activation=None,padding="same")(out)
        out=tf.keras.layers.LeakyReLU(alpha=ALPHA_LEAKY)(out)
        if i!=(len(specs)-1):
            out=tf.keras.layers.Dropout(DROP_PROB)(out)
        else:
            out=tf.keras.activations.tanh(out)
    model=keras.Model(inputs=net_input,outputs=out,name=name)
    return model


# In[6]:


def xnetLoss(x,y,xhat,yhat,xcycle,ycycle,parameters):
    
    trans_loss_x= tf.compat.v1.losses.mean_squared_error(x, xhat)
    trans_loss_y= tf.compat.v1.losses.mean_squared_error(y, yhat)
    cycle_loss_x= tf.compat.v1.losses.mean_squared_error(x, xcycle)
    cycle_loss_y= tf.compat.v1.losses.mean_squared_error(y, ycycle)
    
    reg_loss = tf.add_n(tf.reduce_mean(tf.nn.l2_loss(v)) for v in parameters)
    
    tot_loss_x = (
                W_CYCLE * cycle_loss_x
                + W_TRAN * trans_loss_x
            )

    tot_loss_y = (
                W_CYCLE * cycle_loss_y
                + W_TRAN * trans_loss_y
            )
    return tot_loss_x+tot_loss_y+ W_REG * reg_loss


# In[7]:


class Xnet(keras.Model):
    def __init__(self, xynet,yxnet,inputs,outputs,name,**kwargs):
        super(Xnet, self).__init__(inputs=inputs,outputs=outputs,name=name)
        self.xynet=xynet
        self.yxnet=yxnet
        self.parameters=self.trainable_variables
        
    def compile(self,optimizer,loss_fn):
        super(Xnet, self).compile(optimizer=optimizer,loss=loss_fn)
        self.optimizer=optimizer
        self.loss_fn=loss_fn

    def train_step(self,data):
        xpatch,ypatch=data
        with tf.GradientTape() as tape:
            yhat,xhat,xcycle,ycycle=self(data,training=True)
            tot_loss=xnetLoss(xpatch,ypatch,xhat,yhat,xcycle,ycycle,self.parameters)
        grads=tape.gradient(tot_loss,self.parameters)
        clipped_grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        self.optimizer.apply_gradients(zip(clipped_grads, self.parameters))
        return {"loss": tot_loss}
        
    def test_step(self,data):
        xpatch,ypatch=data
        yhat,xhat,xcycle,ycycle=self(data,training=True)
        tot_loss=xnetLoss(xpatch,ypatch,xhat,yhat,xcycle,ycycle,self.parameters)
        return {"loss": tot_loss}
    
    def __call__(self,data,training=False):
        xpatch,ypatch=data
        xhat=self.yxnet(ypatch)
        yhat=self.xynet(xpatch)
        if training:
            xcycle=self.yxnet(yhat)
            ycycle=self.xynet(xhat)
            return yhat,xhat,xcycle,ycycle
        else:
            return yhat,xhat
            


# In[8]:


LEARNING_RATE = 10e-4
EPOCHS = 1000
BATCH_SIZE = 32


# In[9]:


netSarOpt=network(sen12_specs_X_to_Y,"SarOpt")
netOptSar=network(sen12_specs_Y_to_X,"OptSar")
xnet=Xnet(xynet=netSarOpt,yxnet=netOptSar,inputs=[netSarOpt.input,netOptSar.input],outputs=[netSarOpt.output,netOptSar.output],name="Sen12_Xnet")
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
xnet.compile(optimizer=optimizer,loss_fn=xnetLoss)
tf.keras.utils.plot_model(xnet, to_file="model.png", show_shapes=True)


# In[12]:


list_IDs,_ = getListOfFiles("Data/Sen12/Sar")
train_list_IDs ,valid_list_IDs = train_test_split(list_IDs,test_size=0.2,shuffle=True)
saropt_train_gen=Sen12DataGenerator(train_list_IDs,SEN12_OPT_PATH,SEN12_SAR_PATH,BATCH_SIZE,(PATCH_SIZE,PATCH_SIZE),CHANNELS_SAR,CHANNELS_OPT,True,False)
saropt_valid_gen=Sen12DataGenerator(valid_list_IDs,SEN12_OPT_PATH,SEN12_SAR_PATH,BATCH_SIZE,(PATCH_SIZE,PATCH_SIZE),CHANNELS_SAR,CHANNELS_OPT,True,False)


# In[13]:


cwd = os.getcwd()

exps_dir = os.path.join(cwd, 'experiments_dir')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

exp_name = 'sen12_xnet'

exp_dir = os.path.join(exps_dir, exp_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
callbacks = []
# Model checkpoint
# ----------------
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
              filepath=os.path.join(ckpt_dir,'mymodel_{epoch}'),
              save_best_only=False,
              save_weights_only=True,
              verbose=1)

callbacks.append(ckpt_callback)

# ----------------

# Visualize Learning on Tensorboard
# ---------------------------------
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
    
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)

es_callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
callbacks.append(es_callback)


lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.1,
                                                   patience=2,
                                                   min_lr=0.000001,
                                                   verbose=1)
callbacks.append(lr_callback)

xnet.fit(saropt_train_gen,
         validation_data=saropt_valid_gen,
         epochs=EPOCHS,
         callbacks=callbacks)

