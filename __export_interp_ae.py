#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:18:10 2021

@author: adrienbitton
"""



import numpy as np
import torch
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.use('Agg') # for the server
from matplotlib import pyplot as plt
import os
import argparse
import timeit
import gin
import shutil
import gc


from __nn_utils import load_data
from __nn_utils_ae import AE, export_interp, latent_visualiser


# TODO: merge export scripts into one
# make configurable for data path etc. if plotting latents too ..

## INTERPOLATION EXPORT SCRIPT FOR THE AEs+GANs


gin.enter_interactive_mode()
device = 'cuda' if torch.cuda.is_available() else 'cpu'



pretrained_dir = "./outputs/"

mnames = ["CAN_12_AE_GAN_3d_BN_BN_crop0_Bye"]

wav_dir = '/Users/adrienbitton/Desktop/drum_beat_gan/raster_beats/Byetone/1bar_120bpm/'
artists_sel = []
bs = 8

for mname in mnames:
    
    mpath = pretrained_dir+mname+"/"
    export_path = mpath+"play_0/"
    
    os.makedirs(export_path)
    
    gin_file = os.path.join(mpath,mname+'_configs.gin')
    args = np.load(os.path.join(mpath,mname+'_args.npy'),allow_pickle=True).item()
    
    
    print("gin.parse_config_file",gin_file)
    with gin.unlock_config():
        gin.parse_config_file(gin_file)
    
    target_sr = gin.query_parameter('%target_sr')
    target_dur = gin.query_parameter('%target_dur')
    
    
    ae = AE()
    ae.load_state_dict(torch.load(os.path.join(mpath,mname+'_AE.pt'), map_location='cpu'))
    ae.encoder.remove_weight_norm()
    ae.decoder.remove_weight_norm()
    
    ae.to(device)
    ae.decoder.device = device
    ae.eval()
    
    
    export_interp(ae,20,20,export_path,"",verbose=True) # by default 20 steps == 40sec
    
    
    train_dataloader,test_dataloader = load_data(wav_dir,bs,target_sr,target_dur,artists_sel)
    latent_visualiser(ae,train_dataloader,test_dataloader,save_path=export_path)















