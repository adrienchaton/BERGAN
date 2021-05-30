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


from __nn_utils import Generator, export_interp
from __WaveGANGP import WaveGANGenerator



gin.enter_interactive_mode()
device = 'cuda' if torch.cuda.is_available() else 'cpu'



pretrained_dir = "./outputs/"

mname = "CAN_02_WGANGP_3d_BN_WN_crop0_drep1"

mpath = pretrained_dir+mname+"/"
export_path = mpath+"play_0/"

os.makedirs(export_path)

gin_file = os.path.join(mpath,mname+'_configs.gin')
args = np.load(os.path.join(mpath,mname+'_args.npy'),allow_pickle=True).item()


print("gin.parse_config_file",gin_file)
with gin.unlock_config():
    gin.parse_config_file(gin_file)


try:
    gan_model = gin.query_parameter('%gan_model')
except:
    gan_model = "WAVEGAN"
    
if gan_model!="WAVEGAN":
    generator = Generator()
    generator.load_state_dict(torch.load(os.path.join(mpath,mname+'_Gene.pt'), map_location='cpu'))
    generator.remove_weight_norm()
else:
    generator = WaveGANGenerator()
    generator.load_state_dict(torch.load(os.path.join(mpath,mname+'_Gene.pt'), map_location='cpu'))

generator.to(device)
generator.device = device
generator.eval()


export_interp(generator,10,20,export_path,"",verbose=True)















