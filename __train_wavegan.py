#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 21:30:50 2021

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


from __nn_utils import load_data, export_samples, export_groundtruth, print_time
from __WaveGANGP import WaveGANGenerator, WaveGANDiscriminator, weights_init, apply_zero_grad, enable_disc_disable_gen, enable_gen_disable_disc, disable_all, wavegan_d_loss, wavegan_g_loss, gradient_check



torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


###############################################################################
## args

# TODO: export interpolations with constant step size
# TODO: add auxiliary classification of artist and conditioning generator ?
# TODO: BERGAN = train on 4/4 techno club music

# running with venv from /fast-2/adrien/torchopenl3

# --wav_dir /fast-1/datasets/raster_16k/output_1bar_120bpm/Byetone/


# CUDA_VISIBLE_DEVICES=2 python __train_wavegan.py --mname test2_WAVEGAN_drep1 --config ./gin_configs/WAVEGAN_default.gin --d_rep 1
# CUDA_VISIBLE_DEVICES=3 python __train_wavegan.py --mname test3_WAVEGAN_small_drep1 --config ./gin_configs/WAVEGAN_default_small.gin --d_rep 1    


parser = argparse.ArgumentParser()
parser.add_argument('--mname',    type=str,   default="test")
parser.add_argument('--config',    type=str,   default="./gin_configs/WAVEGAN_default_small.gin")
parser.add_argument('--wav_dir',    type=str,   default="/fast-1/datasets/raster_16k/output_1bar_120bpm/")
parser.add_argument('--N_epochs',    type=int,   default=800)
parser.add_argument('--d_rep',    type=int,   default=1)
parser.add_argument('--bs',    type=int,   default=16)
parser.add_argument('--export_step',    type=int,   default=2)
parser.add_argument('--n_export',    type=int,   default=32)
args = parser.parse_args()
print(args)


gin_file = args.config
print("gin.parse_config_file",gin_file)
with gin.unlock_config():
    gin.parse_config_file(gin_file)


mname = args.mname
wav_dir = args.wav_dir

N_epochs = args.N_epochs
d_rep = args.d_rep
bs = args.bs

lr_g = gin.query_parameter('%lr_g')
lr_d = gin.query_parameter('%lr_d')

beta1 = gin.query_parameter('%beta1')
beta2 = gin.query_parameter('%beta2')
gan_model = gin.query_parameter('%gan_model')
gp_weight = gin.query_parameter('%gp_weight')

target_sr = gin.query_parameter('%target_sr')
target_dur = gin.query_parameter('%target_dur')

export_step = args.export_step
n_export = args.n_export


###############################################################################
## data

train_dataloader,test_dataloader = load_data(wav_dir,bs,target_sr,target_dur)


###############################################################################
## models

generator = WaveGANGenerator()
generator.to(device)
generator.apply(weights_init)
generator.device = device

discriminator = WaveGANDiscriminator()
discriminator.to(device)
discriminator.apply(weights_init)
discriminator.device = device

optim_g = torch.optim.Adam(generator.parameters(),
        lr=lr_g, betas=(beta1, beta2))

optim_d = torch.optim.Adam(discriminator.parameters(),
        lr=lr_d, betas=(beta1, beta2))


mpath = "./"+mname+"/"
export_dir = mpath+'export/'
os.makedirs(mpath)
os.makedirs(export_dir)

np.save(os.path.join(mpath,mname+'_args.npy'),vars(args))
shutil.copy(gin_file,os.path.join(mpath,mname+'_configs.gin'))


###############################################################################
## init gradient check

gradient_check(generator,discriminator,optim_g,optim_d,gan_model,gp_weight,train_dataloader,bs)

# TODO: check zero init gradient in biases (less with BN)

gc.collect()
torch.cuda.empty_cache()


###############################################################################
## training

start_time = timeit.default_timer()

total_iter = 0

train_losses_g = []
train_losses_d = []
test_losses_g = []
test_losses_d = []
train_losslog_g = 0
train_losslog_d = 0
test_losslog_g = 0
test_losslog_d = 0
tr_count = 0
te_count = 0


for epoch in range(N_epochs):
    
    #### training epoch
    generator.train()
    discriminator.train()
    for _,mb in enumerate(train_dataloader):
        
        # discriminator step
        enable_disc_disable_gen(generator,discriminator)
        real_audio = mb[0].unsqueeze(1).to(device)
        loss_tot_d = 0.0
        for di in range(d_rep):
            fake_audio = generator(bs).unsqueeze(1)
            apply_zero_grad(generator,optim_g,discriminator,optim_d)
            loss_d,loss_gp = wavegan_d_loss(discriminator, gp_weight, fake_audio, real_audio, disable_GP=False)
            loss_d = loss_d+loss_gp
            loss_d.backward()
            optim_d.step()
            # loss_tot_d += loss_d
            loss_tot_d += loss_d.item()
        
        # generator step
        apply_zero_grad(generator,optim_g,discriminator,optim_d)
        enable_gen_disable_disc(generator,discriminator)
        loss_g = wavegan_g_loss(generator,discriminator,bs)
        loss_g.backward()
        optim_g.step()
        
        # train_losslog_g += loss_g
        train_losslog_g += loss_g.item()
        train_losslog_d += loss_tot_d/d_rep
        tr_count += 1
        total_iter += 1
        if (total_iter+1)%100==0:
            h, m, s = print_time(timeit.default_timer()-start_time)
            print('current total iterations '+str(total_iter+1)+' and elapsed time '+'%d:%02d:%02d' % (h, m, s))
    
    #### test epoch
    disable_all(generator,discriminator)
    apply_zero_grad(generator,optim_g,discriminator,optim_d)
    generator.eval()
    discriminator.eval()
    for _,mb in enumerate(test_dataloader):
        with torch.no_grad():
            
            # discriminator step
            real_audio = mb[0].unsqueeze(1).to(device)
            fake_audio = generator(bs).unsqueeze(1)
            loss_d,_ = wavegan_d_loss(discriminator, gp_weight, fake_audio, real_audio, disable_GP=True)
            
            # generator step
            loss_g = wavegan_g_loss(generator,discriminator,bs)
            
            # test_losslog_g += loss_g
            # test_losslog_d += loss_d
            test_losslog_g += loss_g.item()
            test_losslog_d += loss_d.item()
            te_count += 1
    
    # train_losses_g.append(train_losslog_g.item()/tr_count)
    # train_losses_d.append(train_losslog_d.item()/tr_count)
    # test_losses_g.append(test_losslog_g.item()/te_count)
    # test_losses_d.append(test_losslog_d.item()/te_count)
    train_losses_g.append(train_losslog_g/tr_count)
    train_losses_d.append(train_losslog_d/tr_count)
    test_losses_g.append(test_losslog_g/te_count)
    test_losses_d.append(test_losslog_d/te_count)
    
    h, m, s = print_time(timeit.default_timer()-start_time)
    print("\n"+mname+'  elapsed time = '+"%d:%02d:%02d" % (h, m, s)+'   out of #epochs = '+str(N_epochs))
    print('training and test losses (G,D) at epoch '+str(epoch)+' and iteration '+str(total_iter))
    print(train_losses_g[-1],train_losses_d[-1])
    print(test_losses_g[-1],test_losses_d[-1])
    
    train_losslog_g = 0
    train_losslog_d = 0
    test_losslog_g = 0
    test_losslog_d = 0
    tr_count = 0
    te_count = 0
    
    if epoch%export_step==0:
        print("\nintermediate export")
        generator.eval()
        export_samples(generator,n_export,export_dir,"")
        for _,mb in enumerate(train_dataloader):
            export_groundtruth(mb[0],target_sr,export_dir,"")
            break
        
        torch.save(generator.state_dict(), mpath+mname+"_Gene.pt")
        torch.save(discriminator.state_dict(), mpath+mname+"_Disc.pt")
        
        plt.figure()
        plt.suptitle("rows = train_losses/test_losses, cols=  generator/discriminator")
        plt.subplot(221)
        plt.plot(train_losses_g)
        plt.subplot(222)
        plt.plot(train_losses_d)
        plt.subplot(223)
        plt.plot(test_losses_g)
        plt.subplot(224)
        plt.plot(test_losses_d)
        plt.savefig(mpath+mname+"losses.pdf")
        plt.close("all")
    
    gc.collect()
    torch.cuda.empty_cache()



###############################################################################
## final export

torch.save(generator.state_dict(), mpath+mname+"_Gene.pt")
torch.save(discriminator.state_dict(), mpath+mname+"_Disc.pt")

plt.figure()
plt.suptitle("rows = train_losses/test_losses, cols=  generator/discriminator")
plt.subplot(221)
plt.plot(train_losses_g)
plt.subplot(222)
plt.plot(train_losses_d)
plt.subplot(223)
plt.plot(test_losses_g)
plt.subplot(224)
plt.plot(test_losses_d)
plt.savefig(mpath+mname+"losses.pdf")
plt.close("all")

plt.figure()
plt.suptitle("rows = train_losses/test_losses, cols=  generator/discriminator")
plt.subplot(221)
plt.plot(train_losses_g[N_epochs//2:])
plt.subplot(222)
plt.plot(train_losses_d[N_epochs//2:])
plt.subplot(223)
plt.plot(test_losses_g[N_epochs//2:])
plt.subplot(224)
plt.plot(test_losses_d[N_epochs//2:])
plt.savefig(mpath+mname+"losses_end.pdf")
plt.close("all")

gradient_check(generator,discriminator,optim_g,optim_d,gan_model,gp_weight,train_dataloader,bs)

generator.eval()

export_samples(generator,n_export,export_dir,"final")
for _,mb in enumerate(train_dataloader):
    export_groundtruth(mb[0],target_sr,export_dir,"")
    break




