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


from __nn_utils import MultiScaleDiscriminator, load_data, print_time          
from __nn_utils import apply_zero_grad,enable_disc_disable_gen,enable_gen_disable_disc,disable_all
from __nn_utils_ae import AE,stft_dist,__ae_loss,__d_loss,gradient_check,export_samples,export_interp,latent_visualiser


## TRAINING SCRIPT FOR THE AEs+GANs

# TODO: add chan out in encoder sub-modules (discriminators) == flatten in larger z
# TODO: export test reconstructions too


torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


###############################################################################
## args

parser = argparse.ArgumentParser()
parser.add_argument('--mname',    type=str,   default="test")
parser.add_argument('--config',    type=str,   default="./gin_configs/AE_GAN_3scales_BN_BN_crop0.gin")
parser.add_argument('--wav_dir',    type=str,   default="/fast-1/datasets/raster_16k/output_1bar_120bpm/")
parser.add_argument('--artists_sel', nargs='+', default=[])
parser.add_argument('--N_epochs',    type=int,   default=300)
parser.add_argument('--bs',    type=int,   default=16)
parser.add_argument('--lr_g',    type=float,   default=0.0001)
parser.add_argument('--lr_d',    type=float,   default=0.0001)
parser.add_argument('--export_step',    type=int,   default=1)
parser.add_argument('--e_interp',    type=int,   default=0) # export interpolation
parser.add_argument('--checkpoint_path',    type=str,   default="none")
args = parser.parse_args()
print(args)


if args.checkpoint_path=="none":
    gin_file = args.config
else:
    gin_file = args.checkpoint_path+'_configs.gin'
print("gin.parse_config_file",gin_file)
with gin.unlock_config():
    gin.parse_config_file(gin_file)


print("RUNNING DEVICE IS ",device)


mname = args.mname
wav_dir = args.wav_dir
artists_sel = args.artists_sel

N_epochs = args.N_epochs
bs = args.bs
lr_g = args.lr_g
lr_d = args.lr_d

beta1 = gin.query_parameter('%beta1')
beta2 = gin.query_parameter('%beta2')

target_sr = gin.query_parameter('%target_sr')
target_dur = gin.query_parameter('%target_dur')

feat_match = gin.query_parameter("%feat_match")

use_stft_dist = gin.query_parameter('%use_stft_dist')
if use_stft_dist is True:
    rec_loss = stft_dist()
else:
    rec_loss = None

target_reg_weight = gin.query_parameter('%target_reg_weight')
if target_reg_weight>0:
    # regularized auto-encoder
    warmup_reg_weight = gin.query_parameter('%warmup_reg_weight')
    if warmup_reg_weight is True and args.checkpoint_path=="none":
        epoch_reg_weight = np.zeros(N_epochs)
        epoch_reg_weight[int(0.3*N_epochs):int(0.8*N_epochs)] = np.linspace(0.,target_reg_weight,num=int(0.8*N_epochs)-int(0.3*N_epochs))
        epoch_reg_weight[int(0.8*N_epochs):] = target_reg_weight
    else:
        # either no warmup or training continuation
        epoch_reg_weight = np.zeros(N_epochs)+target_reg_weight
else:
    epoch_reg_weight = np.zeros(N_epochs)

export_step = args.export_step
e_interp = bool(args.e_interp)


###############################################################################
## data

train_dataloader,test_dataloader = load_data(wav_dir,bs,target_sr,target_dur,artists_sel)


###############################################################################
## models

ae = AE()
if args.checkpoint_path!="none":
    ae.load_state_dict(torch.load(args.checkpoint_path+'_AE.pt', map_location='cpu'))
    print("loaded pretrained AE weights in ",args.checkpoint_path+'_AE.pt')
ae.to(device)
ae.decoder.device = device

discriminator = MultiScaleDiscriminator()
if args.checkpoint_path!="none":
    discriminator.load_state_dict(torch.load(args.checkpoint_path+'_Disc.pt', map_location='cpu'))
    print("loaded pretrained Discriminators weights in ",args.checkpoint_path+'_Disc.pt')
discriminator.to(device)
discriminator.device = device

optim_ae = torch.optim.Adam(ae.parameters(),
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

gradient_check(ae,discriminator,optim_ae,optim_d,train_dataloader,feat_match,rec_loss,target_reg_weight)
# gc.collect()
# torch.cuda.empty_cache()

# TODO: check bias gradients in discriminator


###############################################################################
## training

start_time = timeit.default_timer()

total_iter = 0

# TODO: clean that and fix tensorboard install
train_losses_g = []
train_losses_reg = []
train_losses_d = []
test_losses_g = []
test_losses_reg = []
test_losses_d = []
train_losslog_g = 0
train_losslog_reg = 0
train_losslog_d = 0
test_losslog_g = 0
test_losslog_reg = 0
test_losslog_d = 0
tr_count = 0
te_count = 0


for epoch in range(N_epochs):
    
    #### training epoch
    ae.train()
    discriminator.train()
    for _,mb in enumerate(train_dataloader):
        
        # discriminator step
        enable_disc_disable_gen(ae,discriminator)
        real_audio = mb[0].unsqueeze(1).to(device)
        fake_audio,_ = ae.forward(real_audio)
        fake_audio = fake_audio.detach()
        apply_zero_grad(ae,optim_ae,discriminator,optim_d)
        loss_d = __d_loss(discriminator,fake_audio,real_audio)
        loss_d.backward()
        optim_d.step()
        
        # generator step
        enable_gen_disable_disc(ae,discriminator)
        apply_zero_grad(ae,optim_ae,discriminator,optim_d)
        loss_g_tot,_,_,loss_reg = __ae_loss(ae,discriminator,real_audio,feat_match=feat_match,rec_loss=rec_loss,reg_weight=epoch_reg_weight[epoch])
        loss_g_tot.backward()
        optim_ae.step()
        
        train_losslog_g += loss_g_tot.item()
        if loss_reg>0:
            train_losslog_reg += loss_reg.item()
        train_losslog_d += loss_d.item()
        tr_count += 1
        total_iter += 1
        if (total_iter+1)%100==0:
            h, m, s = print_time(timeit.default_timer()-start_time)
            print('current total iterations '+str(total_iter+1)+' and elapsed time '+'%d:%02d:%02d' % (h, m, s))
            # gc.collect()
            # torch.cuda.empty_cache()
    
    #### test epoch
    ae.eval()
    discriminator.eval()
    disable_all(ae,discriminator)
    apply_zero_grad(ae,optim_ae,discriminator,optim_d)
    for _,mb in enumerate(test_dataloader):
        with torch.no_grad():
            
            # discriminator step
            real_audio = mb[0].unsqueeze(1).to(device)
            fake_audio,_ = ae.forward(real_audio)
            loss_d = __d_loss(discriminator,fake_audio,real_audio)
            
            # generator step
            loss_g_tot,_,_,loss_reg = __ae_loss(ae,discriminator,real_audio,feat_match=feat_match,rec_loss=rec_loss,reg_weight=epoch_reg_weight[epoch])
            
            test_losslog_g += loss_g_tot.item()
            if loss_reg>0:
                test_losslog_reg += loss_reg.item()
            test_losslog_d += loss_d.item()
            te_count += 1
    
    train_losses_g.append(train_losslog_g/tr_count)
    train_losses_reg.append(train_losslog_reg/tr_count)
    train_losses_d.append(train_losslog_d/tr_count)
    test_losses_g.append(test_losslog_g/te_count)
    test_losses_reg.append(test_losslog_reg/te_count)
    test_losses_d.append(test_losslog_d/te_count)
    
    h, m, s = print_time(timeit.default_timer()-start_time)
    print("\n"+mname+'  elapsed time = '+"%d:%02d:%02d" % (h, m, s)+'   out of #epochs = '+str(N_epochs))
    print('training and test losses (G,REG,D) at epoch '+str(epoch)+' and iteration '+str(total_iter))
    print(train_losses_g[-1],train_losses_reg[-1],train_losses_d[-1])
    print(test_losses_g[-1],test_losses_reg[-1],test_losses_d[-1])
    
    train_losslog_g = 0
    train_losslog_reg = 0
    train_losslog_d = 0
    test_losslog_g = 0
    test_losslog_reg = 0
    test_losslog_d = 0
    tr_count = 0
    te_count = 0
    
    if epoch%export_step==0:
        print("\nintermediate export")
        ae.eval()
        
        for _,mb in enumerate(train_dataloader):
            export_samples(ae,mb[0].unsqueeze(1).to(device),export_dir,"") # both reconstructions and random samples
            break
        
        if e_interp is True:
            export_interp(ae,bs,20,export_dir,"") # by default 20 steps == 40sec
        
        torch.save(ae.state_dict(), mpath+mname+"_AE.pt")
        torch.save(discriminator.state_dict(), mpath+mname+"_Disc.pt")
        
        plt.figure()
        plt.suptitle("rows = train_losses/test_losses, cols=  generator/regularizer/discriminator")
        plt.subplot(231)
        plt.plot(train_losses_g)
        plt.subplot(232)
        plt.plot(train_losses_reg)
        plt.subplot(233)
        plt.plot(train_losses_d)
        plt.subplot(234)
        plt.plot(test_losses_g)
        plt.subplot(235)
        plt.plot(test_losses_reg)
        plt.subplot(236)
        plt.plot(test_losses_d)
        plt.savefig(mpath+mname+"losses.pdf")
        plt.close("all")
        
        latent_visualiser(ae,train_dataloader,test_dataloader,save_path=export_dir)
    
    # gc.collect()
    # torch.cuda.empty_cache()



###############################################################################
## final export

torch.save(ae.state_dict(), mpath+mname+"_AE.pt")
torch.save(discriminator.state_dict(), mpath+mname+"_Disc.pt")

plt.figure()
plt.suptitle("rows = train_losses/test_losses, cols=  generator/regularizer/discriminator")
plt.subplot(231)
plt.plot(train_losses_g)
plt.subplot(232)
plt.plot(train_losses_reg)
plt.subplot(233)
plt.plot(train_losses_d)
plt.subplot(234)
plt.plot(test_losses_g)
plt.subplot(235)
plt.plot(test_losses_reg)
plt.subplot(236)
plt.plot(test_losses_d)
plt.savefig(mpath+mname+"losses.pdf")
plt.close("all")

plt.figure()
plt.suptitle("rows = train_losses/test_losses, cols=  generator/regularizer/discriminator")
plt.subplot(231)
plt.plot(train_losses_g[N_epochs//2:])
plt.subplot(232)
plt.plot(train_losses_reg[N_epochs//2:])
plt.subplot(233)
plt.plot(train_losses_d[N_epochs//2:])
plt.subplot(234)
plt.plot(test_losses_g[N_epochs//2:])
plt.subplot(235)
plt.plot(test_losses_reg[N_epochs//2:])
plt.subplot(236)
plt.plot(test_losses_d[N_epochs//2:])
plt.savefig(mpath+mname+"losses_end.pdf")
plt.close("all")

latent_visualiser(ae,train_dataloader,test_dataloader,save_path=export_dir)

gradient_check(ae,discriminator,optim_ae,optim_d,train_dataloader,feat_match,rec_loss,target_reg_weight)

ae.eval()
ae.encoder.remove_weight_norm()
ae.decoder.remove_weight_norm()

for _,mb in enumerate(train_dataloader):
    export_samples(ae,mb[0].unsqueeze(1).to(device),export_dir,"final") # both reconstructions and random samples
    break
export_interp(ae,bs,20,export_dir,"final") # by default 20 steps == 40sec


