
import numpy as np
import torch, sys, os, copy, argparse
import datetime
import matplotlib.pyplot as plt
sys.path.append('./')

from loaders          import Channels
from dotmap           import DotMap

import DMCE
from DMCE.utils import cmplx2real

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--train', type=str, default='quadriga')
args = parser.parse_args()

# random seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'

config = DotMap()
# Model config

# Data
config.data.channel        = args.train
config.data.channels       = 2 # {Re, Im}
config.data.noise_std      = 0
config.data.image_size     = [16, 64] # [Nt, Nr] for the transposed channel
config.data.num_pilots     = config.data.image_size[1]
config.data.norm_channels  = 'global'
config.data.spacing_list   = [0.5] # Training and validation

return_all_timesteps = False  # evaluates all intermediate MSEs
fft_pre = True  # learn channel distribution in angular domain through Fourier transform
mode = '2D'
complex_data = True

# Logging
los_type = '/LOS'
domain_dir = '/angle_domain' if fft_pre else ''
time_stamp = DMCE.utils.get_timestamp()
config.log_path = os.path.join('./models%s/diffusion_cnn/%s%s' % (domain_dir, args.train, los_type), time_stamp)
os.makedirs(config.log_path, exist_ok=True)

# Seeds for train and test datasets
train_seed, val_seed = 1234, 4321

# Get datasets and loaders for channels
num_train_samples, num_val_samples, num_test_samples = 100_000, 10_000, 10_000
dataset = Channels(train_seed, config, norm=config.data.norm_channels if args.train != 'quadriga' else [0, 1],
                   length=num_train_samples)
val_config = copy.deepcopy(config)
val_config.data.noisy = False  # validate using clean data
val_config.data.masked = False
val_dataset = Channels(val_seed, val_config,
                       norm=[dataset.mean, dataset.std] if args.train != 'quadriga' else [0, 1], length=num_val_samples)
data_train = torch.from_numpy(np.asarray(dataset.channels[:, None, :]))
data_train = cmplx2real(data_train, dim=1, new_dim=False).float()
data_val = torch.from_numpy(np.asarray(val_dataset.channels[:, None, :]))
data_val = cmplx2real(data_val, dim=1, new_dim=False).float()

bin_dir = os.path.join(os.getcwd(), 'data')
data_shape = tuple(data_train.shape[1:])

# data parameter dictionary, which is saved in 'sim_params.json'
data_dict = {
    'bin_dir': str(bin_dir),
    'num_train_samples': num_train_samples,
    'num_val_samples': num_val_samples,
    'num_test_samples': num_test_samples,
    'train_dataset': args.train,
    'test_dataset': args.train,
    'n_antennas': config.data.image_size[1],
    'mode': mode,
    'data_shape': data_shape,
    'complex_data': complex_data
}

# set diffusion model params
num_timesteps = 100  #int(np.random.choice([100, 300, 500, 1_000, 2_000, 10_000]))
loss_type = 'l2'
which_schedule = 'quad'

max_snr_dB = 40
beta_start = (1 - 10**(max_snr_dB/10) / (1 + 10**(max_snr_dB/10))) # if masked is False else 10 ** (-snrdb / 10)
if num_timesteps == 5:
    beta_end = 0.95  # -22.5dB
elif num_timesteps == 10:
    beta_end = 0.7  # -22.5dB
elif num_timesteps == 50:
    beta_end = 0.2  # -22.5dB
elif num_timesteps == 100:
    beta_end = 0.1 # -22.5dB
elif num_timesteps == 300:
    beta_end = 0.035  # -23dB
elif num_timesteps == 500:
    beta_end = 0.02 #-22dB
elif num_timesteps == 1_000:
    beta_end = 0.01 #-22dB
elif num_timesteps == 2000:
    beta_end = 0.005  # -22.15dB
elif num_timesteps == 10_000:
    beta_end = 0.001 #-24dB
else:
    beta_end = 0.035
objective = 'pred_noise'  # one of 'pred_noise' (L_n), 'pred_x_0' (L_h), 'pred_post_mean' (L_mu)
loss_weighting = False # bool(np.random.choice([True, False]))
clipping = False
reverse_method = 'reverse_mean'  # either 'reverse_mean' or 'ground_truth'
reverse_add_random = False  # True: PDF Sampling method | False: Reverse Mean Forwarding method

masks = dataset.masks
# masks, noisy, masked = None, False, False  # for training naive model

# diffusion model parameter dictionary, which is saved in 'sim_params.json'
diff_model_dict = {
    'data_shape': data_shape,
    'complex_data': complex_data,
    'loss_type': loss_type,
    'which_schedule': which_schedule,
    'num_timesteps': num_timesteps,
    'beta_start': beta_start,
    'beta_end': beta_end,
    'objective': objective,
    'loss_weighting': loss_weighting,
    'clipping': clipping,
    'reverse_method': reverse_method,
    'reverse_add_random': reverse_add_random
}

kernel_size = (3, 3)
n_layers_pre = 2
max_filter = 64
ch_layers_pre = np.linspace(start=1, stop=max_filter, num=n_layers_pre+1, dtype=int)
ch_layers_pre[0] = 2
ch_layers_pre = tuple(ch_layers_pre)
ch_layers_pre = tuple(int(x) for x in ch_layers_pre)
n_layers_post = 3
ch_layers_post = np.linspace(start=1, stop=max_filter, num=n_layers_post+1, dtype=int)
ch_layers_post[0] = 2
ch_layers_post = ch_layers_post[::-1]
ch_layers_post = tuple(ch_layers_post)
ch_layers_post = tuple(int(x) for x in ch_layers_post)
n_layers_time = 1
ch_init_time = 16
batch_norm = False
downsamp_fac = 1

# batch_norm = True
cnn_dict = {
    'data_shape': data_shape,
    'n_layers_pre': n_layers_pre,
    'n_layers_post': n_layers_post,
    'ch_layers_pre': ch_layers_pre,
    'ch_layers_post': ch_layers_post,
    'n_layers_time': n_layers_time,
    'ch_init_time': ch_init_time,
    'kernel_size': kernel_size,
    'mode': mode,
    'batch_norm': batch_norm,
    'downsamp_fac': downsamp_fac,
    'device': device,
}

# set Trainer params
batch_size = 128
lr_init = 1e-4
lr_step_multiplier = 1.0
epochs_until_lr_step = 150
num_epochs = 500
val_every_n_batches = 2000
num_min_epochs = 50
num_epochs_no_improve = 20
track_val_loss = True
track_fid_score = False
track_mmd = False
use_fixed_gen_noise = True
use_ray = False
save_mode = 'best' # newest, all
dir_result = config.log_path
timestamp = DMCE.utils.get_timestamp()

# Trainer parameter dictionary, which is saved in 'sim_params.json'
trainer_dict = {
    'batch_size': batch_size,
    'lr_init': lr_init,
    'lr_step_multiplier': lr_step_multiplier,
    'epochs_until_lr_step': epochs_until_lr_step,
    'num_epochs': num_epochs,
    'val_every_n_batches': val_every_n_batches,
    'track_val_loss': track_val_loss,
    'track_fid_score': track_fid_score,
    'track_mmd': track_mmd,
    'use_fixed_gen_noise': use_fixed_gen_noise,
    'save_mode': save_mode,
    'mode': mode,
    'dir_result': str(dir_result),
    'use_ray': use_ray,
    'complex_data': complex_data,
    'num_min_epochs': num_min_epochs,
    'num_epochs_no_improve': num_epochs_no_improve,
    'fft_pre': fft_pre,
}

# instantiate CNN, DiffusionModel, Trainer and Tester
cnn = DMCE.CNN(**cnn_dict)
diffusion_model = DMCE.DiffusionModel(cnn, **diff_model_dict)
trainer = DMCE.Trainer(diffusion_model, data_train, data_val, **trainer_dict)

# Print number of trainable parameters
print(f'Number of trainable model parameters: {diffusion_model.num_parameters}')

# other parameters dictionary, which is saved in 'sim_params.json'
misc_dict = {'num_parameters': diffusion_model.num_parameters}

# save the simulation parameters as a JSON file
sim_dict = {
    'data_dict': data_dict,
    'diff_model_dict': diff_model_dict,
    'unet_dict': cnn_dict,
    'trainer_dict': trainer_dict,
    'misc_dict': misc_dict
}

DMCE.utils.save_params(dir_result=dir_result, filename='sim_params', params=sim_dict)

# run training routine
train_dict = trainer.train()
DMCE.utils.save_params(dir_result=dir_result, filename='train_results', params=train_dict)

date_time_now = datetime.datetime.now()
date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')
file_name = os.path.join(dir_result, f'{date_time}_{args.train}_dim={config.data.image_size[0]}'
                                     f'x{config.data.image_size[1]}_valdata={num_val_samples}_'
                                     f'T={num_timesteps}_loss.png')
plt.figure()
plt.semilogy(range(1, len(train_dict['train_losses'])+1), train_dict['train_losses'], label='train-loss')
plt.semilogy(range(1, len(train_dict['val_losses'])+1), train_dict['val_losses'], label='val-loss')
#plt.plot(range(1, params['epochs'] + 1), losses_all_test, label='val-loss')
plt.legend(['train-loss', 'val-loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(file_name)
