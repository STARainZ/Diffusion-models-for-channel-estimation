
import numpy as np
import torch, sys, os, itertools, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm

from loaders import Channels
from dotmap import DotMap
from torch.utils.data import DataLoader
from scipy.linalg import dft, circulant

import DMCE

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='quadriga')
parser.add_argument('--test', type=str, default='quadriga')
parser.add_argument('--save_channels', type=int, default=0)
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[0.6])
parser.add_argument('--verbose', nargs='+', type=int, default=1)  # 0: tqdm 1: print gradient
parser.add_argument('--antennas', nargs='+', type=int, default=[16, 64])
parser.add_argument('--model', type=str, default='')
parser.add_argument('--domain', type=str, default='angle')
parser.add_argument('--reverse_add_random', type=bool, default=False)
parser.add_argument('--model_file', type=str, default='final_model.pt')
parser.add_argument('--param_file', type=str, default='sim_params')
parser.add_argument('--los_type', type=str, default='/LOS')
parser.add_argument('--num_timesteps', nargs='+', type=int, default=100)
parser.add_argument('--condition_method', type=str, default='DMPS')
parser.add_argument('--ddim', type=bool, default=False)
parser.add_argument('--infer_timesteps', nargs='+', type=int, default=100)
parser.add_argument('--save_h', type=bool, default=False)
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

num_timesteps = args.num_timesteps  # diff_model_dict['num_timesteps']

# Target file
domain_dir = '/angle_domain' if args.domain == 'angle' else ''
target_dir  = './models%s/diffusion_cnn/%s%s' % (domain_dir, args.train, args.los_type)
target_file = os.path.join(target_dir, args.model_file)
contents    = torch.load(target_file, map_location=device)
config      = DotMap()
config.data.image_size = args.antennas
config.data.channel        = args.train
config.data.channels       = 2  # {Re, Im}
config.data.noise_std      = 0
config.data.num_pilots     = config.data.image_size[1]
config.data.norm_channels  = 'global'
config.data.spacing_list   = [0.5]  # Training and validation

# load model parameter dictionaries
sim_params = DMCE.utils.load_params(os.path.join(target_dir, args.param_file))
cnn_dict = sim_params['unet_dict']
diff_model_dict = sim_params['diff_model_dict']
diff_model_dict['reverse_add_random'] = args.reverse_add_random
diff_model_dict['data_shape'] = tuple([2] + args.antennas)
cnn_dict['device'] = device

if args.ddim:
    diff_model_dict['reverse_method'] = 'ground_truth'  # DDIM


class DMPS():
    def __init__(self, alpha_t, alpha_bar):
        self.scale = 0.8
        self.alpha_bar = np.clip(alpha_bar, 1e-16, 1 - 1e-16)
        self.alpha_t = np.clip(alpha_t, 1e-16, 1 - 1e-16)
        self.scale_S = (1 - self.alpha_bar) / self.alpha_bar
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        if args.ddim is False:
            self.coef = (1 - self.alpha_t) / np.sqrt(self.alpha_t)
        else:
            self.coef = (1 - self.alpha_bar) / np.sqrt(self.alpha_t) - \
                        np.sqrt(1 - self.alpha_bar) * np.sqrt(1 - np.concatenate(([1], self.alpha_bar[:-1])))

    def conditioning(self, x_t, y, A_funcs, local_noise, t, x_prev, eps_t, **kwargs):
        S = A_funcs['S2']
        S_vector = 1 / (self.scale_S[t] * S + local_noise).reshape(-1, 1)
        # depend whether to use x0 for likelihood gradient based on SNR
        # use_x0 = 10 * np.log10(config.data.image_size[1] / local_noise) > 15
        use_x0 = True
        if use_x0:
            x0 = (x_prev - np.sqrt(1 - self.alpha_bar[t]) * eps_t) / np.sqrt(self.alpha_bar[t])
            temp = torch.matmul(A_funcs['U'].conj().t(), y - torch.matmul(A_funcs['A'], x0) / self.sqrt_alpha_bar[t])
        else:  # note: use xt instead of x_{t-1} for better empirical performance
            temp = torch.matmul(A_funcs['U'].conj().t(), y - torch.matmul(A_funcs['A'], x_t) / self.sqrt_alpha_bar[t])
        grad_value = torch.matmul(A_funcs['A'].conj().t(),
                                  torch.matmul(A_funcs['U'], S_vector * temp)) / self.sqrt_alpha_bar[t]
        x_t += self.scale * 2 * grad_value * self.coef[t]
        return x_t


# Instantiate model
cnn = DMCE.CNN(**cnn_dict)
diffusion_model = DMCE.DiffusionModel(cnn, **diff_model_dict)

# load the parameters of the pre-trained model into the DiffusionModel instance
diffusion_model.cuda()
diffusion_model.load_state_dict(contents['model'])
diffusion_model.eval()
conditioning_method = DMPS(alpha_t=diffusion_model.alphas.cpu().numpy(),
                           alpha_bar=diffusion_model.alphas_cumprod.cpu().numpy())

# Train and validation seeds
train_seed, val_seed = 1234, 4321
# Get training dataset for normalization
dataset = Channels(train_seed, config, norm=config.data.norm_channels if args.test != 'quadriga' else [0, 1],
                   length=10000)

# Range of SNR, test channels and hyper-parameters
snr_range          = np.arange(30, 32.5, 5)
spacing_range      = np.asarray(args.spacing) # From a pre-defined index
pilot_alpha_range  = np.asarray(args.pilot_alpha)
noise_range        = 10 ** (-snr_range / 10.) * config.data.image_size[1]
# Number of validation channels
num_channels = 100
# FFT matrix
A_R = dft(config.data.image_size[0]) / np.sqrt(config.data.image_size[0])
A_T = dft(config.data.image_size[1]) / np.sqrt(config.data.image_size[1])
A_R, A_T = torch.tensor(A_R, dtype=torch.complex64, device=device),\
           torch.tensor(A_T, dtype=torch.complex64, device=device)

# scale_range = [2.0, 2.0, 0.8, 0.6, 0.6]  # for T100 R3, alpha0.2-1, 30 dB
# scale_range = [2.0, 2.0, 1.5, 0.6, 0.6]  # for T100, alpha0.2-1, 30 dB
scale_range = [2.0, 2.0, 1.5, 0.6, 0.6]  # for T1000, alpha0.2-1, 30 dB

# Global results
nmse_log = np.zeros((len(spacing_range), len(pilot_alpha_range),
                     len(snr_range), int(num_timesteps), num_channels))
result_dir = './results/%sx%s/diffusion_cnn/train-%s_test-%s%s' % (args.antennas[0], args.antennas[1],
    args.train, args.test, domain_dir)
os.makedirs(result_dir, exist_ok=True)

# Wrap sparsity, steps and spacings
meta_params = itertools.product(spacing_range, pilot_alpha_range)

if args.save_h:
    saved_H = np.zeros((len(spacing_range), len(pilot_alpha_range),
                        len(snr_range), num_channels, config.data.image_size[0], config.data.image_size[1]),
                       dtype=np.complex64)
else:
    saved_H = []
    oracle_H = []

# For each hyper-combo
for meta_idx, (spacing, pilot_alpha) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, pilot_alpha_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(pilot_alpha_range)))

    # conditioning_method.scale = scale_range[pilot_alpha_idx]

    # Get validation dataset
    val_config = copy.deepcopy(config)
    val_config.data.channel = args.test
    val_config.data.spacing_list = [spacing]
    val_config.data.num_pilots = int(np.floor(config.data.image_size[1] * pilot_alpha))
    val_dataset = Channels(val_seed, val_config,
                           norm=[dataset.mean, dataset.std] if args.test != 'quadriga' else [0, 1])
    val_loader = DataLoader(val_dataset, batch_size=num_channels,
                            shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader)
    print('There are %d validation channels' % len(val_dataset))

    # Get all validation pilots and channels
    val_sample = next(val_iter)
    val_P = val_sample['P'].cuda()
    val_H = val_sample['H'].cuda()
    val_H = val_H[:, 0] + 1j * val_H[:, 1]

    # Save oracle once
    if args.save_h:
        oracle_H = val_H.cpu().numpy()

    # Initial estimates
    init_val_H = torch.randn_like(val_H)

    # should use fixed P for each channel realization
    val_P = torch.tile(val_P[0].unsqueeze(0), (num_channels, 1, 1))  # QPSK
    # val_P = torch.tile(torch.eye(args.antennas[1], dtype=val_P.dtype,
    #                              device=val_P.device).view(1, args.antennas[1], args.antennas[1]), (num_channels, 1, 1))
    # val_P = torch.tile(torch.tensor(dft(config.data.image_size[1])[:, :val_config.data.num_pilots], dtype=val_P.dtype),
    #                    (num_channels, 1, 1)).to(val_P.device)  # DFT
    # val_P_numpy = circulant(np.exp(1j*np.arange(config.data.image_size[1]) ** 2 * np.pi /
    #                                config.data.image_size[1]))[:, :val_config.data.num_pilots]
    # val_P = torch.tile(torch.tensor(val_P_numpy, dtype=val_P.dtype), (num_channels, 1, 1)).to(val_P.device)  # zadoff-chu
    # resolve_conj() only when using dft
    A = torch.tensor(np.kron(val_P[0].t().cpu().numpy(),
                             np.eye(config.data.image_size[0])), dtype=val_P.dtype).to(val_P.device)  # (nr*np, nr*nt)
    if args.domain == 'angle':
        A = torch.matmul(A, torch.kron(A_T.conj(), A_R))
    U, S, Vh = torch.linalg.svd(A)
    A_funcs = {'A': A, 'U': U, 'S': S, 'V': Vh.conj().t(), 'S2': S * S}

    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        # conditioning_method.scale = scale_range[snr_idx]

        # Get received pilots at correct SNR
        val_Y = torch.matmul(val_H, val_P)
        val_Y = val_Y + \
                np.sqrt(local_noise) * torch.randn_like(val_Y)  # randn_like -> var=1 (not 2)
        current = init_val_H.clone()
        y = torch.reshape(torch.transpose(val_Y, -1, -2), (num_channels, -1, 1))
        norm = [0., 1.]
        oracle = torch.matmul(A_R.conj().t(),
                              torch.matmul(val_H, A_T)) if args.domain == 'angle' else val_H  # Ground truth channels
        # Count every step
        trailing_idx = 0

        skip = num_timesteps // args.infer_timesteps
        seq = range(0, num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        t_start = num_timesteps  # todo: plug into the right place
        if args.verbose == 0:
            bar = tqdm(zip(reversed(seq), reversed(seq_next)))
        else:
            bar = zip(reversed(seq), reversed(seq_next))
        round = 1
        # current_sum = 0
        for step_idx, next_t in bar:
            if step_idx == (t_start - 1) // 2:
                round = 10
            for k in range(round):
                current_real = torch.view_as_real(current).permute(0, 3, 1, 2).requires_grad_(True)  # (bs, 2, nr, nt)

                out = diffusion_model.reverse_step(x_t=current_real, t=step_idx,
                                                   add_random=diff_model_dict['reverse_add_random'])

                ht_cplx = torch.transpose(torch.view_as_complex(out['sample'].permute(0, 2, 3, 1).contiguous()),
                                          -1, -2).reshape(num_channels, -1, 1)
                # x0 = torch.transpose(torch.view_as_complex(out['x0'].permute(0, 2, 3, 1).contiguous()),
                #                      -1, -2).reshape(num_channels, -1, 1)
                x_prev = torch.transpose(current, -1, -2).reshape(num_channels, -1, 1)
                eps_t = torch.transpose(torch.view_as_complex(out['eps_t'].permute(0, 2, 3, 1).contiguous()),
                                        -1, -2).reshape(num_channels, -1, 1)
                ht_cplx = conditioning_method.conditioning(ht_cplx, y, A_funcs, local_noise, step_idx,
                                                           x_prev, eps_t)

                current = torch.transpose(ht_cplx.view(-1, config.data.image_size[1], config.data.image_size[0]),
                                          -1, -2).detach()
                # if step_idx < 10:
                #     current_sum += current
            # Store loss
            nmse_log[spacing_idx, pilot_alpha_idx, snr_idx, trailing_idx] = \
                (torch.sum(torch.square(torch.abs(current - oracle)), dim=(-1, -2)) /
                 torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).cpu().numpy()

            if args.verbose == 1:
                print("recovered value: class: {}, mean {}, nmse {} dB".format(trailing_idx,
                                                                               current.abs().mean(),
                                                                               10 * np.log10(np.mean(nmse_log[
                                                                                                         spacing_idx, pilot_alpha_idx, snr_idx, trailing_idx]))))
            trailing_idx += 1

        if args.save_h:
            current_spatial = torch.matmul(A_R,
                              torch.matmul(current, A_T.conj().t())) if args.domain == 'angle' else current
            saved_H[spacing_idx, pilot_alpha_idx, snr_idx] = copy.deepcopy(current_spatial.cpu().numpy())

# Use average estimation error to select best number of steps
avg_nmse = np.mean(nmse_log, axis=-1)
best_nmse = np.min(avg_nmse, axis=-1)  # select the best

# Save results to file based on noise
save_dict = {'nmse_log': nmse_log,
             'avg_nmse': avg_nmse,
             'best_nmse': best_nmse,
             'spacing_range': spacing_range,
             'pilot_alpha_range': pilot_alpha_range,
             'snr_range': snr_range,
             'val_config': val_config,
             'saved_H': saved_H,
             'oracle_H': oracle_H,
             }
if args.save_h:
    import scipy.io as sio
    sio.savemat(os.path.join(result_dir, 'results.mat'), save_dict)
else:
    torch.save(save_dict, os.path.join(result_dir, 'results.pt'))
