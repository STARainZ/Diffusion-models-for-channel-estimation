#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, hdf5storage
from torch.utils.data import Dataset
import numpy as np
from scipy.linalg import dft

class Channels(Dataset):
    """MIMO Channels"""

    def __init__(self, seed, config, norm=None, domain=None, length=10000):
        # Get spacings
        target_spacings = config.data.spacing_list
        target_channel  = config.data.channel
        # Output channels
        self.channels  = []
        self.spacings  = np.copy(target_spacings)
        self.filenames = []
        self.masks = None
        self.toeptx = None
        self.toeprx = None

        print('Load channels ...')
        # For each spacing
        for spacing in target_spacings:
            # Get local filename
            if target_channel == '3gpp':
                filename = './data/%s_Nt%s_Nr%s_ULA%.2f_seed%d.npy' % (
                target_channel, config.data.image_size[1], config.data.image_size[0], spacing, seed)
                self.filenames.append(filename)
                channels = np.load(filename)
                channels = np.reshape(channels, (-1, config.data.image_size[0], config.data.image_size[1]), 'F')
                f_topetx = './data/%s_Nt%s_Nr%s_ULA%.2f_seed%d_toeptx.npy' % (
                target_channel, config.data.image_size[1], config.data.image_size[0], spacing, seed)
                f_toeprx = './data/%s_Nt%s_Nr%s_ULA%.2f_seed%d_toeprx.npy' % (
                target_channel, config.data.image_size[1], config.data.image_size[0], spacing, seed)
                self.toeptx = np.load(f_topetx)
                self.toeprx = np.load(f_toeprx)
            else:
                filename = './data/%s_Nt%s_Nr%s_ULA%.2f_seed%d.mat' % (
                    target_channel, config.data.image_size[1], config.data.image_size[0], spacing, seed)
                # Log
                self.filenames.append(filename)

                # Preload file and serialize
                contents = hdf5storage.loadmat(filename)
                channels = np.asarray(contents['output_h'], dtype=np.complex64)

            # Use only first subcarrier of each symbol
            if target_channel.startswith('CDL') or target_channel.startswith('Mixed'):
                self.channels.append(channels[:, 0])
            else:
                channels = channels[:min(length, channels.shape[0])]
                if target_channel != '3gpp':
                    channels = (channels - np.mean(channels, axis=(1, 2), keepdims=True)) / np.std(channels, axis=(1, 2), keepdims=True)
                self.channels.append(channels)
        print('Done')
                
        # Convert to array
        self.channels = np.asarray(self.channels)
        self.channels = np.reshape(self.channels,
               (-1, self.channels.shape[-2], self.channels.shape[-1]))

        if domain == 'angle':
            A_R = dft(config.data.image_size[0]) / np.sqrt(config.data.image_size[0])
            A_T = dft(config.data.image_size[1]) / np.sqrt(config.data.image_size[1])
            self.channels = np.conj(A_R).T @ self.channels @ A_T
            
        # Normalize
        if type(norm) == list:
            self.mean = norm[0]
            self.std  = norm[1]
        elif norm == 'entrywise':
            self.mean = np.mean(self.channels, axis=0)
            self.std  = np.std(self.channels, axis=0)
        elif norm == 'global':
            self.mean = 0.
            self.std  = np.std(self.channels)
            
        # Generate random QPSK pilots
        self.pilots = 1/np.sqrt(2) * (2 * np.random.binomial(1, 0.5, size=(
            self.channels.shape[0], config.data.image_size[1], config.data.num_pilots)) - 1 +
                1j * (2 * np.random.binomial(1, 0.5, size=(
            self.channels.shape[0], config.data.image_size[1], config.data.num_pilots)) - 1))
            
        # Complex noise power
        self.noise_power = 1/np.sqrt(2) * config.data.noise_std
            
    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Normalize
        H_cplx = self.channels[idx]
        H_cplx_norm = (H_cplx - self.mean) / self.std
        
        # Convert to reals
        H_real_norm = \
            np.stack((np.real(H_cplx_norm), np.imag(H_cplx_norm)), axis=0)
        
        # Get complex pilots and create noisy Y
        P = self.pilots[idx]
        Y = np.matmul(H_cplx, P)
        N = self.noise_power * (np.random.normal(size=Y.shape) + \
                                1j * np.random.normal(size=Y.shape))
        Y = Y + N
        
        # Compute largest eigenvalue of normal operator
        eigvals = np.real(
            np.linalg.eigvals(np.matmul(
                P, np.conj(P.T))))
        
        # Also get Hermitian H, real-viewed
        H_herm_norm = np.conj(np.transpose(H_cplx_norm))
        H_real_herm_norm = \
            np.stack((np.real(H_herm_norm), np.imag(H_herm_norm)), axis=0)

        # And more Hermitians
        P_herm = np.conj(np.transpose(P))
        Y_herm = np.conj(np.transpose(Y))

        sample = {'H': H_real_norm.astype(np.float32),
                  'H_herm': H_real_herm_norm.astype(np.float32),
                  'H_herm_cplx': H_herm_norm.astype(np.complex64),
                  'P': P.astype(np.complex64),
                  'P_herm': P_herm.astype(np.complex64),
                  'Y': Y.astype(np.complex64),
                  'Y_herm': Y_herm.astype(np.complex64),
                  'eig1': eigvals[0].astype(np.float32),
                  'sigma_n': self.noise_power.astype(np.float32),
                  'idx': int(idx)}
        if self.toeptx is not None:
            sample['toeptx'] = self.toeptx[idx]
            sample['toeprx'] = self.toeprx[idx]
        return sample