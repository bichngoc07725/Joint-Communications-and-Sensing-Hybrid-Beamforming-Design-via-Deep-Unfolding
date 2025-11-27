# utility.py
import torch
import numpy as np
import h5py
import scipy.io
import sys
import os

# ===================================================================
# CÁC HẰNG SỐ TOÀN CỤC (sẽ được ghi đè bởi system_config.py khi import *)
# ===================================================================
from system_config import *

# ===================================================================
# HÀM TÍNH LOSS CHÍNH (BẮT BUỘC PHẢI CÓ)
# ===================================================================
def get_sum_loss(F, W, H, R, Pt, batch_size=None):
    """
    Loss = -SumRate + λ * τ  (λ = 10 là ổn định nhất)
    """
    rate = get_sum_rate(H, F, W, Pt)
    tau = get_beam_error(H, F, W, R, Pt)
    return -rate + 10.0 * tau

# ===================================================================
# KHỞI TẠO F, W THEO CÁC PHƯƠNG PHÁP
# ===================================================================
def initialize(H, R, Pt, normalization):
    if init_scheme == 'conv':
        F = torch.randn(len(H[0]), Nt, Nrf, dtype=torch.double) + 1j * torch.randn(len(H[0]), Nt, Nrf, dtype=torch.double)
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)
    elif init_scheme == 'prop':
        if Nrf == M:
            F = H[K // 2, :, :, :] / torch.abs(H[K // 2, :, :, :])
            F = torch.transpose(F, 1, 2)
            F = torch.cat(((F[None, :, :, :],) * K), 0)
            W = torch.zeros(K, len(H[0]), Nrf, M, dtype=torch.cfloat)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj().transpose(1, 2)
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.norm(Wtmp, 'fro')
                W[k] = Wtmp / (Wtmp_norm + 1e-12)
        else:
            sys.stderr.write('Error: init_scheme prop chỉ hỗ trợ Nrf == M hiện tại\n')
            sys.exit(1)
    elif init_scheme == 'svd':
        U, S, VH = torch.linalg.svd(H)
        F = VH[:, :, :, :Nrf]
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)
    else:
        F = H / torch.abs(H)
        F = torch.transpose(F, 2, 3)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)

    if normalization == 1:
        F, W = normalize(F, W, H, Pt)
    else:
        norm2_FW = torch.sum(torch.linalg.norm(F @ W, 'fro') ** 2)
        W = W * torch.sqrt(Pt * K / (norm2_FW + 1e-12)).reshape(len(H[0]), 1, 1)

    rate_init = get_sum_rate(H, F, W, Pt)
    beam_error_init = get_beam_error(H, F, W, R, Pt)
    return rate_init, beam_error_init, F, W

# ===================================================================
# TÍNH SUM RATE
# ===================================================================
def get_sum_rate(H, F, W, Pt):
    F, W = normalize(F, W, H, Pt)
    F_H = F.conj().transpose(2, 3)
    W_H = W.conj().transpose(2, 3)
    rate = torch.zeros(len(H[0]), dtype=torch.float64, device=H.device)

    for m in range(M):
        W_m = W.clone()
        W_m[:, :, :, m] = 0
        V_m = W_m @ W_m.conj().transpose(2, 3)

        h_mk = H[:, :, m:m+1, :]           # [K, B, 1, Nt]
        h_mk_H = h_mk.conj().transpose(2, 3)
        Htilde_mk = h_mk @ h_mk_H            # [K, B, 1, 1]

        signal = torch.real(get_trace(F @ (W @ W_H) @ F_H @ Htilde_mk))
        interference = torch.real(get_trace(F @ V_m @ F_H @ Htilde_mk))
        rate += torch.log2(signal + sigma2) - torch.log2(interference + sigma2)

    return rate.mean()

# ===================================================================
# TÍNH BEAMFORMING ERROR (τ)
# ===================================================================
def get_beam_error(H, F, W, R, Pt):
    F, W = normalize(F, W, H, Pt)
    X = F @ W
    X_H = X.conj().transpose(2, 3)
    if normalize_tau == 1:
        error = torch.linalg.norm(X @ X_H - R, 'fro') ** 2 / (torch.linalg.norm(R, 'fro') ** 2 + 1e-12)
    else:
        error = torch.linalg.norm(X @ X_H - R, 'fro') ** 2
    return error.mean()

# ===================================================================
# TÍNH MSE CỦA RADAR BEAMPATTERN
# ===================================================================
def get_MSE(F, W, at, R, Pt):
    X = F @ W
    X_H = X.conj().transpose(2, 3)
    at_H = at.conj().transpose(2, 3)
    beampattern = torch.real(torch.diagonal(at_H @ X @ X_H @ at, dim1=-2, dim2=-1)) / Pt
    beam_mean = beampattern.mean(dim=(0, 1))

    beam_benchmark = torch.real(torch.diagonal(at_H @ R @ at, dim1=-2, dim2=-1)) / Pt
    beam_bm_mean = beam_benchmark.mean(dim=(0, 1))

    MSE = ((beam_mean - beam_bm_mean) ** 2).mean()
    return 10 * torch.log10(MSE + 1e-12)

# ===================================================================
# HÀM HỖ TRỢ
# ===================================================================
def get_trace(A):
    return torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)

def normalize(F, W, H, Pt):
    F = F / (torch.abs(F) + 1e-12)
    norm2 = torch.sum(torch.linalg.norm(F @ W, 'fro') ** 2)
    scale = torch.sqrt(Pt * K / (norm2 + 1e-12)).reshape(len(H[0]), 1, 1)
    W = W * scale
    return F, W

# ===================================================================
# GRADIENT CỦA PGA (dùng trong unfolded PGA)
# ===================================================================
def get_grad_F_com(H, F, W):
    # Placeholder – trong thực tế bạn cần implement đúng công thức từ paper
    return torch.zeros_like(F, dtype=torch.cfloat)

def get_grad_F_rad(F, W, R):
    return torch.zeros_like(F, dtype=torch.cfloat)

def get_grad_W_com(H, F, W):
    return torch.zeros_like(W, dtype=torch.cfloat)

def get_grad_W_rad(F, W, R):
    return torch.zeros_like(W, dtype=torch.cfloat)

# ===================================================================
# LOAD DỮ LIỆU
# ===================================================================
def get_radar_data(snr_dB, H):
    idx = np.where(snr_dB_list == snr_dB)[0][0]
    radar_file = directory_data + 'radar_data.mat'
    data = scipy.io.loadmat(radar_file)
    R = torch.from_numpy(data['J'][:, :, :, idx]).to(torch.cfloat)
    at = torch.from_numpy(data['a']).to(torch.cfloat)
    theta = data['theta']
    ideal = data['Pd_theta'][0]
    return R, at, theta, ideal

def load_data_matlab():
    train = scipy.io.loadmat(data_path_train)['H_train']
    test = scipy.io.loadmat(data_path_test)['H_test']
    return torch.from_numpy(train).to(torch.cfloat), torch.from_numpy(test).to(torch.cfloat)

def get_data_tensor(source):
    if source == 'matlab':
        return load_data_matlab()
    else:
        with h5py.File(data_path_train, 'r') as f:
            train = f['train_set'][()]
        with h5py.File(data_path_test, 'r') as f:
            test = f['test_set'][()]
        return torch.from_numpy(train).to(torch.cfloat), torch.from_numpy(test).to(torch.cfloat)

print("utility.py loaded successfully!")
