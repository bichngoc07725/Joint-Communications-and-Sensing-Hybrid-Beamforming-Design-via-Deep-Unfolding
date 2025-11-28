
import torch
import numpy as np
import scipy.io
from system_config import *

sigma2 = 1.0

def get_data_tensor(source):
    if source == 'matlab':
        train = scipy.io.loadmat(directory_data + "train_data_matlab.mat")['H_train']
        test = scipy.io.loadmat(directory_data + "test_data_matlab.mat")['H_test']
    return torch.from_numpy(train).cfloat(), torch.from_numpy(test).cfloat()

def get_radar_data(snr_dB, H):
    data = scipy.io.loadmat(directory_data + "radar_data.mat")
    idx = min(int((snr_dB + 10)/5), 6)
    R = torch.from_numpy(data['J'][:,:,:,idx]).cfloat()
    at = torch.from_numpy(data['a']).cfloat()
    theta = data['theta']
    ideal = data['Pd_theta'][0]
    return R, at, theta, ideal

# HÀM normalize ĐÃ FIX HOÀN TOÀN – KHÔNG CÒN LỖI BATCH
def normalize(F, W, H, Pt):
    """
    F: [K, B, Nt, Nrf], W: [K, B, Nrf, M]
    """
    F = F / (F.abs() + 1e-12)                                   # constant modulus
    X = torch.matmul(F, W)                                      # [K, B, Nt, M]
    power = torch.sum(torch.abs(X)**2)                          # tổng công suất
    scale = torch.sqrt(Pt * K / (power + 1e-12))
    W = W * scale
    return F, W

def get_sum_rate(H, F, W, Pt):
    F, W = normalize(F, W, H, Pt)
    rate = torch.zeros(H.shape[1], device=H.device)
    for m in range(M):
        W_m = W.clone()
        W_m[:, :, :, m] = 0
        signal = torch.sum(torch.abs(H[..., m:m+1].conj().transpose(2,3) @ F @ W)**2, dim=-1).squeeze(-1)
        interference = torch.sum(torch.abs(H[..., m:m+1].conj().transpose(2,3) @ F @ W_m)**2, dim=-1).squeeze(-1)
        rate += torch.log2(1 + signal / (interference + sigma2))
    return rate.mean()

def get_beam_error(H, F, W, R, Pt):
    F, W = normalize(F, W, H, Pt)
    X = torch.matmul(F, W)
    return torch.mean(torch.abs(X @ X.conj().transpose(2,3) - R)**2)

def get_sum_loss(F, W, H, R, Pt):
    return -get_sum_rate(H, F, W, Pt) + 10 * get_beam_error(H, F, W, R, Pt)

# HÀM initialize HOÀN HẢO – KHÔNG DÙNG BIẾN TOÀN CỤC, KHÔNG LỖI SHAPE
def initialize(H, R, Pt, normalization=0):
    K, B, M_users, Nt = H.shape
    device = H.device

    # F: constant modulus, khởi tạo ngẫu nhiên
    phase = torch.randn(K, B, Nt, Nrf, device=device)
    F = torch.exp(1j * phase)

    # W: ZF digital precoder
    W = torch.zeros(K, B, Nrf, M_users, dtype=torch.cfloat, device=device)
    for k in range(K):
        Hk = H[k]                                               # [B, M, Nt]
        Fk = F[k]                                               # [B, Nt, Nrf]
        A = torch.bmm(Fk.mH, Fk) + 1e-6 * torch.eye(Nrf, device=device).unsqueeze(0)
        B = torch.bmm(Fk.mH, Hk)
        W[k] = torch.linalg.solve(A, B)

    # Chuẩn hóa công suất
    X = torch.matmul(F, W)
    power = torch.sum(torch.abs(X)**2)
    scale = torch.sqrt(Pt * K / (power + 1e-12))
    W = W * scale

    rate0 = get_sum_rate(H, F, W, Pt)
    tau0  = get_beam_error(H, F, W, R, Pt)
    return rate0, tau0, F, W

print("utility.py loaded successfully!")
