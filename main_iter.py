# plot_Q_PGA_results.py  ←  FILE MỚI, CHỈ CHẠY FILE NÀY LÀ XONG HẾT
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from utility import *
from Q_PGA_models import Q_PGA_Unfold
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================== CẤU HÌNH ==============================
run_Q_PGA = 1
n_iter_outer = 20
n_iter_inner = 20
model_path = './results/Q_PGA_best.pth'   # model đã train tốt nhất
snr_dB = 10                               # SNR để vẽ beampattern & convergence
snr = 10 ** (snr_dB / 10)

# Load test data
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]
R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, :test_size, :, :]

print("Loading Q-PGA model và chạy test...")
# ============================== LOAD & RUN Q-PGA ==============================
if run_Q_PGA:
    model = Q_PGA_Unfold(n_iter_outer=n_iter_outer, n_iter_inner=n_iter_inner,
                         K=K, Nt=Nt, Nrf=Nrf, M=M)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        rates_Q, taus_Q, F_Q, W_Q = model.execute_PGA(H_test, R, snr, n_iter_outer, n_iter_inner)

    # Trung bình theo user và batch
    rate_iter_Q = (rates_Q.sum(dim=0) / rates_Q.shape[0]).cpu().numpy()
    tau_iter_Q  = (taus_Q.sum(dim=0)  / taus_Q.shape[0]).cpu().numpy()

    # Beampattern
    beam_Q = get_beampattern(F_Q, W_Q, at, snr)

# ============================== BENCHMARK (ZF, SCA) ==============================
benchmark = 1
if benchmark:
    bm = scipy.io.loadmat(directory_benchmark + 'result_benchmark')
    idx = np.where(snr_dB_list == snr_dB)[0][0]
    rate_ZF  = bm['rate_ZF_mean'][0, idx]
    rate_SCA = bm['rate_SCA_mean'][0, idx]
    tau_ZF   = bm['tau_ZF_mean'][0, idx]
    tau_SCA  = bm['tau_SCA_mean'][0, idx]
    beam_ZF  = np.squeeze(bm['beam_ZF_mean'][:, idx])
    beam_SCA = np.squeeze(bm['beam_SCA_mean'][:, idx])

    # Dùng giá trị constant để vẽ đường ngang
    rate_ZF_line  = np.ones(n_iter_outer + 1) * rate_ZF
    rate_SCA_line = np.ones(n_iter_outer + 1) * rate_SCA
    tau_ZF_line   = np.ones(n_iter_outer + 1) * tau_ZF
    tau_SCA_line  = np.ones(n_iter_outer + 1) * tau_SCA

# ============================== VẼ ĐỒ THỊ ==============================
iter_vec = np.arange(n_iter_outer + 1)
angles_theta = theta[0, :] * 180 / np.pi
system_params = f'N={Nt}, M={M}, N_RF={Nrf}, SNR={snr_dB} dB, ω={OMEGA}'

# ------------------- 1. Rate vs Iteration -------------------
plt.figure(figsize=(7, 4.5))
plt.plot(iter_vec, rate_iter_Q, '-s', markevery=3, color='cyan', linewidth=3, markersize=8, label='Q-PGA (Ours)')
if benchmark:
    plt.plot(iter_vec, rate_SCA_line, '--x', markevery=5, color='black', linewidth=2.5, label='SCA')
    plt.plot(iter_vec, rate_ZF_line,  '--o', markevery=5, color='purple', linewidth=2.5, label='ZF')
plt.xlabel('Number of iterations/layers ($I$)', fontsize=14)
plt.ylabel('$R$ [bits/s/Hz]', fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)
plt.title(system_params, fontsize=12)
plt.tight_layout()
plt.savefig(f'./results/Q_PGA_rate_vs_iter_{Nt}_{OMEGA}.png', dpi=300)
plt.savefig(f'./results/Q_PGA_rate_vs_iter_{Nt}_{OMEGA}.eps')

# ------------------- 2. Beam Error vs Iteration -------------------
plt.figure(figsize=(7, 4.5))
plt.plot(iter_vec, tau_iter_Q, '-s', markevery=3, color='cyan', linewidth=3, markersize=8, label='Q-PGA (Ours)')
if benchmark:
    plt.plot(iter_vec, tau_SCA_line, '--x', markevery=5, color='black', linewidth=2.5, label='SCA')
    plt.plot(iter_vec, tau_ZF_line,  '--o', markevery=5, color='purple', linewidth=2.5, label='ZF')
plt.xlabel('Number of iterations/layers ($I$)', fontsize=14)
plt.ylabel(r'$\bar{\tau}$', fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)
plt.title(system_params, fontsize=12)
plt.tight_layout()
plt.savefig(f'./results/Q_PGA_tau_vs_iter_{Nt}_{OMEGA}.png', dpi=300)
plt.savefig(f'./results/Q_PGA_tau_vs_iter_{Nt}_{OMEGA}.eps')

# ------------------- 3. Trade-off -------------------
plt.figure(figsize=(6.5, 5))
plt.plot(tau_iter_Q, rate_iter_Q, '-s', markevery=3, color='cyan', linewidth=3, markersize=9, label='Q-PGA (Ours)')
if benchmark:
    plt.plot(tau_SCA, rate_SCA, 'x', color='black', markersize=12, markeredgewidth=3, label='SCA')
    plt.plot(tau_ZF,  rate_ZF,  'o', color='purple', markersize=10, markeredgewidth=3, label='ZF')
plt.xlabel(r'$\bar{\tau}$', fontsize=14)
plt.ylabel('$R$ [bits/s/Hz]', fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)
plt.title(system_params, fontsize=12)
plt.tight_layout()
plt.savefig(f'./results/Q_PGA_tradeoff_{Nt}_{OMEGA}.png', dpi=300)
plt.savefig(f'./results/Q_PGA_tradeoff_{Nt}_{OMEGA}.eps')

# ------------------- 4. Beampattern -------------------
plt.figure(figsize=(8, 4.5))
plt.plot(angles_theta, np.real(beam_Q), '-', color='cyan', linewidth=3.5, label='Q-PGA (Ours)')
if benchmark:
    plt.plot(angles_theta, beam_SCA, '--', color='black', linewidth=2, label='SCA')
    plt.plot(angles_theta, beam_ZF,  ':',  color='purple', linewidth=2, label='ZF')
plt.plot(angles_theta, np.real(ideal_beam), '--', color='green', linewidth=2, label='Ideal')
plt.xlabel(r'Angle $\theta_t$ [degrees]', fontsize=14)
plt.ylabel('Normalized beampattern', fontsize=14)
plt.xlim([-90, 90])
plt.xticks(np.arange(-90, 91, 30))
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)
plt.title(system_params, fontsize=12)
plt.tight_layout()
plt.savefig(f'./results/Q_PGA_beampattern_{Nt}_{OMEGA}.png', dpi=300)
plt.savefig(f'./results/Q_PGA_beampattern_{Nt}_{OMEGA}.eps')

plt.show()

print("HOÀN TẤT! Tất cả hình đã lưu vào thư mục ./results/")
