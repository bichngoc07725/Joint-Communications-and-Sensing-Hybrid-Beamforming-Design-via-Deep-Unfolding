# plot_Q_PGA_results.py ← CHỈ CHẠY FILE NÀY LÀ XONG HẾT (ĐÃ SỬA HOÀN CHỈNH)
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from utility import *
from Q_PGA_models import Q_PGA_Unfold

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs('./results', exist_ok=True)

# ============================== CẤU HÌNH ==============================
run_Q_PGA      = True
n_iter_outer   = 20          # phải trùng với lúc train
n_iter_inner   = 20
model_path     = './results/Q_PGA_best.pth'
snr_dB         = 10          # bạn muốn vẽ ở SNR nào thì đổi ở đây
snr            = 10 ** (snr_dB / 10)

# ------------------- Load dữ liệu test -------------------
print("Đang load dữ liệu test...")
H_train, H_test_all = get_data_tensor(data_source)                # [K, total_samples, M, Nt]
H_test = H_test_all[:, :test_size, :, :]                          # [K, test_size, M, Nt]

R, at_all, theta, ideal_beam = get_radar_data(snr_dB, H_test)     # R: [K, test_size, Nt, Nt]
at = at_all[:, :test_size, :, :]                                 # [1, test_size, Nt, Ntheta]

# ------------------- Load & chạy Q-PGA -------------------
if run_Q_PGA:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}\n"
                                f"Vui lòng chạy test_Q_PGA.py để train và lưu model trước!")

    print("Đang load model Q-PGA và chạy inference...")
    model = Q_PGA_Unfold(n_iter_outer=n_iter_outer, n_iter_inner=n_iter_inner)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        rates_Q, taus_Q, F_Q, W_Q = model.execute_PGA(H_test, R, snr,
                                                     n_iter_outer, n_iter_inner)

    # Trung bình theo batch (test_size)
    rate_iter_Q = rates_Q.mean(dim=0).cpu().numpy()      # shape = (n_iter+1,)
    tau_iter_Q  = taus_Q.mean(dim=0).cpu().numpy()

    # ------------------- Beampattern -------------------
    def get_beampattern(F, W, at, Pt):
        X   = F @ W                                       # [K, B, Nt, M]
        X_H = X.conj().transpose(2, 3)
        at_H = at.conj().transpose(2, 3)                  # [1, B, Ntheta, Nt]
        B = at_H @ X @ X_H @ at                            # [1, B, Ntheta, Ntheta]
        Bdiag = torch.diagonal(B, dim1=-2, dim2=-1).real / Pt
        return Bdiag.mean(dim=(0,1)).cpu().numpy()       # trung bình batch + K → [Ntheta]

    beam_Q = get_beampattern(F_Q, W_Q, at, snr)

# ------------------- Benchmark (ZF & SCA) – tự động bỏ qua nếu không có -------------------
benchmark = True
bm_path = os.path.join(directory_benchmark, 'result_benchmark.mat') if 'directory_benchmark' in globals() else None

if benchmark and bm_path and os.path.exists(bm_path):
    try:
        bm = scipy.io.loadmat(bm_path)
        idx = np.where(np.isclose(snr_dB_list, snr_dB))[0][0]

        rate_ZF  = float(bm['rate_ZF_mean'][0, idx])
        rate_SCA = float(bm['rate_SCA_mean'][0, idx])
        tau_ZF   = float(bm['tau_ZF_mean'][0, idx])
        tau_SCA  = float(bm['tau_SCA_mean'][0, idx])

        beam_ZF  = np.squeeze(bm['beam_ZF_mean'][:, idx])
        beam_SCA = np.squeeze(bm['beam_SCA_mean'][:, idx])

        # Đường ngang cho các benchmark
        rate_ZF_line  = np.ones_like(rate_iter_Q) * rate_ZF
        rate_SCA_line = np.ones_like(rate_iter_Q) * rate_SCA
        tau_ZF_line   = np.ones_like(tau_iter_Q)  * tau_ZF
        tau_SCA_line  = np.ones_like(tau_iter_Q)  * tau_SCA

        print("Đã load thành công benchmark ZF & SCA")
    except Exception as e:
        print("Không load được benchmark → chỉ vẽ Q-PGA")
        benchmark = False
else:
    print("Không tìm thấy file benchmark → chỉ vẽ Q-PGA")
    benchmark = False

# ============================== VẼ ĐỒ THỊ ==============================
iter_vec     = np.arange(n_iter_outer + 1)
angles_theta = theta[0, :] * 180 / np.pi
system_params = f'N_t={Nt}, M={M}, N_RF={Nrf}, SNR={snr_dB} dB, ω={OMEGA}'

# ------------------- 1. Sum Rate vs Iteration -------------------
plt.figure(figsize=(7.5, 4.5))
plt.plot(iter_vec, rate_iter_Q, '-s', markevery=3, color='cyan', linewidth=3, markersize=8, label='Q-PGA (Ours)')
if benchmark:
    plt.plot(iter_vec, rate_SCA_line, '--x', markevery=5, color='black',  linewidth=2.5, markersize=9, label='SCA-ManOpt')
    plt.plot(iter_vec, rate_ZF_line,  '--o', markevery=5, color='purple', linewidth=2.5, markersize=7, label='ZF')
plt.xlabel('Number of iterations/layers ($I$)', fontsize=14)
plt.ylabel('Sum Rate $R$ [bits/s/Hz]', fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)
plt.title(f'Sum Rate Convergence\n{system_params}', fontsize=12)
plt.tight_layout()
plt.savefig(f'./results/Q_PGA_rate_vs_iter_{snr_dB}dB.png', dpi=300, bbox_inches='tight')
plt.savefig(f'./results/Q_PGA_rate_vs_iter_{snr_dB}dB.eps', bbox_inches='tight')

# ------------------- 2. Beam Error vs Iteration -------------------
plt.figure(figsize=(7.5, 4.5))
plt.plot(iter_vec, tau_iter_Q, '-s', markevery=3, color='cyan', linewidth=3, markersize=8, label='Q-PGA (Ours)')
if benchmark:
    plt.plot(iter_vec, tau_SCA_line, '--x', markevery=5, color='black',  linewidth=2.5, markersize=9, label='SCA-ManOpt')
    plt.plot(iter_vec, tau_ZF_line,  '--o', markevery=5, color='purple', linewidth=2.5, markersize=7, label='ZF')
plt.xlabel('Number of iterations/layers ($I$)', fontsize=14)
plt.ylabel(r'Beamforming Error $\bar{\tau}$', fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)
plt.title(f'Beamforming Error Convergence\n{system_params}', fontsize=12)
plt.tight_layout()
plt.savefig(f'./results/Q_PGA_tau_vs_iter_{snr_dB}dB.png', dpi=300, bbox_inches='tight')
plt.savefig(f'./results/Q_PGA_tau_vs_iter_{snr_dB}dB.eps', bbox_inches='tight')

# ------------------- 3. Trade-off Curve -------------------
plt.figure(figsize=(6.8, 5.2))
plt.plot(tau_iter_Q, rate_iter_Q, '-s', markevery=3, color='cyan', linewidth=3.5, markersize=9, label='Q-PGA (Ours)')
if benchmark:
    plt.plot(tau_SCA, rate_SCA, 'x', color='black',  markersize=14, markeredgewidth=3, label='SCA-ManOpt')
    plt.plot(tau_ZF,  rate_ZF,  'o', color='purple', markersize=11, markeredgewidth=3, label='ZF')
plt.xlabel(r'Beamforming Error $\bar{\tau}$', fontsize=14)
plt.ylabel('Sum Rate $R$ [bits/s/Hz]', fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)
plt.title(f'Communication-Sensing Trade-off\n{system_params}', fontsize=12)
plt.tight_layout()
plt.savefig(f'./results/Q_PGA_tradeoff_{snr_dB}dB.png', dpi=300, bbox_inches='tight')
plt.savefig(f'./results/Q_PGA_tradeoff_{snr_dB}dB.eps', bbox_inches='tight')

# ------------------- 4. Beampattern -------------------
plt.figure(figsize=(8.5, 4.8))
plt.plot(angles_theta, beam_Q, '-', color='cyan', linewidth=3.5, label='Q-PGA (Ours)')
if benchmark:
    plt.plot(angles_theta, beam_SCA, '--', color='black',  linewidth=2.5, label='SCA-ManOpt')
    plt.plot(angles_theta, beam_ZF,  ':',  color='purple', linewidth=2.5, label='ZF')
plt.plot(angles_theta, np.real(ideal_beam), '--', color='green', linewidth=2.5, alpha=0.8, label='Ideal Beam')
plt.xlabel(r'Angle $\theta_t$ [degrees]', fontsize=14)
plt.ylabel('Normalized Beampattern', fontsize=14)
plt.xlim([-90, 90])
plt.xticks(np.arange(-90, 91, 30))
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)
plt.title(f'Radar Beampattern (SNR = {snr_dB} dB)\n{system_params}', fontsize=12)
plt.tight_layout()
plt.savefig(f'./results/Q_PGA_beampattern_{snr_dB}dB.png', dpi=300, bbox_inches='tight')
plt.savefig(f'./results/Q_PGA_beampattern_{snr_dB}dB.eps', bbox_inches='tight')

plt.show()

print("\nHOÀN TẤT! 4 hình publication-quality đã được lưu trong thư mục ./results/")
print("Bạn có thể nộp paper ngay mà không cần chỉnh gì thêm!")
