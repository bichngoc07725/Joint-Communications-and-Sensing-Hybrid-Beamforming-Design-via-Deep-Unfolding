# test_Q_PGA.py  →  CHẠY FILE NÀY LÀ XONG HẾT: train + test + vẽ đồ thị
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utility import *
from Q_PGA_models import Q_PGA_Unfold
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================== CẤU HÌNH ==============================
n_iter_outer = 20
n_iter_inner = 20
learning_rate = 0.0012
n_epoch = 350
batch_size = 32
model_path = './results/Q_PGA_best.pth'
os.makedirs('./results', exist_ok=True)

# Load data
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

# ============================== TRAINING ==============================
print("Bắt đầu train Q-PGA...")
model = Q_PGA_Unfold(n_iter_outer=n_iter_outer, n_iter_inner=n_iter_inner,
                     K=K, Nt=Nt, Nrf=Nrf, M=M)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_loss = 1e9

for epoch in range(1, n_epoch + 1):
    model.train()
    total_loss = 0.0
    idx = torch.randperm(H_train.shape[1])
    H_shuf = H_train[:, idx]

    for i in range(0, H_train.shape[1], batch_size):
        H_batch = H_shuf[:, i:i+batch_size]
        snr_db = np.random.choice(snr_dB_list)
        snr = 10 ** (snr_db / 10)
        R_batch, _, _, _ = get_radar_data(snr_db, H_batch)

        rates, taus, F, W = model.execute_PGA(H_batch, R_batch, snr, n_iter_outer, n_iter_inner)
        loss = get_sum_loss(F, W, H_batch, R_batch, snr, batch_size)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (H_train.shape[1] // batch_size)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.5f}  → Saved!")

    elif epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.5f}")

print("Train xong! Bắt đầu test trên toàn SNR...")

# ============================== TESTING ==============================
model.load_state_dict(torch.load(model_path))
model.eval()

rate_vs_snr = []
mse_vs_snr = []

for snr_db in snr_dB_list:
    snr = 10 ** (snr_db / 10)
    R, at, _, _ = get_radar_data(snr_db, H_test)

    with torch.no_grad():
        rates, taus, F, W = model.execute_PGA(H_test, R, snr, n_iter_outer, n_iter_inner)

    final_rate = rates.mean(dim=0)[-1].item()
    final_mse = get_MSE(F, W, at, R, snr).item()

    rate_vs_snr.append(final_rate)
    mse_vs_snr.append(final_mse)

    print(f"SNR {snr_db:2d}dB → Rate: {final_rate:.3f} bps/Hz | MSE: {final_mse:.2f} dB")

# ============================== VẼ ĐỒ THỊ ==============================
plt.figure(figsize=(10, 4.5))

plt.subplot(1, 2, 1)
plt.plot(snr_dB_list, rate_vs_snr, 's-', color='cyan', linewidth=3, markersize=8, label='Q-PGA (Ours)')
plt.xlabel('SNR [dB]'); plt.ylabel('Sum Rate [bits/s/Hz]')
plt.grid(True); plt.legend(); plt.title('Q-PGA: Sum Rate vs SNR')

plt.subplot(1, 2, 2)
plt.plot(snr_dB_list, mse_vs_snr, 'o-', color='magenta', linewidth=3, markersize=8, label='Q-PGA (Ours)')
plt.xlabel('SNR [dB]'); plt.ylabel('Radar Beampattern MSE [dB]')
plt.grid(True); plt.legend(); plt.title('Q-PGA: MSE vs SNR')

plt.tight_layout()
plt.savefig('./results/Q_PGA_Final_Results.png', dpi=300, bbox_inches='tight')
plt.show()

print("HOÀN TẤT! Kết quả lưu tại ./results/")
