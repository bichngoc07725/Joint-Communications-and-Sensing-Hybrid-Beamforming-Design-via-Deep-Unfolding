# test_Q_PGA.py
import torch, os, numpy as np, matplotlib.pyplot as plt
from utility import *
from Q_PGA_models import Q_PGA_Unfold
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

model = Q_PGA_Unfold(n_iter_outer=n_iter_outer, n_iter_inner=n_iter_inner_J20)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
best = 1e9
path = './results/Q_PGA_best.pth'
os.makedirs('./results', exist_ok=True)

for epoch in range(1, n_epoch+1):
    model.train()
    loss_sum = 0.0
    idx = torch.randperm(H_train.shape[1])
    H_shuf = H_train[:, idx]
    for i in range(0, H_train.shape[1], batch_size):
        H_b = H_shuf[:, i:i+batch_size]
        snr_db = np.random.choice(snr_dB_list)
        R_b, _, _, _ = get_radar_data(snr_db, H_b)
        _, _, F, W = model.execute_PGA(H_b, R_b, snr, n_iter_outer, n_iter_inner_J20)
        loss = get_sum_loss(F, W, H_b, R_b, snr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    avg = loss_sum / (H_train.shape[1]//batch_size + 1)
    if avg < best:
        best = avg
        torch.save(model.state_dict(), path)
        print(f"Epoch {epoch} | Loss {avg:.5f} → Saved")
    else:
        print(f"Epoch {epoch} | Loss {avg:.5f}")

model.load_state_dict(torch.load(path))
model.eval()
rate_list, mse_list = [], []
for sdb in snr_dB_list:
    R, at, _, _ = get_radar_data(sdb, H_test)
    with torch.no_grad():
        rates, _, F, W = model.execute_PGA(H_test, R, snr, n_iter_outer, n_iter_inner_J20)
    rate_list.append(rates.mean(0)[-1].item())
    mse_list.append(get_MSE(F, W, at, R, snr).item())
    print(f"SNR {sdb}dB → Rate: {rate_list[-1]:.3f} | MSE: {mse_list[-1]:.2f} dB")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(snr_dB_list, rate_list, 's-', color='cyan', linewidth=3, markersize=8)
plt.xlabel('SNR [dB]'); plt.ylabel('Sum Rate'); plt.grid(); plt.title('Q-PGA Sum Rate')
plt.subplot(1,2,2); plt.plot(snr_dB_list, mse_list, 'o-', color='magenta', linewidth=3, markersize=8)
plt.xlabel('SNR [dB]'); plt.ylabel('MSE [dB]'); plt.grid(); plt.title('Q-PGA Radar MSE')
plt.tight_layout(); plt.savefig('./results/Q_PGA_Result.png', dpi=300); plt.show()
