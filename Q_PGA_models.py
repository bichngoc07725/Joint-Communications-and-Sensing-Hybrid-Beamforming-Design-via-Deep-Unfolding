# Q_PGA_models.py
import torch
import torch.nn as nn
import pennylane as qml
from utility import *

n_qubits = 8
dev = qml.device("default.qubit.torch", wires=n_qubits, shots=None)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)
        qml.RZ(inputs[..., i + n_qubits], wires=i)
    for layer in range(3):
        for i in range(n_qubits):
            qml.Rot(*weights[layer, i], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumBlock(nn.Module):
    def __init__(self, in_features=80, out_features=128):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_features, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 2*n_qubits)
        )
        self.qnode = qml.qnn.TorchLayer(quantum_circuit, {"weights": (3, n_qubits, 3)})
        self.post = nn.Sequential(nn.Linear(n_qubits, out_features))

    def forward(self, x):
        x = self.pre(x)
        return self.post(self.qnode(x))

class Q_PGA_Unfold(nn.Module):
    def __init__(self, n_iter_outer=120, n_iter_inner=20):
        super().__init__()
        self.n_iter_outer = n_iter_outer
        self.n_iter_inner = n_iter_inner
        self.step = nn.Parameter(torch.tensor(0.02))
        self.qF = nn.ModuleList([QuantumBlock(80, Nt*Nrf*2) for _ in range(n_iter_inner)])
        self.qW = QuantumBlock(80, K*Nrf*M*2)
        self.proj = nn.Linear(500, 80)

    def _state(self, H, F, W, R):
        s = torch.cat([F.abs().mean(dim=(0,1)).flatten(),
                       F.angle().mean(dim=(0,1)).flatten(),
                       W.abs().mean(dim=(0,1,2)), W.angle().mean(dim=(0,1,2)),
                       H.abs().mean(dim=(0,1,2)).flatten(),
                       R.abs().mean(dim=(0,1)).flatten()])
        return self.proj(s.unsqueeze(0)).expand(H.shape[1], -1)

    def _complex(self, x):
        return torch.complex(x[...,0::2], x[...,1::2])

    def execute_PGA(self, H, R, Pt, n_iter_outer=None, n_iter_inner=None):
        n_iter_outer = n_iter_outer or self.n_iter_outer
        n_iter_inner = n_iter_inner or self.n_iter_inner
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        B = H.shape[1]
        rates, taus = [get_sum_rate(H, F, W, Pt)], [get_beam_error(H, F, W, R, Pt)]

        for _ in range(n_iter_outer):
            state = self._state(H, F, W, R)
            for j in range(n_iter_inner):
                qF = self.qF[j](state)
                dF = self._complex(qF).view(B, Nt, Nrf).permute(1,0,2).unsqueeze(0)
                F = F + self.step * torch.randn_like(F) * 1e-3 + dF * 1e-3
                F = F / (F.abs() + 1e-12)
            qW = self.qW(state)
            dW = self._complex(qW).view(B, K, Nrf, M).permute(1,0,2,3)
            W = W + self.step * torch.randn_like(W) * 1e-3 + dW * 1e-3
            F, W = normalize(F, W, H, Pt)
            rates.append(get_sum_rate(H, F, W, Pt))
            taus.append(get_beam_error(H, F, W, R, Pt))

        return torch.stack(rates).T, torch.stack(taus).T, F, W
