"""IA Eng - treino da especialista em enigmas.

A cada passo o gerador cria enigmas novos (a IA nunca vê o mesmo duas vezes),
a rede chuta, compara com a resposta e se ajusta. De tempos em tempos mede o
acerto em enigmas separados e salva o "cérebro" em modelo.pt.
Uso: python treinar.py [passos]
"""
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from gerador import ALTURA, CARACTERES, LARGURA, TAMANHO, enigma


def para_tensor(img):
    return torch.from_numpy(np.asarray(img.convert("L"), dtype=np.float32) / 255.0).unsqueeze(0)


class Fluxo(IterableDataset):
    def __iter__(self):
        while True:
            img, texto = enigma()
            yield para_tensor(img), torch.tensor([CARACTERES.index(c) for c in texto])


class Especialista(nn.Module):
    def __init__(self):
        super().__init__()
        def bloco(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.olhos = nn.Sequential(bloco(1, 32), bloco(32, 64), bloco(64, 128), bloco(128, 192))
        self.cabeca = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(192 * (ALTURA // 16) * (LARGURA // 16), 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, TAMANHO * len(CARACTERES)),
        )

    def forward(self, x):
        return self.cabeca(self.olhos(x)).view(-1, TAMANHO, len(CARACTERES))


def avaliar(modelo, prova):
    modelo.eval()
    with torch.no_grad():
        x, y = prova
        p = modelo(x).argmax(-1)
    modelo.train()
    return (p == y).all(1).float().mean().item(), (p == y).float().mean().item()


if __name__ == "__main__":
    passos = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    torch.manual_seed(0)
    dados = iter(DataLoader(Fluxo(), batch_size=128, num_workers=4))
    prova = next(iter(DataLoader(Fluxo(), batch_size=1000)))  # 1000 enigmas que nunca entram no treino

    modelo = Especialista()
    otim = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-4)
    agenda = torch.optim.lr_scheduler.OneCycleLR(otim, max_lr=2e-3, total_steps=passos)
    perda_fn = nn.CrossEntropyLoss()
    inicio = time.time()

    for passo in range(1, passos + 1):
        x, y = next(dados)
        perda = perda_fn(modelo(x).reshape(-1, len(CARACTERES)), y.reshape(-1))
        otim.zero_grad()
        perda.backward()
        otim.step()
        agenda.step()
        if passo % 250 == 0 or passo == passos:
            inteiro, letra = avaliar(modelo, prova)
            print(f"passo {passo:5d} | {passo * 128:7d} enigmas vistos | perda {perda.item():.3f} | "
                  f"acerto enigma inteiro {inteiro:6.1%} | por letra {letra:6.1%} | {(time.time() - inicio) / 60:.1f} min",
                  flush=True)
            torch.save(modelo.state_dict(), "modelo.pt")
    print("pronto: modelo.pt")
