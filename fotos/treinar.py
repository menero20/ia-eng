"""IA Eng fase 2 - ajuste fino: parte de uma IA que já enxerga (MobileNetV3, ImageNet)
e ensina só as nossas classes de rua. Salva fotos/modelo_fotos.pt."""
import os
import sys
import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

AQUI = os.path.dirname(os.path.abspath(__file__))
EPOCAS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

treino = datasets.ImageFolder(os.path.join(AQUI, "dados", "treino"), transforms.Compose([
    transforms.RandomResizedCrop(144, scale=(0.7, 1)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3), transforms.ToTensor(), NORM]))
prova = datasets.ImageFolder(os.path.join(AQUI, "dados", "prova"), transforms.Compose([
    transforms.Resize(160), transforms.CenterCrop(144), transforms.ToTensor(), NORM]))

conta = Counter(treino.targets)
pesos = [1.0 / conta[t] for t in treino.targets]  # classes equilibradas: carro não domina
if __name__ == "__main__":
    torch.set_num_threads(8)
    dl = DataLoader(treino, batch_size=64, sampler=WeightedRandomSampler(pesos, len(pesos)), num_workers=4)
    dlp = DataLoader(prova, batch_size=128, num_workers=2)

    modelo = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, len(treino.classes))
    otim = torch.optim.AdamW(modelo.parameters(), lr=5e-4, weight_decay=1e-4)
    agenda = torch.optim.lr_scheduler.OneCycleLR(otim, max_lr=5e-4, total_steps=EPOCAS * len(dl))
    perda_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    inicio = time.time()
    print("classes:", treino.classes, "| treino", len(treino), "| prova", len(prova), flush=True)

    for ep in range(1, EPOCAS + 1):
        modelo.train()
        for x, y in dl:
            perda = perda_fn(modelo(x), y)
            otim.zero_grad()
            perda.backward()
            otim.step()
            agenda.step()
        modelo.eval()
        certo = Counter()
        total = Counter(prova.targets)
        with torch.no_grad():
            for x, y in dlp:
                p = modelo(x).argmax(1)
                for a, b in zip(p.tolist(), y.tolist()):
                    certo[b] += a == b
        geral = sum(certo.values()) / len(prova)
        por = " ".join(f"{c} {certo[i] / total[i]:.0%}" for i, c in enumerate(prova.classes))
        print(f"época {ep}/{EPOCAS} | acerto geral {geral:.1%} | {por} | {(time.time() - inicio) / 60:.0f} min", flush=True)
        torch.save({"pesos": modelo.state_dict(), "classes": treino.classes}, os.path.join(AQUI, "modelo_fotos.pt"))
    print("pronto: fotos/modelo_fotos.pt", flush=True)
