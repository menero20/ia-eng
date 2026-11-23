"""IA Eng - usa o cérebro treinado. Uso: python resolver.py imagem.png [...]"""
import sys

import torch
from PIL import Image

from gerador import CARACTERES
from treinar import Especialista, para_tensor

modelo = Especialista()
modelo.load_state_dict(torch.load("modelo.pt"))
modelo.eval()
for caminho in sys.argv[1:]:
    with torch.no_grad():
        p = modelo(para_tensor(Image.open(caminho)).unsqueeze(0)).argmax(-1)[0]
    print(caminho, "->", "".join(CARACTERES[i] for i in p))
