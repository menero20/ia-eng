"""IA Eng fase 2 - monta o banco de fotos de rua a partir do COCO (cocodataset.org, uso de estudo).

Baixa as anotações oficiais, escolhe fotos com as classes de interesse, baixa só essas fotos
e recorta cada objeto em fotos/dados/<treino|prova>/<classe>/. Também recorta pedaços sem
nenhum desses objetos para a classe "nada".
"""
import io
import json
import os
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
CLASSES = {"fire hydrant": "hidrante", "traffic light": "semaforo", "bus": "onibus", "bicycle": "bicicleta",
           "car": "carro", "motorcycle": "moto", "stop sign": "placa_pare"}
POR_CLASSE = 1200
ANOT = os.path.join(AQUI, "cache", "instances_train2017.json")
random.seed(7)


def baixar_anotacoes():
    if os.path.exists(ANOT):
        return
    os.makedirs(os.path.dirname(ANOT), exist_ok=True)
    print("baixando anotações (241 MB)...", flush=True)
    dados = urllib.request.urlopen("http://images.cocodataset.org/annotations/annotations_trainval2017.zip").read()
    with zipfile.ZipFile(io.BytesIO(dados)) as z:
        with open(ANOT, "wb") as f:
            f.write(z.read("annotations/instances_train2017.json"))


def sobrepoe(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def main():
    baixar_anotacoes()
    coco = json.load(open(ANOT))
    nome_cat = {c["id"]: CLASSES[c["name"]] for c in coco["categories"] if c["name"] in CLASSES}
    imgs = {i["id"]: i for i in coco["images"]}
    caixas = {}  # imagem -> [(classe, bbox)]
    for a in coco["annotations"]:
        if a["category_id"] in nome_cat and not a["iscrowd"]:
            caixas.setdefault(a["image_id"], []).append((nome_cat[a["category_id"]], a["bbox"]))

    escolhidas, conta = [], {c: 0 for c in CLASSES.values()}
    ids = list(caixas)
    random.shuffle(ids)
    total = {c: sum(1 for v in caixas.values() for k, _ in v if k == c) for c in conta}
    ids.sort(key=lambda i: min(total[c] for c, _ in caixas[i]))  # fotos com classe rara primeiro
    for i in ids:
        grandes = [(c, b) for c, b in caixas[i] if b[2] >= 32 and b[3] >= 32 and conta[c] < POR_CLASSE]
        if grandes:
            escolhidas.append(i)
            for c, _ in grandes:
                conta[c] += 1
        if all(v >= POR_CLASSE for v in conta.values()):
            break
    print(f"{len(escolhidas)} fotos para baixar; objetos por classe: {conta}", flush=True)

    def processar(i):
        info = imgs[i]
        parte = "prova" if i % 5 == 0 else "treino"  # 20% separado por foto, nunca visto no treino
        try:
            img = Image.open(io.BytesIO(urllib.request.urlopen(info["coco_url"], timeout=60).read())).convert("RGB")
        except Exception as e:
            return f"falhou {i}: {e}"
        W, H = img.size
        for n, (c, (x, y, w, h)) in enumerate(caixas[i]):
            if w < 32 or h < 32:
                continue
            m = 0.15
            corte = img.crop((max(0, x - m * w), max(0, y - m * h), min(W, x + w * (1 + m)), min(H, y + h * (1 + m))))
            d = os.path.join(AQUI, "dados", parte, c)
            os.makedirs(d, exist_ok=True)
            corte.resize((160, 160)).save(os.path.join(d, f"{i}_{n}.jpg"), quality=90)
        for n in range(2):  # pedaço da mesma rua sem nenhum objeto de interesse
            for _ in range(20):
                s = random.randint(64, max(65, min(W, H) // 2))
                x, y = random.randint(0, W - s), random.randint(0, H - s)
                if not any(sobrepoe((x, y, s, s), b) for _, b in caixas[i]):
                    d = os.path.join(AQUI, "dados", parte, "nada")
                    os.makedirs(d, exist_ok=True)
                    img.crop((x, y, x + s, y + s)).resize((160, 160)).save(os.path.join(d, f"{i}_{n}.jpg"), quality=90)
                    break
        return None

    with ThreadPoolExecutor(16) as ex:
        for k, erro in enumerate(ex.map(processar, escolhidas), 1):
            if erro:
                print(erro, flush=True)
            if k % 500 == 0:
                print(f"{k}/{len(escolhidas)} fotos", flush=True)
    for parte in ("treino", "prova"):
        base = os.path.join(AQUI, "dados", parte)
        print(parte, {c: len(os.listdir(os.path.join(base, c))) for c in sorted(os.listdir(base))}, flush=True)


if __name__ == "__main__":
    main()
