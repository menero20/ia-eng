"""IA Eng - gerador de enigmas (letras distorcidas) com a resposta certa.

Cada chamada a `enigma()` cria uma imagem nova e devolve (imagem, texto).
Rodar direto salva exemplos em amostras/ para ver como ficam.
"""
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

CARACTERES = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sem O/0 e I/1, que confundem
TAMANHO = 5
LARGURA, ALTURA = 160, 60
FONTES = [f for f in (
    "arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "cour.ttf", "courbd.ttf",
    "verdana.ttf", "verdanab.ttf", "georgia.ttf", "georgiab.ttf", "comic.ttf", "tahoma.ttf",
    "trebuc.ttf", "calibri.ttf", "segoeui.ttf",
) if os.path.exists(os.path.join("C:/Windows/Fonts", f))]


def _cor(clara):
    base = random.randint(170, 255) if clara else random.randint(0, 110)
    return tuple(min(255, max(0, base + random.randint(-40, 40))) for _ in range(3))


def enigma():
    texto = "".join(random.choice(CARACTERES) for _ in range(TAMANHO))
    img = Image.new("RGB", (LARGURA, ALTURA), _cor(True))
    d = ImageDraw.Draw(img)

    for _ in range(random.randint(0, 300)):  # ruído de pontos
        d.point((random.randrange(LARGURA), random.randrange(ALTURA)), fill=_cor(random.random() < 0.5))

    x = random.randint(2, 14)
    for ch in texto:  # cada letra com fonte, tamanho, cor e giro próprios
        fonte = ImageFont.truetype(os.path.join("C:/Windows/Fonts", random.choice(FONTES)), random.randint(28, 40))
        letra = Image.new("RGBA", (50, 60), (0, 0, 0, 0))
        ImageDraw.Draw(letra).text((8, 4), ch, font=fonte, fill=_cor(False))
        letra = letra.rotate(random.uniform(-30, 30), resample=Image.BICUBIC, expand=False)
        img.paste(letra, (x - 8, random.randint(-8, 6)), letra)
        x += random.randint(22, 30)

    for _ in range(random.randint(1, 4)):  # riscos por cima
        d.line([(random.randrange(LARGURA), random.randrange(ALTURA)) for _ in range(2)],
               fill=_cor(False), width=random.randint(1, 3))

    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    return img, texto


if __name__ == "__main__":
    os.makedirs("amostras", exist_ok=True)
    for i in range(12):
        img, texto = enigma()
        img.save(f"amostras/{i:02d}_{texto}.png")
    print("12 exemplos em amostras/")
