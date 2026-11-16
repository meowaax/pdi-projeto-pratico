"""Demo dos Requisitos 1 e 3: média, Gaussiano e Sobel com correlate2d_manual.

Entradas : Imagens/Shapes.png e Imagens/testpat.1k.color2.tif
Saída    : outputs/req1/<imagem>_<filtro>.png, outputs/req1/resumo.png
           e outputs/req1/shapes_bordas.png (mapa de bordas do Shapes.png)

Os kernels e filtros vêm de src/kernels.py e src/filters.py.
"""
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.correlation import correlate2d_manual
from src.filters import bordas_sobel, gradiente_sobel, suavizar_gaussiano
from src.kernels import kernel_media

RAIZ = Path(__file__).parent
SAIDA = RAIZ / "outputs" / "req1"
IMAGENS = [RAIZ / "Imagens" / "Shapes.png", RAIZ / "Imagens" / "testpat.1k.color2.tif"]


def para_cinza(bgr):
    """Luminância BT.601 a partir de uma imagem BGR."""
    b, g, r = (bgr[..., i].astype(np.float64) for i in range(3))
    return 0.299 * r + 0.587 * g + 0.114 * b


def normalizar(x):
    """Leva qualquer resultado para uint8 [0, 255] só para visualização."""
    x = x - x.min()
    return (255 * x / x.max()).astype(np.uint8) if x.max() > 0 else x.astype(np.uint8)


def processar(caminho):
    cinza = para_cinza(cv2.imread(str(caminho), cv2.IMREAD_COLOR))
    t0 = time.perf_counter()
    gx, gy, magnitude, _ = gradiente_sobel(cinza)
    resultados = {
        "original": cinza,
        "media_5x5": correlate2d_manual(cinza, kernel_media(5)),
        "gaussiano_s1.5": suavizar_gaussiano(cinza, 1.5),
        "sobel_x": gx,
        "sobel_y": gy,
        "sobel_magnitude": magnitude,
    }
    print(f"{caminho.name}: {cinza.shape}, 5 filtros em {time.perf_counter() - t0:.3f} s")
    for nome, img in resultados.items():
        cv2.imwrite(str(SAIDA / f"{caminho.stem}_{nome}.png"), normalizar(img))
    return resultados


def mapa_de_bordas_shapes():
    """Triângulo, quadrado e losango: borda com e sem suavização prévia."""
    cinza = para_cinza(cv2.imread(str(IMAGENS[0]), cv2.IMREAD_COLOR))
    fig, axs = plt.subplots(1, 3, figsize=(18, 3.2))
    for ax, (titulo, img) in zip(axs, [("original", cinza),
                                       ("bordas Sobel", bordas_sobel(cinza)),
                                       ("bordas Sobel (Gaussiano σ=1.5 antes)", bordas_sobel(cinza, sigma=1.5))]):
        ax.imshow(img, cmap="gray")
        ax.set_title(titulo)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(SAIDA / "shapes_bordas.png", dpi=100)
    plt.close(fig)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(len(IMAGENS), 6, figsize=(24, 4.5 * len(IMAGENS)))
    for linha, caminho in zip(axs, IMAGENS):
        for ax, (nome, img) in zip(linha, processar(caminho).items()):
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{caminho.stem}\n{nome}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(SAIDA / "resumo.png", dpi=90)
    plt.close(fig)
    mapa_de_bordas_shapes()
    print(f"Resultados em {SAIDA}")


if __name__ == "__main__":
    main()
