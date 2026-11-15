"""Demo do Requisito 1: aplica média, Gaussiano e Sobel com correlate2d_manual.

Entradas : Imagens/Shapes.png e Imagens/testpat.1k.color2.tif
Saída    : outputs/req1/<imagem>_<filtro>.png e outputs/req1/resumo.png

Os kernels abaixo são locais ao demo. A Task 3 cria o módulo definitivo de máscaras.
"""
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.correlation import correlate2d_manual

RAIZ = Path(__file__).parent
SAIDA = RAIZ / "outputs" / "req1"
IMAGENS = [RAIZ / "Imagens" / "Shapes.png", RAIZ / "Imagens" / "testpat.1k.color2.tif"]


def para_cinza(bgr):
    """Luminância BT.601 a partir de uma imagem BGR."""
    b, g, r = (bgr[..., i].astype(np.float64) for i in range(3))
    return 0.299 * r + 0.587 * g + 0.114 * b


def kernel_gaussiano(tamanho, sigma):
    eixo = np.arange(tamanho) - (tamanho - 1) / 2
    g = np.exp(-(eixo[:, None] ** 2 + eixo[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T
MEDIA_5 = np.ones((5, 5)) / 25.0
GAUSS_7 = kernel_gaussiano(7, 1.5)


def normalizar(x):
    """Leva qualquer resultado para uint8 [0, 255] só para visualização."""
    x = x - x.min()
    return (255 * x / x.max()).astype(np.uint8) if x.max() > 0 else x.astype(np.uint8)


def processar(caminho):
    bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    cinza = para_cinza(bgr)
    t0 = time.perf_counter()
    gx = correlate2d_manual(cinza, SOBEL_X)
    gy = correlate2d_manual(cinza, SOBEL_Y)
    resultados = {
        "original": cinza,
        "media_5x5": correlate2d_manual(cinza, MEDIA_5),
        "gaussiano_7x7_s1.5": correlate2d_manual(cinza, GAUSS_7),
        "sobel_x": gx,
        "sobel_y": gy,
        "sobel_magnitude": np.hypot(gx, gy),
    }
    dt = time.perf_counter() - t0
    print(f"{caminho.name}: {cinza.shape}, 5 correlações em {dt:.3f} s")
    for nome, img in resultados.items():
        cv2.imwrite(str(SAIDA / f"{caminho.stem}_{nome}.png"), normalizar(img))
    return resultados


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
    print(f"Resultados em {SAIDA}")


if __name__ == "__main__":
    main()
