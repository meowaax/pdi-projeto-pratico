"""Demo da Task 6: sinal limpo, sem grade, eixos, textos nem anotações.

Saída em outputs/:
  clean_a.png, clean_b.png, clean_c.png : curva isolada, por painel
  etapas_limpeza.png                    : original, sem grade e sinal limpo, lado a lado

Uso: python demo_limpeza.py
"""
import logging

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.cleaning import remover_anotacoes, remover_grade
from src.roi import recortar, segmentar_paineis


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.SAIDA.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()

    paineis = segmentar_paineis(rgb, usar_fallback=False)
    fig, axs = plt.subplots(len(paineis), 3, figsize=(20, 3.6 * len(paineis)))
    for linha, p in zip(axs, paineis):
        recorte = recortar(rgb, p)
        sem_grade = remover_grade(recorte)
        limpo = remover_anotacoes(sem_grade)
        cv2.imwrite(str(config.SAIDA / f"clean_{p.nome}.png"), limpo.imagem)
        print(f"painel {p.nome}: {len(limpo.anotacoes)} anotações removidas, curva com "
              f"{int(limpo.mascara.sum())} pixels e {100 * limpo.largura_coberta:.0f}% da largura")
        for ax, img, titulo in zip(linha, (recorte, sem_grade.imagem, limpo.imagem),
                                   ("original", "sem grade", "sinal limpo")):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=255)
            ax.set_title(f"painel {p.nome} ({config.PAINEIS[p.nome]['nome']}): {titulo}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.SAIDA / "etapas_limpeza.png", dpi=100)
    plt.close(fig)
    print(f"Imagens salvas em {config.SAIDA}")


if __name__ == "__main__":
    main()
