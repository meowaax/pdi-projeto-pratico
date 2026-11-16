"""Demo da Task 5: remoção da grade em cada painel.

Saída em outputs/:
  sem_grade_a.png, sem_grade_b.png, sem_grade_c.png : curva e anotações sem a grade
  grade_antes_depois.png                            : original x sem grade, por painel

Uso: python demo_grade.py
"""
import logging

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.cleaning import remover_grade
from src.roi import recortar, segmentar_paineis


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.SAIDA.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()

    paineis = segmentar_paineis(rgb, usar_fallback=False)
    fig, axs = plt.subplots(len(paineis), 2, figsize=(14, 3.6 * len(paineis)))
    for linha, p in zip(axs, paineis):
        recorte = recortar(rgb, p)
        r = remover_grade(recorte)
        cv2.imwrite(str(config.SAIDA / f"sem_grade_{p.nome}.png"), r.imagem)
        print(f"painel {p.nome}: {r.mascara.sum()} pixels azuis de {r.mascara.size} "
              f"({100 * r.mascara.mean():.1f}%), referência de azulado = {r.azul_de_referencia:.0f}")
        linha[0].imshow(recorte)
        linha[0].set_title(f"painel {p.nome} ({config.PAINEIS[p.nome]['nome']}): original")
        linha[1].imshow(r.imagem, cmap="gray", vmin=0, vmax=255)
        linha[1].set_title(f"painel {p.nome}: sem grade")
        for ax in linha:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.SAIDA / "grade_antes_depois.png", dpi=110)
    plt.close(fig)
    print(f"Imagens salvas em {config.SAIDA}")


if __name__ == "__main__":
    main()
