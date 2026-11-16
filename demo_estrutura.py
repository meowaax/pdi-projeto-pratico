"""Demo da Task 7: suavização, bordas, binarização e afinamento da curva.

Saída em outputs/:
  estrutura_<a|b|c>.png    : sinal limpo, suavizado, bordas de Sobel, binarizado e esqueleto
  esqueleto_<a|b|c>.png    : curva de 1 px (branco sobre preto não; preto sobre branco)
  varredura_sigma.md       : tabela de métricas por painel, σ e limiar

Uso: python demo_estrutura.py
"""
import logging

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.cleaning import limpar_recorte
from src.roi import recortar, segmentar_paineis
from src.structure import realcar_curva, varrer_parametros

SIGMAS = (0.6, 0.8, 1.0, 1.5, 2.0)
LIMIARES = (0.06, "otsu")


def tabela_markdown(nome, linhas):
    cab = ("| σ | limiar | componentes | colunas vazias | espessura mediana (px) | erro dos extremos (px) "
           "| viés do pico (px) | pixels do esqueleto |")
    sep = "|---|---|---|---|---|---|---|---|"
    corpo = []
    for m in linhas:
        lim = "Otsu" if m["limiar_param"] == "otsu" else f"{m['limiar_param']:g}"
        if m["limiar_param"] == "otsu":
            lim += f" ({m['limiar']:.3f})"
        corpo.append(f"| {m['sigma']:g} | {lim} | {m['componentes']} | {m['colunas_vazias']} | "
                     f"{m['espessura_mediana_px']:.0f} | {m['erro_extremos_px']:.2f} | {m['vies_pico_px']:+.1f} | "
                     f"{m['esqueleto_pixels']} |")
    return f"### Painel {nome} ({config.PAINEIS[nome]['nome']})\n\n" + "\n".join([cab, sep] + corpo) + "\n"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.SAIDA.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()

    tabelas = []
    for p in segmentar_paineis(rgb, usar_fallback=False):
        limpo = limpar_recorte(recortar(rgb, p))
        e = realcar_curva(limpo.cobertura)  # parâmetros padrão

        fig, axs = plt.subplots(5, 1, figsize=(12, 11))
        mostrar = [("sinal limpo (Task 6)", limpo.imagem, "gray"),
                   (f"suavizado (Gaussiano σ={e.sigma:g})", 255 - 255 * e.suavizada, "gray"),
                   ("bordas (magnitude de Sobel)", e.bordas, "gray_r"),
                   (f"binarizado (limiar {e.limiar:g})", ~e.mascara, "gray"),
                   ("esqueleto (1 px)", ~e.esqueleto, "gray")]
        for ax, (titulo, img, cmap) in zip(axs, mostrar):
            ax.imshow(img, cmap=cmap)
            ax.set_title(f"painel {p.nome} ({config.PAINEIS[p.nome]['nome']}): {titulo}", fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(config.SAIDA / f"estrutura_{p.nome}.png", dpi=100)
        plt.close(fig)
        cv2.imwrite(str(config.SAIDA / f"esqueleto_{p.nome}.png"), np.where(e.esqueleto, 0, 255).astype(np.uint8))

        linhas = varrer_parametros(limpo.cobertura, limpo.mascara, SIGMAS, LIMIARES)
        tabelas.append(tabela_markdown(p.nome, linhas))
        print(f"painel {p.nome}: máscara {int(e.mascara.sum())} px, esqueleto {int(e.esqueleto.sum())} px")

    (config.SAIDA / "varredura_sigma.md").write_text("\n".join(tabelas), encoding="utf-8")
    print(f"Imagens e tabela salvas em {config.SAIDA}")


if __name__ == "__main__":
    main()
