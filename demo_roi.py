"""Demo da Task 4: segmenta os três gráficos da Figure1.png.

Saída em outputs/:
  roi_a.png, roi_b.png, roi_c.png : recortes do interior de cada gráfico
  rois_detectadas.png             : figura original com as ROIs desenhadas

Uso: python demo_roi.py
"""
import logging

import cv2

from src import config
from src.roi import recortar, segmentar_paineis, desenhar_rois


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.SAIDA.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)
    rgb = bgr[:, :, ::-1].copy()

    paineis = segmentar_paineis(rgb, usar_fallback=False)  # a demo deve usar a detecção automática
    for p in paineis:
        recorte = recortar(rgb, p)
        cv2.imwrite(str(config.SAIDA / f"roi_{p.nome}.png"), recorte[:, :, ::-1])
        x0, y0, x1, y1 = p.roi
        print(f"painel {p.nome} ({config.PAINEIS[p.nome]['nome']}): ROI x[{x0}, {x1}] y[{y0}, {y1}] "
              f"-> {recorte.shape[1]}x{recorte.shape[0]} px")

    cv2.imwrite(str(config.SAIDA / "rois_detectadas.png"), desenhar_rois(rgb, paineis)[:, :, ::-1])
    print(f"Imagens salvas em {config.SAIDA}")


if __name__ == "__main__":
    main()
