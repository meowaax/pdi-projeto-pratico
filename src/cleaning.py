"""Limpeza dos recortes dos gráficos (grade, eixos, textos).

Este módulo começa pela remoção da grade (Task 5). Eixos, ticks e textos entram na Task 6.

Como a grade é removida
-----------------------
A grade, os eixos e os textos pretos são neutros (R = G = B), enquanto a curva e as
anotações são azuis. Medimos o "azulado" de cada pixel:

    azulado = max(0, B - max(R, G))

Num pixel neutro o azulado é 0. Num pixel que mistura a curva (opacidade a) com um
fundo neutro qualquer (branco ou cinza da grade), o azulado é a * azulado_da_curva,
e não depende do fundo. Por isso o azulado dá a cobertura da curva em cada pixel,
mesmo onde ela cruza uma linha da grade, e as bordas suavizadas (antialiasing) da
curva sobrevivem sem deixar rastro da grade.

Na Figure1.png o azulado dos pixels da grade e do fundo chega a 9 (ruído de compressão),
então o limiar padrão é 12.
"""
from dataclasses import dataclass

import numpy as np

LIMIAR_AZUL = 12        # azulado máximo do ruído de compressão (medido: 9) com folga
PERCENTIL_REFERENCIA = 99.0


@dataclass(frozen=True)
class SemGrade:
    """Recorte sem grade: só o que é azul (curva e anotações) permanece."""
    cobertura: np.ndarray   # float64 em [0, 1], 1 = azul pleno
    mascara: np.ndarray     # bool, True onde há azul acima do limiar
    imagem: np.ndarray      # uint8 em tons de cinza, azul escuro sobre fundo branco
    azul_de_referencia: float  # azulado que corresponde a cobertura 1


def azulado(rgb):
    """Azulado de cada pixel: max(0, B - max(R, G)), como int16."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Esperava imagem RGB (H, W, 3), recebido {rgb.shape}")
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return np.maximum(b - np.maximum(r, g), 0)


def remover_grade(rgb, limiar=LIMIAR_AZUL):
    """Remove grade, eixos e textos pretos de um recorte RGB, mantendo a curva.

    A cobertura (0 a 1) é linear no azulado acima do limiar e satura no percentil 99
    dos pixels azuis. A imagem de saída desenha a cobertura como azul escuro sobre
    branco, que é o formato esperado pelos filtros das próximas tasks.
    """
    az = azulado(rgb).astype(np.float64)
    mascara = az > limiar
    if not mascara.any():
        cobertura = np.zeros(az.shape)
        referencia = limiar + 1.0
    else:
        referencia = float(np.percentile(az[mascara], PERCENTIL_REFERENCIA))
        referencia = max(referencia, limiar + 1.0)
        cobertura = np.clip((az - limiar) / (referencia - limiar), 0.0, 1.0)
        cobertura[~mascara] = 0.0
    imagem = np.rint(255.0 * (1.0 - cobertura)).astype(np.uint8)
    return SemGrade(cobertura=cobertura, mascara=mascara, imagem=imagem, azul_de_referencia=referencia)
