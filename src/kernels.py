"""Máscaras (kernels) de filtragem espacial para uso com correlate2d_manual.

Todas retornam arrays float64 2D. Os kernels de suavização somam 1 (preservam o
brilho médio) e os de derivada (Sobel) somam 0 (resposta nula em regiões constantes).
"""
import math

import numpy as np

# Sobel como máscara de correlação: resposta positiva quando o brilho AUMENTA no
# sentido do eixo (x para a direita em SOBEL_X, y para baixo em SOBEL_Y).
SOBEL_X = np.array([[-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.T.copy()


def kernel_media(tamanho):
    """Média (box filter) tamanho x tamanho, com soma 1."""
    tamanho = _tamanho_impar(tamanho)
    return np.full((tamanho, tamanho), 1.0 / (tamanho * tamanho))


def tamanho_gaussiano(sigma):
    """Menor tamanho ímpar que cobre ±3σ (cerca de 99,7% da massa por eixo)."""
    _checar_sigma(sigma)
    return 2 * math.ceil(3.0 * sigma) + 1


def kernel_gaussiano(sigma, tamanho=None):
    """Gaussiana 2D isotrópica, normalizada para somar 1.

    sigma   : desvio padrão em pixels (> 0).
    tamanho : lado do kernel (ímpar). Se omitido, usa `tamanho_gaussiano(sigma)`.
    """
    _checar_sigma(sigma)
    tamanho = tamanho_gaussiano(sigma) if tamanho is None else _tamanho_impar(tamanho)
    eixo = np.arange(tamanho, dtype=np.float64) - (tamanho - 1) / 2.0
    g1d = np.exp(-(eixo ** 2) / (2.0 * sigma ** 2))
    g2d = np.outer(g1d, g1d)  # a Gaussiana 2D é separável
    return g2d / g2d.sum()


def _tamanho_impar(tamanho):
    if int(tamanho) != tamanho or tamanho < 1 or tamanho % 2 == 0:
        raise ValueError(f"O tamanho do kernel deve ser um inteiro ímpar >= 1, recebido {tamanho!r}")
    return int(tamanho)


def _checar_sigma(sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma deve ser um número finito > 0, recebido {sigma!r}")
