"""Redução de ruído e realce da estrutura da curva (Task 7).

Entrada : cobertura da curva (0 a 1, "tinta" sobre fundo 0), saída de `cleaning.limpar_recorte`.
Etapas  : suavização Gaussiana -> bordas de Sobel -> binarização -> afinamento.

Toda a filtragem espacial usa `correlate2d_manual` (Requisito 1). O fundo vale 0 na
cobertura, então o zero padding da correlação equivale a continuar o fundo branco.

Por que σ pequeno
-----------------
O traço tem 2 a 3 px de espessura e os picos de S1/S2 têm 1 px de largura. Um Gaussiano
largo achata esses picos e engrossa a curva, o que distorce a amplitude do áudio. A figura
já chega sem o ruído de compressão (ele foi cortado pelo limiar de azulado na Task 5),
então a suavização só precisa fechar as falhas de 1 px do antialiasing. Os valores padrão
vieram da varredura de σ e do limiar documentada em docs/filtragem_sigma.md.
"""
from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_otsu

from src.filters import bordas_sobel, suavizar_gaussiano

SIGMA_PADRAO = 0.6      # kernel 5x5 (±3σ)
LIMIAR_PADRAO = 0.06    # fração da cobertura máxima suavizada acima da qual há curva


@dataclass(frozen=True)
class Estrutura:
    """Resultado do realce da curva."""
    sigma: float
    limiar: float            # limiar efetivamente usado na binarização
    suavizada: np.ndarray    # float64, cobertura após o Gaussiano
    bordas: np.ndarray       # float64, magnitude do gradiente de Sobel da suavizada
    mascara: np.ndarray      # bool, curva binarizada (2 a 4 px de espessura)


def calcular_limiar(suavizada, limiar=LIMIAR_PADRAO):
    """Converte o parâmetro `limiar` em um valor numérico: 'otsu' ou um número em (0, 1)."""
    if isinstance(limiar, str):
        if limiar.lower() != "otsu":
            raise ValueError(f"limiar deve ser 'otsu' ou um número, recebido {limiar!r}")
        return float(threshold_otsu(suavizada))
    limiar = float(limiar)
    if not 0.0 < limiar < 1.0:
        raise ValueError(f"limiar deve estar em (0, 1), recebido {limiar}")
    return limiar


def binarizar(suavizada, limiar=LIMIAR_PADRAO):
    """Máscara booleana dos pixels de curva (suavizada > limiar)."""
    return suavizada > calcular_limiar(suavizada, limiar)


def realcar_curva(cobertura, sigma=SIGMA_PADRAO, limiar=LIMIAR_PADRAO):
    """Suaviza, detecta bordas e binariza a cobertura da curva."""
    cobertura = np.asarray(cobertura, dtype=np.float64)
    if cobertura.ndim != 2:
        raise ValueError(f"A cobertura deve ser 2D, recebido ndim={cobertura.ndim}")
    suavizada = suavizar_gaussiano(cobertura, sigma)
    bordas = bordas_sobel(suavizada)
    valor = calcular_limiar(suavizada, limiar)
    return Estrutura(sigma=float(sigma), limiar=valor, suavizada=suavizada, bordas=bordas,
                     mascara=suavizada > valor)
