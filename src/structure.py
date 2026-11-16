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

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

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
    esqueleto: np.ndarray    # bool, curva afinada para 1 px de espessura


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
    mascara = suavizada > valor
    return Estrutura(sigma=float(sigma), limiar=valor, suavizada=suavizada, bordas=bordas,
                     mascara=mascara, esqueleto=afinar(mascara))


def afinar(mascara, margem=4):
    """Afina a máscara para uma curva de 1 px de espessura (esqueleto de Lee, 8-conexo).

    O afinamento encurta as pontas de um traço. Como a curva atravessa o recorte de lado
    a lado, a máscara é estendida `margem` pixels replicando a borda antes de afinar e
    depois recortada de volta, para que o esqueleto continue cobrindo todas as colunas.
    """
    mascara = np.asarray(mascara, dtype=bool)
    estendida = np.pad(mascara, margem, mode="edge")
    esqueleto = skeletonize(estendida, method="lee").astype(bool)
    return esqueleto[margem:-margem, margem:-margem]


def tem_espessura_um(esqueleto):
    """True se não existe nenhum bloco 2x2 totalmente preenchido (curva realmente fina)."""
    e = np.asarray(esqueleto, dtype=bool)
    return not (e[:-1, :-1] & e[1:, :-1] & e[:-1, 1:] & e[1:, 1:]).any()


# ----------------------------------------------------------------------------
# Métricas para comparar valores de σ e de limiar
# ----------------------------------------------------------------------------
def _extremos_por_coluna(mascara):
    """(topo, base) de cada coluna, e a máscara das colunas com pelo menos um pixel."""
    cheias = mascara.any(axis=0)
    topo = np.where(cheias, mascara.argmax(axis=0), -1).astype(float)
    base = np.where(cheias, mascara.shape[0] - 1 - mascara[::-1].argmax(axis=0), -1).astype(float)
    return topo, base, cheias


def _componentes(mascara):
    return cv2.connectedComponents(mascara.astype(np.uint8), connectivity=8)[0] - 1


def metricas(referencia, estrutura, n_picos=40):
    """Compara a máscara realçada com a máscara de referência (curva original sem grade).

    erro_extremos_px : diferença média, em pixels, entre topo/base de cada coluna.
    vies_pico_px     : viés do topo/base nas `n_picos` colunas com maior extensão vertical.
                       Positivo = curva ficou mais curta que a original (pico achatado),
                       negativo = ficou mais alta (pico alongado pelo borramento).
    """
    ref_topo, ref_base, ref_cheias = _extremos_por_coluna(referencia)
    topo, base, cheias = _extremos_por_coluna(estrutura.mascara)
    comuns = ref_cheias & cheias
    erro = 0.5 * (np.abs(topo - ref_topo)[comuns].mean() + np.abs(base - ref_base)[comuns].mean())
    picos = np.argsort(-(ref_base - ref_topo))[:n_picos]
    picos = picos[comuns[picos]]
    vies = 0.5 * ((topo[picos] - ref_topo[picos]).mean() + (ref_base[picos] - base[picos]).mean())
    return {
        "sigma": estrutura.sigma,
        "limiar": round(estrutura.limiar, 4),
        "componentes": int(_componentes(estrutura.mascara)),
        "colunas_vazias": int((~cheias).sum()),
        "espessura_mediana_px": float(np.median(estrutura.mascara.sum(axis=0)[cheias])),
        "erro_extremos_px": float(erro),
        "vies_pico_px": float(vies),
        "esqueleto_pixels": int(estrutura.esqueleto.sum()),
        "esqueleto_componentes": int(_componentes(estrutura.esqueleto)),
    }


def varrer_parametros(cobertura, referencia, sigmas, limiares):
    """Lista de métricas para cada combinação (σ, limiar). `limiares` aceita 'otsu'."""
    return [metricas(referencia, realcar_curva(cobertura, s, l)) | {"limiar_param": l}
            for s in sigmas for l in limiares]
