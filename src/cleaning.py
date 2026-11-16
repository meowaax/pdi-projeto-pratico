"""Limpeza dos recortes dos gráficos (grade, eixos, textos e anotações).

Remoção da grade (Task 5) e das anotações azuis (Task 6).

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
import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

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


# ----------------------------------------------------------------------------
# Remoção das anotações azuis (S1, S2, SM, VSD, "systole")
# ----------------------------------------------------------------------------
# Eixos, marcas de escala, números dos eixos e a legenda "Diaphragm mode; Normalized" são
# neutros (pretos ou cinza) e já saem em `remover_grade`. Restam as anotações em azul, que
# têm a mesma cor da curva e por isso são separadas pela forma:
#   curva     : traço fino e comprido, ocupa menos de ~15% da sua caixa envolvente;
#   anotação  : letras compactas, caixa pequena (~12 px de altura) e ocupação > 60%.
ALTURA_MAX_ANOTACAO = 20      # px (as letras da figura têm 8 a 12 px de altura)
LARGURA_MAX_ANOTACAO = 45     # px (o maior rótulo, "VSD", tem 30 px)
PREENCHIMENTO_MIN_ANOTACAO = 0.5   # área / (largura * altura) da caixa; a curva fica abaixo de 0,15
LARGURA_MIN_CURVA = 0.98      # fração da largura da ROI que a curva deve atravessar


@dataclass(frozen=True)
class SoCurva:
    """Recorte com apenas a curva (sem grade, eixos, textos e anotações)."""
    cobertura: np.ndarray       # float64 em [0, 1]
    mascara: np.ndarray         # bool
    imagem: np.ndarray          # uint8, curva escura sobre fundo branco
    anotacoes: tuple            # caixas (x, y, largura, altura) das anotações removidas
    caixa_curva: tuple          # caixa (x, y, largura, altura) do componente da curva
    largura_coberta: float      # fração das colunas do recorte atravessada pela curva


def classificar_componentes(mascara):
    """Rotula os componentes conexos (8 vizinhos) e separa curva de anotações.

    Retorna (rotulos, indice_da_curva, indices_das_anotacoes, estatisticas). A curva é o
    componente de maior área. Os outros são anotações só se forem pequenos e compactos;
    pedaços finos ou longos continuam, pois podem ser fragmentos da curva.
    """
    n, rotulos, est, _ = cv2.connectedComponentsWithStats(mascara.astype(np.uint8), connectivity=8)
    if n <= 1:
        return rotulos, None, [], est
    curva = 1 + int(np.argmax(est[1:, cv2.CC_STAT_AREA]))
    anotacoes = []
    for i in range(1, n):
        if i == curva:
            continue
        x, y, w, h, area = est[i]
        if h <= ALTURA_MAX_ANOTACAO and w <= LARGURA_MAX_ANOTACAO and area / (w * h) >= PREENCHIMENTO_MIN_ANOTACAO:
            anotacoes.append(i)
    return rotulos, curva, anotacoes, est


def remover_anotacoes(sem_grade):
    """Apaga as anotações de um `SemGrade` e devolve um `SoCurva`.

    Emite um aviso no log se a curva restante não atravessar a largura inteira do recorte
    (sinal de que uma anotação encostou na curva ou de que a curva ficou fragmentada).
    """
    mascara = sem_grade.mascara
    rotulos, curva, anotacoes, est = classificar_componentes(mascara)
    cobertura = sem_grade.cobertura.copy()
    caixas = []
    for i in anotacoes:
        cobertura[rotulos == i] = 0.0
        caixas.append(tuple(int(v) for v in est[i, :4]))
    nova_mascara = cobertura > 0.0
    if curva is None:
        caixa_curva, coberta = (0, 0, 0, 0), 0.0
    else:
        caixa_curva = tuple(int(v) for v in est[curva, :4])
        coberta = caixa_curva[2] / mascara.shape[1]
    if coberta < LARGURA_MIN_CURVA:
        log.warning("A curva cobre só %.1f%% da largura do recorte; confira anotações coladas ou fragmentos.",
                    100 * coberta)
    imagem = np.rint(255.0 * (1.0 - cobertura)).astype(np.uint8)
    return SoCurva(cobertura=cobertura, mascara=nova_mascara, imagem=imagem,
                   anotacoes=tuple(caixas), caixa_curva=caixa_curva, largura_coberta=float(coberta))


def limpar_recorte(rgb):
    """Atalho: remove grade, eixos, textos e anotações de um recorte RGB."""
    return remover_anotacoes(remover_grade(rgb))
