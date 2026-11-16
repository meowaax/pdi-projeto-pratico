"""Segmentação automática dos gráficos (ROIs) da Figure1.png.

Etapas:
  1. `detectar_faixas_paineis`: acha as faixas verticais (y0, y1) dos três painéis
     usando a barra cinza de rótulos de amplitude, que existe à esquerda de cada
     painel e não existe nas legendas "(a)", "(b)" e "(c)".
  2. `analisar_painel`: dentro de cada faixa, mede a grade
     e a faixa de rótulos de tempo para obter a ROI interior e a calibração.

Nenhuma coordenada fixa é usada na detecção. `src/config.py` só serve de fallback.
"""
import logging
from dataclasses import dataclass

import numpy as np

from src import config

log = logging.getLogger(__name__)

# Pixel é "não branco" se a soma dos canais ficar abaixo disto (3 * 250).
LIMITE_NAO_BRANCO = 750


class RoiError(RuntimeError):
    """A detecção automática não encontrou a estrutura esperada."""


def faixas(mascara):
    """Trechos consecutivos True de um vetor booleano, como lista de (inicio, fim)."""
    mascara = np.asarray(mascara, dtype=bool)
    if mascara.size == 0:
        return []
    ext = np.concatenate(([False], mascara, [False])).astype(np.int8)
    d = np.diff(ext)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def mascara_nao_branco(rgb):
    """True onde o pixel não é (quase) branco."""
    return rgb.astype(np.int32).sum(axis=2) < LIMITE_NAO_BRANCO


def detectar_barra_lateral(nao_branco, fracao=0.5, largura_minima=10):
    """Colunas (x0, x1) da barra cinza de rótulos de amplitude, à esquerda dos painéis.

    É o primeiro trecho de colunas largo em que mais da metade dos pixels não é branca.
    """
    cobertura = nao_branco.mean(axis=0)
    for x0, x1 in faixas(cobertura > fracao):
        if x1 - x0 + 1 >= largura_minima:
            return int(x0), int(x1)
    raise RoiError("barra lateral dos painéis não encontrada")


def detectar_faixas_paineis(rgb, altura_minima_rel=0.1):
    """Faixas verticais (y0, y1) dos painéis, de cima para baixo.

    Um painel é um trecho de linhas em que a barra lateral é não branca e que tem
    pelo menos `altura_minima_rel` da altura da imagem. Isso descarta as legendas
    e a descrição da figura.
    """
    nao_branco = mascara_nao_branco(rgb)
    x0, x1 = detectar_barra_lateral(nao_branco)
    # Usa só o miolo da barra para ignorar suas bordas com antialiasing.
    miolo = nao_branco[:, x0 + 2:x1 - 1]
    altura_min = altura_minima_rel * rgb.shape[0]
    paineis = [(int(a), int(b)) for a, b in faixas(miolo.mean(axis=1) > 0.5) if b - a + 1 >= altura_min]
    if not paineis:
        raise RoiError("nenhum painel encontrado")
    return paineis


# ----------------------------------------------------------------------------
# Análise de cada painel: grade, faixa de rótulos de tempo e ROI interior
# ----------------------------------------------------------------------------
N_INTERVALOS_TEMPO = 10     # a grade tem 10 intervalos verticais por painel
N_INTERVALOS_AMPLITUDE = 6  # de +75% a -75%, em passos de 25%


@dataclass(frozen=True)
class Painel:
    """Resultado da segmentação de um painel (coordenadas em pixels da imagem)."""
    nome: str
    faixa: tuple          # (y0, y1) do painel inteiro, com moldura e rótulos de tempo
    roi: tuple            # (x0, y0, x1, y1) interior do gráfico, sem moldura nem rótulos
    x_esq: float          # x da borda esquerda do eixo de tempo
    x_dir: float          # x da borda direita do eixo de tempo
    y_75: float           # y da linha de +75%
    y_zero: float         # y da linha de 0%
    y_m75: float          # y da linha de -75%
    y_rotulos_tempo: int  # primeira linha da faixa de rótulos de tempo


def mascara_cinza(rgb):
    """Pixels cinza claro (R≈G≈B entre 180 e 249): grade e faixas de rótulos."""
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return (np.abs(r - g) < 10) & (np.abs(g - b) < 10) & (r >= 180) & (r < 250)


def _mesclar(trechos, folga=3):
    """Une trechos separados por no máximo `folga` posições (linhas de grade partidas)."""
    saida = []
    for a, b in trechos:
        if saida and a - saida[-1][1] <= folga:
            saida[-1] = (saida[-1][0], b)
        else:
            saida.append((a, b))
    return saida


def _centro(trecho):
    return (trecho[0] + trecho[1]) / 2.0


def _calibrar_amplitude(linhas):
    """Posições y de +75..-75% a partir das linhas horizontais da grade (de cima para baixo).

    A última linha é -75%. A linha de 0% costuma ficar coberta pela curva, então os
    valores são obtidos por ajuste linear sobre o índice de cada linha detectada.
    """
    ultimas = np.asarray(linhas[-N_INTERVALOS_AMPLITUDE:], dtype=float)
    if len(ultimas) < 4:
        raise RoiError("linhas horizontais da grade insuficientes")
    dif = np.diff(ultimas)
    passo0 = float(np.mean(dif[dif < 1.5 * dif.min()]))
    ks = np.round((ultimas[-1] - ultimas) / passo0).astype(int)  # 0 = -75%, 6 = +75%
    uteis = (ks >= 0) & (ks <= N_INTERVALOS_AMPLITUDE)
    if uteis.sum() < 4:
        raise RoiError("não foi possível indexar as linhas de amplitude")
    a, b = np.polyfit(ks[uteis], ultimas[uteis], 1)  # y = a*k + b
    ys = [a * k + b for k in range(N_INTERVALOS_AMPLITUDE, -1, -1)]  # +75 ... -75
    return ys, -float(a)


def analisar_painel(rgb, faixa, nome, x_inicio):
    """Mede a grade de um painel e devolve a ROI interior e a calibração geométrica.

    faixa    : (y0, y1) devolvido por detectar_faixas_paineis.
    x_inicio : primeira coluna depois da barra lateral de rótulos de amplitude.
    """
    y0, y1 = faixa
    cinza = mascara_cinza(rgb)[y0:y1 + 1, :]
    altura, largura = cinza.shape

    # Faixa de rótulos de tempo = último trecho largo de linhas quase todas cinza.
    por_linha = cinza.sum(axis=1)
    largos = [(a, b) for a, b in faixas(por_linha > 0.35 * largura) if b - a + 1 >= 8]
    if not largos:
        raise RoiError(f"painel {nome}: faixa de rótulos de tempo não encontrada")
    base = int(largos[-1][0] + y0)

    # Linhas horizontais da grade acima da faixa de rótulos
    linhas = [_centro(t) + y0 for t in faixas(por_linha > 0.7 * largura) if t[1] + y0 < base]
    ys, passo = _calibrar_amplitude(linhas)

    # Linhas verticais da grade: a k-ésima fica em x_esq + k * passo_x (k = 1..10)
    por_coluna = cinza[:, x_inicio:].sum(axis=0)
    colunas = [x_inicio + _centro(t) for t in _mesclar(faixas(por_coluna > 0.5 * altura))]
    if len(colunas) != N_INTERVALOS_TEMPO:
        raise RoiError(f"painel {nome}: {len(colunas)} linhas verticais, esperado {N_INTERVALOS_TEMPO}")
    k = np.arange(1, N_INTERVALOS_TEMPO + 1)
    passo_x, x_esq = np.polyfit(k, colunas, 1)
    x_dir = x_esq + N_INTERVALOS_TEMPO * passo_x

    # ROI: do topo (linha de 100% = +75% menos um passo) até acima dos rótulos de tempo
    roi_ = (int(round(x_esq)) + 2, int(round(ys[0] - passo)) + 2, int(round(x_dir)) - 1, base - 1)
    return Painel(nome=nome, faixa=(y0, y1), roi=roi_, x_esq=float(x_esq), x_dir=float(x_dir),
                  y_75=float(ys[0]), y_zero=float(ys[3]), y_m75=float(ys[-1]), y_rotulos_tempo=int(base))


def _painel_do_config(nome):
    c = config.PAINEIS[nome]
    return Painel(nome=nome, faixa=(c["roi"][1], c["roi"][3]), roi=tuple(c["roi"]), x_esq=c["x_esq"],
                  x_dir=c["x_dir"], y_75=c["y_75"], y_zero=c["y_zero"], y_m75=c["y_m75"],
                  y_rotulos_tempo=c["roi"][3] + 1)


def segmentar_paineis(rgb, usar_fallback=True):
    """Segmenta os três gráficos (a, b, c) da figura.

    Se a detecção automática falhar e `usar_fallback` for True, usa as coordenadas de
    src/config.py (medidas na Task 1B) e registra um aviso no log.
    """
    nomes = tuple(config.PAINEIS)
    try:
        x0, x1 = detectar_barra_lateral(mascara_nao_branco(rgb))
        faixas_ = detectar_faixas_paineis(rgb)
        if len(faixas_) != len(nomes):
            raise RoiError(f"{len(faixas_)} painéis detectados, esperado {len(nomes)}")
        return [analisar_painel(rgb, f, n, x1 + 1) for f, n in zip(faixas_, nomes)]
    except RoiError as erro:
        if not usar_fallback:
            raise
        log.warning("Detecção automática falhou (%s). Usando coordenadas do config.", erro)
        return [_painel_do_config(n) for n in nomes]


def recortar(rgb, painel):
    """Recorte do interior do gráfico (a ROI é inclusiva em x1 e y1)."""
    x0, y0, x1, y1 = painel.roi
    return rgb[y0:y1 + 1, x0:x1 + 1]


def desenhar_rois(rgb, paineis, cor=(255, 0, 0), espessura=1):
    """Cópia da figura com o retângulo da ROI e o nome de cada painel."""
    import cv2  # só esta função de visualização depende do OpenCV
    saida = np.ascontiguousarray(rgb.copy())
    for p in paineis:
        x0, y0, x1, y1 = p.roi
        cv2.rectangle(saida, (x0, y0), (x1, y1), cor, espessura)
        cv2.putText(saida, p.nome, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    return saida
