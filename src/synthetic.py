"""Sinais sintéticos com gabarito conhecido, para validar a extração por ida e volta.

O sinal é desenhado como a Figure1 (traço de cerca de 2,5 px com antialiasing, 1333 colunas
por segundo) e depois passa pelo mesmo pipeline de extração. Como o sinal original é
conhecido, dá para medir o erro da reconstrução.
"""
import cv2
import numpy as np

from src.curve import Calibracao

FS_GABARITO = 8000      # taxa do sinal original, igual à gravação do artigo
FORMA_PADRAO = (208, 664)


def calibracao_sintetica(forma=FORMA_PADRAO, px_por_s=1333.2, px_por_pct=1.06):
    """Calibração sem deslocamento: coluna 0 em t = 0, linha de 0% no meio do recorte."""
    altura, largura = forma
    y_zero = altura / 2.0 - 0.5
    return Calibracao(x0=0, y0=0, x_esq=0.0, x_dir=0.5 * px_por_s, t_ini=0.0, duracao=0.5,
                      y_zero=y_zero, y_75=y_zero - 75 * px_por_pct, y_m75=y_zero + 75 * px_por_pct)


def sinal_sintetico(tipo="burst", semente=0):
    """Sinal de 0,5 s a 8 kHz com amplitude máxima 0,85 (normalizado).

    tipo = 'burst' : dois estouros amortecidos (como S1 e S2), de 60 a 160 Hz;
           'sopro' : os dois estouros mais ruído de 100 a 350 Hz entre eles;
           'denso' : os dois estouros mais ruído de 150 a 600 Hz (como o painel c).
    """
    if tipo not in ("burst", "sopro", "denso"):
        raise ValueError(f"tipo desconhecido: {tipo!r}")
    rng = np.random.default_rng(semente)
    t = np.arange(0, 0.5, 1.0 / FS_GABARITO)
    s = np.zeros_like(t)
    for t0, amp, f, larg in [(0.12, 0.7, rng.uniform(60, 120), 0.012), (0.37, 0.9, rng.uniform(80, 160), 0.010)]:
        s += amp * np.exp(-(((t - t0) / larg) ** 2)) * np.sin(2 * np.pi * f * (t - t0))
    if tipo != "burst":
        faixa, ganho, largura = ((100, 350), 0.35, 0.09) if tipo == "sopro" else ((150, 600), 0.7, 0.12)
        espectro = np.fft.rfft(rng.normal(size=t.size))
        freq = np.fft.rfftfreq(t.size, 1.0 / FS_GABARITO)
        espectro[(freq < faixa[0]) | (freq > faixa[1])] = 0
        ruido = np.fft.irfft(espectro, t.size)
        s += ganho * ruido / np.abs(ruido).max() * np.exp(-(((t - 0.25) / largura) ** 2))
    return t, 0.85 * s / np.abs(s).max()


def renderizar(t, s, cal, forma=FORMA_PADRAO, espessura=2.5, supersampling=8):
    """Desenha o sinal (amplitude normalizada de -1 a 1) e devolve a cobertura (0 a 1) do recorte."""
    altura, largura = forma
    x = (cal.coluna(t) + 0.5) * supersampling          # +0,5: o pixel i cobre [i, i+1)
    y = (cal.linha(100.0 * s) + 0.5) * supersampling
    pontos = np.rint(np.stack([x, y], axis=1)).astype(np.int32)
    img = np.zeros((altura * supersampling, largura * supersampling), dtype=np.uint8)
    cv2.polylines(img, [pontos], False, 255, int(round(espessura * supersampling)), cv2.LINE_8)
    return cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
