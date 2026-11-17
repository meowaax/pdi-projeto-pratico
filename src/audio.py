"""Reamostragem do sinal extraído e exportação dos arquivos .wav (Task 9).

Etapas: reamostrar para a grade uniforme de 5000 Hz -> remover nível DC -> normalizar ->
quantizar em 16 bits -> gravar PCM mono com o módulo `wave` da biblioteca padrão.

Os parâmetros de áudio (5000 Hz, 16 bits, mono, 0,5 s) vêm de `src/config.py`.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from src import config

METODOS = ("linear", "cubica")


def grade_uniforme(t_ini, duracao=config.DURACAO_S, fs=config.FS_AUDIO):
    """Instantes (s) das amostras: `round(duracao * fs)` pontos, o primeiro em `t_ini`.

    O último fica a uma amostra antes de `t_ini + duracao`, como numa gravação de 0,5 s.
    """
    return t_ini + np.arange(int(round(duracao * fs))) / fs


def reamostrar(t, amplitude, t_ini, duracao=config.DURACAO_S, fs=config.FS_AUDIO, metodo="linear"):
    """Interpola o sinal (uma amostra por coluna da figura) para a grade uniforme de `fs` Hz.

    metodo = 'linear' (padrão) ou 'cubica' (spline cúbica). Nos testes com sinais conhecidos
    a linear erra menos e preserva a amplitude dos picos, e a cúbica ultrapassa os picos
    em cerca de 10% (ver docs/audio.md).

    O sinal extraído começa e termina um pouco dentro do trecho (a ROI deixa de fora a moldura).
    Antes do primeiro e depois do último ponto vale o valor da borda.
    Retorna (instantes, amplitudes).
    """
    if metodo not in METODOS:
        raise ValueError(f"metodo deve ser um de {METODOS}, recebido {metodo!r}")
    t = np.asarray(t, dtype=np.float64)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if t.ndim != 1 or t.shape != amplitude.shape:
        raise ValueError("t e amplitude devem ser vetores com o mesmo tamanho")
    if t.size < 2:
        raise ValueError("são necessárias pelo menos 2 amostras para interpolar")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t deve ser estritamente crescente")
    if not (np.isfinite(t).all() and np.isfinite(amplitude).all()):
        raise ValueError("t e amplitude não podem ter NaN nem infinitos")
    grade = grade_uniforme(t_ini, duracao, fs)
    if metodo == "linear":
        y = np.interp(grade, t, amplitude)          # fora do intervalo repete a borda
    else:
        dentro = (grade >= t[0]) & (grade <= t[-1])
        y = np.interp(grade, t, amplitude)          # borda repetida fora do intervalo
        y[dentro] = CubicSpline(t, amplitude)(grade[dentro])
    return grade, y
