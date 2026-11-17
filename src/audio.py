"""Reamostragem do sinal extraído e exportação dos arquivos .wav (Task 9).

Etapas: reamostrar para a grade uniforme de 5000 Hz -> remover nível DC -> normalizar ->
quantizar em 16 bits -> gravar PCM mono com o módulo `wave` da biblioteca padrão.

Os parâmetros de áudio (5000 Hz, 16 bits, mono, 0,5 s) vêm de `src/config.py`.
"""
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from src import config

METODOS = ("linear", "cubica")
NORMALIZACOES = ("pico", "escala")
ALVO_PADRAO = config.ALVO_NORMALIZACAO   # fração do fundo de escala (0,95 = -0,45 dBFS), com folga contra clipping
PCM_MAX = 32767          # escala simétrica: +1,0 -> 32767 e -1,0 -> -32767


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


# ----------------------------------------------------------------------------
# Nível DC, normalização e quantização
# ----------------------------------------------------------------------------
def remover_dc(y):
    """Subtrai a média, para o áudio oscilar em torno de zero."""
    y = np.asarray(y, dtype=np.float64)
    return y - y.mean()


def normalizar(y, modo="pico", alvo=ALVO_PADRAO):
    """Ajusta o ganho do sinal (já sem DC).

    modo = 'pico'   : o maior valor absoluto vira `alvo` (usa todo o intervalo de 16 bits).
           'escala' : ganho fixo, em que 100% da escala do gráfico (amplitude 1,0) vale `alvo`.
                      Preserva o volume relativo entre painéis diferentes.
    Um sinal todo zero é devolvido sem alteração.
    """
    if modo not in NORMALIZACOES:
        raise ValueError(f"modo deve ser um de {NORMALIZACOES}, recebido {modo!r}")
    if not 0.0 < alvo <= 1.0:
        raise ValueError(f"alvo deve estar em (0, 1], recebido {alvo}")
    y = np.asarray(y, dtype=np.float64)
    if modo == "escala":
        return y * alvo
    pico = np.abs(y).max()
    return y if pico == 0 else y * (alvo / pico)


def quantizar(y):
    """Quantiza amostras em [-1, 1] para int16 (16 bits com sinal). Valores fora são saturados."""
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("o sinal não pode ter NaN nem infinitos")
    return np.clip(np.rint(y * PCM_MAX), -PCM_MAX - 1, PCM_MAX).astype(np.int16)


def sinal_para_pcm(t, amplitude, t_ini, metodo="linear", normalizacao="pico", alvo=ALVO_PADRAO,
                   duracao=config.DURACAO_S, fs=config.FS_AUDIO):
    """Cadeia completa: reamostra, remove DC, normaliza e quantiza. Retorna um array int16."""
    _, y = reamostrar(t, amplitude, t_ini, duracao, fs, metodo)
    return quantizar(normalizar(remover_dc(y), normalizacao, alvo))


# ----------------------------------------------------------------------------
# Arquivos .wav (PCM, mono, 16 bits)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class InfoWav:
    """Conteúdo e parâmetros de um arquivo .wav."""
    taxa: int          # amostras por segundo
    canais: int
    bits: int
    amostras: np.ndarray   # int16, uma coluna por canal se houver mais de um

    @property
    def n_amostras(self):
        return int(self.amostras.shape[0])

    @property
    def duracao(self):
        return self.n_amostras / self.taxa


def gravar_wav(caminho, pcm, fs=config.FS_AUDIO):
    """Grava `pcm` (int16, 1D) como WAV PCM mono de 16 bits, com o módulo `wave` da biblioteca padrão."""
    pcm = np.asarray(pcm)
    if pcm.dtype != np.int16 or pcm.ndim != 1:
        raise ValueError("pcm deve ser um vetor int16 (use `quantizar`)")
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(caminho), "wb") as arq:
        arq.setnchannels(config.CANAIS)
        arq.setsampwidth(config.BITS // 8)
        arq.setframerate(int(fs))
        arq.writeframes(pcm.astype("<i2").tobytes())      # little-endian, como exige o formato WAV
    return caminho


def ler_wav(caminho):
    """Lê um .wav PCM de 16 bits e devolve um `InfoWav`."""
    with wave.open(str(caminho), "rb") as arq:
        canais, largura, taxa, n = arq.getnchannels(), arq.getsampwidth(), arq.getframerate(), arq.getnframes()
        bruto = arq.readframes(n)
    if largura != 2:
        raise ValueError(f"esperava 16 bits por amostra, o arquivo tem {8 * largura}")
    amostras = np.frombuffer(bruto, dtype="<i2").astype(np.int16)
    if canais > 1:
        amostras = amostras.reshape(-1, canais)
    return InfoWav(taxa=taxa, canais=canais, bits=8 * largura, amostras=amostras)


def exportar_wavs(sinais, pasta, metodo="linear", normalizacao="pico", alvo=ALVO_PADRAO):
    """Grava um .wav por painel, com os nomes de `config.ARQUIVOS_WAV` (normal, still e vsd).

    sinais : dict {'a': sinal, 'b': sinal, 'c': sinal}. Cada sinal precisa de `t`, `amplitude` e
             `calibracao.t_ini` (como o `curve.SinalExtraido`).
    Retorna um dict {nome: {'caminho', 'pico_pct', 'rms_pct'}} com o nível do sinal ANTES da
    normalização, em % da escala do gráfico, para comparar o volume relativo entre os painéis.
    """
    pasta = Path(pasta)
    saida = {}
    for nome, sinal in sinais.items():
        if nome not in config.ARQUIVOS_WAV:
            raise ValueError(f"painel desconhecido: {nome!r}")
        pcm = sinal_para_pcm(sinal.t, sinal.amplitude, sinal.calibracao.t_ini, metodo, normalizacao, alvo)
        caminho = gravar_wav(pasta / config.ARQUIVOS_WAV[nome], pcm)
        _, y = reamostrar(sinal.t, sinal.amplitude, sinal.calibracao.t_ini, metodo=metodo)
        y = remover_dc(y) * 100.0
        saida[nome] = {"caminho": caminho, "pico_pct": float(np.abs(y).max()),
                       "rms_pct": float(np.sqrt(np.mean(y ** 2)))}
    return saida
