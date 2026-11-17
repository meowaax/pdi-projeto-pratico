"""Análise comparativa dos sinais reconstruídos (Task 11).

Compara os três sinais (Normal, Still e VSD) no tempo e na frequência:

  * localiza S1 e S2 e divide cada sinal em janelas (S1, sopro sistólico, S2 e diástole);
  * mede nível (pico, RMS, fator de crista) em cada janela;
  * mede o espectro de cada janela (frequência dominante, centroide, largura de banda,
    distribuição de energia por faixa, planura espectral);
  * calcula espectrogramas;
  * confronta os resultados com o que o artigo afirma sobre cada tipo de sopro.

Os espectros usam só `numpy.fft`. Os níveis estão em porcentagem da escala do gráfico, depois de
remover o DC, como saem de `audio.exportar_wavs`.

Convenção das janelas
---------------------
S1 e S2 são localizados pelas anotações "S1" e "S2" que a própria figura traz: a caixa mais à esquerda
e a mais à direita entre as anotações removidas na Task 6. O instante do evento é o máximo da envolvente
RMS (10 ms) entre 10 ms antes e 50 ms depois do centro do rótulo.
"""
from dataclasses import dataclass

import numpy as np

from src import audio, config

FS = config.FS_AUDIO
MEIA_JANELA_S = 0.040          # S1 e S2 ocupam 80 ms, centrados no evento
ENVOLVENTE_MS = 10.0           # janela da envolvente RMS
ANTES_DO_ROTULO_S = 0.010      # o evento fica de 0 a 30 ms depois do rótulo na figura
DEPOIS_DO_ROTULO_S = 0.050


@dataclass(frozen=True)
class SinalAnalise:
    """Sinal de um painel pronto para a análise: grade de 5 kHz, sem DC, em % da escala do gráfico."""
    nome: str
    t: np.ndarray        # segundos, no eixo do gráfico (começa em t_ini)
    y: np.ndarray        # amplitude em %, média zero
    t_ini: float

    @property
    def fs(self):
        return FS

    @property
    def duracao(self):
        return len(self.y) / FS


@dataclass(frozen=True)
class Eventos:
    """Instantes (s, no eixo do gráfico) de S1 e S2."""
    t_s1: float
    t_s2: float

    @property
    def intervalo_ms(self):
        return 1000.0 * (self.t_s2 - self.t_s1)


def preparar_sinal(resultado, metodo=config.METODO_INTERPOLACAO):
    """Monta o `SinalAnalise` de um `pipeline.ResultadoPainel`."""
    s = resultado.sinal
    t_ini = s.calibracao.t_ini
    t, y = audio.reamostrar(s.t, s.amplitude, t_ini, metodo=metodo)
    return SinalAnalise(nome=resultado.nome, t=t, y=audio.remover_dc(y) * 100.0, t_ini=t_ini)


def envolvente_rms(y, ms=ENVOLVENTE_MS, fs=FS):
    """Envolvente RMS por média móvel centrada de `ms` milissegundos (bordas com janela parcial)."""
    y = np.asarray(y, dtype=np.float64)
    n = max(1, int(round(fs * ms / 1000.0)))
    soma = np.concatenate(([0.0], np.cumsum(y * y)))
    idx = np.arange(len(y))
    ini = np.clip(idx - n // 2, 0, len(y))
    fim = np.clip(idx - n // 2 + n, 0, len(y))
    return np.sqrt((soma[fim] - soma[ini]) / (fim - ini))


def tempos_dos_rotulos(anotacoes, calibracao):
    """Instantes (s) dos rótulos S1 e S2: o centro da caixa mais à esquerda e o da mais à direita.

    `anotacoes` são as caixas (x, y, largura, altura) do recorte devolvidas por `cleaning.remover_anotacoes`.
    """
    caixas = sorted(anotacoes, key=lambda b: b[0])
    if len(caixas) < 2:
        raise ValueError("são necessárias ao menos duas anotações (S1 e S2) para localizar os eventos")
    centro = lambda b: b[0] + b[2] / 2.0
    return float(calibracao.tempo(centro(caixas[0]))), float(calibracao.tempo(centro(caixas[-1])))


def localizar_eventos(sinal, t_rotulo_s1, t_rotulo_s2, antes=ANTES_DO_ROTULO_S, depois=DEPOIS_DO_ROTULO_S):
    """S1 e S2 como o máximo da envolvente perto de cada rótulo."""
    env = envolvente_rms(sinal.y)

    def pico(t_rotulo):
        janela = (sinal.t >= t_rotulo - antes) & (sinal.t <= t_rotulo + depois)
        if not janela.any():
            raise ValueError(f"o rótulo em {t_rotulo:.3f} s está fora do sinal")
        return float(sinal.t[np.argmax(np.where(janela, env, -np.inf))])

    return Eventos(t_s1=pico(t_rotulo_s1), t_s2=pico(t_rotulo_s2))


def janelas(sinal, eventos, meia_janela=MEIA_JANELA_S):
    """Máscaras booleanas das janelas: 's1', 's2', 'sopro' (entre S1 e S2) e 'diastole' (antes de S1 e depois de S2)."""
    t, j = sinal.t, meia_janela
    fim = sinal.t_ini + sinal.duracao
    if eventos.t_s2 - eventos.t_s1 <= 2 * j:
        raise ValueError("S1 e S2 estão próximos demais para separar as janelas")
    entre = lambda a, b: (t >= a) & (t < b)
    return {
        "s1": entre(eventos.t_s1 - j, eventos.t_s1 + j),
        "s2": entre(eventos.t_s2 - j, eventos.t_s2 + j),
        "sopro": entre(eventos.t_s1 + j, eventos.t_s2 - j),
        "diastole": entre(sinal.t_ini, eventos.t_s1 - j) | entre(eventos.t_s2 + j, fim),
    }


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def razao_db(a, b):
    """20*log10(a/b); -inf ou +inf se um dos valores for zero."""
    if a == 0 and b == 0:
        return 0.0
    if b == 0:
        return float("inf")
    if a == 0:
        return float("-inf")
    return float(20.0 * np.log10(a / b))


def metricas_tempo(sinal, eventos):
    """Níveis em cada janela e razões entre elas. Todos os níveis estão em % da escala do gráfico."""
    jan = janelas(sinal, eventos)
    y = sinal.y
    nivel = {nome: rms(y[m]) for nome, m in jan.items()}
    pp = lambda m: float(y[m].max() - y[m].min())
    pico, geral = float(np.abs(y).max()), rms(y)
    sop = y[jan["sopro"]]
    cruzamentos = np.count_nonzero(np.diff(np.signbit(sop - sop.mean()).astype(np.int8)))
    return {
        "pico_pct": pico,
        "rms_pct": geral,
        "fator_de_crista": pico / geral if geral else float("inf"),
        "intervalo_s1_s2_ms": eventos.intervalo_ms,
        "rms_s1_pct": nivel["s1"], "rms_s2_pct": nivel["s2"],
        "rms_sopro_pct": nivel["sopro"], "rms_diastole_pct": nivel["diastole"],
        "pico_a_pico_s1_pct": pp(jan["s1"]), "pico_a_pico_s2_pct": pp(jan["s2"]),
        "s2_sobre_s1_db": razao_db(nivel["s2"], nivel["s1"]),
        "sopro_sobre_diastole_db": razao_db(nivel["sopro"], nivel["diastole"]),
        "sopro_sobre_s1_db": razao_db(nivel["sopro"], nivel["s1"]),
        "cruzamentos_por_zero_sopro_hz": float(cruzamentos / (2.0 * len(sop) / FS)) if len(sop) else 0.0,
    }
