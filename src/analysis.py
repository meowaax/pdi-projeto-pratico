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


# ----------------------------------------------------------------------------
# Espectro
# ----------------------------------------------------------------------------
# A figura só entrega conteúdo abaixo de ~667 Hz (1333 colunas por segundo) e abaixo de ~40 Hz há
# energia de nível DC residual e de oscilações lentas, então o espectro é medido de 40 a 660 Hz.
BANDA_HZ = (40.0, 660.0)
FAIXAS_HZ = ((40.0, 80.0), (80.0, 200.0), (200.0, 500.0), (500.0, 660.0))    # as do artigo: <80 ruído, <200 Still, 80-500 VSD
NFFT = 8192                  # zero padding: o espectro fica liso, a resolução real é 1 / duração da janela
PLANURA_HZ = (80.0, 500.0)   # faixa em que se mede a planura espectral


def psd(x, fs=FS, nfft=NFFT):
    """Densidade espectral de potência (periodograma com janela de Hann), sem a média do trecho.

    Retorna (frequências em Hz, potência). Espectro unilateral, normalizado de modo que a soma de P * df
    é a potência média de x (a variância, pois a média é retirada).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        raise ValueError("trecho curto demais para calcular o espectro")
    w = np.hanning(x.size)
    X = np.fft.rfft((x - x.mean()) * w, nfft)
    p = (np.abs(X) ** 2) / (fs * np.sum(w * w))
    p[1:-1] *= 2.0                      # espectro unilateral: a energia das frequências negativas vem para cá
    return np.fft.rfftfreq(nfft, 1.0 / fs), p


def _contido(f, a, b):
    return (f >= a) & (f < b)


def metricas_espectro(f, p, banda=BANDA_HZ, faixas=FAIXAS_HZ):
    """Resumo de um espectro dentro de `banda`.

    frequencia_dominante_hz : posição do maior pico.
    centroide_hz            : média das frequências ponderada pela potência.
    largura_meia_potencia_hz: largura do pico dominante, onde a potência cai à metade.
    energia_faixas          : fração da energia em cada faixa de `faixas` (somam 1 se as faixas cobrem a banda).
    planura                 : média geométrica / média aritmética da potência de 80 a 500 Hz. Perto de 0 é tom
                              puro; em ruído branco, ~0,56 (efeito do periodograma de uma só realização).
    """
    em_banda = (f >= banda[0]) & (f <= banda[1])
    fb, pb = f[em_banda], p[em_banda]
    if not pb.any():
        raise ValueError("espectro sem energia na banda de análise")
    i = int(np.argmax(pb))
    metade = pb[i] / 2.0
    esq, dir_ = i, i
    while esq > 0 and pb[esq - 1] >= metade:
        esq -= 1
    while dir_ < len(pb) - 1 and pb[dir_ + 1] >= metade:
        dir_ += 1
    total = pb.sum()
    plano = p[_contido(f, *PLANURA_HZ)]
    return {
        "frequencia_dominante_hz": float(fb[i]),
        "centroide_hz": float(np.sum(fb * pb) / total),
        "largura_meia_potencia_hz": float(fb[dir_] - fb[esq]),
        "energia_faixas": {f"{int(a)}-{int(b)}": float(p[_contido(f, a, b)].sum() / total) for a, b in faixas},
        "planura": float(np.exp(np.mean(np.log(plano + 1e-30))) / np.mean(plano)),
    }


def espectro_da_janela(sinal, mascara):
    """(f, p) do trecho do sinal selecionado pela máscara booleana."""
    return psd(sinal.y[mascara], sinal.fs)


def espectrograma(y, fs=FS, janela_ms=25.0, passo_ms=2.0, nfft=1024, fmax=700.0):
    """Espectrograma de potência (STFT com janela de Hann).

    Retorna (instantes dos centros das janelas em s a partir do início de y, frequências em Hz até `fmax`,
    matriz de potência com uma linha por frequência e uma coluna por instante).
    """
    y = np.asarray(y, dtype=np.float64)
    n, passo = int(round(fs * janela_ms / 1000.0)), max(1, int(round(fs * passo_ms / 1000.0)))
    if y.size < n:
        raise ValueError("sinal mais curto que a janela do espectrograma")
    w = np.hanning(n)
    inicios = np.arange(0, y.size - n + 1, passo)
    quadros = np.stack([y[i:i + n] * w for i in inicios])
    quadros -= quadros.mean(axis=1, keepdims=True) * w / w.mean()      # tira a média de cada quadro
    P = (np.abs(np.fft.rfft(quadros, nfft, axis=1)) ** 2 / (fs * np.sum(w * w))).T
    P[1:-1] *= 2.0
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    manter = f <= fmax
    return (inicios + n / 2.0) / fs, f[manter], P[manter]


# ----------------------------------------------------------------------------
# Análise completa e confronto com o artigo
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalisePainel:
    """Tudo o que a análise mede em um painel."""
    sinal: SinalAnalise
    eventos: Eventos
    janelas: dict
    tempo: dict          # metricas_tempo
    espectro: dict       # {'sopro': metricas_espectro, 's1': ..., 's2': ...}
    psd_sopro: tuple     # (f, p) do sopro sistólico

    @property
    def nome(self):
        return self.sinal.nome


def analisar_painel(resultado):
    """Analisa um `pipeline.ResultadoPainel`."""
    sinal = preparar_sinal(resultado)
    t1, t2 = tempos_dos_rotulos(resultado.limpo.anotacoes, resultado.sinal.calibracao)
    eventos = localizar_eventos(sinal, t1, t2)
    jan = janelas(sinal, eventos)
    espectros = {nome: espectro_da_janela(sinal, jan[nome]) for nome in ("sopro", "s1", "s2")}
    return AnalisePainel(sinal=sinal, eventos=eventos, janelas=jan, tempo=metricas_tempo(sinal, eventos),
                         espectro={n: metricas_espectro(*fp) for n, fp in espectros.items()},
                         psd_sopro=espectros["sopro"])


def analisar(resultados):
    """Analisa todos os painéis. Devolve {'a': AnalisePainel, ...}."""
    return {r.nome: analisar_painel(r) for r in resultados}


def conferir_artigo(analises):
    """Confronta as medidas com as afirmações de Tomas et al. (2011) sobre cada tipo de sopro.

    Cada item traz: painel, afirmação do artigo, critério numérico usado, valor medido e se foi atendido.
    As afirmações valem para 20 gravações de cada tipo. Aqui há uma gravação de cada, e a figura limita o
    conteúdo a ~667 Hz, então um critério não atendido não contradiz o artigo.
    """
    a, b, c = (analises[n] for n in "abc")
    sop = lambda an: an.espectro["sopro"]
    fx = lambda an, ini, fim: sum(v for k, v in sop(an)["energia_faixas"].items() if ini <= float(k.split("-")[0]) and float(k.split("-")[1]) <= fim)
    itens = []

    def item(painel, afirmacao, criterio, medido, atendido, unidade=""):
        itens.append({"painel": painel, "afirmacao": afirmacao, "criterio": criterio,
                      "medido": f"{medido:.3g}{unidade}", "atendido": bool(atendido)})

    f_still = sop(b)["frequencia_dominante_hz"]
    item("b", "Still tem ressonância entre 90 e 170 Hz", "frequência dominante do sopro em 90 a 170 Hz", f_still, 90 <= f_still <= 170, " Hz")
    item("b", "A frequência do tom de Still fica na maioria dos casos entre 110 e 130 Hz (média de 118,75 Hz)",
         "frequência dominante do sopro em 110 a 130 Hz", f_still, 110 <= f_still <= 130, " Hz")
    item("b", "A energia de Still fica abaixo de 200 Hz, em banda estreita", "pelo menos 80% da energia do sopro abaixo de 200 Hz",
         fx(b, 40, 200), fx(b, 40, 200) >= 0.8)
    item("a", "A energia do Normal fica abaixo de 200 Hz", "pelo menos 80% da energia da janela sistólica abaixo de 200 Hz",
         fx(a, 40, 200), fx(a, 40, 200) >= 0.8)
    item("a", "O Normal não tem sopro (sem ressonância)", "sopro menos de 3 dB acima do nível da diástole",
         a.tempo["sopro_sobre_diastole_db"], a.tempo["sopro_sobre_diastole_db"] < 3, " dB")
    item("c", "A energia do VSD se distribui em banda larga de 80 a 500 Hz", "pelo menos 80% da energia do sopro de 80 a 500 Hz",
         fx(c, 80, 500), fx(c, 80, 500) >= 0.8)
    item("c", "O VSD é mais disperso em frequência que o Still", "planura espectral do VSD maior que a do Still",
         sop(c)["planura"] / sop(b)["planura"], sop(c)["planura"] > sop(b)["planura"], "x")
    item("c", "A frequência do maior pico do VSD passa de 200 Hz", "frequência dominante do sopro acima de 200 Hz",
         sop(c)["frequencia_dominante_hz"], sop(c)["frequencia_dominante_hz"] > 200, " Hz")
    return itens
