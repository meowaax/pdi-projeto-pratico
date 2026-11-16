"""Extração da curva por coluna e calibração de tempo e amplitude (Task 8).

Convenção de coordenadas: todas as posições são índices de pixel (o pixel i tem centro em i),
a mesma usada em `roi.py` para medir as linhas da grade. Assim, as retas de calibração
passam exatamente pelas posições medidas, sem correção de meio pixel.

Tempo      : t = t_ini + (x - x_esq) * DURACAO_S / (x_dir - x_esq)   (x = coluna na figura)
Amplitude  : a(%) = (y_zero - y) * 150 / (y_m75 - y_75)             (y = linha na figura)
"""
from dataclasses import dataclass

import numpy as np

from src import config


@dataclass(frozen=True)
class Calibracao:
    """Mapeia pixels do recorte (ROI) de um painel para segundos e porcentagem."""
    x0: int              # coluna da figura onde começa o recorte
    y0: int              # linha da figura onde começa o recorte
    x_esq: float         # x da borda esquerda do eixo de tempo (t = t_ini)
    x_dir: float         # x da borda direita do eixo de tempo (t = t_ini + duração)
    t_ini: float         # segundos
    duracao: float       # segundos entre x_esq e x_dir
    y_zero: float        # y da linha de 0%
    y_75: float          # y da linha de +75%
    y_m75: float         # y da linha de -75%

    @classmethod
    def de_painel(cls, painel, t_ini=None, duracao=None):
        """Monta a calibração a partir de um `roi.Painel`.

        `t_ini` vem de config.PAINEIS (lido dos rótulos do eixo, sem OCR) se não for informado.
        """
        t_ini = config.PAINEIS[painel.nome]["t_ini"] if t_ini is None else t_ini
        duracao = config.DURACAO_S if duracao is None else duracao
        x0, y0, _, _ = painel.roi
        return cls(x0=x0, y0=y0, x_esq=painel.x_esq, x_dir=painel.x_dir, t_ini=float(t_ini),
                   duracao=float(duracao), y_zero=painel.y_zero, y_75=painel.y_75, y_m75=painel.y_m75)

    @property
    def s_por_px(self):
        """Segundos por coluna."""
        return self.duracao / (self.x_dir - self.x_esq)

    @property
    def px_por_s(self):
        return (self.x_dir - self.x_esq) / self.duracao

    @property
    def pct_por_px(self):
        """Porcentagem de amplitude por linha (positivo: o eixo y da imagem cresce para baixo)."""
        return 150.0 / (self.y_m75 - self.y_75)

    @property
    def t_fim(self):
        return self.t_ini + self.duracao

    def tempo(self, coluna):
        """Tempo (s) de uma coluna do RECORTE (aceita escalar ou array)."""
        return self.t_ini + (coluna + self.x0 - self.x_esq) * self.s_por_px

    def amplitude_pct(self, linha):
        """Amplitude (%) de uma linha do RECORTE (aceita escalar ou array)."""
        return (self.y_zero - (linha + self.y0)) * self.pct_por_px

    def coluna(self, t):
        """Inverso de `tempo`: coluna do recorte (fracionária) de um instante t (s)."""
        return (t - self.t_ini) / self.s_por_px + self.x_esq - self.x0

    def linha(self, amplitude_pct):
        """Inverso de `amplitude_pct`: linha do recorte (fracionária) de uma amplitude (%)."""
        return self.y_zero - amplitude_pct / self.pct_por_px - self.y0


# ----------------------------------------------------------------------------
# Extração do sinal
# ----------------------------------------------------------------------------
LIMITE_EXTENSAO_PX = 6   # extensão vertical da coluna acima da qual ela pode conter um pico (traço = 3 a 4 px)
REGRAS = ("picos", "centro")


@dataclass(frozen=True)
class SinalExtraido:
    """Sinal de um painel, com uma amostra por coluna do recorte."""
    t: np.ndarray               # segundos, crescente
    amplitude_pct: np.ndarray   # porcentagem da escala do gráfico
    amplitude: np.ndarray       # normalizada em [-1, 1] (amplitude_pct / 100)
    topo_pct: np.ndarray        # amplitude do topo da coluna
    base_pct: np.ndarray        # amplitude da base da coluna
    colunas_interpoladas: int   # colunas sem curva, preenchidas por interpolação
    regra: str
    calibracao: Calibracao


def extremos_por_coluna(mascara):
    """Linha do topo e da base da curva em cada coluna (índices) e quais colunas têm curva."""
    mascara = np.asarray(mascara, dtype=bool)
    validas = mascara.any(axis=0)
    topo = np.where(validas, mascara.argmax(axis=0), 0).astype(np.float64)
    base = np.where(validas, mascara.shape[0] - 1 - mascara[::-1].argmax(axis=0), 0).astype(np.float64)
    return topo, base, validas


def preencher_colunas_vazias(topo, base, validas):
    """Interpola linearmente topo e base nas colunas sem curva. Devolve cópias."""
    if not validas.any():
        raise ValueError("a máscara não tem nenhum pixel de curva")
    x = np.arange(len(validas))
    return np.interp(x, x[validas], topo[validas]), np.interp(x, x[validas], base[validas])


def colapsar_centro(topo_pct, base_pct, limite_pct=None):
    """Uma amostra por coluna: o ponto médio entre topo e base. Achata os picos."""
    return 0.5 * (topo_pct + base_pct)


def colapsar_picos(topo_pct, base_pct, limite_pct):
    """Uma amostra por coluna: o ponto médio, trocado pelo extremo nas colunas de virada.

    Numa subida ou descida a coluna tem um trecho inclinado e o ponto médio é uma boa estimativa.
    Num máximo (ou mínimo) local a curva passa pelo extremo e volta, e o ponto médio perde o
    pico (em testes sintéticos, 30% da amplitude). Por isso, nas colunas com extensão maior que
    `limite_pct` que são máximo local do topo, vale o topo; as que são mínimo local da base, a base.
    O topo vence quando sobe sobre os vizinhos mais do que a base desce abaixo deles.
    """
    topo, base = np.asarray(topo_pct, dtype=float), np.asarray(base_pct, dtype=float)
    centro = 0.5 * (topo + base)
    saida = centro.copy()
    if len(topo) < 3:
        return saida
    c_ant, c_prox = centro[:-2], centro[2:]
    t_ant, t_prox = topo[:-2], topo[2:]
    b_ant, b_prox = base[:-2], base[2:]
    t, b, c = topo[1:-1], base[1:-1], centro[1:-1]
    larga = (t - b) > limite_pct
    max_local = (t >= t_ant) & (t >= t_prox)
    min_local = (b <= b_ant) & (b <= b_prox)
    sobe = t - np.maximum(c_ant, c_prox)
    desce = np.minimum(c_ant, c_prox) - b
    usa_topo = larga & max_local & (sobe >= desce)
    usa_base = larga & ~usa_topo & min_local
    meio = saida[1:-1]
    meio[usa_topo] = t[usa_topo]
    meio[usa_base] = b[usa_base]
    return saida


def extrair_sinal(mascara, calibracao, regra="picos", limite_px=LIMITE_EXTENSAO_PX):
    """Extrai o sinal calibrado (tempo e amplitude) da máscara binária da curva de um painel.

    mascara    : bool 2D, recorte da ROI com a curva (saída de structure.realcar_curva).
    calibracao : `Calibracao.de_painel(painel)`.
    regra      : 'picos' (padrão) ou 'centro'; ver `colapsar_picos`.
    """
    if regra not in REGRAS:
        raise ValueError(f"regra deve ser uma de {REGRAS}, recebido {regra!r}")
    topo, base, validas = extremos_por_coluna(mascara)
    topo, base = preencher_colunas_vazias(topo, base, validas)
    topo_pct = calibracao.amplitude_pct(topo)
    base_pct = calibracao.amplitude_pct(base)
    limite_pct = limite_px * abs(calibracao.pct_por_px)
    colapsar = colapsar_picos if regra == "picos" else colapsar_centro
    amplitude_pct = colapsar(topo_pct, base_pct, limite_pct)
    t = calibracao.tempo(np.arange(mascara.shape[1]))
    return SinalExtraido(t=t, amplitude_pct=amplitude_pct, amplitude=np.clip(amplitude_pct / 100.0, -1.0, 1.0),
                         topo_pct=topo_pct, base_pct=base_pct, colunas_interpoladas=int((~validas).sum()),
                         regra=regra, calibracao=calibracao)
