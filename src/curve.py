"""Extração da curva por coluna e calibração de tempo e amplitude (Task 8).

Convenção de coordenadas: todas as posições são índices de pixel (o pixel i tem centro em i),
a mesma usada em `roi.py` para medir as linhas da grade. Assim, as retas de calibração
passam exatamente pelas posições medidas, sem correção de meio pixel.

Tempo      : t = t_ini + (x - x_esq) * DURACAO_S / (x_dir - x_esq)   (x = coluna na figura)
Amplitude  : a(%) = (y_zero - y) * 150 / (y_m75 - y_75)             (y = linha na figura)
"""
from dataclasses import dataclass

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
