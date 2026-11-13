"""Parâmetros do pipeline, medidos em Figure1.png pela Task 1B (explore_data.py).

Coordenadas em pixels da imagem original (x para a direita, y para baixo).
tests/test_exploration.py confere que estes valores continuam batendo com as medições.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FIGURA = RAIZ / "Figure1.png"
SAIDA = RAIZ / "outputs"

# Parâmetros de áudio exigidos pela especificação
FS_AUDIO = 5000       # amostras/segundo
BITS = 16             # PCM com sinal
CANAIS = 1            # mono
DURACAO_S = 0.5       # cada gráfico mostra 0,5 s (10 intervalos de 0,05 s)
N_AMOSTRAS = int(FS_AUDIO * DURACAO_S)  # 2500

# Nomes dos arquivos de saída
ARQUIVOS_WAV = {"a": "normal.wav", "b": "still.wav", "c": "vsd.wav"}

# Valores de amplitude (%) rotulados no eixo vertical, de cima para baixo
AMPLITUDES_PCT = (75, 50, 25, 0, -25, -50, -75)

# Cada painel tem sua própria calibração (o painel c é ~4 px mais à esquerda e tem
# escala vertical diferente).
#   roi            : (x0, y0, x1, y1) interior do gráfico, sem moldura nem rótulos
#   x_esq, x_dir   : x das bordas esquerda (t_ini) e direita (t_fim) do eixo de tempo
#   t_ini          : tempo (s) na borda esquerda
#   y_75, y_m75    : y das linhas de +75% e -75% (y_zero é a de 0%)
#   legenda        : caixa (x0, y0, x1, y1) do texto "Diaphragm mode; Normalized"
PAINEIS = {
    "a": {
        "nome": "Normal",
        "roi": (34, 7, 698, 230),
        "x_esq": 32.4, "x_dir": 698.9, "t_ini": 5.665,
        "y_75": 33.2, "y_zero": 119.0, "y_m75": 204.8,
        "legenda": (43, 218, 216, 229),
    },
    "b": {
        "nome": "Still",
        "roi": (34, 279, 697, 486),
        "x_esq": 32.3, "x_dir": 698.2, "t_ini": 1.345,
        "y_75": 303.6, "y_zero": 382.8, "y_m75": 462.1,
        "legenda": (43, 475, 215, 485),
    },
    "c": {
        "nome": "VSD",
        "roi": (31, 534, 694, 741),
        "x_esq": 28.8, "x_dir": 695.4, "t_ini": 1.085,
        "y_75": 558.2, "y_zero": 637.6, "y_m75": 717.0,
        "legenda": (39, 730, 212, 739),
    },
}

# Valores aproximados em todos os painéis
PX_POR_SEGUNDO = 1333.0   # 66,65 px a cada 0,05 s
