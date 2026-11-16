"""Segmentação automática dos gráficos (ROIs) da Figure1.png.

Etapas:
  1. `detectar_faixas_paineis`: acha as faixas verticais (y0, y1) dos três painéis
     usando a barra cinza de rótulos de amplitude, que existe à esquerda de cada
     painel e não existe nas legendas "(a)", "(b)" e "(c)".
  2. `analisar_painel` (próximo passo do módulo): dentro de cada faixa, mede a grade
     e a faixa de rótulos de tempo para obter a ROI interior e a calibração.

Nenhuma coordenada fixa é usada na detecção. `src/config.py` só serve de fallback.
"""
import numpy as np

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
