"""Testes da segmentação automática das ROIs."""
import cv2
import numpy as np
import pytest

from src import config
from src import roi


@pytest.fixture(scope="module")
def rgb():
    return cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def test_faixas_em_vetor_booleano():
    assert roi.faixas([0, 1, 1, 0, 1, 0, 0, 1, 1]) == [(1, 2), (4, 4), (7, 8)]
    assert roi.faixas([0, 0]) == []
    assert roi.faixas([1, 1]) == [(0, 1)]
    assert roi.faixas([]) == []


def test_barra_lateral_encontrada(rgb):
    x0, x1 = roi.detectar_barra_lateral(roi.mascara_nao_branco(rgb))
    assert 5 <= x0 <= 15
    assert 28 <= x1 <= 36


def test_tres_paineis_de_cima_para_baixo(rgb):
    paineis = roi.detectar_faixas_paineis(rgb)
    assert len(paineis) == 3
    assert [p[0] for p in paineis] == sorted(p[0] for p in paineis)
    # sem sobreposição e com folga para a legenda "(a)", "(b)"
    for (_, fim), (ini, _) in zip(paineis, paineis[1:]):
        assert ini - fim > 10


def test_faixas_batem_com_o_config_a_menos_da_moldura(rgb):
    paineis = roi.detectar_faixas_paineis(rgb)
    for (y0, y1), nome in zip(paineis, "abc"):
        _, ry0, _, ry1 = config.PAINEIS[nome]["roi"]
        assert abs(y0 - ry0) <= 6 or y0 < ry0          # a faixa inclui a moldura superior
        assert y1 >= ry1                                # e a faixa de rótulos de tempo


def test_legendas_nao_viram_paineis(rgb):
    # A descrição "Figure 1. ..." começa depois de y=800 e não pode ser detectada.
    assert all(y1 < 800 for _, y1 in roi.detectar_faixas_paineis(rgb))


def test_deteccao_acompanha_deslocamento_da_imagem(rgb):
    # Prova que não há coordenada fixa: com 40 px de borda branca, tudo sobe 40 px.
    com_borda = np.pad(rgb, ((40, 25), (30, 0), (0, 0)), constant_values=255)
    base = roi.detectar_faixas_paineis(rgb)
    deslocado = roi.detectar_faixas_paineis(com_borda)
    assert [(a + 40, b + 40) for a, b in base] == deslocado


def test_imagem_toda_branca_levanta_erro():
    with pytest.raises(roi.RoiError):
        roi.detectar_faixas_paineis(np.full((100, 100, 3), 255, dtype=np.uint8))
