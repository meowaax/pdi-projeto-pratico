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


# ----------------------------------------------------------------------------
# Análise de cada painel e segmentação completa
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def paineis(rgb):
    return roi.segmentar_paineis(rgb, usar_fallback=False)


def test_tres_paineis_a_b_c(paineis):
    assert [p.nome for p in paineis] == ["a", "b", "c"]


def test_rois_automaticas_batem_com_as_medidas_da_task_1b(paineis):
    for p in paineis:
        c = config.PAINEIS[p.nome]
        assert p.roi == tuple(c["roi"])
        assert p.x_esq == pytest.approx(c["x_esq"], abs=0.2)
        assert p.x_dir == pytest.approx(c["x_dir"], abs=0.2)
        assert p.y_75 == pytest.approx(c["y_75"], abs=0.2)
        assert p.y_zero == pytest.approx(c["y_zero"], abs=0.2)
        assert p.y_m75 == pytest.approx(c["y_m75"], abs=0.2)


def test_rois_sem_sobreposicao_e_com_tamanho_esperado(paineis):
    for (_, _, _, fim), (_, ini, _, _) in zip((p.roi for p in paineis), (p.roi for p in paineis[1:])):
        assert fim < ini
    for p in paineis:
        x0, y0, x1, y1 = p.roi
        assert 650 <= x1 - x0 <= 670       # 0,5 s a ~1333 px/s
        assert 200 <= y1 - y0 <= 235


def test_roi_exclui_legendas_e_rotulos_de_tempo(rgb, paineis):
    for p in paineis:
        _, _, _, y1 = p.roi
        assert y1 < p.y_rotulos_tempo
        assert p.y_rotulos_tempo <= p.faixa[1]
    # Nenhuma ROI chega perto da legenda "(a)" / "(b)" / "(c)" nem da descrição da figura.
    assert max(p.roi[3] for p in paineis) < 760


def test_calibracao_de_amplitude_e_monotona_e_igualmente_espacada(paineis):
    for p in paineis:
        assert p.y_75 < p.y_zero < p.y_m75
        assert (p.y_zero - p.y_75) == pytest.approx(p.y_m75 - p.y_zero, abs=0.5)


def test_analise_acompanha_deslocamento_da_imagem(rgb, paineis):
    com_borda = np.pad(rgb, ((40, 25), (30, 0), (0, 0)), constant_values=255)
    deslocados = roi.segmentar_paineis(com_borda, usar_fallback=False)
    for p, d in zip(paineis, deslocados):
        x0, y0, x1, y1 = p.roi
        assert d.roi == (x0 + 30, y0 + 40, x1 + 30, y1 + 40)
        assert d.x_esq == pytest.approx(p.x_esq + 30, abs=0.01)
        assert d.y_zero == pytest.approx(p.y_zero + 40, abs=0.01)


def test_fallback_usa_o_config_quando_a_deteccao_falha(caplog):
    branca = np.full((867, 707, 3), 255, dtype=np.uint8)
    with pytest.raises(roi.RoiError):
        roi.segmentar_paineis(branca, usar_fallback=False)
    with caplog.at_level("WARNING"):
        paineis = roi.segmentar_paineis(branca)
    assert [p.roi for p in paineis] == [tuple(config.PAINEIS[n]["roi"]) for n in "abc"]
    assert "Detecção automática falhou" in caplog.text


def test_paineis_sao_imutaveis(paineis):
    with pytest.raises(Exception):
        paineis[0].roi = (0, 0, 1, 1)
