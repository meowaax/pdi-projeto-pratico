"""Testes da extração da curva e da calibração de tempo e amplitude."""
import cv2
import numpy as np
import pytest

from src import config, curve, roi


@pytest.fixture(scope="module")
def paineis():
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    return {p.nome: p for p in roi.segmentar_paineis(rgb, usar_fallback=False)}


@pytest.fixture(scope="module")
def cals(paineis):
    return {n: curve.Calibracao.de_painel(p) for n, p in paineis.items()}


# ----------------------------------------------------------------------------
# Calibração
# ----------------------------------------------------------------------------
def test_bordas_do_eixo_de_tempo_viram_o_inicio_e_o_fim_do_trecho(paineis, cals):
    for nome, c in cals.items():
        p = paineis[nome]
        assert c.tempo(p.x_esq - c.x0) == pytest.approx(config.PAINEIS[nome]["t_ini"], abs=1e-9)
        assert c.tempo(p.x_dir - c.x0) == pytest.approx(config.PAINEIS[nome]["t_ini"] + 0.5, abs=1e-9)


def test_duracao_total_e_meio_segundo(cals):
    for c in cals.values():
        assert c.t_fim - c.t_ini == pytest.approx(0.5)
        assert c.px_por_s == pytest.approx(1333, rel=0.003)
        assert c.s_por_px * c.px_por_s == pytest.approx(1.0)


def test_linhas_da_grade_viram_75_0_menos_75(paineis, cals):
    for nome, c in cals.items():
        p = paineis[nome]
        assert c.amplitude_pct(p.y_75 - c.y0) == pytest.approx(75.0, abs=0.5)
        assert c.amplitude_pct(p.y_zero - c.y0) == pytest.approx(0.0, abs=1e-9)
        assert c.amplitude_pct(p.y_m75 - c.y0) == pytest.approx(-75.0, abs=0.5)


def test_amplitude_positiva_para_cima(cals):
    for c in cals.values():
        assert c.amplitude_pct(10) > c.amplitude_pct(20)    # linha menor = mais alto na imagem


def test_escalas_por_painel_sao_diferentes(cals):
    # O painel a tem ~1,14 px por 1% e os painéis b e c ~1,06 (Task 1B).
    assert 1 / cals["a"].pct_por_px == pytest.approx(1.144, abs=0.01)
    assert 1 / cals["b"].pct_por_px == pytest.approx(1.056, abs=0.01)
    assert 1 / cals["c"].pct_por_px == pytest.approx(1.059, abs=0.01)


def test_inversos_sao_consistentes(cals):
    for c in cals.values():
        for col in (0, 100.5, 663):
            assert c.coluna(c.tempo(col)) == pytest.approx(col)
        for lin in (0, 50.25, 200):
            assert c.linha(c.amplitude_pct(lin)) == pytest.approx(lin)


def test_t_ini_pode_ser_informado(paineis):
    c = curve.Calibracao.de_painel(paineis["a"], t_ini=0.0, duracao=1.0)
    assert c.tempo(paineis["a"].x_esq - c.x0) == pytest.approx(0.0)
    assert c.tempo(paineis["a"].x_dir - c.x0) == pytest.approx(1.0)


def test_tempo_e_amplitude_aceitam_arrays(cals):
    c = cals["b"]
    t = c.tempo(np.arange(5))
    assert t.shape == (5,) and np.all(np.diff(t) > 0)
    assert c.amplitude_pct(np.arange(5)).shape == (5,)
