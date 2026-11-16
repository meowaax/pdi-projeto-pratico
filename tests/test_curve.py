"""Testes da extração da curva e da calibração de tempo e amplitude."""
import cv2
import numpy as np
import pytest

from src import cleaning, config, curve, roi, structure, synthetic


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


# ----------------------------------------------------------------------------
# Extração por coluna
# ----------------------------------------------------------------------------
def test_extremos_por_coluna():
    m = np.zeros((10, 5), dtype=bool)
    m[2:5, 0] = True
    m[7, 1] = True
    m[0, 3] = m[9, 3] = True      # topo e base na mesma coluna, sem o meio
    topo, base, validas = curve.extremos_por_coluna(m)
    assert validas.tolist() == [True, True, False, True, False]
    assert topo[[0, 1, 3]].tolist() == [2, 7, 0]
    assert base[[0, 1, 3]].tolist() == [4, 7, 9]


def test_colunas_vazias_sao_interpoladas():
    m = np.zeros((20, 7), dtype=bool)
    m[4, 0] = m[4, 1] = True
    m[10, 5] = m[10, 6] = True
    topo, base, validas = curve.extremos_por_coluna(m)
    topo, base = curve.preencher_colunas_vazias(topo, base, validas)
    assert topo.tolist() == pytest.approx([4, 4, 4 + 6 / 4, 4 + 12 / 4, 4 + 18 / 4, 10, 10])
    cal = synthetic.calibracao_sintetica((20, 7))
    s = curve.extrair_sinal(m, cal)
    assert s.colunas_interpoladas == 3 and not np.isnan(s.amplitude).any()


def test_mascara_vazia_levanta_erro():
    with pytest.raises(ValueError):
        curve.extrair_sinal(np.zeros((10, 10), dtype=bool), synthetic.calibracao_sintetica((10, 10)))


def test_regra_invalida():
    with pytest.raises(ValueError):
        curve.extrair_sinal(np.ones((4, 4), dtype=bool), synthetic.calibracao_sintetica((4, 4)), regra="x")


def test_centro_e_ponto_medio_e_picos_preserva_o_pico():
    # Onda que sobe de 0 a 60 e volta: a coluna 3 contém o pico (entra em 40, chega a 60, sai em 40).
    topo = np.array([0.0, 10, 40, 60, 40, 10, 0])
    base = np.array([0.0, 0, 10, 40, 10, 0, 0])
    np.testing.assert_allclose(curve.colapsar_centro(topo, base), 0.5 * (topo + base))
    saida = curve.colapsar_picos(topo, base, limite_pct=6.0)
    assert saida[3] == 60                       # coluna de virada: vale o topo
    assert saida[2] == pytest.approx(25.0)      # subida: ponto médio
    assert saida[4] == pytest.approx(25.0)      # descida: ponto médio
    assert saida[0] == 0 and saida[-1] == 0     # bordas sem vizinhos


def test_picos_preserva_vale_tambem():
    topo = -np.array([0.0, 0, 10, 40, 10, 0, 0])
    base = -np.array([0.0, 10, 40, 60, 40, 10, 0])
    saida = curve.colapsar_picos(topo, base, 6.0)
    assert saida[3] == -60                      # vale: vale a base
    assert saida[2] == pytest.approx(-25.0)


def test_colunas_finas_ficam_no_centro():
    # Extensão menor que o limite = só a espessura do traço: sem troca pelo extremo.
    topo = np.array([0.0, 1, 4, 2, 1, 0, 0])
    base = np.array([-2.0, -1, 0, -2, -1, -2, -2])
    np.testing.assert_allclose(curve.colapsar_picos(topo, base, 6.0), 0.5 * (topo + base))


def metricas_ida_e_volta(tipo, regra, semente=0):
    cal = synthetic.calibracao_sintetica()
    t, s = synthetic.sinal_sintetico(tipo, semente)
    est = structure.realcar_curva(synthetic.renderizar(t, s, cal))
    sinal = curve.extrair_sinal(est.mascara, cal, regra)
    tg = np.arange(0.002, 0.498, 1 / 5000)
    e, v = np.interp(tg, sinal.t, sinal.amplitude), np.interp(tg, t, s)
    e, v = e - e.mean(), v - v.mean()
    return {"corr": np.corrcoef(e, v)[0, 1], "nrmse": np.sqrt(np.mean((e - v) ** 2) / np.mean(v ** 2)),
            "pico": np.abs(e).max() / np.abs(v).max()}


@pytest.mark.parametrize("tipo, corr_min, nrmse_max", [("burst", 0.97, 0.30), ("sopro", 0.95, 0.35)])
def test_ida_e_volta_sintetica_reconstroi_o_sinal(tipo, corr_min, nrmse_max):
    m = metricas_ida_e_volta(tipo, "picos")
    assert m["corr"] > corr_min
    assert m["nrmse"] < nrmse_max
    assert 0.92 < m["pico"] < 1.12


@pytest.mark.parametrize("tipo", ["burst", "sopro", "denso"])
def test_regra_de_picos_preserva_a_amplitude_e_o_centro_nao(tipo):
    picos, centro = metricas_ida_e_volta(tipo, "picos"), metricas_ida_e_volta(tipo, "centro")
    assert picos["pico"] > 0.9
    assert centro["pico"] < 0.8
    assert picos["nrmse"] <= centro["nrmse"] * 1.1     # sem piorar o erro geral


# ----------------------------------------------------------------------------
# Figure1.png real
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sinais(paineis):
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    saida = {}
    for nome, p in paineis.items():
        est = structure.realcar_curva(cleaning.limpar_recorte(roi.recortar(rgb, p)).cobertura)
        saida[nome] = curve.extrair_sinal(est.mascara, curve.Calibracao.de_painel(p))
    return saida


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_cobre_quase_todo_o_meio_segundo(sinais, nome):
    s, c = sinais[nome], sinais[nome].calibracao
    assert 660 <= len(s.t) <= 668                                  # uma amostra por coluna da ROI
    assert np.all(np.diff(s.t) > 0)
    assert s.t[0] - c.t_ini < 0.003 and c.t_fim - s.t[-1] < 0.003    # perde só a moldura das pontas
    assert s.t[-1] - s.t[0] == pytest.approx(0.5, abs=0.004)


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_sem_nan_sem_interpolacao_e_dentro_da_escala(sinais, nome):
    s = sinais[nome]
    assert not np.isnan(s.amplitude_pct).any() and not np.isnan(s.t).any()
    assert s.colunas_interpoladas == 0
    assert np.abs(s.amplitude_pct).max() <= 100.0
    assert np.abs(s.amplitude).max() <= 1.0
    assert np.all(s.topo_pct >= s.base_pct)


def test_real_picos_do_painel_a_batem_com_a_leitura_da_figura(sinais):
    # Valores lidos a olho na Figure1(a): S1 chega a +75% em ~5,80 s, S2 vai de +92% a -72% em ~6,045 s.
    s = sinais["a"]
    s1 = (s.t > 5.78) & (s.t < 5.83)
    s2 = (s.t > 6.03) & (s.t < 6.07)
    assert s.amplitude_pct[s1].max() == pytest.approx(75, abs=3)
    assert s.t[s1][np.argmax(s.amplitude_pct[s1])] == pytest.approx(5.80, abs=0.01)
    assert s.amplitude_pct[s2].max() == pytest.approx(92, abs=3)
    assert s.amplitude_pct[s2].min() == pytest.approx(-72, abs=3)
    assert s.t[s2][np.argmax(s.amplitude_pct[s2])] == pytest.approx(6.045, abs=0.01)


def test_real_picos_dos_paineis_b_e_c(sinais):
    b, c = sinais["b"], sinais["c"]
    assert b.amplitude_pct.max() == pytest.approx(44, abs=3)      # S1 do painel b
    assert b.amplitude_pct.min() == pytest.approx(-55, abs=3)     # S2 do painel b
    assert c.amplitude_pct.max() == pytest.approx(87, abs=4)      # murmúrio do painel c
    assert c.amplitude_pct.min() == pytest.approx(-85, abs=4)
    assert 1.32 < c.t[np.argmax(c.amplitude_pct)] < 1.34


def test_real_a_linha_de_base_fica_perto_de_zero(sinais):
    # Trechos de silêncio (início do painel a, sem som): amplitude pequena em torno de 0%.
    s = sinais["a"]
    silencio = s.t < 5.74
    assert np.abs(s.amplitude_pct[silencio]).max() < 6
    assert abs(np.median(s.amplitude_pct[silencio])) < 2
