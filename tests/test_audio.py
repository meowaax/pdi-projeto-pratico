"""Testes da reamostragem, quantização e exportação de áudio."""
import numpy as np
import pytest

from src import audio, config


# ----------------------------------------------------------------------------
# Grade uniforme e reamostragem
# ----------------------------------------------------------------------------
def test_grade_tem_2500_amostras_a_5_khz():
    g = audio.grade_uniforme(5.665)
    assert len(g) == 2500
    assert g[0] == pytest.approx(5.665)
    assert np.diff(g) == pytest.approx(1 / 5000)
    assert g[-1] == pytest.approx(5.665 + 0.5 - 1 / 5000)


def test_reamostrar_rampa_linear_e_exata_nos_dois_metodos():
    t = np.arange(0, 0.5, 1 / 1333.0)
    y = 0.3 + 0.8 * t                       # rampa: interpolação linear e cúbica reproduzem exatamente
    for metodo in audio.METODOS:
        g, r = audio.reamostrar(t, y, 0.0, metodo=metodo)
        dentro = g <= t[-1]
        np.testing.assert_allclose(r[dentro], 0.3 + 0.8 * g[dentro], atol=1e-12)


def test_cubica_reproduz_polinomio_cubico_e_linear_nao():
    t = np.linspace(0, 0.5, 400)
    y = 4 * (t - 0.25) ** 3
    g, c = audio.reamostrar(t, y, 0.0, metodo="cubica")
    _, l = audio.reamostrar(t, y, 0.0, metodo="linear")
    dentro = g <= t[-1]
    ref = 4 * (g[dentro] - 0.25) ** 3
    assert np.abs(c[dentro] - ref).max() < 1e-9
    assert np.abs(l[dentro] - ref).max() > 1e-7


def test_senoide_de_baixa_frequencia_e_reconstruida_com_erro_pequeno():
    t = np.arange(0.001, 0.499, 1 / 1333.2)
    y = np.sin(2 * np.pi * 50 * t)
    g, r = audio.reamostrar(t, y, 0.0)
    dentro = (g >= t[0]) & (g <= t[-1])
    # Cota do erro da interpolação linear: h² / 8 * max|f''| = (1/1333,2)² / 8 * (2π·50)² = 0,0069.
    cota = (1 / 1333.2) ** 2 / 8 * (2 * np.pi * 50) ** 2
    erro = np.abs(r[dentro] - np.sin(2 * np.pi * 50 * g[dentro])).max()
    assert erro <= cota * 1.001
    assert erro > 0.5 * cota                # e a cota é atingida: o teste não é frouxo


def test_pontas_repetem_o_valor_da_borda():
    t = np.linspace(0.002, 0.497, 100)
    y = np.linspace(0.2, 0.7, 100)
    g, r = audio.reamostrar(t, y, 0.0)
    assert np.all(r[g < t[0]] == 0.2)
    assert np.all(r[g > t[-1]] == 0.7)
    assert len(r) == 2500 and not np.isnan(r).any()


def test_linear_nao_ultrapassa_os_picos_e_cubica_pode():
    t = np.arange(0, 0.5, 1 / 1333.2)
    y = np.zeros_like(t)
    y[300] = 1.0                                       # pico isolado
    _, lin = audio.reamostrar(t, y, 0.0, metodo="linear")
    _, cub = audio.reamostrar(t, y, 0.0, metodo="cubica")
    assert lin.max() <= 1.0 and lin.min() >= 0.0
    assert cub.min() < -0.01                           # a cúbica oscila ao redor do pico


@pytest.mark.parametrize("t, y", [
    (np.array([0.0]), np.array([1.0])),                             # poucas amostras
    (np.array([0.0, 0.1, 0.1]), np.array([0.0, 1.0, 2.0])),         # t repetido
    (np.array([0.0, 0.2, 0.1]), np.array([0.0, 1.0, 2.0])),         # t decrescente
    (np.array([0.0, 0.1, 0.2]), np.array([0.0, 1.0])),              # tamanhos diferentes
    (np.array([0.0, 0.1, 0.2]), np.array([0.0, np.nan, 1.0])),      # NaN
])
def test_entradas_invalidas(t, y):
    with pytest.raises(ValueError):
        audio.reamostrar(t, y, 0.0)


def test_metodo_invalido():
    with pytest.raises(ValueError):
        audio.reamostrar(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0, metodo="sinc")


def test_nao_altera_as_entradas():
    t = np.linspace(0.001, 0.499, 50)
    y = np.sin(10 * t)
    t0, y0 = t.copy(), y.copy()
    audio.reamostrar(t, y, 0.0)
    np.testing.assert_array_equal(t, t0)
    np.testing.assert_array_equal(y, y0)
