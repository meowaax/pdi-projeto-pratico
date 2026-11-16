"""Testes das máscaras de filtragem."""
import numpy as np
import pytest

from src.correlation import correlate2d_manual
from src.kernels import (SOBEL_X, SOBEL_Y, kernel_gaussiano, kernel_media,
                         tamanho_gaussiano)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 2.0, 3.7])
def test_gaussiano_soma_um_simetrico_e_centrado(sigma):
    k = kernel_gaussiano(sigma)
    assert k.sum() == pytest.approx(1.0, abs=1e-12)
    assert k.shape[0] == k.shape[1] == tamanho_gaussiano(sigma)
    np.testing.assert_allclose(k, k[::-1, ::-1])
    np.testing.assert_allclose(k, k.T)
    c = k.shape[0] // 2
    assert k[c, c] == k.max()


def test_gaussiano_tamanho_padrao_e_explicito():
    assert tamanho_gaussiano(1.0) == 7
    assert tamanho_gaussiano(1.5) == 11
    assert kernel_gaussiano(1.5, tamanho=7).shape == (7, 7)
    assert kernel_gaussiano(1.5, tamanho=7).sum() == pytest.approx(1.0)


def test_gaussiano_maior_sigma_e_mais_espalhado():
    assert kernel_gaussiano(2.0, 9)[4, 4] < kernel_gaussiano(1.0, 9)[4, 4]


def test_gaussiano_e_separavel_e_segue_a_formula():
    sigma = 1.2
    k = kernel_gaussiano(sigma, 5)
    x = np.arange(-2, 3)
    esperado = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    np.testing.assert_allclose(k, esperado / esperado.sum())
    assert np.linalg.matrix_rank(k) == 1


@pytest.mark.parametrize("tamanho", [1, 3, 5, 9])
def test_media_soma_um(tamanho):
    k = kernel_media(tamanho)
    assert k.shape == (tamanho, tamanho)
    assert k.sum() == pytest.approx(1.0)
    assert np.unique(k).size == 1


@pytest.mark.parametrize("kernel", [SOBEL_X, SOBEL_Y])
def test_sobel_soma_zero_e_zera_em_regiao_constante(kernel):
    assert kernel.sum() == 0
    saida = correlate2d_manual(np.full((9, 9), 123.0), kernel)
    np.testing.assert_array_equal(saida[1:-1, 1:-1], 0.0)  # só as bordas sofrem com o zero padding


def test_sobel_y_e_transposto_de_x():
    np.testing.assert_array_equal(SOBEL_Y, SOBEL_X.T)


def test_sobel_sentido_das_rampas():
    # Rampa crescente para a direita: Sobel X > 0 e Sobel Y = 0 no interior.
    rampa_x = np.tile(np.arange(10, dtype=float), (10, 1))
    gx = correlate2d_manual(rampa_x, SOBEL_X)[1:-1, 1:-1]
    gy = correlate2d_manual(rampa_x, SOBEL_Y)[1:-1, 1:-1]
    assert np.all(gx == 8.0)  # 4 (peso total) * 2 (diferença entre colunas vizinhas)
    assert np.all(gy == 0.0)
    # Rampa crescente para baixo: o contrário.
    gx2 = correlate2d_manual(rampa_x.T, SOBEL_X)[1:-1, 1:-1]
    gy2 = correlate2d_manual(rampa_x.T, SOBEL_Y)[1:-1, 1:-1]
    assert np.all(gx2 == 0.0) and np.all(gy2 == 8.0)


@pytest.mark.parametrize("valor", [0, -1, 2, 4, 2.5])
def test_tamanho_invalido(valor):
    with pytest.raises(ValueError):
        kernel_media(valor)
    with pytest.raises(ValueError):
        kernel_gaussiano(1.0, valor)


@pytest.mark.parametrize("sigma", [0, -1.0, float("nan"), float("inf")])
def test_sigma_invalido(sigma):
    with pytest.raises(ValueError):
        kernel_gaussiano(sigma)
    with pytest.raises(ValueError):
        tamanho_gaussiano(sigma)
