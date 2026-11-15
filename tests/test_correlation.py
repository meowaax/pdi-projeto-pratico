"""Testes do Requisito 1.

scipy e cv2.filter2D são usados somente aqui, como referência de validação.
"""
import cv2
import numpy as np
import pytest
from scipy.signal import correlate2d

from src.correlation import correlate2d_manual, correlate2d_naive

RAIZ = __import__("pathlib").Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)


def referencia_cv2(imagem, kernel):
    return cv2.filter2D(
        imagem.astype(np.float64), cv2.CV_64F, kernel.astype(np.float64),
        borderType=cv2.BORDER_CONSTANT,
    )


@pytest.mark.parametrize("forma_k", [(1, 1), (3, 3), (5, 5), (3, 5), (7, 1), (1, 7), (2, 2), (4, 4), (2, 3), (4, 5)])
def test_manual_igual_naive(forma_k):
    img = rng.integers(0, 256, (13, 17)).astype(float)
    k = rng.normal(size=forma_k)
    np.testing.assert_allclose(correlate2d_manual(img, k), correlate2d_naive(img, k), atol=1e-9)


@pytest.mark.parametrize("forma_k", [(3, 3), (5, 5), (3, 5), (7, 1), (1, 7), (9, 9)])
def test_igual_scipy_kernels_impares(forma_k):
    img = rng.integers(0, 256, (20, 25)).astype(float)
    k = rng.normal(size=forma_k)
    esperado = correlate2d(img, k, mode="same", boundary="fill", fillvalue=0)
    np.testing.assert_allclose(correlate2d_manual(img, k), esperado, atol=1e-9)


@pytest.mark.parametrize("forma_k", [(1, 1), (3, 3), (5, 5), (3, 5), (7, 1), (2, 2), (4, 4), (2, 3), (4, 5)])
def test_igual_cv2_filter2d(forma_k):
    img = rng.integers(0, 256, (20, 25)).astype(float)
    k = rng.normal(size=forma_k)
    np.testing.assert_allclose(correlate2d_manual(img, k), referencia_cv2(img, k), atol=1e-9)


def test_tamanho_preservado():
    img = np.ones((11, 7))
    for forma in [(1, 1), (3, 3), (4, 6), (11, 7), (15, 15)]:
        assert correlate2d_manual(img, np.ones(forma)).shape == img.shape


def test_resposta_ao_impulso_e_o_proprio_kernel_espelhado():
    # Correlação com um delta reproduz o kernel invertido nas duas direções.
    img = np.zeros((9, 9))
    img[4, 4] = 1.0
    k = np.arange(1, 10, dtype=float).reshape(3, 3)
    saida = correlate2d_manual(img, k)
    np.testing.assert_allclose(saida[3:6, 3:6], k[::-1, ::-1])
    assert saida.sum() == pytest.approx(k.sum())


def test_kernel_delta_e_identidade():
    img = rng.integers(0, 256, (8, 10)).astype(float)
    k = np.zeros((5, 5))
    k[2, 2] = 1.0
    np.testing.assert_array_equal(correlate2d_manual(img, k), img)


def test_zero_padding_nas_bordas():
    # Média 3x3 de uma imagem de 1s: o interior vale 9, as bordas e cantos têm menos vizinhos.
    saida = correlate2d_manual(np.ones((5, 5)), np.ones((3, 3)))
    assert saida[2, 2] == 9
    assert saida[0, 2] == 6   # borda: 6 vizinhos dentro da imagem
    assert saida[0, 0] == 4   # canto: 4 vizinhos


def test_kernel_maior_que_a_imagem():
    img = rng.normal(size=(3, 4))
    k = rng.normal(size=(7, 9))
    np.testing.assert_allclose(correlate2d_manual(img, k), correlate2d_naive(img, k), atol=1e-9)


def test_linearidade_e_entrada_inteira():
    a = rng.integers(0, 256, (10, 10))
    b = rng.integers(0, 256, (10, 10))
    k = rng.normal(size=(3, 3))
    esperado = 2 * correlate2d_manual(a, k) + 3 * correlate2d_manual(b, k)
    np.testing.assert_allclose(correlate2d_manual(2 * a + 3 * b, k), esperado, atol=1e-9)
    assert correlate2d_manual(a.astype(np.uint8), k).dtype == np.float64


def test_nao_altera_as_entradas():
    img = rng.integers(0, 256, (6, 6)).astype(np.uint8)
    k = rng.normal(size=(3, 3))
    img0, k0 = img.copy(), k.copy()
    correlate2d_manual(img, k)
    np.testing.assert_array_equal(img, img0)
    np.testing.assert_array_equal(k, k0)


@pytest.mark.parametrize("funcao", [correlate2d_manual, correlate2d_naive])
def test_entradas_invalidas(funcao):
    with pytest.raises(ValueError):
        funcao(np.zeros((4, 4, 3)), np.ones((3, 3)))
    with pytest.raises(ValueError):
        funcao(np.zeros((4, 4)), np.ones(3))
    with pytest.raises(ValueError):
        funcao(np.zeros((0, 4)), np.ones((3, 3)))


def _cinza(caminho):
    bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR).astype(float)
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def test_shapes_png_contra_referencias():
    img = _cinza(RAIZ / "Imagens" / "Shapes.png")
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    media = np.ones((5, 5)) / 25
    for k in (sobel_x, sobel_x.T, media):
        np.testing.assert_allclose(correlate2d_manual(img, k), referencia_cv2(img, k), atol=1e-9)


def test_imagem_de_teste_colorida_por_canal():
    bgr = cv2.imread(str(RAIZ / "Imagens" / "testpat.1k.color2.tif"), cv2.IMREAD_COLOR)
    k = np.ones((7, 7)) / 49
    for c in range(3):
        canal = bgr[..., c]
        np.testing.assert_allclose(correlate2d_manual(canal, k), referencia_cv2(canal, k), atol=1e-6)
