"""Testes dos filtros de alto nível."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.filters import (bordas_sobel, gradiente_sobel, magnitude_gradiente,
                         suavizar_gaussiano)

RAIZ = Path(__file__).resolve().parent.parent


def shapes_cinza():
    return cv2.imread(str(RAIZ / "Imagens" / "Shapes.png"), cv2.IMREAD_GRAYSCALE).astype(float)


def test_magnitude_e_pitagorica():
    gx = np.array([[3.0, 0.0], [-5.0, 1.0]])
    gy = np.array([[4.0, 2.0], [12.0, 0.0]])
    np.testing.assert_allclose(magnitude_gradiente(gx, gy), [[5.0, 2.0], [13.0, 1.0]])


def test_gradiente_em_degrau_vertical():
    # Degrau claro-escuro vertical no meio: borda só na direção x.
    img = np.zeros((10, 10))
    img[:, 5:] = 100.0
    gx, gy, mag, ang = gradiente_sobel(img)
    assert np.all(gx[2:-2, 4:6] == 400.0)  # 4 (peso) * 100 (salto)
    assert np.all(gy[2:-2, :] == 0.0)
    assert np.all(mag[2:-2, :4] == 0.0) and np.all(mag[2:-2, 6:-1] == 0.0)  # a última coluna sente o zero padding
    assert np.all(ang[2:-2, 4:6] == 0.0)  # brilho cresce para a direita


def test_gradiente_em_degrau_horizontal():
    img = np.zeros((10, 10))
    img[5:, :] = 100.0
    gx, gy, mag, ang = gradiente_sobel(img)
    assert np.all(gy[4:6, 2:-2] == 400.0)
    assert np.all(gx[:, 2:-2] == 0.0)
    assert np.all(ang[4:6, 2:-2] == pytest.approx(np.pi / 2))


def test_gaussiano_preserva_media_no_interior_e_reduz_ruido():
    rng = np.random.default_rng(1)
    ruido = rng.normal(100, 20, (60, 60))
    suave = suavizar_gaussiano(ruido, 1.5)
    assert suave.shape == ruido.shape
    assert suave[10:-10, 10:-10].std() < 0.5 * ruido[10:-10, 10:-10].std()
    assert suave[10:-10, 10:-10].mean() == pytest.approx(ruido[10:-10, 10:-10].mean(), abs=2.0)


def test_gaussiano_em_imagem_constante_nao_altera_o_interior():
    suave = suavizar_gaussiano(np.full((30, 30), 77.0), 1.0)
    np.testing.assert_allclose(suave[5:-5, 5:-5], 77.0)


def test_bordas_do_shapes_so_aparecem_no_contorno():
    img = shapes_cinza()
    mag = bordas_sobel(img)
    assert mag.shape == img.shape
    contorno = mag > 0.25 * mag.max()
    assert 0 < contorno.mean() < 0.03        # poucos pixels de borda
    # No interior do triângulo (preto) e no fundo longe das formas, a resposta é nula.
    assert mag[250, 240] == 0.0   # dentro do triângulo
    assert mag[30, 400] == 0.0    # fundo
    assert mag[200, 525] == 0.0   # dentro do losango branco


def test_suavizar_antes_do_sobel_reduz_a_resposta_a_ruido():
    rng = np.random.default_rng(2)
    img = np.zeros((40, 40))
    img[:, 20:] = 100.0
    ruidosa = img + rng.normal(0, 10, img.shape)
    sem = bordas_sobel(ruidosa)[5:-5, :10].mean()           # região plana, longe da borda
    com = bordas_sobel(ruidosa, sigma=1.5)[5:-5, :10].mean()
    assert com < sem
