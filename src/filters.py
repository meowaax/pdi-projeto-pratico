"""Filtros espaciais de alto nível, todos baseados em correlate2d_manual."""
import numpy as np

from src.correlation import correlate2d_manual
from src.kernels import SOBEL_X, SOBEL_Y, kernel_gaussiano


def suavizar_gaussiano(imagem, sigma, tamanho=None):
    """Suaviza a imagem com uma Gaussiana de desvio `sigma` (zero padding nas bordas)."""
    return correlate2d_manual(imagem, kernel_gaussiano(sigma, tamanho))


def gradiente_sobel(imagem):
    """Aplica Sobel X e Y e devolve (gx, gy, magnitude, direcao).

    magnitude = sqrt(gx² + gy²); direcao em radianos, np.arctan2(gy, gx), no
    intervalo (-π, π], medida a partir do eixo x (para a direita) com y para baixo.
    """
    gx = correlate2d_manual(imagem, SOBEL_X)
    gy = correlate2d_manual(imagem, SOBEL_Y)
    return gx, gy, magnitude_gradiente(gx, gy), np.arctan2(gy, gx)


def magnitude_gradiente(gx, gy):
    """Magnitude euclidiana do gradiente a partir das componentes."""
    return np.hypot(gx, gy)


def bordas_sobel(imagem, sigma=None):
    """Mapa de bordas (magnitude do Sobel), com suavização Gaussiana opcional antes."""
    if sigma is not None:
        imagem = suavizar_gaussiano(imagem, sigma)
    return gradiente_sobel(imagem)[2]
