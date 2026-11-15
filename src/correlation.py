"""Requisito 1: correlação 2D não normalizada, feita do zero no domínio do espaço.

Definição (h com M linhas e N colunas, âncora no "centro" (ay, ax)):

    g[i, j] = soma_{m, n} h[m, n] * I[i + m - ay, j + n - ax]

onde I vale 0 fora da imagem (zero padding). A saída tem o mesmo tamanho de I.

Convenção da âncora: ay = M // 2 e ax = N // 2. Para kernels ímpares é o centro
exato. Para kernels pares é o elemento logo após o centro (mesma convenção do
cv2.filter2D), de modo que o padding é M // 2 acima e M - 1 - M // 2 abaixo
(idem para as colunas).

Não usa scipy.signal.correlate2d nem cv2.filter2D. Os testes usam as duas
funções apenas como referência de validação.
"""
import numpy as np


def _validar(imagem, kernel):
    """Converte para float64 e confere se as entradas são matrizes 2D não vazias."""
    imagem = np.asarray(imagem, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if imagem.ndim != 2:
        raise ValueError(f"A imagem deve ser 2D (tons de cinza), recebido ndim={imagem.ndim}")
    if kernel.ndim != 2:
        raise ValueError(f"O kernel deve ser 2D, recebido ndim={kernel.ndim}")
    if imagem.size == 0 or kernel.size == 0:
        raise ValueError("Imagem e kernel não podem ser vazios")
    return imagem, kernel


def _paddings(kernel):
    """Retorna (acima, abaixo, esquerda, direita) para manter o tamanho da imagem."""
    M, N = kernel.shape
    ay, ax = M // 2, N // 2
    return ay, M - 1 - ay, ax, N - 1 - ax


def correlate2d_naive(imagem, kernel):
    """Versão de referência com quatro laços explícitos (lenta, usar em imagens pequenas).

    Percorre cada pixel de saída (i, j) e soma os produtos com a vizinhança,
    tratando os pixels fora da imagem como zero.
    """
    imagem, kernel = _validar(imagem, kernel)
    H, W = imagem.shape
    M, N = kernel.shape
    ay, ax = M // 2, N // 2
    saida = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for m in range(M):
                for n in range(N):
                    y = i + m - ay
                    x = j + n - ax
                    if 0 <= y < H and 0 <= x < W:  # fora da imagem = 0 (zero padding)
                        acc += kernel[m, n] * imagem[y, x]
            saida[i, j] = acc
    return saida


def correlate2d_manual(imagem, kernel):
    """Correlação 2D não normalizada com zero padding, mesmo tamanho da imagem.

    Implementação no domínio do espaço, equivalente à versão com laços, mas
    percorrendo os M*N elementos do kernel e somando a imagem deslocada
    (com padding de zeros) multiplicada por cada coeficiente. Os laços do kernel
    são explícitos, e o deslocamento da imagem inteira usa fatias do NumPy.

    Parâmetros
    ----------
    imagem : array 2D (tons de cinza), qualquer tipo numérico.
    kernel : array 2D (M x N), qualquer tamanho, par ou ímpar.

    Retorna
    -------
    array float64 com o mesmo shape de `imagem`.
    """
    imagem, kernel = _validar(imagem, kernel)
    H, W = imagem.shape
    M, N = kernel.shape
    acima, abaixo, esq, dir_ = _paddings(kernel)

    # Zero padding: a borda extra garante que toda janela M x N caiba na imagem.
    padded = np.zeros((H + acima + abaixo, W + esq + dir_), dtype=np.float64)
    padded[acima:acima + H, esq:esq + W] = imagem

    saida = np.zeros((H, W), dtype=np.float64)
    for m in range(M):
        for n in range(N):
            coef = kernel[m, n]
            if coef != 0.0:
                saida += coef * padded[m:m + H, n:n + W]
    return saida
