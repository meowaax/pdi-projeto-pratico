"""Testes da remoção de grade."""
import numpy as np
import pytest

from src import cleaning


def imagem(cor, forma=(6, 8)):
    return np.full(forma + (3,), cor, dtype=np.uint8)


def test_azulado_de_pixels_neutros_e_zero():
    for cinza in (0, 128, 223, 240, 255):
        assert cleaning.azulado(imagem((cinza, cinza, cinza))).max() == 0


def test_azulado_de_azul_puro_e_amarelo():
    assert cleaning.azulado(imagem((0, 0, 200)))[0, 0] == 200
    assert cleaning.azulado(imagem((255, 255, 0)))[0, 0] == 0   # nunca negativo
    assert cleaning.azulado(imagem((100, 50, 150)))[0, 0] == 50


def test_azulado_aceita_somente_rgb():
    with pytest.raises(ValueError):
        cleaning.azulado(np.zeros((4, 4)))


def test_grade_cinza_some_por_completo():
    img = imagem((255, 255, 255), (20, 30))
    img[5, :] = (223, 223, 223)           # linha horizontal da grade
    img[:, 10:12] = (230, 230, 230)       # linha vertical grossa
    img[0, :] = (0, 0, 0)                 # texto/eixo preto
    r = cleaning.remover_grade(img)
    assert not r.mascara.any()
    assert np.all(r.cobertura == 0.0)
    assert np.all(r.imagem == 255)


def test_ruido_de_compressao_abaixo_do_limiar_e_ignorado():
    img = imagem((240, 240, 240))
    img[2, 3] = (236, 236, 245)  # azulado 9, como o ruído medido na figura
    assert not cleaning.remover_grade(img).mascara.any()


def test_curva_azul_e_preservada_sobre_fundo_branco_e_sobre_a_grade():
    img = imagem((255, 255, 255), (10, 10))
    img[:, 4] = (223, 223, 223)               # grade vertical
    img[3, 1:9] = (30, 30, 200)               # curva horizontal cruzando a grade
    r = cleaning.remover_grade(img)
    assert r.mascara[3, 1:9].all()
    assert r.mascara.sum() == 8               # nada além da curva
    assert np.all(r.imagem[3, 1:9] == r.imagem[3, 1])  # o cruzamento é igual ao resto
    assert r.imagem[3, 4] == r.imagem[3, 3]
    assert r.imagem[3, 1] < 100               # curva escura
    assert np.all(r.imagem[r.mascara == 0] == 255)


def test_cobertura_proporcional_ao_azulado_independe_do_fundo():
    # Mesma mistura de 40% de azul sobre branco e sobre cinza da grade: mesma cobertura.
    azul = np.array([0.0, 0.0, 200.0])
    cobre = []
    for fundo in (255.0, 223.0):
        px = 0.4 * azul + 0.6 * fundo
        img = imagem((255, 255, 255), (8, 8))
        img[4, 4] = np.rint(px).astype(np.uint8)
        img[1, 1] = (0, 0, 200)               # azul pleno de referência
        cobre.append(cleaning.remover_grade(img).cobertura[4, 4])
    assert cobre[0] == pytest.approx(cobre[1], abs=0.02)
    assert 0 < cobre[0] < 1


def test_sem_pixels_azuis_nao_quebra():
    r = cleaning.remover_grade(imagem((255, 255, 255)))
    assert r.cobertura.sum() == 0
    assert r.azul_de_referencia > cleaning.LIMIAR_AZUL


def test_tipos_e_formas_de_saida():
    img = imagem((255, 255, 255), (7, 9))
    img[2, 2] = (0, 0, 220)
    r = cleaning.remover_grade(img)
    assert r.cobertura.shape == r.mascara.shape == r.imagem.shape == (7, 9)
    assert r.imagem.dtype == np.uint8
    assert r.mascara.dtype == bool
    assert r.cobertura.max() <= 1.0 and r.cobertura.min() >= 0.0


def test_nao_altera_a_entrada():
    img = imagem((255, 255, 255))
    img[1, 1] = (0, 0, 200)
    antes = img.copy()
    cleaning.remover_grade(img)
    np.testing.assert_array_equal(img, antes)
