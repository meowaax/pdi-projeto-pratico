"""Testes da remoção de anotações azuis (casos sintéticos)."""
import logging

import numpy as np
import pytest

from src import cleaning

AZUL = (20, 20, 220)


def recorte_vazio(h=60, w=200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def com_curva(img, y=30):
    img[y, :] = AZUL            # traço fino que atravessa toda a largura
    img[y + 1, :] = AZUL
    return img


def com_rotulo(img, x, y, w=12, h=10):
    img[y:y + h, x:x + w] = AZUL    # bloco compacto, como uma letra em negrito
    return img


def test_curva_fica_e_rotulos_compactos_saem():
    img = com_curva(recorte_vazio())
    com_rotulo(img, 20, 5)
    com_rotulo(img, 120, 8, 25, 10)
    r = cleaning.limpar_recorte(img)
    assert len(r.anotacoes) == 2
    assert set(r.anotacoes) == {(20, 5, 12, 10), (120, 8, 25, 10)}
    assert r.mascara[30:32, :].all()
    assert r.mascara.sum() == 2 * 200          # só a curva (2 linhas x 200 colunas)
    assert not r.mascara[5:15, 20:32].any()
    assert r.largura_coberta == 1.0


def test_imagem_de_saida_e_branca_fora_da_curva():
    img = com_rotulo(com_curva(recorte_vazio()), 50, 5)
    r = cleaning.limpar_recorte(img)
    assert np.all(r.imagem[~r.mascara] == 255)
    assert np.all(r.imagem[r.mascara] < 128)


def test_trecho_fino_e_comprido_da_curva_nao_e_confundido_com_anotacao():
    # Curva quebrada por uma lacuna de 4 px: o trecho menor (80 px de largura, 2 de altura)
    # é um fragmento da curva, não uma letra, e precisa ficar.
    img = recorte_vazio()
    img[30:32, 0:80] = AZUL
    img[30:32, 84:200] = AZUL
    r = cleaning.limpar_recorte(img)
    assert r.anotacoes == ()
    assert r.mascara[30:32, 0:80].all() and r.mascara[30:32, 84:200].all()


def test_rotulo_maior_que_o_limite_nao_e_removido():
    img = com_curva(recorte_vazio())
    com_rotulo(img, 10, 5, 12, cleaning.ALTURA_MAX_ANOTACAO + 5)
    r = cleaning.limpar_recorte(img)
    assert len(r.anotacoes) == 0


def test_componente_pequeno_mas_esparso_nao_e_anotacao():
    # Diagonal fina e conectada: caixa de 10 x 10 com só 10 pixels (preenchimento 0,1), longe de uma letra.
    img = com_curva(recorte_vazio())
    for k in range(10):
        img[5 + k, 20 + k] = AZUL
    r = cleaning.limpar_recorte(img)
    assert r.anotacoes == ()
    assert r.mascara[5, 20] and r.mascara[14, 29]


def test_aviso_quando_a_curva_nao_atravessa_o_recorte(caplog):
    img = recorte_vazio()
    img[30:32, 0:100] = AZUL        # curva só até a metade
    with caplog.at_level(logging.WARNING):
        r = cleaning.limpar_recorte(img)
    assert r.largura_coberta == pytest.approx(0.5)
    assert "cobre só" in caplog.text


def test_sem_nada_azul_nao_quebra():
    r = cleaning.limpar_recorte(recorte_vazio())
    assert r.mascara.sum() == 0
    assert r.anotacoes == ()
    assert r.caixa_curva == (0, 0, 0, 0)


def test_texto_preto_e_grade_ja_saem_na_primeira_etapa():
    img = recorte_vazio()
    img[10:20, 10:60] = (0, 0, 0)       # legenda preta
    img[40, :] = (223, 223, 223)        # grade
    com_curva(img, 25)
    r = cleaning.limpar_recorte(img)
    assert r.mascara.sum() == 2 * 200
    assert not r.mascara[10:20, 10:60].any()


def test_nao_altera_a_entrada():
    img = com_rotulo(com_curva(recorte_vazio()), 20, 5)
    antes = img.copy()
    cleaning.limpar_recorte(img)
    np.testing.assert_array_equal(img, antes)
