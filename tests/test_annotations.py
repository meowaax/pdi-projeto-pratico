"""Testes da remoção de anotações azuis (casos sintéticos)."""
import logging

import cv2
import numpy as np
import pytest
from scipy import ndimage as ndi

from src import cleaning, config, roi

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


# ----------------------------------------------------------------------------
# Figure1.png real
# ----------------------------------------------------------------------------
# Anotações esperadas por painel: a = S1, S2 e as 7 letras de "systole"; b = S1, SM, S2; c = S1, VSD, S2.
ANOTACOES_ESPERADAS = {"a": 9, "b": 3, "c": 3}


@pytest.fixture(scope="module")
def reais():
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    saida = {}
    for p in roi.segmentar_paineis(rgb, usar_fallback=False):
        recorte = roi.recortar(rgb, p)
        saida[p.nome] = (p, recorte, cleaning.remover_grade(recorte), cleaning.limpar_recorte(recorte))
    return saida


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_numero_de_anotacoes_removidas(reais, nome):
    assert len(reais[nome][3].anotacoes) == ANOTACOES_ESPERADAS[nome]


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_resta_um_unico_componente_que_atravessa_o_painel(reais, nome):
    p, recorte, _, r = reais[nome]
    n, _, est, _ = cv2.connectedComponentsWithStats(r.mascara.astype(np.uint8), connectivity=8)
    assert n - 1 == 1
    assert est[1, cv2.CC_STAT_WIDTH] == recorte.shape[1]
    assert r.largura_coberta == 1.0


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_nenhum_pixel_da_curva_foi_apagado(reais, nome):
    _, _, sem_grade, r = reais[nome]
    rotulos, curva, _, _ = cleaning.classificar_componentes(sem_grade.mascara)
    np.testing.assert_array_equal(r.mascara, rotulos == curva)
    # e a cobertura da curva é a mesma de antes
    np.testing.assert_array_equal(r.cobertura[r.mascara], sem_grade.cobertura[r.mascara])


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_anotacoes_ficam_longe_da_curva(reais, nome):
    # Margem que justifica separar por componentes: nenhuma letra encosta na curva.
    _, _, sem_grade, r = reais[nome]
    rotulos, curva, anotacoes, _ = cleaning.classificar_componentes(sem_grade.mascara)
    distancia = ndi.distance_transform_edt(rotulos != curva)
    for i in anotacoes:
        assert distancia[rotulos == i].min() >= 5.0


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_nao_sobra_texto_nem_eixo(reais, nome):
    p, recorte, _, r = reais[nome]
    # nada na faixa da legenda "Diaphragm mode; Normalized" além da própria curva
    x0, y0, _, _ = p.roi
    lx0, ly0, lx1, ly1 = config.PAINEIS[nome]["legenda"]
    legenda = r.mascara[ly0 - y0:ly1 - y0 + 1, lx0 - x0:lx1 - x0 + 1]
    assert legenda.sum() == 0
    # nenhuma coluna é uma linha vertical (eixo ou borda): a curva nunca ocupa >90% da altura de uma coluna
    assert r.mascara.sum(axis=0).max() < 0.9 * r.mascara.shape[0]


def test_real_barra_do_painel_c_fica_fora_da_roi(reais):
    # À direita do painel c há uma barra azul-escura (x1+3 a x1+5) que não pode entrar no recorte.
    p, recorte, _, r = reais["c"]
    assert not r.mascara[:, -1].all()
    assert r.mascara[:, -3:].sum(axis=0).max() < 0.5 * r.mascara.shape[0]
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1]
    x1 = p.roi[2]
    barra = cleaning.azulado(rgb[p.roi[1]:p.roi[3] + 1, x1 + 1:x1 + 8]) > cleaning.LIMIAR_AZUL
    assert barra.any(axis=0).sum() >= 1                   # a barra existe logo à direita
    assert barra[:, 0].sum() < 0.5 * barra.shape[0]       # mas não toca a coluna x1 + 1 da ROI
