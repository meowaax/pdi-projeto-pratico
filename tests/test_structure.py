"""Testes do realce da estrutura da curva."""
import cv2
import numpy as np
import pytest

from src import cleaning, config, roi, structure


def linha_horizontal(h=30, w=80, y=15, espessura=2):
    cob = np.zeros((h, w))
    cob[y:y + espessura, :] = 1.0
    return cob


def test_suavizada_preserva_a_tinta_no_interior():
    cob = linha_horizontal()
    e = structure.realcar_curva(cob, sigma=1.0)
    # Gaussiana soma 1: a tinta total se conserva (sem perdas nas bordas laterais de uma linha longa)
    assert e.suavizada[:, 10:-10].sum() == pytest.approx(cob[:, 10:-10].sum(), rel=1e-6)
    assert e.suavizada.max() <= 1.0 and e.suavizada.min() >= 0.0


def test_bordas_de_uma_linha_ficam_dos_dois_lados_do_traco():
    e = structure.realcar_curva(linha_horizontal(y=15, espessura=2), sigma=1.0)
    col = e.bordas[:, 40]
    topo, base = col[:15].argmax(), 15 + col[15:].argmax()
    assert topo < 15 and base > 15          # um pico de borda acima e outro abaixo do centro
    assert col[topo] == pytest.approx(col[base], rel=1e-6)   # simétrico (linha de 2 px centrada em 15,5)
    assert col[15] == pytest.approx(col[16], rel=1e-6)       # as duas linhas do traço são simétricas
    assert col[15] < col[topo]                               # e a resposta no centro do traço é menor que nas bordas


def test_binarizacao_mantem_a_linha_e_descarta_o_fundo():
    e = structure.realcar_curva(linha_horizontal(), sigma=0.6)
    assert e.mascara[15:17, 5:-5].all()
    assert not e.mascara[:12, :].any() and not e.mascara[19:, :].any()


def test_sigma_maior_engrossa_a_mascara():
    cob = linha_horizontal()
    fino = structure.realcar_curva(cob, sigma=0.6).mascara.sum()
    grosso = structure.realcar_curva(cob, sigma=2.0).mascara.sum()
    assert grosso > fino


def test_limiar_otsu_e_numerico():
    cob = linha_horizontal()
    e = structure.realcar_curva(cob, sigma=1.0, limiar="otsu")
    assert 0.0 < e.limiar < 1.0
    assert e.mascara[15:17, 5:-5].all()


@pytest.mark.parametrize("limiar", [0.0, 1.0, -0.1, 2.0, "invalido"])
def test_limiar_invalido(limiar):
    with pytest.raises(ValueError):
        structure.realcar_curva(linha_horizontal(), limiar=limiar)


def test_entrada_deve_ser_2d():
    with pytest.raises(ValueError):
        structure.realcar_curva(np.zeros((4, 4, 3)))


def test_nao_altera_a_entrada():
    cob = linha_horizontal()
    antes = cob.copy()
    structure.realcar_curva(cob)
    np.testing.assert_array_equal(cob, antes)


@pytest.fixture(scope="module")
def reais():
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    saida = {}
    for p in roi.segmentar_paineis(rgb, usar_fallback=False):
        so_curva = cleaning.limpar_recorte(roi.recortar(rgb, p))
        saida[p.nome] = (so_curva, structure.realcar_curva(so_curva.cobertura))
    return saida


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_curva_continua_e_sem_colunas_vazias(reais, nome):
    so_curva, e = reais[nome]
    n, _, est, _ = cv2.connectedComponentsWithStats(e.mascara.astype(np.uint8), connectivity=8)
    assert n - 1 == 1                                  # uma única curva, sem lacunas
    assert est[1, cv2.CC_STAT_WIDTH] == e.mascara.shape[1]
    assert e.mascara.any(axis=0).all()


@pytest.mark.parametrize("nome", ["a", "b", "c"])
def test_real_picos_preservados_com_sigma_padrao(reais, nome):
    # Com o σ padrão, os extremos da curva se mantêm a poucos pixels dos originais.
    so_curva, e = reais[nome]
    ref = so_curva.mascara
    topo_ref = np.flatnonzero(ref.any(axis=1))[[0, -1]]
    topo_new = np.flatnonzero(e.mascara.any(axis=1))[[0, -1]]
    assert np.abs(topo_ref - topo_new).max() <= 2
