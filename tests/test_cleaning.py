"""Testes da remoção de grade."""
import cv2
import numpy as np
import pytest

from src import cleaning, config, roi


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


# ----------------------------------------------------------------------------
# Validação com curva sintética (gabarito conhecido) e com a Figure1.png real
# ----------------------------------------------------------------------------
def curva_sintetica(forma=(120, 300), com_grade=False, semente=0):
    """Curva azul com antialiasing sobre fundo branco, opcionalmente sobre uma grade cinza."""
    rng = np.random.default_rng(semente)
    h, w = forma
    x = np.arange(0, w, 3)
    y = h / 2 + 35 * np.sin(x / 18.0) * np.exp(-((x - w / 2) / 90.0) ** 2) + rng.normal(0, 2, x.size)
    pontos = np.stack([x, y], axis=1).astype(np.float32)
    # Desenha em 8x e reduz: gera bordas suavizadas como as da figura.
    grande = np.full((h * 8, w * 8, 3), 255, dtype=np.uint8)
    cv2.polylines(grande, [np.rint(pontos * 8).astype(np.int32)], False, (40, 40, 205), 14, cv2.LINE_AA)
    curva = cv2.resize(grande, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    if not com_grade:
        return np.rint(curva).astype(np.uint8)
    # Mistura a curva com o fundo: onde a curva é branca, mostra a grade por baixo.
    fundo = np.full((h, w, 3), 255.0)
    fundo[::25, :] = 223.0
    fundo[:, ::40] = 230.0
    fundo[h // 2, :] = 223.0
    # curva = a*cor + (1-a)*255  ->  a = (255 - curva_B_menos_R) ... via canal R
    a = (255.0 - curva[..., 0]) / (255.0 - 40.0)
    return np.rint(a[..., None] * np.array([40, 40, 205.0]) + (1 - a[..., None]) * fundo).astype(np.uint8)


def test_sintetico_grade_removida_e_curva_igual_a_sem_grade():
    limpa = cleaning.remover_grade(curva_sintetica(com_grade=False))
    com = cleaning.remover_grade(curva_sintetica(com_grade=True))
    assert limpa.mascara.sum() > 500
    assert np.abs(com.cobertura - limpa.cobertura).max() < 0.03
    assert abs(int(com.mascara.sum()) - int(limpa.mascara.sum())) <= 0.01 * limpa.mascara.sum()
    assert np.all(com.imagem[~limpa.mascara] == 255)   # tudo fora da curva ficou branco


def test_sintetico_cruzamento_com_a_grade_mantem_a_curva_continua():
    com = cleaning.remover_grade(curva_sintetica(com_grade=True))
    n, _, estat, _ = cv2.connectedComponentsWithStats(com.mascara.astype(np.uint8), connectivity=8)
    maior = estat[1:, cv2.CC_STAT_AREA].max()
    assert maior >= 0.95 * com.mascara.sum()   # uma única curva, sem quebras nas linhas da grade


@pytest.fixture(scope="module")
def recortes():
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    paineis = roi.segmentar_paineis(rgb, usar_fallback=False)
    return paineis, {p.nome: roi.recortar(rgb, p) for p in paineis}


def test_real_linhas_da_grade_ficam_brancas_fora_da_curva(recortes):
    paineis, rec = recortes
    for p in paineis:
        r = cleaning.remover_grade(rec[p.nome])
        x0, y0, _, _ = p.roi
        # Linhas horizontais da grade (±25%, ±50%): fora das colunas onde há curva, a imagem é toda branca.
        passo = (p.y_m75 - p.y_75) / 6
        for k in (1, 2, 4, 5):  # +50, +25, -25, -50
            y = int(round(p.y_75 + k * passo)) - y0
            sem_curva = ~r.mascara[max(y - 2, 0):y + 3, :].any(axis=0)
            assert sem_curva.sum() > 0.5 * sem_curva.size
            assert np.all(r.imagem[max(y - 2, 0):y + 3, sem_curva] == 255)


def test_real_nada_neutro_sobrevive(recortes):
    paineis, rec = recortes
    for p in paineis:
        r = cleaning.remover_grade(rec[p.nome])
        neutro = roi.mascara_cinza(rec[p.nome]) & (cleaning.azulado(rec[p.nome]) <= 9)
        assert not (r.mascara & neutro).any()
        assert r.cobertura[neutro].max() == 0


def test_real_curva_continua_por_toda_a_largura_do_painel(recortes):
    paineis, rec = recortes
    for p in paineis:
        r = cleaning.remover_grade(rec[p.nome])
        n, _, estat, _ = cv2.connectedComponentsWithStats(r.mascara.astype(np.uint8), connectivity=8)
        i = 1 + int(np.argmax(estat[1:, cv2.CC_STAT_AREA]))
        assert estat[i, cv2.CC_STAT_WIDTH] == rec[p.nome].shape[1]   # a curva cruza todas as linhas da grade
        assert r.mascara.any(axis=0).all()                            # toda coluna tem curva


def test_real_mantem_pixels_azuis_da_figura(recortes):
    paineis, rec = recortes
    for p in paineis:
        r = cleaning.remover_grade(rec[p.nome])
        azul = (cleaning.azulado(rec[p.nome]) > 40)
        assert azul.sum() > 0
        assert r.mascara[azul].all()      # todo azul de verdade foi mantido
        # e o limiar baixo só acrescenta bordas suavizadas (poucos pixels a mais)
        assert r.mascara.sum() < 2.0 * azul.sum()
