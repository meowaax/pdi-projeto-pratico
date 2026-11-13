"""Testes da Task 1B: as medições de explore_data.py batem com src/config.py."""
import cv2
import numpy as np
import pytest

import explore_data as ed
from src import config


@pytest.fixture(scope="module")
def medidas():
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1]
    azul, cinza, escuro = ed.mascaras_cor(rgb)
    return {n: ed.medir_painel(n, rgb, azul, cinza, escuro) for n in ed.PAINEIS}


def test_figura_tem_tres_paineis(medidas):
    assert set(medidas) == {"a", "b", "c"}


def test_dez_linhas_verticais_e_escala_de_tempo(medidas):
    for m in medidas.values():
        assert m["n_linhas_verticais_detectadas"] == 10
        assert m["residuo_ajuste_x_px"] < 1.5
        assert m["px_por_segundo"] == pytest.approx(config.PX_POR_SEGUNDO, abs=3)
        duracao = (m["x_borda_direita_px"] - m["x_borda_esquerda_px"]) / m["px_por_segundo"]
        assert duracao == pytest.approx(config.DURACAO_S, abs=0.005)


def test_config_bate_com_medicoes(medidas):
    for nome, m in medidas.items():
        c = config.PAINEIS[nome]
        assert list(c["roi"]) == m["roi_xyxy"]
        assert c["x_esq"] == pytest.approx(m["x_borda_esquerda_px"], abs=0.2)
        assert c["x_dir"] == pytest.approx(m["x_borda_direita_px"], abs=0.2)
        assert c["t_ini"] == m["t_inicial_s"]
        assert c["y_75"] == pytest.approx(m["amplitude_y_px"]["75"], abs=0.2)
        assert c["y_zero"] == pytest.approx(m["amplitude_y_px"]["0"], abs=0.2)
        assert c["y_m75"] == pytest.approx(m["amplitude_y_px"]["-75"], abs=0.2)
        assert list(c["legenda"]) == m["legenda_escura_bbox_xyxy"]


def test_rois_nao_se_sobrepoem():
    caixas = [config.PAINEIS[n]["roi"] for n in "abc"]
    for (_, ya0, _, ya1), (_, yb0, _, yb1) in zip(caixas, caixas[1:]):
        assert ya1 < yb0


def test_toda_coluna_do_grafico_tem_curva(medidas):
    for m in medidas.values():
        assert m["colunas_com_azul"] == m["colunas_totais"]


def test_imagens_de_teste_do_req1_abrem():
    for caminho in ed.IMAGENS_TESTE:
        info = ed.descrever_imagem(caminho)
        assert info["dtype"] == "uint8"
        assert info["shape"][2] == 3


def test_exploracao_gera_arquivos(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "SAIDA", tmp_path)
    ed.main()
    for nome in ("measurements.json", "rois_anotadas.png", "mascaras_cor.png", "componentes_azuis.png"):
        assert (tmp_path / nome).exists()
    assert np.isfinite(config.N_AMOSTRAS)
