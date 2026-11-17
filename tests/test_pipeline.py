"""Testes do pipeline integrado."""
from dataclasses import replace

import numpy as np
import pytest

from src import config, pipeline, roi


@pytest.fixture(scope="module")
def rgb():
    return pipeline.carregar_figura()


@pytest.fixture(scope="module")
def resultados(rgb):
    return pipeline.processar_figura(rgb)


def test_carregar_figura_devolve_rgb_uint8(rgb):
    assert rgb.shape == (867, 707, 3) and rgb.dtype == np.uint8
    assert rgb.flags["C_CONTIGUOUS"]


def test_figura_inexistente_levanta_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="inexistente.png"):
        pipeline.carregar_figura(tmp_path / "inexistente.png")


def test_processa_os_tres_paineis_na_ordem(resultados):
    assert [r.nome for r in resultados] == ["a", "b", "c"]
    for r in resultados:
        assert r.recorte.ndim == 3 and r.recorte.shape[2] == 3
        assert r.sinal.t.shape == r.sinal.amplitude.shape
        assert 660 <= len(r.sinal.t) <= 668
        assert not np.isnan(r.sinal.amplitude).any()


def test_resultado_e_reprodutivel_e_bate_com_as_tasks_anteriores(rgb, resultados):
    outra = pipeline.processar_figura(rgb)
    for r, o in zip(resultados, outra):
        np.testing.assert_array_equal(r.sinal.amplitude, o.sinal.amplitude)
    a = resultados[0].sinal
    assert a.amplitude_pct.max() == pytest.approx(92.7, abs=0.2)     # S2 do painel a (Task 8)
    assert a.amplitude_pct.min() == pytest.approx(-72.6, abs=0.2)
    assert [len(r.limpo.anotacoes) for r in resultados] == [9, 3, 3]  # Task 6


def test_parametros_padrao_vem_do_config():
    p = pipeline.Parametros()
    assert (p.sigma, p.limiar_binarizacao, p.limiar_azul) == (config.SIGMA_GAUSSIANO, config.LIMIAR_BINARIZACAO, config.LIMIAR_AZUL)
    assert (p.regra, p.interpolacao, p.normalizacao, p.alvo) == (
        config.REGRA_COLAPSO, config.METODO_INTERPOLACAO, config.NORMALIZACAO, config.ALVO_NORMALIZACAO)


def test_parametros_chegam_as_etapas(rgb):
    centro = pipeline.processar_figura(rgb, replace(pipeline.Parametros(), regra="centro"))
    assert centro[0].sinal.regra == "centro"
    assert centro[0].sinal.amplitude_pct.max() < 80          # a regra 'centro' achata o pico de +93%
    largo = pipeline.processar_figura(rgb, replace(pipeline.Parametros(), sigma=2.0))
    assert largo[0].estrutura.sigma == 2.0
    assert largo[0].estrutura.mascara.sum() > pipeline.processar_figura(rgb)[0].estrutura.mascara.sum()


def test_sem_fallback_a_imagem_sem_paineis_levanta_erro():
    branca = np.full((867, 707, 3), 255, dtype=np.uint8)
    with pytest.raises(roi.RoiError):
        pipeline.processar_figura(branca, replace(pipeline.Parametros(), usar_fallback=False))


def test_resultados_sao_imutaveis(resultados):
    with pytest.raises(Exception):
        resultados[0].sinal = None
