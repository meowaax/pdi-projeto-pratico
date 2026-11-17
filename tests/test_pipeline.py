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


# ----------------------------------------------------------------------------
# Execução completa e arquivos gerados
# ----------------------------------------------------------------------------
import json

import cv2

from src import audio


@pytest.fixture(scope="module")
def execucao(tmp_path_factory):
    return pipeline.executar(saida=tmp_path_factory.mktemp("saida"))


def test_gera_os_tres_wav_com_os_nomes_pedidos(execucao):
    nomes = sorted(p.name for p in (execucao.pasta / "audio").iterdir())
    assert nomes == ["normal.wav", "still.wav", "vsd.wav"]
    for nome in nomes:
        w = audio.ler_wav(execucao.pasta / "audio" / nome)
        assert (w.taxa, w.bits, w.canais, w.n_amostras) == (5000, 16, 1, 2500)


def test_gera_as_imagens_de_cada_etapa(execucao):
    esperadas = {"rois_detectadas.png"} | {
        f"{n}_{e}.png" for n in "abc" for e in
        ("01_original", "02_sem_grade", "03_sinal_limpo", "04_suavizado", "05_bordas", "06_binario", "07_esqueleto", "08_extracao")}
    assert set(execucao.imagens) == esperadas
    assert {p.name for p in (execucao.pasta / "etapas").iterdir()} == esperadas
    for caminho in execucao.imagens.values():
        assert cv2.imread(str(caminho), cv2.IMREAD_UNCHANGED) is not None


def test_imagens_de_cada_painel_tem_o_tamanho_do_recorte(execucao):
    for r in execucao.resultados:
        h, w = r.recorte.shape[:2]
        for etapa in ("02_sem_grade", "03_sinal_limpo", "04_suavizado", "05_bordas", "06_binario", "07_esqueleto"):
            img = cv2.imread(str(execucao.imagens[f"{r.nome}_{etapa}.png"]), cv2.IMREAD_UNCHANGED)
            assert img.shape == (h, w), (r.nome, etapa)
        orig = cv2.imread(str(execucao.imagens[f"{r.nome}_01_original.png"]), cv2.IMREAD_COLOR)[:, :, ::-1]
        np.testing.assert_array_equal(orig, r.recorte)


def test_imagem_do_esqueleto_e_so_preto_e_branco(execucao):
    img = cv2.imread(str(execucao.imagens["a_07_esqueleto.png"]), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(img)) == {0, 255}
    assert (img == 0).sum() == execucao.resultados[0].estrutura.esqueleto.sum()


def test_resumo_json_tem_os_numeros_principais(execucao):
    resumo = json.loads((execucao.pasta / "resumo.json").read_text(encoding="utf-8"))
    assert resumo == json.loads(json.dumps(execucao.resumo))
    assert resumo["audio"] == {"fs": 5000, "bits": 16, "canais": 1, "amostras": 2500, "duracao_s": 0.5}
    assert set(resumo["paineis"]) == {"a", "b", "c"}
    assert resumo["paineis"]["a"]["wav"]["arquivo"] == "normal.wav"
    assert resumo["paineis"]["a"]["anotacoes_removidas"] == 9
    assert resumo["parametros"]["sigma"] == config.SIGMA_GAUSSIANO


def test_wav_reflete_os_parametros_de_normalizacao(tmp_path):
    ex = pipeline.executar(saida=tmp_path, params=replace(pipeline.Parametros(), normalizacao="escala"))
    pico_a = np.abs(audio.ler_wav(tmp_path / "audio" / "normal.wav").amostras).max()
    pico_b = np.abs(audio.ler_wav(tmp_path / "audio" / "still.wav").amostras).max()
    assert pico_a / pico_b == pytest.approx(ex.wavs["a"]["pico_pct"] / ex.wavs["b"]["pico_pct"], rel=0.02)


def test_executar_e_idempotente(tmp_path):
    pipeline.executar(saida=tmp_path)
    antes = {p.name: p.read_bytes() for p in (tmp_path / "audio").iterdir()}
    pipeline.executar(saida=tmp_path)
    depois = {p.name: p.read_bytes() for p in (tmp_path / "audio").iterdir()}
    assert antes == depois


def test_executar_com_figura_inexistente_nao_cria_saidas(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.executar(figura=tmp_path / "nao_existe.png", saida=tmp_path / "saida")
    assert not (tmp_path / "saida").exists()


def test_executar_registra_as_etapas_no_log(tmp_path, caplog):
    with caplog.at_level("INFO"):
        pipeline.executar(saida=tmp_path)
    assert "3 painéis segmentados" in caplog.text
    assert "normal.wav gravado (2500 amostras a 5000 Hz, 16 bits, mono" in caplog.text
    assert "pronto em" in caplog.text
