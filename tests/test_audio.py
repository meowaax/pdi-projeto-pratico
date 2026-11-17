"""Testes da reamostragem, quantização e exportação de áudio."""
import numpy as np
import pytest

from src import audio, config


# ----------------------------------------------------------------------------
# Grade uniforme e reamostragem
# ----------------------------------------------------------------------------
def test_grade_tem_2500_amostras_a_5_khz():
    g = audio.grade_uniforme(5.665)
    assert len(g) == 2500
    assert g[0] == pytest.approx(5.665)
    assert np.diff(g) == pytest.approx(1 / 5000)
    assert g[-1] == pytest.approx(5.665 + 0.5 - 1 / 5000)


def test_reamostrar_rampa_linear_e_exata_nos_dois_metodos():
    t = np.arange(0, 0.5, 1 / 1333.0)
    y = 0.3 + 0.8 * t                       # rampa: interpolação linear e cúbica reproduzem exatamente
    for metodo in audio.METODOS:
        g, r = audio.reamostrar(t, y, 0.0, metodo=metodo)
        dentro = g <= t[-1]
        np.testing.assert_allclose(r[dentro], 0.3 + 0.8 * g[dentro], atol=1e-12)


def test_cubica_reproduz_polinomio_cubico_e_linear_nao():
    t = np.linspace(0, 0.5, 400)
    y = 4 * (t - 0.25) ** 3
    g, c = audio.reamostrar(t, y, 0.0, metodo="cubica")
    _, l = audio.reamostrar(t, y, 0.0, metodo="linear")
    dentro = g <= t[-1]
    ref = 4 * (g[dentro] - 0.25) ** 3
    assert np.abs(c[dentro] - ref).max() < 1e-9
    assert np.abs(l[dentro] - ref).max() > 1e-7


def test_senoide_de_baixa_frequencia_e_reconstruida_com_erro_pequeno():
    t = np.arange(0.001, 0.499, 1 / 1333.2)
    y = np.sin(2 * np.pi * 50 * t)
    g, r = audio.reamostrar(t, y, 0.0)
    dentro = (g >= t[0]) & (g <= t[-1])
    # Cota do erro da interpolação linear: h² / 8 * max|f''| = (1/1333,2)² / 8 * (2π·50)² = 0,0069.
    cota = (1 / 1333.2) ** 2 / 8 * (2 * np.pi * 50) ** 2
    erro = np.abs(r[dentro] - np.sin(2 * np.pi * 50 * g[dentro])).max()
    assert erro <= cota * 1.001
    assert erro > 0.5 * cota                # e a cota é atingida: o teste não é frouxo


def test_pontas_repetem_o_valor_da_borda():
    t = np.linspace(0.002, 0.497, 100)
    y = np.linspace(0.2, 0.7, 100)
    g, r = audio.reamostrar(t, y, 0.0)
    assert np.all(r[g < t[0]] == 0.2)
    assert np.all(r[g > t[-1]] == 0.7)
    assert len(r) == 2500 and not np.isnan(r).any()


def test_linear_nao_ultrapassa_os_picos_e_cubica_pode():
    t = np.arange(0, 0.5, 1 / 1333.2)
    y = np.zeros_like(t)
    y[300] = 1.0                                       # pico isolado
    _, lin = audio.reamostrar(t, y, 0.0, metodo="linear")
    _, cub = audio.reamostrar(t, y, 0.0, metodo="cubica")
    assert lin.max() <= 1.0 and lin.min() >= 0.0
    assert cub.min() < -0.01                           # a cúbica oscila ao redor do pico


@pytest.mark.parametrize("t, y", [
    (np.array([0.0]), np.array([1.0])),                             # poucas amostras
    (np.array([0.0, 0.1, 0.1]), np.array([0.0, 1.0, 2.0])),         # t repetido
    (np.array([0.0, 0.2, 0.1]), np.array([0.0, 1.0, 2.0])),         # t decrescente
    (np.array([0.0, 0.1, 0.2]), np.array([0.0, 1.0])),              # tamanhos diferentes
    (np.array([0.0, 0.1, 0.2]), np.array([0.0, np.nan, 1.0])),      # NaN
])
def test_entradas_invalidas(t, y):
    with pytest.raises(ValueError):
        audio.reamostrar(t, y, 0.0)


def test_metodo_invalido():
    with pytest.raises(ValueError):
        audio.reamostrar(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0, metodo="sinc")


def test_nao_altera_as_entradas():
    t = np.linspace(0.001, 0.499, 50)
    y = np.sin(10 * t)
    t0, y0 = t.copy(), y.copy()
    audio.reamostrar(t, y, 0.0)
    np.testing.assert_array_equal(t, t0)
    np.testing.assert_array_equal(y, y0)


# ----------------------------------------------------------------------------
# DC, normalização, quantização
# ----------------------------------------------------------------------------
def test_remover_dc_zera_a_media_e_preserva_a_forma():
    y = np.sin(np.linspace(0, 20, 500)) + 0.3
    r = audio.remover_dc(y)
    assert abs(r.mean()) < 1e-12
    np.testing.assert_allclose(r - r.mean(), y - y.mean())


def test_normalizar_pico_leva_o_maior_valor_ao_alvo():
    y = np.array([0.1, -0.4, 0.2, 0.05])
    r = audio.normalizar(y, "pico", 0.9)
    assert np.abs(r).max() == pytest.approx(0.9)
    assert r[1] < 0                                  # o sinal do pico é mantido
    np.testing.assert_allclose(r / y, np.full(4, 0.9 / 0.4))


def test_normalizar_escala_preserva_a_razao_entre_sinais():
    a, b = np.array([0.0, 0.8, -0.2]), np.array([0.0, 0.4, -0.1])
    ra, rb = audio.normalizar(a, "escala"), audio.normalizar(b, "escala")
    assert np.abs(ra).max() / np.abs(rb).max() == pytest.approx(2.0)
    assert np.abs(ra).max() == pytest.approx(0.8 * audio.ALVO_PADRAO)


def test_normalizar_silencio_nao_divide_por_zero():
    r = audio.normalizar(np.zeros(10))
    assert np.all(r == 0) and not np.isnan(r).any()


@pytest.mark.parametrize("modo, alvo", [("x", 0.9), ("pico", 0.0), ("pico", 1.5), ("pico", -0.1)])
def test_normalizar_parametros_invalidos(modo, alvo):
    with pytest.raises(ValueError):
        audio.normalizar(np.ones(3), modo, alvo)


def test_quantizar_extremos_e_saturacao():
    pcm = audio.quantizar(np.array([1.0, -1.0, 0.0, 0.5, 2.0, -3.0]))
    assert pcm.dtype == np.int16
    assert pcm.tolist() == [32767, -32767, 0, 16384, 32767, -32768]


def test_quantizar_erro_menor_que_meio_degrau():
    y = np.random.default_rng(0).uniform(-1, 1, 5000)
    pcm = audio.quantizar(y)
    assert np.abs(pcm / audio.PCM_MAX - y).max() <= 0.5 / audio.PCM_MAX + 1e-12


def test_quantizar_rejeita_nan():
    with pytest.raises(ValueError):
        audio.quantizar(np.array([0.0, np.nan]))


def test_sinal_para_pcm_cadeia_completa():
    t = np.linspace(0.0015, 0.4985, 665)
    y = 0.4 * np.sin(2 * np.pi * 40 * t) + 0.2             # com DC
    pcm = audio.sinal_para_pcm(t, y, 0.0)
    assert pcm.dtype == np.int16 and pcm.shape == (2500,)
    assert abs(pcm.mean()) < 0.02 * np.abs(pcm).max()      # DC removido
    assert np.abs(pcm).max() == pytest.approx(audio.ALVO_PADRAO * audio.PCM_MAX, abs=1)
    assert np.abs(pcm).max() < 32767                        # sem clipping


# ----------------------------------------------------------------------------
# Arquivo .wav
# ----------------------------------------------------------------------------
def test_gravar_e_ler_wav_ida_e_volta(tmp_path):
    pcm = audio.quantizar(np.sin(np.linspace(0, 60, 2500)) * 0.8)
    caminho = audio.gravar_wav(tmp_path / "x.wav", pcm)
    info = audio.ler_wav(caminho)
    assert (info.taxa, info.canais, info.bits, info.n_amostras) == (5000, 1, 16, 2500)
    assert info.duracao == pytest.approx(0.5)
    np.testing.assert_array_equal(info.amostras, pcm)


def test_wav_confere_com_leitor_independente(tmp_path):
    from scipy.io import wavfile
    pcm = audio.quantizar(np.cos(np.linspace(0, 40, 2500)) * 0.7)
    caminho = audio.gravar_wav(tmp_path / "y.wav", pcm)
    taxa, dados = wavfile.read(caminho)
    assert taxa == 5000 and dados.dtype == np.int16
    np.testing.assert_array_equal(dados, pcm)
    assert caminho.stat().st_size == 44 + 2 * 2500          # cabeçalho PCM de 44 bytes + 2 bytes por amostra


def test_gravar_wav_cria_pastas_e_rejeita_tipo_errado(tmp_path):
    audio.gravar_wav(tmp_path / "a" / "b" / "z.wav", np.zeros(10, dtype=np.int16))
    assert (tmp_path / "a" / "b" / "z.wav").exists()
    with pytest.raises(ValueError):
        audio.gravar_wav(tmp_path / "w.wav", np.zeros(10))              # float
    with pytest.raises(ValueError):
        audio.gravar_wav(tmp_path / "w.wav", np.zeros((10, 2), dtype=np.int16))   # 2D


def test_ler_wav_rejeita_8_bits(tmp_path):
    import wave
    caminho = tmp_path / "oito.wav"
    with wave.open(str(caminho), "wb") as arq:
        arq.setnchannels(1), arq.setsampwidth(1), arq.setframerate(5000)
        arq.writeframes(bytes(10))
    with pytest.raises(ValueError):
        audio.ler_wav(caminho)


# ----------------------------------------------------------------------------
# Exportação a partir da Figure1.png real
# ----------------------------------------------------------------------------
import cv2
from types import SimpleNamespace

from src import cleaning, curve, roi, structure


@pytest.fixture(scope="module")
def sinais_reais():
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    saida = {}
    for p in roi.segmentar_paineis(rgb, usar_fallback=False):
        est = structure.realcar_curva(cleaning.limpar_recorte(roi.recortar(rgb, p)).cobertura)
        saida[p.nome] = curve.extrair_sinal(est.mascara, curve.Calibracao.de_painel(p))
    return saida


@pytest.fixture(scope="module")
def wavs(sinais_reais, tmp_path_factory):
    pasta = tmp_path_factory.mktemp("audio")
    return pasta, audio.exportar_wavs(sinais_reais, pasta)


def test_gera_os_tres_arquivos_com_os_nomes_pedidos(wavs):
    pasta, info = wavs
    assert sorted(p.name for p in pasta.iterdir()) == ["normal.wav", "still.wav", "vsd.wav"]
    assert {n: i["caminho"].name for n, i in info.items()} == {"a": "normal.wav", "b": "still.wav", "c": "vsd.wav"}


@pytest.mark.parametrize("arquivo", ["normal.wav", "still.wav", "vsd.wav"])
def test_wav_real_tem_o_formato_exigido(wavs, arquivo):
    pasta, _ = wavs
    w = audio.ler_wav(pasta / arquivo)
    assert (w.taxa, w.bits, w.canais) == (5000, 16, 1)
    assert w.n_amostras == 2500 and w.duracao == pytest.approx(0.5)


@pytest.mark.parametrize("arquivo", ["normal.wav", "still.wav", "vsd.wav"])
def test_wav_real_sem_clipping_e_sem_dc(wavs, arquivo):
    pasta, _ = wavs
    x = audio.ler_wav(pasta / arquivo).amostras.astype(float)
    assert np.abs(x).max() < 32767                                  # nenhuma amostra saturada
    assert np.abs(x).max() == pytest.approx(audio.ALVO_PADRAO * 32767, abs=1)
    assert abs(x.mean()) < 0.01 * np.abs(x).max()                   # DC removido


def test_wav_real_segue_o_sinal_extraido(wavs, sinais_reais):
    pasta, _ = wavs
    for nome, arquivo in config.ARQUIVOS_WAV.items():
        s = sinais_reais[nome]
        x = audio.ler_wav(pasta / arquivo).amostras.astype(float)
        _, y = audio.reamostrar(s.t, s.amplitude, s.calibracao.t_ini)
        assert np.corrcoef(x, y)[0, 1] > 0.9999


def test_niveis_relativos_sao_informados(wavs):
    _, info = wavs
    assert info["c"]["rms_pct"] > info["a"]["rms_pct"]              # VSD (denso) é mais forte que o normal
    assert info["a"]["pico_pct"] > info["b"]["pico_pct"]
    for i in info.values():
        assert 0 < i["rms_pct"] < i["pico_pct"] <= 100


def test_normalizacao_escala_preserva_o_volume_relativo(sinais_reais, tmp_path):
    info = audio.exportar_wavs(sinais_reais, tmp_path, normalizacao="escala")
    rms = {n: np.sqrt(np.mean(audio.ler_wav(i["caminho"]).amostras.astype(float) ** 2)) for n, i in info.items()}
    for n in "abc":
        assert rms[n] / rms["a"] == pytest.approx(info[n]["rms_pct"] / info["a"]["rms_pct"], rel=0.01)


def test_exportar_rejeita_painel_desconhecido(tmp_path):
    falso = SimpleNamespace(t=np.array([0.0, 0.5]), amplitude=np.array([0.0, 0.0]), calibracao=SimpleNamespace(t_ini=0.0))
    with pytest.raises(ValueError):
        audio.exportar_wavs({"z": falso}, tmp_path)


def test_exportar_sinal_silencioso_gera_arquivo_valido(tmp_path):
    mudo = SimpleNamespace(t=np.linspace(0.001, 0.499, 100), amplitude=np.zeros(100), calibracao=SimpleNamespace(t_ini=0.0))
    info = audio.exportar_wavs({"a": mudo}, tmp_path)
    assert not audio.ler_wav(info["a"]["caminho"]).amostras.any()
