"""Testes da análise comparativa."""
import numpy as np
import pytest

from src import analysis, config, pipeline


def sinal_com_s1_s2(ruido_sopro=0.0, t_s1=0.12, t_s2=0.37, t_ini=2.0, semente=0):
    """Dois estouros (S1 e S2) em ruído de fundo fraco; opcionalmente ruído entre eles."""
    rng = np.random.default_rng(semente)
    t = t_ini + np.arange(2500) / 5000
    y = 0.8 * rng.normal(size=t.size)                              # ruído de fundo ~0,8%
    for t0, a in ((t_s1, 40.0), (t_s2, 50.0)):
        y += a * np.exp(-(((t - t_ini - t0) / 0.012) ** 2)) * np.sin(2 * np.pi * 110 * (t - t_ini - t0))
    entre = (t - t_ini > t_s1 + 0.05) & (t - t_ini < t_s2 - 0.05)
    y[entre] += ruido_sopro * rng.normal(size=entre.sum())
    return analysis.SinalAnalise("x", t, y - y.mean(), t_ini)


# ----------------------------------------------------------------------------
# Envolvente
# ----------------------------------------------------------------------------
def test_envolvente_de_senoide_e_a_amplitude_sobre_raiz_de_dois():
    t = np.arange(5000) / 5000
    y = 30 * np.sin(2 * np.pi * 100 * t)
    env = analysis.envolvente_rms(y, ms=20)
    assert env[500:-500] == pytest.approx(30 / np.sqrt(2), rel=0.01)


def test_envolvente_de_constante_e_do_valor_absoluto_inclusive_nas_bordas():
    env = analysis.envolvente_rms(np.full(100, -3.0))
    np.testing.assert_allclose(env, 3.0)


def test_envolvente_tem_o_mesmo_tamanho_e_nao_altera_a_entrada():
    y = np.random.default_rng(0).normal(size=333)
    y0 = y.copy()
    assert analysis.envolvente_rms(y).shape == y.shape
    np.testing.assert_array_equal(y, y0)


# ----------------------------------------------------------------------------
# Eventos e janelas
# ----------------------------------------------------------------------------
def test_localiza_s1_e_s2_perto_dos_estouros():
    s = sinal_com_s1_s2()
    # os rótulos ficam de 0 a 30 ms antes do estouro; aqui 20 ms antes
    ev = analysis.localizar_eventos(s, s.t_ini + 0.12 - 0.02, s.t_ini + 0.37 - 0.02)
    assert abs(ev.t_s1 - (s.t_ini + 0.12)) < 0.012
    assert abs(ev.t_s2 - (s.t_ini + 0.37)) < 0.012
    assert ev.intervalo_ms == pytest.approx(250, abs=15)


def test_rotulo_fora_do_sinal_levanta_erro():
    s = sinal_com_s1_s2()
    with pytest.raises(ValueError):
        analysis.localizar_eventos(s, 0.0, s.t_ini + 0.37)


def test_tempos_dos_rotulos_usam_a_caixa_mais_a_esquerda_e_a_mais_a_direita():
    from src.synthetic import calibracao_sintetica
    cal = calibracao_sintetica()
    caixas = [(400, 10, 15, 12), (100, 10, 15, 12), (250, 20, 30, 11), (260, 20, 10, 11)]    # fora de ordem
    t1, t2 = analysis.tempos_dos_rotulos(caixas, cal)
    assert t1 == pytest.approx(cal.tempo(107.5)) and t2 == pytest.approx(cal.tempo(407.5))
    with pytest.raises(ValueError):
        analysis.tempos_dos_rotulos([(1, 1, 1, 1)], cal)


def test_janelas_nao_se_sobrepoem_e_cobrem_o_sinal():
    s = sinal_com_s1_s2()
    ev = analysis.Eventos(s.t_ini + 0.12, s.t_ini + 0.37)
    j = analysis.janelas(s, ev)
    assert set(j) == {"s1", "s2", "sopro", "diastole"}
    total = sum(m.astype(int) for m in j.values())
    assert total.max() == 1                                   # nenhuma amostra em duas janelas
    assert total.min() == 1                                   # e nenhuma fora de todas
    assert j["s1"].sum() == pytest.approx(0.08 * 5000, abs=1) and j["s2"].sum() == pytest.approx(0.08 * 5000, abs=1)


def test_janelas_com_eventos_proximos_demais_levantam_erro():
    s = sinal_com_s1_s2()
    with pytest.raises(ValueError):
        analysis.janelas(s, analysis.Eventos(s.t_ini + 0.20, s.t_ini + 0.26))


# ----------------------------------------------------------------------------
# Métricas no tempo
# ----------------------------------------------------------------------------
def test_razao_db():
    assert analysis.razao_db(2.0, 1.0) == pytest.approx(6.0206, abs=1e-3)
    assert analysis.razao_db(1.0, 10.0) == pytest.approx(-20.0)
    assert analysis.razao_db(0.0, 0.0) == 0.0
    assert analysis.razao_db(1.0, 0.0) == float("inf") and analysis.razao_db(0.0, 1.0) == float("-inf")


def test_metricas_de_senoide_pura_na_janela():
    t = 2.0 + np.arange(2500) / 5000
    y = 20 * np.sin(2 * np.pi * 100 * (t - 2.0))
    s = analysis.SinalAnalise("x", t, y - y.mean(), 2.0)
    m = analysis.metricas_tempo(s, analysis.Eventos(2.12, 2.37))
    assert m["rms_sopro_pct"] == pytest.approx(20 / np.sqrt(2), rel=0.02)
    assert m["pico_pct"] == pytest.approx(20, rel=0.01)
    assert m["fator_de_crista"] == pytest.approx(np.sqrt(2), rel=0.02)
    assert m["cruzamentos_por_zero_sopro_hz"] == pytest.approx(100, rel=0.05)
    assert m["intervalo_s1_s2_ms"] == pytest.approx(250)


def test_periodicidade_de_tom_puro_e_de_ruido():
    coef, freq = analysis.periodicidade(tom(120.0, n=900))
    assert coef > 0.8 and freq == pytest.approx(120, abs=3)
    ruido = np.random.default_rng(3).normal(size=900)
    assert analysis.periodicidade(ruido)[0] < 0.3


def test_periodicidade_de_trem_de_pulsos():
    x = np.zeros(1000)
    x[::40] = 1.0                                          # um pulso a cada 8 ms = 125 Hz
    coef, freq = analysis.periodicidade(x)
    assert freq == pytest.approx(125, abs=1) and coef > 0.8


def test_periodicidade_rejeita_trecho_inutil():
    with pytest.raises(ValueError):
        analysis.periodicidade(np.zeros(500))
    with pytest.raises(ValueError):
        analysis.periodicidade(np.ones(4))


def test_sopro_sobre_diastole_e_positivo_quando_ha_sopro_e_nulo_quando_nao_ha():
    ev = analysis.Eventos(2.12, 2.37)
    com = analysis.metricas_tempo(sinal_com_s1_s2(ruido_sopro=6.0), ev)
    sem = analysis.metricas_tempo(sinal_com_s1_s2(ruido_sopro=0.0), ev)
    assert com["sopro_sobre_diastole_db"] > 12
    assert abs(sem["sopro_sobre_diastole_db"]) < 2.5
    assert com["rms_s1_pct"] > com["rms_sopro_pct"] > com["rms_diastole_pct"]


def test_metricas_nao_alteram_o_sinal():
    s = sinal_com_s1_s2()
    antes = s.y.copy()
    analysis.metricas_tempo(s, analysis.Eventos(2.12, 2.37))
    np.testing.assert_array_equal(s.y, antes)


# ----------------------------------------------------------------------------
# Figure1.png real
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reais():
    saida = {}
    for r in pipeline.processar_figura(pipeline.carregar_figura()):
        s = analysis.preparar_sinal(r)
        t1, t2 = analysis.tempos_dos_rotulos(r.limpo.anotacoes, r.sinal.calibracao)
        ev = analysis.localizar_eventos(s, t1, t2)
        saida[r.nome] = (s, ev, analysis.metricas_tempo(s, ev))
    return saida


def test_real_sinais_de_analise_tem_o_formato_do_audio(reais):
    for s, _, _ in reais.values():
        assert len(s.y) == 2500 and s.fs == 5000 and s.duracao == pytest.approx(0.5)
        assert abs(s.y.mean()) < 1e-9


def test_real_s1_e_s2_batem_com_a_leitura_da_figura(reais):
    # Instantes lidos na Figure1: (a) 5,795 e 6,047 s; (b) 1,461 e 1,737 s; (c) 1,145 e 1,455 s.
    esperado = {"a": (5.795, 6.047), "b": (1.461, 1.737), "c": (1.145, 1.455)}
    for nome, (_, ev, _) in reais.items():
        assert ev.t_s1 == pytest.approx(esperado[nome][0], abs=0.02), nome
        assert ev.t_s2 == pytest.approx(esperado[nome][1], abs=0.02), nome
        assert ev.intervalo_ms == pytest.approx({"a": 254, "b": 276, "c": 310}[nome], abs=15)   # intervalos lidos na figura


def test_real_normal_nao_tem_sopro_e_still_e_vsd_tem(reais):
    assert abs(reais["a"][2]["sopro_sobre_diastole_db"]) < 3
    assert reais["b"][2]["sopro_sobre_diastole_db"] > 6
    assert reais["c"][2]["sopro_sobre_diastole_db"] > 6


def test_real_vsd_tem_sopro_holossistolico_forte(reais):
    # O sopro do VSD tem nível parecido com o do S1; o do Still é bem menor; o Normal quase não tem.
    assert reais["c"][2]["sopro_sobre_s1_db"] > -3
    assert -12 < reais["b"][2]["sopro_sobre_s1_db"] < -3
    assert reais["a"][2]["sopro_sobre_s1_db"] < -15
    assert reais["c"][2]["rms_sopro_pct"] > 3 * reais["b"][2]["rms_sopro_pct"]


def test_real_still_e_periodico_e_vsd_e_ruidoso(reais):
    normal, still, vsd = (reais[n][2] for n in "abc")
    assert still["periodicidade_sopro"] > 0.5 and still["frequencia_periodica_sopro_hz"] == pytest.approx(120, abs=6)
    assert vsd["periodicidade_sopro"] < 0.3
    assert normal["periodicidade_sopro"] < 0.4
    assert still["periodicidade_sopro"] > 2 * vsd["periodicidade_sopro"]


def test_real_niveis_batem_com_o_demo_de_audio(reais):
    # Os mesmos RMS e picos que audio.exportar_wavs imprimiu na Task 9.
    assert reais["a"][2]["rms_pct"] == pytest.approx(10.87, abs=0.05)
    assert reais["b"][2]["rms_pct"] == pytest.approx(8.51, abs=0.05)
    assert reais["c"][2]["rms_pct"] == pytest.approx(17.64, abs=0.05)
    assert reais["a"][2]["pico_pct"] == pytest.approx(92.7, abs=0.2)


# ----------------------------------------------------------------------------
# Espectro
# ----------------------------------------------------------------------------
def tom(freq, amp=1.0, n=900, fs=5000):
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / fs)


def test_psd_preserva_a_potencia_media():
    x = np.random.default_rng(0).normal(0, 3, 4000)
    f, p = analysis.psd(x, nfft=8192)
    df = f[1] - f[0]
    assert np.sum(p) * df == pytest.approx(np.var(x), rel=0.1)         # Parseval: potência total ≈ variância


@pytest.mark.parametrize("freq", [60.0, 120.0, 250.0, 480.0])
def test_tom_puro_tem_o_pico_na_frequencia_certa(freq):
    f, p = analysis.psd(tom(freq))
    m = analysis.metricas_espectro(f, p)
    assert m["frequencia_dominante_hz"] == pytest.approx(freq, abs=1.5)
    assert m["centroide_hz"] == pytest.approx(freq, abs=5)
    assert m["planura"] < 0.01


def test_largura_a_meia_potencia_do_tom_e_a_do_lobo_da_janela_de_hann():
    n = 900
    f, p = analysis.psd(tom(120.0, n=n))
    largura = analysis.metricas_espectro(f, p)["largura_meia_potencia_hz"]
    assert largura == pytest.approx(1.44 * 5000 / n, rel=0.15)         # -3 dB da janela de Hann (potência)


def test_energia_por_faixa_de_dois_tons():
    x = tom(60.0, 1.0) + tom(300.0, 1.0)                                 # mesma amplitude: metade da energia em cada
    f, p = analysis.psd(x)
    e = analysis.metricas_espectro(f, p)["energia_faixas"]
    assert e["40-80"] == pytest.approx(0.5, abs=0.03) and e["200-500"] == pytest.approx(0.5, abs=0.03)
    assert e["80-200"] < 0.02 and e["500-660"] < 0.02
    assert sum(e.values()) == pytest.approx(1.0, abs=1e-9)


def test_planura_separa_tom_de_ruido_branco():
    ruido = np.random.default_rng(1).normal(size=2000)
    plano = analysis.metricas_espectro(*analysis.psd(ruido))["planura"]
    assert 0.4 < plano < 0.7                                              # periodograma de ruído branco: ~0,56
    assert analysis.metricas_espectro(*analysis.psd(tom(150.0)))["planura"] < 0.05


def test_espectro_sem_energia_na_banda_levanta_erro():
    f, p = analysis.psd(tom(1200.0))                                       # só energia acima da banda
    with pytest.raises(ValueError):
        analysis.metricas_espectro(f, np.where(f > 700, p, 0.0))


def test_psd_rejeita_trecho_curto():
    with pytest.raises(ValueError):
        analysis.psd(np.ones(4))


def test_espectrograma_acompanha_a_mudanca_de_frequencia():
    y = np.concatenate([tom(100.0, n=1250), tom(300.0, n=1250)])          # 0,25 s em 100 Hz e 0,25 s em 300 Hz
    t, f, P = analysis.espectrograma(y)
    assert P.shape == (len(f), len(t)) and f.max() <= 700 and np.all(P >= 0)
    primeira, segunda = P[:, t < 0.2], P[:, t > 0.3]
    assert f[np.argmax(primeira.mean(axis=1))] == pytest.approx(100, abs=6)
    assert f[np.argmax(segunda.mean(axis=1))] == pytest.approx(300, abs=6)
    assert t[0] == pytest.approx(0.0125) and np.diff(t) == pytest.approx(0.002)


def test_espectrograma_rejeita_sinal_curto():
    with pytest.raises(ValueError):
        analysis.espectrograma(np.ones(50))


# ----------------------------------------------------------------------------
# Espectro dos sinais reais
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def espectros(reais):
    saida = {}
    for nome, (s, ev, _) in reais.items():
        jan = analysis.janelas(s, ev)
        saida[nome] = analysis.metricas_espectro(*analysis.espectro_da_janela(s, jan["sopro"]))
    return saida


def test_real_still_tem_tom_em_torno_de_120_hz(espectros):
    # O artigo mede 118,75 Hz em média (90 a 170 Hz) para o sopro de Still.
    assert 90 <= espectros["b"]["frequencia_dominante_hz"] <= 170
    assert espectros["b"]["frequencia_dominante_hz"] == pytest.approx(120, abs=15)
    assert espectros["b"]["energia_faixas"]["80-200"] > 0.8


def test_real_vsd_e_mais_largo_e_mais_agudo_que_still(espectros):
    vsd, still = espectros["c"], espectros["b"]
    assert vsd["planura"] > 2 * still["planura"]
    assert vsd["centroide_hz"] > still["centroide_hz"] + 40
    assert vsd["energia_faixas"]["200-500"] > 0.3 > still["energia_faixas"]["200-500"]


def test_real_energias_por_faixa_somam_um(espectros):
    for m in espectros.values():
        assert sum(m["energia_faixas"].values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# Análise completa e confronto com o artigo
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def analises():
    return analysis.analisar(pipeline.processar_figura(pipeline.carregar_figura()))


def test_analisar_devolve_os_tres_paineis(analises):
    assert list(analises) == ["a", "b", "c"]
    for n, an in analises.items():
        assert an.nome == n
        assert set(an.espectro) == {"sopro", "s1", "s2"}
        f, p = an.psd_sopro
        assert f.shape == p.shape and f[0] == 0


def test_analisar_concorda_com_as_etapas_separadas(analises, reais):
    for n in "abc":
        assert analises[n].tempo == reais[n][2]
        assert analises[n].eventos == reais[n][1]


def test_conferir_artigo_tem_itens_bem_formados(analises):
    itens = analysis.conferir_artigo(analises)
    assert len(itens) == 8
    for it in itens:
        assert set(it) == {"painel", "afirmacao", "criterio", "medido", "atendido"}
        assert it["painel"] in "abc" and isinstance(it["atendido"], bool)


def test_conferir_artigo_resultado_real(analises):
    # 7 de 8 afirmações do artigo são reproduzidas. A que falha: a frequência do maior pico do VSD fica abaixo de 200 Hz.
    por_criterio = {it["criterio"]: it for it in analysis.conferir_artigo(analises)}
    falhas = [c for c, it in por_criterio.items() if not it["atendido"]]
    assert falhas == ["frequência dominante do sopro acima de 200 Hz"]
    assert por_criterio["frequência dominante do sopro em 110 a 130 Hz"]["atendido"]


def test_conferir_artigo_detecta_quando_o_criterio_falha(analises):
    from dataclasses import replace
    espectro = {**analises["b"].espectro, "sopro": {**analises["b"].espectro["sopro"], "frequencia_dominante_hz": 260.0}}
    trocado = {**analises, "b": replace(analises["b"], espectro=espectro)}
    itens = {it["criterio"]: it for it in analysis.conferir_artigo(trocado)}
    assert not itens["frequência dominante do sopro em 90 a 170 Hz"]["atendido"]
    assert not itens["frequência dominante do sopro em 110 a 130 Hz"]["atendido"]
