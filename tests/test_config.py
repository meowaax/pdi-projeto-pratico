"""Os padrões dos módulos vêm de src/config.py (fonte única dos parâmetros)."""
from src import audio, cleaning, config, curve, structure


def test_padroes_dos_modulos_vem_do_config():
    assert cleaning.LIMIAR_AZUL == config.LIMIAR_AZUL == 12
    assert structure.SIGMA_PADRAO == config.SIGMA_GAUSSIANO == 1.0
    assert structure.LIMIAR_PADRAO == config.LIMIAR_BINARIZACAO == "otsu"
    assert audio.ALVO_PADRAO == config.ALVO_NORMALIZACAO == 0.95


def test_opcoes_do_config_sao_validas():
    assert config.REGRA_COLAPSO in curve.REGRAS
    assert config.METODO_INTERPOLACAO in audio.METODOS
    assert config.NORMALIZACAO in audio.NORMALIZACOES
    assert 0 < config.ALVO_NORMALIZACAO <= 1


def test_parametros_de_audio_exigidos_pela_especificacao():
    assert (config.FS_AUDIO, config.BITS, config.CANAIS) == (5000, 16, 1)
    assert config.ARQUIVOS_WAV == {"a": "normal.wav", "b": "still.wav", "c": "vsd.wav"}
