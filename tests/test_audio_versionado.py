"""Os três .wav versionados em audio/ são os que o código atual gera (entregável do trabalho)."""
from pathlib import Path

import numpy as np
import pytest

from src import audio, config, pipeline

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "audio"
NOMES = ["normal.wav", "still.wav", "vsd.wav"]
TOLERANCIA = 2      # níveis de quantização de 16 bits; cobre diferenças de ponto flutuante entre plataformas


def test_a_pasta_audio_tem_exatamente_os_tres_arquivos_pedidos():
    assert sorted(p.name for p in PASTA.iterdir()) == NOMES


@pytest.mark.parametrize("nome", NOMES)
def test_arquivo_versionado_tem_o_formato_exigido(nome):
    w = audio.ler_wav(PASTA / nome)
    assert (w.taxa, w.bits, w.canais, w.n_amostras) == (config.FS_AUDIO, config.BITS, config.CANAIS, config.N_AMOSTRAS)
    assert np.abs(w.amostras.astype(int)).max() < 32767          # sem clipping


def test_arquivos_versionados_sao_os_gerados_pelo_codigo(tmp_path):
    """Se o código mudar, regenere com `python main.py` e copie outputs/audio/*.wav para audio/."""
    pipeline.executar(saida=tmp_path)
    for nome in NOMES:
        novo = audio.ler_wav(tmp_path / "audio" / nome).amostras.astype(int)
        versionado = audio.ler_wav(PASTA / nome).amostras.astype(int)
        assert novo.shape == versionado.shape, nome
        assert np.abs(novo - versionado).max() <= TOLERANCIA, nome
