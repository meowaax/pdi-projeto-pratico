"""Demo da Task 9: reamostra os sinais extraídos e grava normal.wav, still.wav e vsd.wav.

Saída em outputs/audio/:
  normal.wav, still.wav, vsd.wav : PCM mono, 5000 Hz, 16 bits, 0,5 s
  formas_de_onda.png             : forma de onda de cada arquivo, lida de volta do .wav
  comparacao_metodos.md          : interpolação linear x cúbica (ida e volta com sinais sintéticos)

Uso: python demo_audio.py
"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import audio, config, pipeline, synthetic
from src.curve import extrair_sinal
from src.structure import realcar_curva

SAIDA = config.SAIDA / "audio"


def extrair_sinais():
    """Roda a cadeia das Tasks 4 a 8 (src/pipeline.py) e devolve {'a': sinal, 'b': sinal, 'c': sinal}."""
    resultados = pipeline.processar_figura(pipeline.carregar_figura(config.FIGURA))
    return {r.nome: r.sinal for r in resultados}


def comparar_metodos():
    """Erro da reconstrução a 5 kHz com cada método de interpolação (média de 4 sinais por tipo)."""
    cal = synthetic.calibracao_sintetica()
    linhas = ["| sinal | método | correlação | erro relativo (NRMSE) | amplitude do pico (razão) |", "|---|---|---|---|---|"]
    grade = audio.grade_uniforme(0.0)
    meio = (grade > 0.003) & (grade < 0.497)
    for tipo in ("burst", "sopro", "denso"):
        for metodo in audio.METODOS:
            res = []
            for semente in range(4):
                t, s = synthetic.sinal_sintetico(tipo, semente)
                sinal = extrair_sinal(realcar_curva(synthetic.renderizar(t, s, cal)).mascara, cal)
                _, y = audio.reamostrar(sinal.t, sinal.amplitude, 0.0, metodo=metodo)
                v = np.interp(grade, t, s)
                e, r = y[meio] - y[meio].mean(), v[meio] - v[meio].mean()
                res.append((np.corrcoef(e, r)[0, 1], np.sqrt(np.mean((e - r) ** 2) / np.mean(r ** 2)),
                            np.abs(y).max() / np.abs(v).max()))
            c, n, p = np.mean(res, axis=0)
            linhas.append(f"| {tipo} | {metodo} | {c:.3f} | {n:.3f} | {p:.2f} |")
    return "\n".join(linhas) + "\n"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    SAIDA.mkdir(parents=True, exist_ok=True)
    sinais = extrair_sinais()
    info = audio.exportar_wavs(sinais, SAIDA)

    fig, axs = plt.subplots(len(info), 1, figsize=(12, 7), sharex=True)
    for ax, (nome, i) in zip(axs, info.items()):
        w = audio.ler_wav(i["caminho"])
        t = np.arange(w.n_amostras) / w.taxa
        ax.plot(t, w.amostras / 32767.0, color="navy", lw=0.7)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("amplitude")
        ax.set_title(f"{i['caminho'].name} ({config.PAINEIS[nome]['nome']}): {w.n_amostras} amostras, {w.taxa} Hz, "
                     f"{w.bits} bits, {w.canais} canal; nível antes da normalização: pico {i['pico_pct']:.0f}%, "
                     f"RMS {i['rms_pct']:.1f}%", fontsize=9)
        print(f"{i['caminho'].name}: {w.n_amostras} amostras, {w.taxa} Hz, {w.bits} bits, {w.canais} canal, "
              f"pico {i['pico_pct']:.1f}%, RMS {i['rms_pct']:.2f}% da escala")
    axs[-1].set_xlabel("tempo desde o início do trecho (s)")
    fig.tight_layout()
    fig.savefig(SAIDA / "formas_de_onda.png", dpi=100)
    plt.close(fig)

    tabela = comparar_metodos()
    (SAIDA / "comparacao_metodos.md").write_text(tabela, encoding="utf-8")
    print(tabela)
    print(f"Arquivos salvos em {SAIDA}")


if __name__ == "__main__":
    main()
