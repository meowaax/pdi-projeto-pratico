"""Demo da Task 11: análise comparativa dos três sinais.

Saída em outputs/analise/:
  formas_de_onda.png   : os três sinais na mesma escala, com S1, S2 e as janelas marcados
  espectros.png        : espectro do sopro sistólico de cada sinal, com as faixas citadas no artigo
  espectrogramas.png   : espectrograma de cada sinal, na mesma escala de cor
  tabela_tempo.md      : níveis por janela
  tabela_espectro.md   : métricas espectrais do sopro
  conferencia_artigo.md: afirmações do artigo x valores medidos

Uso: python demo_analise.py
"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import analysis, config, pipeline

SAIDA = config.SAIDA / "analise"
CORES = {"a": "tab:green", "b": "tab:blue", "c": "tab:red"}
NOME = {n: config.PAINEIS[n]["nome"] for n in "abc"}


def figura_formas_de_onda(analises):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (n, an) in zip(axs, analises.items()):
        s, ev = an.sinal, an.eventos
        t = s.t - s.t_ini
        ax.plot(t, s.y, color=CORES[n], lw=0.7)
        for nome, cor in (("s1", "gold"), ("sopro", "lightgray"), ("s2", "gold")):
            m = an.janelas[nome]
            ax.axvspan(t[m][0], t[m][-1], color=cor, alpha=0.35, lw=0)
        for rot, tev in (("S1", ev.t_s1), ("S2", ev.t_s2)):
            ax.axvline(tev - s.t_ini, color="k", lw=0.8, ls="--")
            ax.text(tev - s.t_ini + 0.003, 85, rot, fontsize=9)
        ax.set_ylim(-100, 100)
        ax.set_ylabel("amplitude (%)")
        ax.set_title(f"{NOME[n]}: pico {an.tempo['pico_pct']:.0f}%, RMS {an.tempo['rms_pct']:.1f}%, "
                     f"sopro {an.tempo['rms_sopro_pct']:.1f}% (diástole {an.tempo['rms_diastole_pct']:.1f}%)", fontsize=10)
    axs[-1].set_xlabel("tempo desde o início do trecho (s). Amarelo: S1 e S2; cinza: sopro sistólico")
    fig.tight_layout()
    fig.savefig(SAIDA / "formas_de_onda.png", dpi=100)
    plt.close(fig)


def figura_espectros(analises):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, escala in zip(axs, ("linear", "log")):
        for n, an in analises.items():
            f, p = an.psd_sopro
            m = (f >= 20) & (f <= 700)
            ax.plot(f[m], p[m], color=CORES[n], label=f"{NOME[n]} (dominante {an.espectro['sopro']['frequencia_dominante_hz']:.0f} Hz)")
        ax.axvspan(90, 170, color="tab:blue", alpha=0.10, lw=0)
        ax.axvspan(80, 500, color="tab:red", alpha=0.05, lw=0)
        ax.set_yscale(escala)
        ax.set_xlabel("frequência (Hz)")
        ax.set_ylabel("densidade espectral (%²/Hz)")
        ax.legend(fontsize=8)
    axs[0].set_title("espectro do sopro sistólico (azul claro: 90 a 170 Hz, o tom de Still; rosa: 80 a 500 Hz, VSD)", fontsize=9)
    axs[1].set_title("mesma curva em escala logarítmica", fontsize=9)
    fig.tight_layout()
    fig.savefig(SAIDA / "espectros.png", dpi=100)
    plt.close(fig)


def figura_espectrogramas(analises):
    dados = {n: analysis.espectrograma(an.sinal.y) for n, an in analises.items()}
    teto = max(P.max() for _, _, P in dados.values())
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (n, (t, f, P)) in zip(axs, dados.items()):
        db = 10 * np.log10(np.maximum(P, teto * 1e-6) / teto)
        im = ax.pcolormesh(t, f, db, vmin=-50, vmax=0, shading="auto", cmap="magma")
        for tev in (analises[n].eventos.t_s1, analises[n].eventos.t_s2):
            ax.axvline(tev - analises[n].sinal.t_ini, color="white", lw=0.7, ls="--")
        ax.set_ylabel("frequência (Hz)")
        ax.set_title(NOME[n], fontsize=10)
    axs[-1].set_xlabel("tempo desde o início do trecho (s)")
    fig.colorbar(im, ax=axs, label="potência relativa ao máximo dos três (dB)")
    fig.savefig(SAIDA / "espectrogramas.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def tabela(cabecalho, linhas):
    return "\n".join(["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)] +
                     ["| " + " | ".join(str(c) for c in l) + " |" for l in linhas]) + "\n"


def tabelas(analises):
    t = lambda n, k: analises[n].tempo[k]
    db = lambda v: f"{v:+.1f}"
    linhas = [
        ("Pico (%)", *[f"{t(n, 'pico_pct'):.1f}" for n in "abc"]),
        ("RMS total (%)", *[f"{t(n, 'rms_pct'):.1f}" for n in "abc"]),
        ("Fator de crista", *[f"{t(n, 'fator_de_crista'):.1f}" for n in "abc"]),
        ("Intervalo S1 a S2 (ms)", *[f"{t(n, 'intervalo_s1_s2_ms'):.0f}" for n in "abc"]),
        ("RMS de S1 (%)", *[f"{t(n, 'rms_s1_pct'):.1f}" for n in "abc"]),
        ("RMS de S2 (%)", *[f"{t(n, 'rms_s2_pct'):.1f}" for n in "abc"]),
        ("RMS do sopro sistólico (%)", *[f"{t(n, 'rms_sopro_pct'):.2f}" for n in "abc"]),
        ("RMS da diástole (%)", *[f"{t(n, 'rms_diastole_pct'):.2f}" for n in "abc"]),
        ("Pico a pico de S1 (%)", *[f"{t(n, 'pico_a_pico_s1_pct'):.0f}" for n in "abc"]),
        ("Pico a pico de S2 (%)", *[f"{t(n, 'pico_a_pico_s2_pct'):.0f}" for n in "abc"]),
        ("S2 sobre S1 (dB)", *[db(t(n, "s2_sobre_s1_db")) for n in "abc"]),
        ("Sopro sobre diástole (dB)", *[db(t(n, "sopro_sobre_diastole_db")) for n in "abc"]),
        ("Sopro sobre S1 (dB)", *[db(t(n, "sopro_sobre_s1_db")) for n in "abc"]),
        ("Taxa de cruzamentos por zero no sopro (Hz equivalentes)", *[f"{t(n, 'cruzamentos_por_zero_sopro_hz'):.0f}" for n in "abc"]),
        ("Periodicidade do sopro (autocorrelação, 0 a 1)", *[f"{t(n, 'periodicidade_sopro'):.2f}" for n in "abc"]),
        ("Frequência da repetição do sopro (Hz)", *[f"{t(n, 'frequencia_periodica_sopro_hz'):.0f}" for n in "abc"]),
    ]
    tab_tempo = tabela(["Medida", *[NOME[n] for n in "abc"]], linhas)

    e = lambda n, k: analises[n].espectro["sopro"][k]
    faixas = list(analises["a"].espectro["sopro"]["energia_faixas"])
    linhas = [
        ("Frequência dominante (Hz)", *[f"{e(n, 'frequencia_dominante_hz'):.0f}" for n in "abc"]),
        ("Centroide (Hz)", *[f"{e(n, 'centroide_hz'):.0f}" for n in "abc"]),
        ("Largura a meia potência (Hz)", *[f"{e(n, 'largura_meia_potencia_hz'):.0f}" for n in "abc"]),
        ("Planura espectral (80 a 500 Hz)", *[f"{e(n, 'planura'):.3f}" for n in "abc"]),
        *[(f"Energia de {k} Hz (%)", *[f"{100 * e(n, 'energia_faixas')[k]:.0f}" for n in "abc"]) for k in faixas],
        ("Frequência dominante de S1 (Hz)", *[f"{analises[n].espectro['s1']['frequencia_dominante_hz']:.0f}" for n in "abc"]),
        ("Frequência dominante de S2 (Hz)", *[f"{analises[n].espectro['s2']['frequencia_dominante_hz']:.0f}" for n in "abc"]),
    ]
    tab_esp = tabela(["Medida (janela do sopro sistólico)", *[NOME[n] for n in "abc"]], linhas)

    tab_art = tabela(["Painel", "Afirmação do artigo", "Critério", "Medido", "Atendido"],
                     [(it["painel"], it["afirmacao"], it["criterio"], it["medido"], "sim" if it["atendido"] else "**não**")
                      for it in analysis.conferir_artigo(analises)])
    return tab_tempo, tab_esp, tab_art


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    SAIDA.mkdir(parents=True, exist_ok=True)
    analises = analysis.analisar(pipeline.processar_figura(pipeline.carregar_figura(config.FIGURA)))
    figura_formas_de_onda(analises)
    figura_espectros(analises)
    figura_espectrogramas(analises)
    tab_tempo, tab_esp, tab_art = tabelas(analises)
    for nome, txt in (("tabela_tempo.md", tab_tempo), ("tabela_espectro.md", tab_esp), ("conferencia_artigo.md", tab_art)):
        (SAIDA / nome).write_text(txt, encoding="utf-8")
        print(f"\n## {nome}\n{txt}")
    print(f"Arquivos salvos em {SAIDA}")


if __name__ == "__main__":
    main()
