"""Demo da Task 8: extração do sinal calibrado de cada painel.

Saída em outputs/:
  extracao_<a|b|c>.png   : curva extraída (vermelho) sobre o recorte original, e o sinal em função do tempo
  sinal_<a|b|c>.npz      : arrays `t` (s) e `amplitude` (-1 a 1), entrada da Task 9
  comparacao_regras.md   : regra 'picos' x 'centro' na ida e volta com sinais sintéticos

Uso: python demo_extracao.py
"""
import logging

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config, synthetic
from src.cleaning import limpar_recorte
from src.curve import Calibracao, REGRAS, extrair_sinal
from src.roi import recortar, segmentar_paineis
from src.structure import realcar_curva

SEMENTES = range(4)
TIPOS = ("burst", "sopro", "denso")


def medir_regras():
    """Qualidade da reconstrução por tipo de sinal e regra: lista de (tipo, regra, correlação, NRMSE, razão do pico).

    Média de 4 sinais sintéticos por tipo, de gabarito conhecido, comparados a 5 kHz.
    """
    cal = synthetic.calibracao_sintetica()
    tg = np.arange(0.002, 0.498, 1 / config.FS_AUDIO)
    linhas = []
    for tipo in TIPOS:
        for regra in REGRAS:
            res = []
            for semente in SEMENTES:
                t, s = synthetic.sinal_sintetico(tipo, semente)
                est = realcar_curva(synthetic.renderizar(t, s, cal))
                sinal = extrair_sinal(est.mascara, cal, regra)
                e, v = np.interp(tg, sinal.t, sinal.amplitude), np.interp(tg, t, s)
                e, v = e - e.mean(), v - v.mean()
                res.append((np.corrcoef(e, v)[0, 1], np.sqrt(np.mean((e - v) ** 2) / np.mean(v ** 2)),
                            np.abs(e).max() / np.abs(v).max()))
            linhas.append((tipo, regra, *np.mean(res, axis=0)))
    return linhas


def comparar_regras():
    """Tabela em Markdown de `medir_regras`."""
    linhas = ["| sinal | regra | correlação | erro relativo (NRMSE) | amplitude do pico (razão) |", "|---|---|---|---|---|"]
    linhas += [f"| {tipo} | {regra} | {c:.3f} | {n:.3f} | {p:.2f} |" for tipo, regra, c, n, p in medir_regras()]
    return "\n".join(linhas) + "\n"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.SAIDA.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(config.FIGURA), cv2.IMREAD_COLOR)[:, :, ::-1].copy()

    for p in segmentar_paineis(rgb, usar_fallback=False):
        recorte = recortar(rgb, p)
        estrutura = realcar_curva(limpar_recorte(recorte).cobertura)
        cal = Calibracao.de_painel(p)
        sinal = extrair_sinal(estrutura.mascara, cal)
        np.savez(config.SAIDA / f"sinal_{p.nome}.npz", t=sinal.t, amplitude=sinal.amplitude,
                 t_ini=cal.t_ini, t_fim=cal.t_fim)

        colunas = np.arange(len(sinal.t))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
        ax1.imshow(recorte)
        ax1.plot(colunas, cal.linha(sinal.amplitude_pct), color="red", lw=0.8)
        ax1.set_title(f"painel {p.nome} ({config.PAINEIS[p.nome]['nome']}): curva extraída (vermelho) sobre o recorte")
        ax1.axis("off")
        ax2.plot(sinal.t, sinal.amplitude_pct, color="navy", lw=0.8)
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_xlim(cal.t_ini, cal.t_fim)
        ax2.set_xlabel("tempo (s)")
        ax2.set_ylabel("amplitude (%)")
        ax2.set_title("sinal calibrado")
        fig.tight_layout()
        fig.savefig(config.SAIDA / f"extracao_{p.nome}.png", dpi=100)
        plt.close(fig)
        print(f"painel {p.nome}: {len(sinal.t)} amostras de {sinal.t[0]:.4f} s a {sinal.t[-1]:.4f} s, "
              f"amplitude de {sinal.amplitude_pct.min():.1f}% a {sinal.amplitude_pct.max():.1f}%")

    tabela = comparar_regras()
    (config.SAIDA / "comparacao_regras.md").write_text(tabela, encoding="utf-8")
    print(tabela)


if __name__ == "__main__":
    main()
