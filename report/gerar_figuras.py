"""Gera as figuras (PNG) e as tabelas (LaTeX) do relatório a partir do pipeline.

As tabelas e as figuras do relatório saem dos mesmos cálculos do código, então os números do texto
não ficam desatualizados quando o código muda.

Uso (a partir da raiz do repositório):
    python report/gerar_figuras.py

Saída:
    report/figuras/*.png   figuras usadas no relatório
    report/gerado/*.tex    tabelas (apenas o ambiente `tabular`, a legenda fica no relatório)
"""
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import demo_analise
import demo_extracao
from demo_req1 import para_cinza
from src import analysis, config, pipeline
from src.correlation import correlate2d_manual
from src.filters import bordas_sobel, gradiente_sobel, suavizar_gaussiano
from src.kernels import kernel_media
from src.roi import desenhar_rois
from src.structure import varrer_parametros

FIGURAS = RAIZ / "report" / "figuras"
GERADO = RAIZ / "report" / "gerado"
SIGMAS = (0.6, 0.8, 1.0, 1.5, 2.0)
NOME = {n: config.PAINEIS[n]["nome"] for n in "abc"}


# ----------------------------------------------------------------------------
# Tabelas em LaTeX
# ----------------------------------------------------------------------------
def escapar(texto):
    """Escapa os caracteres especiais do LaTeX em um texto simples."""
    for velho, novo in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        texto = texto.replace(velho, novo)
    return texto


def tabular(alinhamento, cabecalho, linhas, escapar_celulas=True):
    """Ambiente `tabular` com booktabs. `linhas` é uma lista de listas de textos."""
    f = escapar if escapar_celulas else (lambda x: x)
    corpo = "\n".join(" & ".join(f(str(c)) for c in l) + r" \\" for l in linhas)
    cab = " & ".join(rf"\textbf{{{f(c)}}}" for c in cabecalho) + r" \\"
    return f"\\begin{{tabular}}{{{alinhamento}}}\n\\toprule\n{cab}\n\\midrule\n{corpo}\n\\bottomrule\n\\end{{tabular}}\n"


def numero(x, casas):
    """Número com vírgula decimal, como no texto em português."""
    return f"{x:.{casas}f}".replace(".", "{,}")


def virgula(texto):
    """Prepara um número já formatado para o LaTeX: vírgula decimal e sinal de menos tipográfico.

    Exemplo: '-9.8 dB' vira '$-$9{,}8 dB'.
    """
    return re.sub(r"(^|\s)-(?=\d)", r"\1$-$", texto.replace(".", "{,}"))


def tabela_calibracao(execucao):
    linhas = []
    for r in execucao.resultados:
        c = r.sinal.calibracao
        x0, y0, x1, y1 = r.painel.roi
        linhas.append([f"{r.nome} ({NOME[r.nome]})", f"({x0}, {y0}, {x1}, {y1})",
                       f"{numero(c.t_ini, 3)} a {numero(c.t_fim, 3)}", numero(c.px_por_s, 0),
                       numero(1 / abs(c.pct_por_px), 2), numero(c.y_zero, 1)])
    return tabular("lllrrr", ["Painel", "ROI $(x_0, y_0, x_1, y_1)$", "Tempo (s)", "Colunas/s", "Linhas por 1\\%", "$y$ de 0\\%"],
                   linhas, escapar_celulas=False)


def tabela_sigma(execucao):
    """Erro dos extremos e viés dos picos, por painel, para cada σ, com limiar de Otsu."""
    por_painel = {}
    for r in execucao.resultados:
        tab = varrer_parametros(r.limpo.cobertura, r.limpo.mascara, SIGMAS, ["otsu"])
        por_painel[r.nome] = {t["sigma"]: t for t in tab}
    linhas = []
    for s in SIGMAS:
        linha = [numero(s, 1)]
        for n in "abc":
            t = por_painel[n][s]
            linha += [numero(t["erro_extremos_px"], 2), virgula(f"{t['vies_pico_px']:+.1f}")]
        linha.append("$\\star$" if s == 1.0 else "")
        linhas.append(linha)
    cab = ["$\\sigma$", "erro (a)", "viés (a)", "erro (b)", "viés (b)", "erro (c)", "viés (c)", ""]
    return tabular("lrrrrrrc", cab, linhas, escapar_celulas=False)


def tabela_regras():
    linhas = [[tipo, regra, numero(c, 3), numero(n, 3), numero(p, 2)] for tipo, regra, c, n, p in demo_extracao.medir_regras()]
    return tabular("llrrr", ["Sinal sintético", "Regra", "Correlação", "NRMSE", "Razão do pico"], linhas)


def tabela_audio(execucao):
    linhas = []
    for r in execucao.resultados:
        w = execucao.wavs[r.nome]
        linhas.append([Path(w["caminho"]).name.replace("_", "\\_"), str(config.N_AMOSTRAS), f"{config.FS_AUDIO}",
                       f"{config.BITS}", "1", numero(w["pico_pct"], 1), numero(w["rms_pct"], 1)])
    return tabular("lrrrrrr", ["Arquivo", "Amostras", "Taxa (Hz)", "Bits", "Canais", "Pico (\\%)", "RMS (\\%)"],
                   linhas, escapar_celulas=False)


def tabela_tempo(analises):
    linhas = [[escapar(l[0]), *[virgula(c) for c in l[1:]]] for l in demo_analise.linhas_tempo(analises)]
    return tabular("lrrr", ["Medida", *[NOME[n] for n in "abc"]], linhas, escapar_celulas=False)


def tabela_espectro(analises):
    linhas = [[escapar(l[0]), *[virgula(c) for c in l[1:]]] for l in demo_analise.linhas_espectro(analises)]
    return tabular("lrrr", ["Medida (sopro sistólico)", *[NOME[n] for n in "abc"]], linhas, escapar_celulas=False)


def tabela_artigo(analises):
    itens = analysis.conferir_artigo(analises)
    linhas = [[it["painel"], it["afirmacao"], virgula(it["medido"].replace("x", "$\\times$")), "sim" if it["atendido"] else "\\textbf{não}"]
              for it in itens]
    for l in linhas:
        l[1] = escapar(l[1])
    return ("\\begin{tabularx}{\\textwidth}{cXrc}\n\\toprule\n\\textbf{Painel} & \\textbf{Afirmação do artigo} & \\textbf{Medido} & \\textbf{Atendido} \\\\\n\\midrule\n"
            + "\n".join(" & ".join(c) + r" \\" for c in linhas) + "\n\\bottomrule\n\\end{tabularx}\n")


def gravar_tabelas(execucao, analises, destino):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    tabelas = {
        "tab_calibracao.tex": tabela_calibracao(execucao),
        "tab_sigma.tex": tabela_sigma(execucao),
        "tab_regras.tex": tabela_regras(),
        "tab_audio.tex": tabela_audio(execucao),
        "tab_tempo.tex": tabela_tempo(analises),
        "tab_espectro.tex": tabela_espectro(analises),
        "tab_artigo.tex": tabela_artigo(analises),
    }
    for nome, texto in tabelas.items():
        (destino / nome).write_text(texto, encoding="utf-8")
    return sorted(tabelas)


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def _salvar(fig, caminho, dpi=130):
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(caminho).name


def figura_filtros(destino):
    """Média, Gaussiano e Sobel com a correlação manual, em Shapes.png e testpat.1k.color2.tif."""
    imagens = [RAIZ / "Imagens" / "Shapes.png", RAIZ / "Imagens" / "testpat.1k.color2.tif"]
    fig, axs = plt.subplots(2, 5, figsize=(15, 6.2))
    for linha, caminho in zip(axs, imagens):
        cinza = para_cinza(cv2.imread(str(caminho), cv2.IMREAD_COLOR))
        gx, gy, mag, _ = gradiente_sobel(cinza)
        for ax, (nome, img) in zip(linha, [("original", cinza), ("média 5×5", correlate2d_manual(cinza, kernel_media(5))),
                                             ("Gaussiano σ = 1,5", suavizar_gaussiano(cinza, 1.5)), ("Sobel em x", gx),
                                             ("magnitude do gradiente", mag)]):
            ax.imshow(img, cmap="gray")
            ax.set_title(nome, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    return _salvar(fig, Path(destino) / "filtros.png")


def figura_bordas_shapes(destino):
    cinza = para_cinza(cv2.imread(str(RAIZ / "Imagens" / "Shapes.png"), cv2.IMREAD_COLOR))
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.4))
    for ax, (nome, img) in zip(axs, [("original", cinza), ("bordas de Sobel", bordas_sobel(cinza)),
                                     ("Gaussiano σ = 1,5 antes do Sobel", bordas_sobel(cinza, sigma=1.5))]):
        ax.imshow(img, cmap="gray")
        ax.set_title(nome, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return _salvar(fig, Path(destino) / "bordas_shapes.png")


def figura_rois(rgb, resultados, destino):
    caminho = Path(destino) / "rois.png"
    if not cv2.imwrite(str(caminho), desenhar_rois(rgb, [r.painel for r in resultados])[:, :, ::-1]):
        raise OSError(f"não foi possível gravar {caminho}")
    return caminho.name


def figura_limpeza(resultados, destino):
    """Original, sem grade e sinal limpo, para os três painéis."""
    fig, axs = plt.subplots(len(resultados), 3, figsize=(15, 3.3 * len(resultados)))
    for linha, r in zip(axs, resultados):
        for ax, img, titulo in zip(linha, (r.recorte, r.sem_grade.imagem, r.limpo.imagem),
                                   ("original", "sem grade, eixos e legenda", "sinal limpo (sem anotações)")):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=255)
            ax.set_title(f"({r.nome}) {NOME[r.nome]}: {titulo}", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    return _salvar(fig, Path(destino) / "limpeza.png", dpi=110)


def figura_estrutura(r, destino):
    """Etapas de suavização, bordas, binarização e afinamento de um painel."""
    e = r.estrutura
    etapas = [("sinal limpo", r.limpo.imagem), (f"suavizado (Gaussiano σ = {numero(e.sigma, 1)})", 255 - 255 * e.suavizada),
              ("bordas (magnitude de Sobel)", e.bordas), ("binarizado (Otsu)", ~e.mascara), ("esqueleto de 1 px", ~e.esqueleto)]
    fig, axs = plt.subplots(len(etapas), 1, figsize=(11, 10))
    for ax, (nome, img) in zip(axs, etapas):
        ax.imshow(img, cmap="gray_r" if "Sobel" in nome else "gray")
        ax.set_title(f"({r.nome}) {NOME[r.nome]}: {nome}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return _salvar(fig, Path(destino) / f"estrutura_{r.nome}.png", dpi=110)


def gerar(destino_figuras=FIGURAS, destino_tabelas=GERADO, trabalho=None):
    """Roda o pipeline e grava figuras e tabelas. `trabalho` recebe as saídas do pipeline (.wav, etapas)."""
    destino_figuras, destino_tabelas = Path(destino_figuras), Path(destino_tabelas)
    destino_figuras.mkdir(parents=True, exist_ok=True)
    trabalho = Path(trabalho) if trabalho else config.SAIDA
    rgb = pipeline.carregar_figura(config.FIGURA)
    execucao = pipeline.executar(config.FIGURA, trabalho)
    resultados = execucao.resultados
    analises = analysis.analisar(resultados)

    feitas = [figura_filtros(destino_figuras), figura_bordas_shapes(destino_figuras), figura_rois(rgb, resultados, destino_figuras),
              figura_limpeza(resultados, destino_figuras), figura_estrutura(resultados[-1], destino_figuras)]
    for r in resultados:                                   # curva extraída sobre o recorte e sinal calibrado
        caminho = destino_figuras / f"extracao_{r.nome}.png"
        pipeline.figura_extracao(r, caminho)
        feitas.append(caminho.name)

    anterior = demo_analise.SAIDA                          # as figuras da análise gravam em demo_analise.SAIDA
    demo_analise.SAIDA = destino_figuras
    try:
        demo_analise.figura_formas_de_onda(analises)
        demo_analise.figura_espectros(analises)
        demo_analise.figura_espectrogramas(analises)
    finally:
        demo_analise.SAIDA = anterior
    feitas += ["formas_de_onda.png", "espectros.png", "espectrogramas.png"]

    tabelas = gravar_tabelas(execucao, analises, destino_tabelas)
    return {"figuras": sorted(feitas), "tabelas": tabelas}


def main():
    resultado = gerar()
    print(f"{len(resultado['figuras'])} figuras em {FIGURAS}")
    print(f"{len(resultado['tabelas'])} tabelas em {GERADO}")


if __name__ == "__main__":
    main()
