"""Task 1B: exploração e mapeamento dos dados de entrada.

Inspeciona Figure1.png, Imagens/Shapes.png e Imagens/testpat.1k.color2.tif,
mede a estrutura dos três gráficos (painéis a, b, c) e salva:
  - outputs/exploration/measurements.json  (valores usados em src/config.py)
  - outputs/exploration/*.png              (figuras anotadas)

Uso: python explore_data.py
"""
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).parent
SAIDA = RAIZ / "outputs" / "exploration"
FIGURA = RAIZ / "Figure1.png"
IMAGENS_TESTE = [RAIZ / "Imagens" / "Shapes.png", RAIZ / "Imagens" / "testpat.1k.color2.tif"]

# Faixas de linhas de cada painel (incluem o gráfico e a faixa de rótulos de tempo,
# mas não a legenda "(a)", "(b)", "(c)"). Definidas a partir dos perfis de linhas.
PAINEIS = {"a": (0, 255), "b": (262, 525), "c": (535, 800)}
# Limites do eixo de tempo lidos visualmente nos rótulos da figura (segundos).
TEMPO_INICIAL = {"a": 5.665, "b": 1.345, "c": 1.085}
DURACAO_S = 0.5
PASSO_GRADE_S = 0.05  # 10 intervalos por gráfico
VALORES_AMPLITUDE = [75, 50, 25, 0, -25, -50, -75]


def faixas(mascara):
    """Retorna (inicio, fim) de cada trecho consecutivo True de um vetor booleano."""
    saida, ini = [], None
    for i, v in enumerate(mascara):
        if v and ini is None:
            ini = i
        elif not v and ini is not None:
            saida.append((ini, i - 1))
            ini = None
    if ini is not None:
        saida.append((ini, len(mascara) - 1))
    return saida


def mesclar(fx, folga=3):
    """Junta trechos separados por no máximo `folga` posições (linhas de grade partidas)."""
    saida = []
    for a, b in fx:
        if saida and a - saida[-1][1] <= folga:
            saida[-1] = (saida[-1][0], b)
        else:
            saida.append((a, b))
    return saida


def centros(fx):
    return [(a + b) / 2 for a, b in fx]


def descrever_imagem(caminho):
    im = cv2.imread(str(caminho), cv2.IMREAD_UNCHANGED)
    info = {
        "arquivo": caminho.name,
        "shape": list(im.shape),
        "dtype": str(im.dtype),
        "min": int(im.min()),
        "max": int(im.max()),
        "cores_unicas": int(len(np.unique(im.reshape(-1, im.shape[-1]), axis=0))),
    }
    if im.ndim == 3:
        bgr = im[:, :, :3]
        info["e_cinza_replicado"] = bool(np.all(bgr[..., 0] == bgr[..., 1]) and np.all(bgr[..., 1] == bgr[..., 2]))
    return info


def mascaras_cor(rgb):
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    azul = (b - np.maximum(r, g)) > 40
    cinza = (np.abs(r - g) < 10) & (np.abs(g - b) < 10) & (r >= 180) & (r < 250)
    escuro = (r < 110) & (g < 110) & (b < 110) & (np.abs(b - r) < 25)  # texto preto, sem a curva azul-marinho
    return azul, cinza, escuro


def medir_painel(nome, rgb, azul, cinza, escuro):
    y0, y1 = PAINEIS[nome]
    sub_cinza = cinza[y0:y1]
    res = {"faixa_linhas": [y0, y1]}

    # Faixa de rótulos de tempo (base do gráfico): trecho cinza largo (>= 8 linhas)
    soma_linhas = sub_cinza.sum(1)
    largas = [(a + y0, b + y0) for a, b in faixas(soma_linhas > 250) if b - a >= 8]
    base = largas[-1]
    res["faixa_rotulos_tempo_y"] = list(base)

    # Linhas horizontais da grade acima da base (podem incluir a moldura superior)
    fx = [(a + y0, b + y0) for a, b in faixas(soma_linhas > 500) if b + y0 < base[0]]
    grade = centros(fx)
    res["linhas_cinza_y"] = [round(v, 1) for v in grade]

    # -75% é a última linha antes da base; passo = média das menores diferenças.
    # A linha de 0% pode ficar encoberta pela curva, então ajustamos por posição inteira k.
    ultimas = grade[-6:]
    dif = np.diff(ultimas)
    passo0 = float(np.mean(dif[dif < 1.5 * dif.min()]))
    y_m75 = ultimas[-1]
    ks = np.array([round((y_m75 - y) / passo0) for y in grade])
    uteis = (ks >= 0) & (ks <= 6)
    coef_y = np.polyfit(ks[uteis], np.array(grade)[uteis], 1)  # y = coef0*k + coef1 (k = 0 em -75%)
    passo = -float(coef_y[0])
    ys = [float(np.polyval(coef_y, k)) for k in range(6, -1, -1)]  # 75 ... -75
    res["linhas_detectadas_com_valor"] = {str(VALORES_AMPLITUDE[6 - int(k)]): round(float(y), 1) for k, y in zip(ks[uteis], np.array(grade)[uteis])}
    res["passo_25pct_px"] = round(passo, 2)
    res["amplitude_y_px"] = {str(v): round(y, 1) for v, y in zip(VALORES_AMPLITUDE, ys)}
    res["y_zero_px"] = round(ys[3], 1)
    res["px_por_pct"] = round(passo / 25, 4)

    # Linhas verticais da grade (k = 1..10; k = 0 é a borda esquerda)
    soma_cols = sub_cinza.sum(0)
    colunas = centros(mesclar(faixas(soma_cols > 120)))
    verticais = [c for c in colunas if c > 40]  # ignora a faixa de rótulos de amplitude
    res["colunas_grade_x"] = [round(c, 1) for c in verticais]
    k = np.arange(1, len(verticais) + 1)
    coef = np.polyfit(k, verticais, 1)
    px_por_passo = float(coef[0])
    x_esq = float(coef[1])
    res["x_borda_esquerda_px"] = round(x_esq, 1)
    res["x_borda_direita_px"] = round(x_esq + 10 * px_por_passo, 1)
    res["px_por_segundo"] = round(px_por_passo / PASSO_GRADE_S, 1)
    res["t_inicial_s"] = TEMPO_INICIAL[nome]
    res["t_final_s"] = round(TEMPO_INICIAL[nome] + DURACAO_S, 3)
    res["n_linhas_verticais_detectadas"] = len(verticais)
    res["residuo_ajuste_x_px"] = round(float(np.abs(np.polyval(coef, k) - verticais).max()), 2)

    # Região interna do gráfico: exclui as faixas dos rótulos
    xi, xf = int(round(x_esq)) + 2, int(round(x_esq + 10 * px_por_passo)) - 1
    # topo do gráfico = linha de 75% menos um passo (moldura superior); exclui legendas vizinhas
    yi, yf = int(round(ys[0] - passo)) + 2, base[0] - 1
    res["y_moldura_superior_px"] = round(ys[0] - passo, 1)
    res["roi_xyxy"] = [xi, yi, xf, yf]

    # Curva e anotações (azul) dentro do gráfico
    reg_azul = azul[yi:yf, xi:xf]
    n_comp, rot, stats, _ = cv2.connectedComponentsWithStats(reg_azul.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n_comp):
        x, y, w, h, area = stats[i]
        comps.append({"bbox_xywh": [int(x + xi), int(y + yi), int(w), int(h)], "area": int(area)})
    comps.sort(key=lambda c: -c["area"])
    res["n_componentes_azuis"] = len(comps)
    res["componentes_azuis_maiores"] = comps[:8]
    res["componentes_azuis_todos_bbox"] = [c["bbox_xywh"] + [c["area"]] for c in comps]
    por_col = reg_azul.sum(0)
    ativas = por_col[por_col > 0]
    res["colunas_com_azul"] = int((por_col > 0).sum())
    res["colunas_totais"] = int(reg_azul.shape[1])
    res["azul_por_coluna"] = {
        "media": round(float(ativas.mean()), 2),
        "mediana": float(np.median(ativas)),
        "p95": float(np.percentile(ativas, 95)),
        "max": int(ativas.max()),
    }
    # texto escuro (legenda "Diaphragm mode; Normalized")
    y_leg = int(round(ys[-1])) + 2  # abaixo da linha de -75%
    ey, ex = np.where(escuro[y_leg:yf, xi:xf])
    if len(ex):
        res["legenda_escura_bbox_xyxy"] = [int(ex.min() + xi), int(ey.min() + y_leg), int(ex.max() + xi), int(ey.max() + y_leg)]
    res["pixels_escuros_acima_da_legenda"] = int(escuro[yi:y_leg, xi:xf].sum())
    # espessura estimada do traço: mediana de pixels azuis por coluna na maior componente
    maior = comps[0]
    mx, my, mw, mh = maior["bbox_xywh"]
    res["curva_principal_bbox_xywh"] = maior["bbox_xywh"]
    # cor dominante da curva
    pix = rgb[yi:yf, xi:xf][reg_azul]
    res["cor_curva_mediana_rgb"] = [int(v) for v in np.median(pix, axis=0)]
    res["cor_curva_mais_escura_rgb"] = [int(v) for v in pix[np.argmin(pix.sum(1))]]
    return res


def figura_anotada(rgb, medidas):
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(rgb)
    for nome, m in medidas.items():
        xi, yi, xf, yf = m["roi_xyxy"]
        ax.add_patch(plt.Rectangle((xi, yi), xf - xi, yf - yi, fill=False, ec="red", lw=1.2))
        for y in m["amplitude_y_px"].values():
            ax.plot([xi, xf], [y, y], color="orange", lw=0.5, ls="--")
        for k in range(11):
            x = m["x_borda_esquerda_px"] + k * (m["x_borda_direita_px"] - m["x_borda_esquerda_px"]) / 10
            ax.plot([x, x], [yi, yf], color="green", lw=0.5, ls="--")
        ax.text(xi + 3, yi + 12, nome, color="red", fontsize=12, weight="bold")
    ax.set_title("ROIs (vermelho), grade de amplitude (laranja) e de tempo (verde)")
    fig.savefig(SAIDA / "rois_anotadas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_mascaras(rgb, azul, cinza, escuro):
    fig, axs = plt.subplots(1, 4, figsize=(16, 7))
    for ax, img, tit in zip(axs, [rgb, azul, cinza, escuro], ["original", "azul (curva+anotações)", "cinza (grade)", "escuro (texto)"]):
        ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
        ax.set_title(tit)
        ax.axis("off")
    fig.savefig(SAIDA / "mascaras_cor.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def figura_componentes(rgb, medidas):
    fig, axs = plt.subplots(3, 1, figsize=(9, 9))
    for ax, (nome, m) in zip(axs, medidas.items()):
        xi, yi, xf, yf = m["roi_xyxy"]
        ax.imshow(rgb[yi:yf, xi:xf], extent=(xi, xf, yf, yi))
        for c in m["componentes_azuis_maiores"]:
            x, y, w, h = c["bbox_xywh"]
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="red", lw=0.8))
        ax.set_title(f"painel {nome}: componentes azuis (8 maiores)")
    fig.tight_layout()
    fig.savefig(SAIDA / "componentes_azuis.png", dpi=130)
    plt.close(fig)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    resultado = {"imagens": [descrever_imagem(p) for p in [FIGURA] + IMAGENS_TESTE]}

    bgr = cv2.imread(str(FIGURA), cv2.IMREAD_COLOR)
    rgb = bgr[:, :, ::-1]
    cores = Counter(map(tuple, rgb.reshape(-1, 3)))
    resultado["figure1_cores_unicas"] = len(cores)
    resultado["figure1_fundo_branco_pct"] = round(100 * cores[(255, 255, 255)] / (rgb.shape[0] * rgb.shape[1]), 1)

    azul, cinza, escuro = mascaras_cor(rgb)
    resultado["pixels"] = {"azul": int(azul.sum()), "cinza": int(cinza.sum()), "escuro": int(escuro.sum())}
    resultado["paineis"] = {n: medir_painel(n, rgb, azul, cinza, escuro) for n in PAINEIS}

    figura_anotada(rgb, resultado["paineis"])
    figura_mascaras(rgb, azul, cinza, escuro)
    figura_componentes(rgb, resultado["paineis"])

    with open(SAIDA / "measurements.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
