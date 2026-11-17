"""Pipeline completo: Figure1.png -> normal.wav, still.wav e vsd.wav (Task 10).

Encadeia as etapas das Tasks 4 a 9:

    segmentar ROIs -> remover grade -> remover anotações -> suavizar e binarizar
    -> extrair o sinal por coluna -> reamostrar a 5 kHz -> gravar .wav

Os parâmetros ajustáveis ficam em `src/config.py` e entram aqui pela classe `Parametros`.
`processar_figura` só calcula; `executar` também grava as imagens intermediárias e os .wav.
"""
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from src import audio, config
from src.cleaning import SemGrade, SoCurva, remover_anotacoes, remover_grade
from src.curve import Calibracao, SinalExtraido, extrair_sinal
from src.roi import Painel, desenhar_rois, recortar, segmentar_paineis
from src.structure import Estrutura, realcar_curva

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Parametros:
    """Parâmetros do pipeline. Os padrões vêm de `src/config.py`."""
    limiar_azul: int = config.LIMIAR_AZUL
    sigma: float = config.SIGMA_GAUSSIANO
    limiar_binarizacao: object = config.LIMIAR_BINARIZACAO     # 'otsu' ou número em (0, 1)
    regra: str = config.REGRA_COLAPSO
    interpolacao: str = config.METODO_INTERPOLACAO
    normalizacao: str = config.NORMALIZACAO
    alvo: float = config.ALVO_NORMALIZACAO
    usar_fallback: bool = True                                  # usa config.PAINEIS se a detecção das ROIs falhar


@dataclass(frozen=True)
class ResultadoPainel:
    """Produtos intermediários e finais de um painel."""
    painel: Painel
    recorte: np.ndarray       # RGB, interior do gráfico
    sem_grade: SemGrade
    limpo: SoCurva
    estrutura: Estrutura
    sinal: SinalExtraido

    @property
    def nome(self):
        return self.painel.nome


def carregar_figura(caminho=config.FIGURA):
    """Lê a figura como RGB uint8. Levanta FileNotFoundError se o arquivo não abrir."""
    bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"não foi possível abrir a figura: {caminho}")
    return bgr[:, :, ::-1].copy()


def processar_painel(rgb, painel, params=Parametros()):
    """Roda as etapas de imagem e de extração em um painel (sem gravar nada)."""
    recorte = recortar(rgb, painel)
    sem_grade = remover_grade(recorte, params.limiar_azul)
    limpo = remover_anotacoes(sem_grade)
    estrutura = realcar_curva(limpo.cobertura, params.sigma, params.limiar_binarizacao)
    sinal = extrair_sinal(estrutura.mascara, Calibracao.de_painel(painel), params.regra)
    log.info("painel %s (%s): ROI %s, %d anotações removidas, curva de %d px, sinal de %d amostras "
             "(%.1f%% a %.1f%%)", painel.nome, config.PAINEIS[painel.nome]["nome"], painel.roi, len(limpo.anotacoes),
             int(estrutura.mascara.sum()), len(sinal.t), sinal.amplitude_pct.min(), sinal.amplitude_pct.max())
    return ResultadoPainel(painel=painel, recorte=recorte, sem_grade=sem_grade, limpo=limpo,
                           estrutura=estrutura, sinal=sinal)


def processar_figura(rgb, params=Parametros()):
    """Segmenta os três painéis e processa cada um. Devolve uma lista de `ResultadoPainel`."""
    paineis = segmentar_paineis(rgb, usar_fallback=params.usar_fallback)
    log.info("%d painéis segmentados", len(paineis))
    return [processar_painel(rgb, p, params) for p in paineis]


# ----------------------------------------------------------------------------
# Gravação das saídas
# ----------------------------------------------------------------------------
def _gravar_png(caminho, imagem):
    """Grava uma imagem uint8 em cinza (2D) ou RGB (3D)."""
    if imagem.ndim == 3:
        imagem = imagem[:, :, ::-1]
    if not cv2.imwrite(str(caminho), np.ascontiguousarray(imagem)):
        raise OSError(f"não foi possível gravar {caminho}")
    return caminho


def _como_tinta(valores):
    """Converte valores em [0, 1] (1 = curva) para uint8 com a curva escura sobre fundo branco."""
    return np.rint(255.0 * (1.0 - np.clip(valores, 0.0, 1.0))).astype(np.uint8)


def _figura_extracao(r, caminho):
    """Curva extraída sobre o recorte original e o sinal calibrado em função do tempo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = r.sinal.calibracao
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
    ax1.imshow(r.recorte)
    ax1.plot(np.arange(len(r.sinal.t)), cal.linha(r.sinal.amplitude_pct), color="red", lw=0.8)
    ax1.set_title(f"painel {r.nome} ({config.PAINEIS[r.nome]['nome']}): curva extraída (vermelho) sobre o recorte")
    ax1.axis("off")
    ax2.plot(r.sinal.t, r.sinal.amplitude_pct, color="navy", lw=0.8)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlim(cal.t_ini, cal.t_fim)
    ax2.set_xlabel("tempo (s)")
    ax2.set_ylabel("amplitude (%)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=100)
    plt.close(fig)
    return caminho


def salvar_etapas(rgb, resultados, pasta):
    """Grava as imagens de cada etapa em `pasta` e devolve {nome_do_arquivo: caminho}.

    rois_detectadas.png e, para cada painel X: X_01_original, X_02_sem_grade, X_03_sinal_limpo,
    X_04_suavizado, X_05_bordas, X_06_binario, X_07_esqueleto e X_08_extracao.
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    gravados = {}

    def gravar(nome, imagem):
        gravados[nome] = _gravar_png(pasta / nome, imagem)

    gravar("rois_detectadas.png", desenhar_rois(rgb, [r.painel for r in resultados]))
    for r in resultados:
        n, e = r.nome, r.estrutura
        gravar(f"{n}_01_original.png", r.recorte)
        gravar(f"{n}_02_sem_grade.png", r.sem_grade.imagem)
        gravar(f"{n}_03_sinal_limpo.png", r.limpo.imagem)
        gravar(f"{n}_04_suavizado.png", _como_tinta(e.suavizada))
        gravar(f"{n}_05_bordas.png", _como_tinta(e.bordas / max(e.bordas.max(), 1e-12)))
        gravar(f"{n}_06_binario.png", _como_tinta(e.mascara))
        gravar(f"{n}_07_esqueleto.png", _como_tinta(e.esqueleto))
        nome = f"{n}_08_extracao.png"
        gravados[nome] = _figura_extracao(r, pasta / nome)
    return gravados


def resumir(resultados, wavs, params):
    """Dicionário serializável com os números principais de cada painel, para o relatório."""
    paineis = {}
    for r in resultados:
        cal, s = r.sinal.calibracao, r.sinal
        paineis[r.nome] = {
            "nome": config.PAINEIS[r.nome]["nome"],
            "roi": [int(v) for v in r.painel.roi],
            "t_inicial_s": cal.t_ini, "t_final_s": cal.t_fim,
            "px_por_segundo": round(cal.px_por_s, 2), "pct_por_pixel": round(abs(cal.pct_por_px), 4),
            "anotacoes_removidas": len(r.limpo.anotacoes),
            "pixels_da_curva": int(r.estrutura.mascara.sum()),
            "amostras_extraidas": int(len(s.t)),
            "amplitude_min_pct": round(float(s.amplitude_pct.min()), 2),
            "amplitude_max_pct": round(float(s.amplitude_pct.max()), 2),
            "wav": {"arquivo": Path(wavs[r.nome]["caminho"]).name, "pico_pct": round(wavs[r.nome]["pico_pct"], 2),
                    "rms_pct": round(wavs[r.nome]["rms_pct"], 2)},
        }
    return {"parametros": asdict(params),
            "audio": {"fs": config.FS_AUDIO, "bits": config.BITS, "canais": config.CANAIS,
                      "amostras": config.N_AMOSTRAS, "duracao_s": config.DURACAO_S},
            "paineis": paineis}


@dataclass(frozen=True)
class Execucao:
    """O que `executar` produziu."""
    resultados: list
    wavs: dict          # {'a': {'caminho', 'pico_pct', 'rms_pct'}, ...}
    imagens: dict       # {nome_do_arquivo: caminho}
    resumo: dict
    pasta: Path


def executar(figura=config.FIGURA, saida=config.SAIDA, params=Parametros()):
    """Roda o pipeline completo e grava tudo em `saida`.

    saida/etapas/*.png   imagens intermediárias de cada painel
    saida/audio/*.wav    normal.wav, still.wav e vsd.wav
    saida/resumo.json    parâmetros e números principais
    """
    t0 = time.perf_counter()
    saida = Path(saida)
    log.info("lendo %s", figura)
    rgb = carregar_figura(figura)
    resultados = processar_figura(rgb, params)

    imagens = salvar_etapas(rgb, resultados, saida / "etapas")
    log.info("%d imagens intermediárias em %s", len(imagens), saida / "etapas")

    wavs = audio.exportar_wavs({r.nome: r.sinal for r in resultados}, saida / "audio",
                               params.interpolacao, params.normalizacao, params.alvo)
    for nome, w in wavs.items():
        log.info("%s gravado (%d amostras a %d Hz, %d bits, mono; pico %.1f%%, RMS %.1f%%)", Path(w["caminho"]).name,
                 config.N_AMOSTRAS, config.FS_AUDIO, config.BITS, w["pico_pct"], w["rms_pct"])

    resumo = resumir(resultados, wavs, params)
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("pronto em %.1f s: arquivos em %s", time.perf_counter() - t0, saida)
    return Execucao(resultados=resultados, wavs=wavs, imagens=imagens, resumo=resumo, pasta=saida)
