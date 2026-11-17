"""Pipeline completo: Figure1.png -> normal.wav, still.wav e vsd.wav (Task 10).

Encadeia as etapas das Tasks 4 a 9:

    segmentar ROIs -> remover grade -> remover anotações -> suavizar e binarizar
    -> extrair o sinal por coluna -> reamostrar a 5 kHz -> gravar .wav

Os parâmetros ajustáveis ficam em `src/config.py` e entram aqui pela classe `Parametros`.
`processar_figura` só calcula; `executar` também grava as imagens intermediárias e os .wav.
"""
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from src import config
from src.cleaning import SemGrade, SoCurva, remover_anotacoes, remover_grade
from src.curve import Calibracao, SinalExtraido, extrair_sinal
from src.roi import Painel, recortar, segmentar_paineis
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
