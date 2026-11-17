"""Pipeline completo: Figure1.png -> normal.wav, still.wav e vsd.wav.

Uso:
    python main.py                       # usa Figure1.png e grava em outputs/
    python main.py --figura outra.png --saida resultado/
    python main.py --silencioso          # só mostra avisos e erros

Saídas:
    <saida>/etapas/*.png   imagens de cada etapa (original, sem grade, sinal limpo, ...)
    <saida>/audio/*.wav    normal.wav, still.wav, vsd.wav (5000 Hz, 16 bits, mono)
    <saida>/resumo.json    parâmetros e números principais

Os parâmetros ajustáveis ficam em src/config.py.
"""
import argparse
import logging
import sys

from src import config, pipeline
from src.roi import RoiError


def criar_parser():
    p = argparse.ArgumentParser(description="Reconstrói os três sinais de áudio da Figure1.png.")
    p.add_argument("--figura", default=str(config.FIGURA), help="imagem de entrada (padrão: %(default)s)")
    p.add_argument("--saida", default=str(config.SAIDA), help="pasta de saída (padrão: %(default)s)")
    p.add_argument("--silencioso", action="store_true", help="mostra só avisos e erros")
    return p


def main(argv=None):
    """Executa o pipeline. Devolve 0 em caso de sucesso e 1 em caso de erro."""
    args = criar_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.silencioso else logging.INFO,
                        format="%(levelname)s: %(message)s", force=True)
    try:
        pipeline.executar(args.figura, args.saida)
    except FileNotFoundError as erro:
        logging.error("%s", erro)
        return 1
    except RoiError as erro:
        logging.error("não foi possível segmentar os gráficos: %s", erro)
        return 1
    except ValueError as erro:      # por exemplo, uma imagem sem nenhuma curva para extrair
        logging.error("não foi possível extrair o sinal: %s", erro)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
