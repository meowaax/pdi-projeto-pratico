"""Teste trivial para validar a configuração do pytest e das dependências."""
import importlib


def test_dependencias_importaveis():
    for nome in ("numpy", "cv2", "skimage", "matplotlib", "scipy"):
        importlib.import_module(nome)


def test_pacote_src_importavel():
    importlib.import_module("src")
