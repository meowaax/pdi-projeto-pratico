"""Requisito 1: o código do pipeline não usa funções prontas de correlação ou filtragem."""
import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVOS = sorted(RAIZ.glob("src/*.py")) + [RAIZ / "main.py"] + sorted(RAIZ.glob("demo_*.py"))
PROIBIDOS = {
    "correlate2d", "convolve2d", "filter2D", "sepFilter2D", "GaussianBlur", "blur", "boxFilter", "Sobel",
    "gaussian_filter", "sobel", "convolve", "correlate", "uniform_filter",
}


def usos_proibidos(caminho):
    """Linhas e nomes proibidos usados como código (comentários e docstrings não contam)."""
    arvore = ast.parse(Path(caminho).read_text(encoding="utf-8"))
    achados = []
    for no in ast.walk(arvore):
        nome = no.attr if isinstance(no, ast.Attribute) else no.id if isinstance(no, ast.Name) else None
        if nome in PROIBIDOS:
            achados.append((no.lineno, nome))
        if isinstance(no, ast.ImportFrom) and no.module and no.module.startswith(("scipy.signal", "scipy.ndimage")):
            achados.append((no.lineno, no.module))
        if isinstance(no, ast.ImportFrom) and any(a.name in PROIBIDOS for a in no.names):
            achados.extend((no.lineno, a.name) for a in no.names if a.name in PROIBIDOS)
    return achados


def test_ha_arquivos_para_verificar():
    assert len(ARQUIVOS) >= 10


def test_pipeline_nao_usa_correlacao_nem_filtros_prontos():
    for arquivo in ARQUIVOS:
        assert usos_proibidos(arquivo) == [], arquivo.name


def test_o_verificador_pega_um_uso_proibido(tmp_path):
    ruim = tmp_path / "ruim.py"
    ruim.write_text("import cv2\nsaida = cv2.filter2D(img, -1, k)\nfrom scipy.signal import correlate2d\n", encoding="utf-8")
    assert {nome for _, nome in usos_proibidos(ruim)} == {"filter2D", "scipy.signal", "correlate2d"}


def test_as_referencias_prontas_ficam_so_nos_testes_de_correlacao():
    texto = (RAIZ / "tests" / "test_correlation.py").read_text(encoding="utf-8")
    assert "correlate2d" in texto and "filter2D" in texto
