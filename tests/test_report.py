"""Testes do gerador de figuras e tabelas do relatório."""
import re
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "report"))

import gerar_figuras as g  # noqa: E402

FIGURAS_ESPERADAS = {
    "bordas_shapes.png", "espectrogramas.png", "espectros.png", "estrutura_c.png", "extracao_a.png",
    "extracao_b.png", "extracao_c.png", "filtros.png", "formas_de_onda.png", "limpeza.png", "rois.png",
}
TABELAS_ESPERADAS = {
    "tab_artigo.tex", "tab_audio.tex", "tab_calibracao.tex", "tab_espectro.tex", "tab_regras.tex", "tab_sigma.tex", "tab_tempo.tex",
}


def test_escapar_caracteres_especiais():
    assert g.escapar("50% de a_b & c #1 $x") == r"50\% de a\_b \& c \#1 \$x"


def test_numero_e_virgula_usam_a_virgula_decimal():
    assert g.numero(3.14159, 2) == "3{,}14"
    assert g.virgula("+9.8") == "+9{,}8"
    assert g.virgula("-0.821 dB") == "$-$0{,}821 dB"
    assert g.virgula("a-b") == "a-b"


def test_tabular_tem_booktabs_e_o_numero_de_colunas_certo():
    t = g.tabular("lr", ["A", "B"], [["x", "1"], ["y", "2"]])
    assert t.startswith("\\begin{tabular}{lr}") and "\\toprule" in t and "\\midrule" in t and t.rstrip().endswith("\\end{tabular}")
    assert t.count(r" \\") == 3                                  # cabeçalho e duas linhas


@pytest.fixture(scope="module")
def gerado(tmp_path_factory):
    base = tmp_path_factory.mktemp("relatorio")
    return base, g.gerar(base / "figuras", base / "tex", base / "trabalho")


def test_gera_todas_as_figuras_e_tabelas(gerado):
    base, resultado = gerado
    assert set(resultado["figuras"]) == FIGURAS_ESPERADAS
    assert set(resultado["tabelas"]) == TABELAS_ESPERADAS
    for nome in FIGURAS_ESPERADAS:
        assert (base / "figuras" / nome).stat().st_size > 5000
    for nome in TABELAS_ESPERADAS:
        assert (base / "tex" / nome).stat().st_size > 100


def test_tabelas_sao_ambientes_bem_formados(gerado):
    base, _ = gerado
    for nome in TABELAS_ESPERADAS:
        texto = (base / "tex" / nome).read_text(encoding="utf-8")
        ambiente = "tabularx" if nome == "tab_artigo.tex" else "tabular"
        assert texto.count(f"\\begin{{{ambiente}}}") == 1 and texto.count(f"\\end{{{ambiente}}}") == 1
        assert not re.search(r"\b(nan|inf)\b", texto.lower())          # nenhum valor inválido nas células


def test_tabela_de_audio_tem_os_tres_arquivos_com_o_formato_pedido(gerado):
    base, _ = gerado
    texto = (base / "tex" / "tab_audio.tex").read_text(encoding="utf-8")
    for nome in ("normal.wav", "still.wav", "vsd.wav"):
        assert nome in texto
    assert texto.count("2500 & 5000 & 16 & 1") == 3


def test_tabelas_do_repositorio_estao_atualizadas(gerado):
    """report/gerado/*.tex é versionado; se o código mudar, rode `python report/gerar_figuras.py` de novo."""
    base, _ = gerado
    for nome in TABELAS_ESPERADAS:
        assert (RAIZ / "report" / "gerado" / nome).read_text(encoding="utf-8") == (base / "tex" / nome).read_text(encoding="utf-8"), nome


def test_gerar_nao_altera_o_demo_analise(gerado):
    import demo_analise
    assert demo_analise.SAIDA.name == "analise"
