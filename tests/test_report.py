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


# ----------------------------------------------------------------------------
# O texto do relatório
# ----------------------------------------------------------------------------
import shutil
import subprocess

TEX = RAIZ / "report" / "relatorio.tex"
SECOES = sorted((RAIZ / "report" / "secoes").glob("*.tex"))


def texto_do_relatorio():
    return "\n".join(p.read_text(encoding="utf-8") for p in [TEX, *SECOES])


def test_relatorio_tem_as_secoes_exigidas_pela_especificacao():
    texto = texto_do_relatorio()
    for secao in ("Introdução", "Materiais e métodos", "Resultados", "Discussão", "Conclusão"):
        assert f"\\section{{{secao}}}" in texto, secao


def test_figuras_e_tabelas_citadas_existem():
    texto = texto_do_relatorio()
    for caminho in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", texto):
        assert (RAIZ / "report" / caminho).exists(), caminho
    for caminho in re.findall(r"\\input\{(gerado/[^}]+)\}", texto):
        assert (RAIZ / "report" / f"{caminho}.tex").exists() or (RAIZ / "report" / caminho).exists(), caminho


def test_todas_as_citacoes_tem_referencia():
    texto = texto_do_relatorio()
    citadas = {c.strip() for grupo in re.findall(r"\\cite\{([^}]+)\}", texto) for c in grupo.split(",")}
    definidas = set(re.findall(r"\\bibitem\{([^}]+)\}", texto))
    assert citadas and citadas <= definidas
    assert definidas <= citadas, "referência listada e nunca citada"


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex não instalado")
def test_relatorio_compila_sem_referencias_indefinidas(tmp_path):
    copia = tmp_path / "report"
    shutil.copytree(RAIZ / "report", copia, ignore=shutil.ignore_patterns("*.pdf", "*.aux", "*.log", "*.out", "*.toc", "__pycache__"))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "relatorio.tex"], cwd=copia,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout[-2000:]
    log = (copia / "relatorio.log").read_text(encoding="utf-8", errors="replace")
    assert "undefined" not in log
    assert "Overfull \\hbox" not in log or all(float(x) < 20 for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide", log))
    assert (copia / "relatorio.pdf").stat().st_size > 50_000
