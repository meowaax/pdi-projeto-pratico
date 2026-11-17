"""O demo da análise gera as figuras e as tabelas, e as tabelas batem com os números do docs/."""
from pathlib import Path

import demo_analise

RAIZ = Path(__file__).resolve().parent.parent


def test_demo_gera_figuras_e_tabelas(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_analise, "SAIDA", tmp_path)
    demo_analise.main()
    for nome in ("formas_de_onda.png", "espectros.png", "espectrogramas.png", "tabela_tempo.md",
                 "tabela_espectro.md", "conferencia_artigo.md"):
        assert (tmp_path / nome).stat().st_size > 0


def test_tabelas_do_docs_sao_as_geradas_pelo_codigo(tmp_path, monkeypatch):
    """docs/analise_comparativa.md copia as tabelas do demo; se o código mudar, o documento precisa ser refeito."""
    monkeypatch.setattr(demo_analise, "SAIDA", tmp_path)
    demo_analise.main()
    doc = (RAIZ / "docs" / "analise_comparativa.md").read_text(encoding="utf-8")
    for nome in ("tabela_tempo.md", "tabela_espectro.md", "conferencia_artigo.md"):
        tabela = (tmp_path / nome).read_text(encoding="utf-8").strip().splitlines()
        for linha in tabela:
            assert linha in doc, linha
