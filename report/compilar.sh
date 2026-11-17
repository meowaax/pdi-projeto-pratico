#!/bin/sh
# Compila o relatório em PDF (report/relatorio.pdf). Roda o pdflatex duas vezes para resolver
# o sumário e as referências cruzadas. Antes, gere as figuras e as tabelas, se o código mudou:
#     python report/gerar_figuras.py
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error relatorio.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error relatorio.tex > /dev/null
if grep -q "undefined" relatorio.log; then
    echo "Há referências ou citações indefinidas em relatorio.log" >&2
    exit 1
fi
echo "report/relatorio.pdf gerado."
