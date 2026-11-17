# Trabalho Prático de PDI: reconstrução de sinais de áudio a partir da Figure 1

Introdução ao Processamento Digital de Imagens, UFPB, período 2026.2.

Extrai automaticamente os três traçados de sons cardíacos da `Figure1.png`
(a: Normal, b: Still, c: VSD) e reconstrói três arquivos `.wav`
(5000 Hz, 16 bits, mono): `normal.wav`, `still.wav` e `vsd.wav`.

A filtragem espacial usa uma função de correlação 2D escrita do zero
(sem `scipy.signal.correlate2d` nem `cv2.filter2D` no pipeline).

## Prazos

- Prazo interno: 25/11/2026
- Entrega oficial: 30/11/2026

O backlog está nas [issues](https://github.com/meowaax/pdi-projeto-pratico/issues)
do milestone "Entrega PDI".

## Estrutura

```
main.py              # pipeline completo (python main.py)
src/correlation.py   # Req. 1: correlação 2D manual
src/kernels.py       # máscaras: média, Gaussiano, Sobel
src/filters.py       # suavização e gradiente sobre a correlação manual
src/roi.py           # segmentação dos três gráficos
src/cleaning.py      # remoção de grade, eixos, textos e anotações
src/structure.py     # suavização, bordas, binarização e afinamento da curva
src/curve.py         # extração da curva + calibração de tempo e amplitude
src/audio.py         # reamostragem a 5 kHz e escrita do .wav
src/pipeline.py      # encadeia tudo e grava as saídas
src/config.py        # parâmetros (áudio, ROIs medidas, σ, limiares, ...)
src/synthetic.py     # sinais sintéticos para validar a extração
tests/               # testes (pytest)
demo_*.py            # demos de cada etapa
explore_data.py      # exploração dos dados de entrada (Task 1B)
docs/                # decisões e medições de cada etapa
outputs/             # saídas geradas (não versionadas)
report/              # relatório técnico
Imagens/             # imagens de teste do Req. 1
```

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py                  # roda tudo e grava em outputs/
python main.py --saida pasta/   # outra pasta de saída
pytest                          # testes
```

O pipeline grava:

- `outputs/audio/normal.wav`, `still.wav`, `vsd.wav` (5000 Hz, 16 bits, mono, 0,5 s)
- `outputs/etapas/*.png`: imagens de cada etapa, por painel (`a_01_original`, `a_02_sem_grade`,
  `a_03_sinal_limpo`, `a_04_suavizado`, `a_05_bordas`, `a_06_binario`, `a_07_esqueleto`,
  `a_08_extracao`, e o mesmo para `b` e `c`) e `rois_detectadas.png`
- `outputs/resumo.json`: parâmetros e números principais

Os parâmetros ajustáveis (σ do Gaussiano, limiares, regra de colapso, interpolação,
normalização) ficam em `src/config.py`.

## Relatório

O relatório técnico está em `report/relatorio.pdf`, escrito em LaTeX (`report/relatorio.tex` e
`report/secoes/`). As figuras (`report/figuras/`) e as tabelas (`report/gerado/`) saem do próprio
código, para os números do texto não ficarem desatualizados:

```bash
python report/gerar_figuras.py   # regenera figuras e tabelas
report/compilar.sh               # compila o PDF (pdflatex, duas passadas)
```

`tests/test_report.py` confere que as tabelas versionadas são as geradas pelo código e que o
relatório compila sem referências indefinidas.

## Requisito 1

A filtragem espacial (média, Gaussiano e Sobel) usa só `correlate2d_manual`, escrita do zero
no domínio do espaço com zero padding. O código do pipeline não chama
`scipy.signal.correlate2d` nem `cv2.filter2D`, o que `tests/test_requisito1.py` verifica.
Essas duas funções aparecem somente nos testes, como referência de validação.

Outras bibliotecas usadas no pipeline: `skimage` para o limiar de Otsu e o afinamento
(esqueleto), e `scipy.interpolate` para a opção de interpolação cúbica.
