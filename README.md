# Trabalho Prático de PDI: reconstrução de sinais de áudio a partir da Figure 1

Extrai os três traçados de sons cardíacos da `Figure1.png` (a: Normal, b: Still, c: VSD) e
reconstrói três arquivos `.wav` (5000 Hz, 16 bits, mono): `normal.wav`, `still.wav` e `vsd.wav`.
A filtragem espacial usa uma correlação 2D escrita do zero, sem `scipy.signal.correlate2d`
nem `cv2.filter2D`.

## Como executar
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py          # pipeline completo, grava em outputs/
python demo_req1.py     # Requisito 1: filtros nas imagens de Imagens/, grava em outputs/req1/
```
`python main.py --saida pasta/` grava em outra pasta. O pipeline gera `outputs/audio/*.wav`,
as imagens de cada etapa em `outputs/etapas/` e `outputs/resumo.json`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `main.py` | Pipeline completo: `Figure1.png` para `normal.wav`, `still.wav` e `vsd.wav` |
| `demo_req1.py` | Teste do Requisito 1 com as imagens de `Imagens/` |
| `Figure1.png` | Entrada do pipeline |
| `Imagens/` | Imagens de teste do Requisito 1 (`Shapes.png`, `testpat.1k.color2.tif`) |
| `audio/` | `.wav` entregues (5000 Hz, 16 bits, mono, 0,5 s) |
| `report/relatorio.pdf` | Relatório técnico |
| `requirements.txt` | Dependências |
| `outputs/` | Criada ao rodar `main.py` e `demo_req1.py` (não versionada) |

Código em `src/`:

| Módulo | Função |
|---|---|
| `correlation.py` | Requisito 1: correlação 2D não normalizada com zero padding, feita do zero |
| `kernels.py` | Máscaras: média, Gaussiana e Sobel |
| `filters.py` | Suavização Gaussiana e gradiente de Sobel, sobre a correlação manual |
| `roi.py` | Segmentação automática dos três gráficos (a, b, c) |
| `cleaning.py` | Remoção de grade, eixos, textos e anotações |
| `structure.py` | Suavização, bordas, binarização e afinamento da curva |
| `curve.py` | Extração da curva e calibração de tempo e amplitude |
| `audio.py` | Reamostragem a 5 kHz, quantização em 16 bits e escrita do `.wav` |
| `pipeline.py` | Encadeia as etapas e grava as saídas |
| `config.py` | Parâmetros (áudio, ROIs, σ, limiares) |

Testes, demos de cada etapa, documentação das decisões e fontes do relatório estão na
branch `extras`.
