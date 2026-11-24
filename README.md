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

## Estrutura de pastas

```text
.
├── main.py                  # pipeline completo (Figure1.png -> .wav)
├── demo_req1.py             # teste do Requisito 1 com as imagens anexas
├── Figure1.png              # entrada do pipeline
├── requirements.txt         # dependências
├── README.md
│
├── Imagens/                 # imagens de teste do Requisito 1
│   ├── Shapes.png
│   └── testpat.1k.color2.tif
│
├── audio/                   # áudios entregues (5000 Hz, 16 bits, mono, 0,5 s)
│   ├── normal.wav
│   ├── still.wav
│   └── vsd.wav
│
├── report/
│   ├── relatorio.pdf        # relatório técnico
│   └── apresentacao.pdf     # slides baseados no relatório
│
├── src/                     # código do pipeline
│   ├── correlation.py       # Requisito 1: correlação 2D manual com zero padding
│   ├── kernels.py           # máscaras: média, Gaussiano, Sobel
│   ├── filters.py           # suavização e gradiente sobre a correlação manual
│   ├── roi.py               # segmentação dos três gráficos
│   ├── cleaning.py          # remoção de grade, eixos, textos e anotações
│   ├── structure.py         # suavização, bordas, binarização e afinamento da curva
│   ├── curve.py             # extração da curva e calibração de tempo e amplitude
│   ├── audio.py             # reamostragem a 5 kHz, quantização e escrita do .wav
│   ├── pipeline.py          # encadeia as etapas e grava as saídas
│   └── config.py            # parâmetros (áudio, ROIs, σ, limiares)
│
└── outputs/                 # criada ao rodar os scripts (não versionada)
    ├── audio/               # .wav gerados por main.py
    ├── etapas/              # imagens de cada etapa do pipeline
    ├── resumo.json
    └── req1/                # resultados de demo_req1.py
```

Testes, demos de cada etapa, documentação das decisões e fontes do relatório estão na
branch `extras`.
