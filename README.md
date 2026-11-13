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
src/correlation.py   # Req. 1: correlação 2D manual
src/roi.py           # segmentação dos gráficos
src/cleaning.py      # remoção de grade, eixos e texto
src/curve.py         # extração da curva + calibração
src/audio.py         # reamostragem + escrita do .wav
src/config.py        # ROIs, limites de tempo
main.py              # pipeline completo
tests/               # testes (pytest)
outputs/             # imagens intermediárias e .wav (gerados)
report/              # relatório técnico
Imagens/             # imagens de teste do Req. 1
```

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest            # testes
python main.py    # pipeline (disponível após a Task 10)
```

`scipy` e `cv2.filter2D` são usados somente nos testes, como referência de validação.
