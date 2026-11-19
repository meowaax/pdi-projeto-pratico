# Task 13: lista de entregáveis

Conferência final contra `Especificacao_do_Trabalho_Pratico.pdf`. Prazo interno 25/11/2026; entrega oficial 30/11/2026
(relatório em PDF e código-fonte enviados até o início da primeira aula de apresentação).

## Requisitos

| Item da especificação | Onde está | Como foi conferido |
|---|---|---|
| Req. 1: correlação 2D não normalizada, zero padding, mesmo tamanho, sem funções prontas | `src/correlation.py` | `tests/test_correlation.py` (contra `scipy` e `cv2` só nos testes) e `tests/test_requisito1.py` (varre o código) |
| Req. 1: testar com as imagens anexas | `demo_req1.py`, relatório (Figuras 2 e 3) | `tests/test_correlation.py` com `Shapes.png` e o TIFF |
| Req. 2.1: segmentação automática das ROIs | `src/roi.py` | `tests/test_roi.py`, figura `rois` do relatório |
| Req. 2.2: remover grade, eixos, marcas, textos e rótulos | `src/cleaning.py` | `tests/test_cleaning.py`, `tests/test_annotations.py` |
| Req. 2.3: Gaussiano e Sobel com a correlação do Req. 1 | `src/filters.py`, `src/structure.py` | `tests/test_structure.py` |
| Req. 2.4: eixo de tempo, mapeamento e reamostragem a 5000 Hz, 16 bits, mono | `src/curve.py`, `src/audio.py` | `tests/test_curve.py`, `tests/test_audio.py` |
| Req. 2.5: gravar `normal.wav`, `still.wav`, `vsd.wav` | `main.py`, `audio/` | `tests/test_main.py`, `tests/test_audio_versionado.py` |

## Entregáveis

| Entregável | Onde está | Situação |
|---|---|---|
| Código-fonte comentado e organizado | `src/`, `main.py`, `README.md` | pronto |
| Relatório técnico (algoritmos e máscaras, imagens intermediárias, análise comparativa) | `report/relatorio.pdf` (26 páginas) | pronto, falta conferir os itens abaixo |
| Seções exigidas: introdução, materiais e métodos, resultados, discussão, conclusão | `report/secoes/` | conferido por `tests/test_report.py` |
| Os 3 arquivos .wav | `audio/normal.wav`, `audio/still.wav`, `audio/vsd.wav` | prontos, idênticos aos gerados por `python main.py` |

## Verificação em ambiente limpo (19/11/2026)

Clone novo da branch `task-13-entrega`, venv novo, `pip install -r requirements.txt`:

- `pytest`: 333 testes passando.
- `python main.py`: código de saída 0, gera os três .wav em `outputs/audio/`, idênticos byte a byte aos de `audio/`.

## Pendências do aluno

Estes itens dependem de você e não podem ser feitos pelo código:

- [ ] Confirmar na capa o nome completo do professor ("Leonardo Vidal", tirado dos metadados do PDF da especificação) e o centro ("Centro de Informática", deduzido do e-mail `ci.ufpb.br`).
- [ ] Conferir quatro das cinco referências bibliográficas (Gonzalez e Woods, Otsu, Lee et al., Oppenheim e Schafer), que foram escritas de memória. A de Tomas et al. (2011) foi conferida contra o PDF do artigo: autores, conferência CONTENT 2011 e ISBN 978-1-61208-157-1.
- [ ] Confirmar com o professor a data e o horário da apresentação.
- [ ] Imprimir o relatório (`report/relatorio.pdf`).
- [ ] Enviar o PDF do relatório e o código-fonte até o início da primeira aula de apresentação.
- [ ] Preparar a apresentação.
