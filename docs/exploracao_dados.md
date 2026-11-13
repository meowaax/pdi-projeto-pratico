# Task 1B: exploração e mapeamento dos dados

Gerado por `python explore_data.py` (saída completa em `docs/exploration/measurements.json`).
Os valores usados pelo pipeline estão em `src/config.py`, e `tests/test_exploration.py`
confere que os dois continuam iguais.

## Arquivos de entrada

| Arquivo | Formato | Observações |
|---|---|---|
| `Figure1.png` | 707x867, RGB, uint8 | 5837 cores distintas (compressão com perdas), 73% de fundo branco. |
| `Imagens/Shapes.png` | 800x400, RGB, uint8 | Só 2 valores (0 e 255), canais idênticos. Triângulo, quadrado e losango. |
| `Imagens/testpat.1k.color2.tif` | 1024x1024, RGB, uint8 | 346 cores, colorida de verdade (canais diferentes). Abre normalmente com OpenCV. |
| `Especificacao_do_Trabalho_Pratico.pdf` | 2 páginas | Texto extraído com `pypdf` (4131 caracteres), com os mesmos requisitos do texto colado pelo usuário. |
| `content_2011_2_30_60024.pdf` | 4 páginas | Artigo (Tomas et al.). |

## Estrutura da Figure1 (3 painéis)

Coordenadas em pixels da imagem original. Valores completos em `src/config.py`.

| Painel | ROI (x0,y0,x1,y1) | Tempo (s) | px/s | y de 0% | px por 1% |
|---|---|---|---|---|---|
| a (Normal) | 34, 7, 698, 230 | 5.665 a 6.165 | 1333,0 | 119,0 | 1,144 |
| b (Still) | 34, 279, 697, 486 | 1.345 a 1.845 | 1331,8 | 382,8 | 1,056 |
| c (VSD) | 31, 534, 694, 741 | 1.085 a 1.585 | 1333,2 | 637,6 | 1,059 |

- Os rótulos de tempo foram conferidos visualmente nas faixas de cada painel: 11 rótulos, de 0,05 em 0,05 s.
- A grade tem 10 intervalos verticais por painel (ajuste linear com resíduo máximo de 0,9 px) e 6 linhas
  horizontais visíveis entre +75% e -75%.
- A linha de 0% fica coberta pela curva e não é detectada. Ela é inferida pelo ajuste dos passos de 25%.
  No painel a, a linha cinza residual em y=120 confirma o valor inferido de 119,0.
- Cores: a curva e as anotações são o mesmo azul (mediana RGB 145,145,219, com traços mais escuros no núcleo).
  A grade é cinza com R=G=B entre 180 e 249. A legenda "Diaphragm mode; Normalized" é a única coisa preta dentro do gráfico.
- Cada coluna do gráfico tem pelo menos um pixel de curva (sem lacunas).
- Pixels azuis por coluna: mediana 3 nos painéis a e b e 11 no c. O máximo é 128, 78 e 139,
  nos picos verticais.
- Anotações azuis: viram componentes conexos separados da curva (10 em a, 5 em b, 4 em c).
  Nenhuma toca a curva com o limiar `b - max(r,g) > 40`.

Figuras: `docs/exploration/rois_anotadas.png`, `componentes_azuis.png`, `mascaras_cor.png`.

## Premissas do plano x observado x ação

| # | Premissa | Observado | Ação |
|---|---|---|---|
| 1 | A curva é azul saturado e a grade é cinza claro. | A curva é azul claro com antialiasing e a imagem tem artefatos de compressão. | Separar por diferença de canais (`b - max(r,g) > 40`) e não por cor exata. Tasks 5 e 6. |
| 2 | Anotações azuis encostam na curva (risco). | São componentes separados, sem contato no limiar usado. | Remover por componentes conexos (área e posição) **antes** do Gaussiano, que poderia fundi-las à curva. Task 6. |
| 3 | Os três gráficos cobrem 0,5 s. | Confirmado, a 1333 px/s. | Usar `DURACAO_S = 0.5` e `N_AMOSTRAS = 2500`. Tasks 8 e 9. |
| 4 | Mesma escala vertical nos painéis. | Falso: 1,144 px/% em a e cerca de 1,06 em b e c. | Calibração de amplitude por painel (`config.PAINEIS`). Task 8. |
| 5 | Mesmo alinhamento horizontal. | O painel c fica cerca de 3,6 px mais à esquerda. | Calibração de tempo por painel. Task 8. |
| 6 | A linha de 0% serve como referência. | Está coberta pela curva. | Inferir a partir das linhas de ±25/50/75%. Já feito em `explore_data.py`. |
| 7 | Há texto preto espalhado. | Só a legenda "Diaphragm mode; Normalized" está dentro do gráfico. Os números dos eixos ficam fora da ROI. | Remover a legenda pela caixa `legenda` do config. Task 6. |
| 8 | A ROI é o retângulo todo de cada painel. | A moldura superior e a faixa de rótulos de tempo não pertencem ao gráfico. A legenda "(a)" invade a faixa de linhas do painel b. | A ROI do config já exclui moldura e rótulos. Task 4 deve detectar isso automaticamente, usando as linhas de grade. |
| 9 | O colapso por coluna pode usar o centroide. | Há colunas com mais de 100 pixels azuis (picos verticais), em que o centroide erra. | Usar extremos (mín e máx) por coluna ou o caminho do traço. Task 8. |
| 10 | O traço é fino. | Cerca de 3 px de espessura por antialiasing. | Gaussiano com σ entre 1 e 1,5 (kernel 5x5 ou 7x7). Sobel 3x3. Task 7. |
| 11 | A reamostragem a 5 kHz recupera mais detalhe. | A imagem só dá 1333 colunas por segundo, então o conteúdo útil fica abaixo de cerca de 667 Hz. | Manter 5 kHz como pede a especificação, mas registrar o limite na análise. Tasks 9 e 11. |
| 12 | Entrada da correlação é uma matriz 2D em cinza. | `Shapes.png` é binária e o TIFF é colorido, ambos em 3 canais uint8. | `correlate2d_manual` aceita 2D float. O script de demo converte para cinza (BT.601) e também testa por canal. Task 2. |
| 13 | A especificação dá fs=5000, 16 bits, mono e os nomes dos arquivos. | Confirmado no texto extraído do PDF. | Constantes em `config.py`. |
| 14 | A barra à direita do painel c. | A figura mostra uma barra vertical azul-escura na borda direita do painel c, vista na inspeção visual. Ela ficou fora da ROI (x1=694), mas não foi medida. | Conferir na Task 6 que ela não entra como curva. |

## Contexto do artigo

- Os sons originais foram gravados a 8000 Hz, 16 bits, mono. O trabalho pede 5000 Hz, então o áudio reconstruído já tem banda menor que o original.
- "Still" é o sopro inocente de Still. VSD é a comunicação interventricular (sopro patológico). "Normal" é sem sopro.
