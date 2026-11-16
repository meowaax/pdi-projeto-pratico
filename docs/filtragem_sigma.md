# Task 7: escolha de σ e do limiar

Gerado por `python demo_estrutura.py` (tabela completa em `outputs/varredura_sigma.md`).

## O que é medido

A referência é a máscara da curva original, depois de remover grade e anotações (Tasks 5 e 6).
Cada linha compara a máscara realçada com essa referência:

- **componentes** e **colunas vazias**: continuidade da curva (o ideal é 1 e 0).
- **espessura mediana**: pixels de curva por coluna. A curva original tem 4 px (painéis a e b) e 15 px (painel c, por ser mais densa).
- **erro dos extremos**: diferença média, em pixels, entre o topo e a base de cada coluna.
- **viés do pico**: erro médio do topo e da base nas 40 colunas de maior extensão vertical (os picos de S1 e S2). Negativo é pico alongado pelo borramento, positivo é pico achatado. 1 px vale cerca de 1% de amplitude (1,06 px por 1%, ver `docs/exploracao_dados.md`).

## Resultado

Valores adotados: **σ = 1,0 (kernel 7x7) e limiar de Otsu**.

- σ de 1,5 ou mais engrossa a curva e alonga os picos em 5 a 21 px (viés de -4,8 a -20,6 px), e com Otsu deixa 1 coluna vazia nos painéis a e c. Isso distorce a amplitude do áudio.
- Com σ = 0,6, o limiar fixo 0,06 tem erro parecido, mas o viés dos picos é maior em módulo (de -2,3 px no painel a e de -1,9 px no c) do que com σ = 1,0 e Otsu (-1,2 px no a e +0,3 px no c).
- O limiar de Otsu se adapta ao brilho de cada painel, e o fixo não.

### Painel a (Normal)

| σ | limiar | componentes | colunas vazias | espessura mediana (px) | erro dos extremos (px) | viés do pico (px) | pixels do esqueleto |
|---|---|---|---|---|---|---|---|
| 0.6 | 0.06 | 1 | 0 | 4 | 0.50 | -2.3 | 1556 |
| 0.6 | Otsu (0.152) | 1 | 0 | 3 | 0.80 | +5.9 | 1642 |
| 0.8 | 0.06 | 1 | 0 | 5 | 1.13 | -8.3 | 1525 |
| 0.8 | Otsu (0.132) | 1 | 0 | 4 | 0.49 | +2.8 | 1611 |
| 1 | 0.06 | 1 | 0 | 5 | 1.70 | -11.0 | 1403 |
| 1 | Otsu (0.116) | 1 | 0 | 4 | 0.55 | -1.2 | 1526 |
| 1.5 | 0.06 | 1 | 0 | 6 | 2.80 | -16.5 | 1243 |
| 1.5 | Otsu (0.089) | 1 | 1 | 5 | 1.87 | -11.7 | 1299 |
| 2 | 0.06 | 1 | 0 | 7 | 3.65 | -20.6 | 1163 |
| 2 | Otsu (0.075) | 1 | 1 | 6 | 3.06 | -18.3 | 1188 |

### Painel b (Still)

| σ | limiar | componentes | colunas vazias | espessura mediana (px) | erro dos extremos (px) | viés do pico (px) | pixels do esqueleto |
|---|---|---|---|---|---|---|---|
| 0.6 | 0.06 | 1 | 0 | 5 | 0.47 | -0.7 | 1355 |
| 0.6 | Otsu (0.137) | 1 | 0 | 3 | 0.75 | +3.1 | 1411 |
| 0.8 | 0.06 | 1 | 0 | 5 | 0.85 | -2.6 | 1323 |
| 0.8 | Otsu (0.117) | 1 | 0 | 4 | 0.49 | +1.6 | 1386 |
| 1 | 0.06 | 1 | 0 | 6 | 1.26 | -3.7 | 1248 |
| 1 | Otsu (0.102) | 1 | 0 | 5 | 0.56 | -0.4 | 1309 |
| 1.5 | 0.06 | 1 | 0 | 7 | 2.21 | -6.3 | 1125 |
| 1.5 | Otsu (0.078) | 1 | 0 | 6 | 1.58 | -4.8 | 1150 |
| 2 | 0.06 | 1 | 0 | 8 | 2.95 | -7.8 | 1074 |
| 2 | Otsu (0.065) | 1 | 0 | 7 | 2.71 | -7.2 | 1079 |

### Painel c (VSD)

| σ | limiar | componentes | colunas vazias | espessura mediana (px) | erro dos extremos (px) | viés do pico (px) | pixels do esqueleto |
|---|---|---|---|---|---|---|---|
| 0.6 | 0.06 | 1 | 0 | 18 | 1.68 | -1.9 | 3144 |
| 0.6 | Otsu (0.196) | 1 | 0 | 11 | 2.46 | +11.1 | 3844 |
| 0.8 | 0.06 | 1 | 0 | 19 | 2.81 | -6.2 | 2972 |
| 0.8 | Otsu (0.174) | 1 | 0 | 13 | 1.73 | +6.2 | 3591 |
| 1 | 0.06 | 1 | 0 | 22 | 3.86 | -8.6 | 2621 |
| 1 | Otsu (0.157) | 1 | 0 | 15 | 1.58 | +0.3 | 3198 |
| 1.5 | 0.06 | 1 | 0 | 26 | 5.97 | -11.4 | 2029 |
| 1.5 | Otsu (0.131) | 1 | 1 | 19 | 3.06 | -7.1 | 2531 |
| 2 | 0.06 | 1 | 0 | 28 | 7.53 | -13.1 | 1835 |
| 2 | Otsu (0.119) | 1 | 1 | 22 | 4.54 | -9.8 | 2005 |

## Aviso: o esqueleto não serve para medir amplitude

O esqueleto de 1 px (`Estrutura.esqueleto`) fica no meio do traço. Nos picos, onde a curva sobe e desce na mesma coluna, ele encurta as pontas e junta traços vizinhos. Comparado com os extremos da máscara original, o esqueleto erra em média cerca de 3 px por coluna nos painéis a e b e cerca de 8 px no painel c, com erros de até 125 px.

A máscara binarizada erra em média 0,5 a 1,6 px. Por isso a Task 8 deve extrair o sinal dos extremos de cada coluna da **máscara** (ou da cobertura suavizada), e não do esqueleto. O esqueleto fica para visualização e para o relatório.
