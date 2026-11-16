# Task 8: extração do sinal e calibração

Gerado por `python demo_extracao.py`.

## Calibração

`curve.Calibracao.de_painel` usa as medidas da Task 4, uma por painel:

- **Tempo:** `t = t_ini + (x - x_esq) * 0,5 / (x_dir - x_esq)`. As linhas verticais da grade ficam a 0,05 s umas das outras. `t_ini` vem de `config.PAINEIS` (5,665, 1,345 e 1,085 s, lidos dos rótulos do eixo).
- **Amplitude:** `a(%) = (y_zero - y) * 150 / (y_m75 - y_75)`, com a linha de 0% obtida por ajuste das linhas de ±25, 50 e 75%.
- Todas as posições são índices de pixel, a mesma convenção usada para medir a grade. Por isso não há correção de meio pixel.

Conferência: as bordas do eixo de tempo viram exatamente `t_ini` e `t_ini + 0,5 s`, e as linhas de ±75% viram ±75% (testes em `tests/test_curve.py`). Os picos extraídos do painel a (S1 em +75% perto de 5,80 s, S2 de +92% a -72% perto de 6,045 s) batem com o que se lê na figura.

## Uma amostra por coluna

A figura tem 1333 colunas por segundo, então o sinal extraído tem uma amostra por coluna (cerca de 664 por painel). A Task 9 reamostra isso para 5000 Hz por interpolação, que não cria informação nova. O conteúdo recuperável fica abaixo de uns 667 Hz.

O sinal começa de 1,2 a 1,6 ms depois do início do trecho e termina de 0,7 a 1,1 ms antes do fim dos 0,5 s, porque a ROI deixa de fora a moldura nas pontas. A Task 9 precisa estender as pontas para completar os 0,5 s.

## Regra de colapso por coluna

Em cada coluna a máscara tem um trecho vertical de topo a base. Em subidas e descidas íngremes, o trecho é longo e o ponto médio é uma boa estimativa. Num pico ou vale, a curva chega ao extremo e volta, e o ponto médio perde a amplitude.

Regras comparadas (`curve.REGRAS`):

- **`centro`:** ponto médio entre topo e base.
- **`picos` (padrão):** ponto médio, trocado pelo topo (ou pela base) nas colunas de virada com extensão maior que 6 px.

Teste de ida e volta: um sinal conhecido a 8 kHz é desenhado como a figura, passa pelo mesmo pipeline e é comparado com o original em 5 kHz (média de 4 sinais por tipo, `src/synthetic.py`).

| sinal | regra | correlação | erro relativo (NRMSE) | amplitude do pico (razão) |
|---|---|---|---|---|
| burst | picos | 0.981 | 0.193 | 1.01 |
| burst | centro | 0.977 | 0.343 | 0.68 |
| sopro | picos | 0.970 | 0.242 | 1.00 |
| sopro | centro | 0.950 | 0.403 | 0.67 |
| denso | picos | 0.780 | 0.674 | 1.02 |
| denso | centro | 0.779 | 0.644 | 0.67 |

`picos` mantém a amplitude do pico (razão 1,0) e reduz o erro de 0,34 para 0,19 nos estouros e de 0,40 para 0,24 com sopro. O sinal denso (ruído de 150 a 600 Hz, como o painel c) é o limite do método: a correlação cai para 0,78 nas duas regras, porque as oscilações são mais rápidas que a resolução da figura. Nesse caso `centro` tem um erro um pouco menor (0,64 contra 0,67), mas perde um terço da amplitude do pico.

Experimento exploratório, fora do código: emitir dois pontos por coluna (topo e base, em ordem escolhida por continuidade ou por Viterbi) manteve o pico, mas adicionou ruído de alta frequência e piorou a correlação (0,73 nos estouros, 0,69 com sopro e 0,46 no denso).

## Limites

- **Painel c:** o painel de VSD é o mais denso. Onde a curva oscila várias vezes por coluna, uma amostra por coluna só acompanha o envelope, e os detalhes mais rápidos se perdem.
- **Espessura do traço:** o topo e a base da máscara incluem a metade da espessura do traço (cerca de 1 px, ou 1% de amplitude).
