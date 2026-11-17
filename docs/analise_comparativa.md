# Task 11: análise comparativa dos três sinais

Gerado por `python demo_analise.py` (figuras e tabelas em `outputs/analise/`). O código está em `src/analysis.py`.

Figuras: `formas_de_onda.png`, `espectros.png` e `espectrogramas.png`.

## Como os sinais foram divididos

Cada sinal (0,5 s, 5 kHz, sem nível DC) é dividido em quatro janelas:

| Janela | Intervalo |
|---|---|
| S1 | 40 ms antes a 40 ms depois do instante de S1 |
| Sopro sistólico | de 40 ms depois de S1 até 40 ms antes de S2 |
| S2 | 40 ms antes a 40 ms depois do instante de S2 |
| Diástole | tudo o que fica antes de S1 ou depois de S2 (nível de fundo) |

S1 e S2 são localizados pelos rótulos "S1" e "S2" que a própria figura traz: as caixas mais à esquerda e mais à direita entre as anotações removidas na Task 6. O instante do evento é o máximo da envolvente RMS (10 ms) entre 10 ms antes e 50 ms depois do centro do rótulo. Os instantes obtidos batem com a leitura da figura dentro de 20 ms (testes em `tests/test_analysis.py`).

Os espectros usam só `numpy.fft` (periodograma com janela de Hann e zero padding), medidos de 40 a 660 Hz, porque a figura só entrega conteúdo abaixo de uns 667 Hz.

## Nível no tempo

Amplitudes em % da escala do gráfico, depois de remover o DC.

| Medida | Normal | Still | VSD |
|---|---|---|---|
| Pico (%) | 92.7 | 54.9 | 88.1 |
| RMS total (%) | 10.9 | 8.5 | 17.6 |
| Fator de crista | 8.5 | 6.5 | 5.0 |
| Intervalo S1 a S2 (ms) | 254 | 276 | 311 |
| RMS de S1 (%) | 17.0 | 11.0 | 10.8 |
| RMS de S2 (%) | 21.1 | 16.5 | 15.2 |
| RMS do sopro sistólico (%) | 0.98 | 4.66 | 23.53 |
| RMS da diástole (%) | 1.07 | 1.51 | 1.00 |
| Pico a pico de S1 (%) | 124 | 71 | 74 |
| Pico a pico de S2 (%) | 165 | 98 | 101 |
| S2 sobre S1 (dB) | +1.9 | +3.5 | +3.0 |
| Sopro sobre diástole (dB) | -0.8 | +9.8 | +27.4 |
| Sopro sobre S1 (dB) | -24.8 | -7.5 | +6.8 |
| Taxa de cruzamentos por zero no sopro (Hz equivalentes) | 109 | 99 | 206 |

**Aviso sobre comparar níveis entre painéis.** A legenda dos gráficos diz "Diaphragm mode; Normalized". Cada gravação foi normalizada pelo programa de origem de um jeito que não conhecemos (os picos dos três painéis não são todos 100%). Por isso as razões dentro de um mesmo sinal (sopro sobre S1, sopro sobre diástole) são confiáveis, e os valores absolutos (pico, RMS) não servem para dizer qual gravação foi mais forte.

## Forma de onda

- **Normal:** dois sons nítidos, S1 e S2, e uma sístole silenciosa. O RMS da sístole (0,98%) é igual ao da diástole (1,07%), a -0,8 dB, ou seja, não há sopro. S2 é 1,9 dB mais forte que S1.
- **Still:** S1 e S2 de amplitude menor e, entre eles, um sopro de baixo nível, 9,8 dB acima da diástole e 7,5 dB abaixo de S1. As oscilações do sopro são regulares: a taxa de cruzamentos por zero equivale a 99 Hz, perto do tom de 120 Hz do espectro.
- **VSD:** o sopro ocupa toda a sístole, de S1 até S2 (sopro holossistólico), com RMS 6,8 dB acima do de S1 e 27,4 dB acima da diástole. Por isso o fator de crista é o menor dos três (5,0): a energia é sustentada, sem picos isolados dominando. A taxa de cruzamentos por zero do sopro equivale a 206 Hz, o dobro da dos outros.
- **Intervalo S1 a S2:** 254, 276 e 311 ms. Esta análise não diz se a diferença é fisiológica, porque são três gravações, provavelmente de crianças diferentes.

## Espectro do sopro sistólico

| Medida (janela do sopro sistólico) | Normal | Still | VSD |
|---|---|---|---|
| Frequência dominante (Hz) | 144 | 120 | 164 |
| Centroide (Hz) | 129 | 122 | 202 |
| Largura a meia potência (Hz) | 9 | 18 | 6 |
| Planura espectral (80 a 500 Hz) | 0.062 | 0.047 | 0.195 |
| Energia de 40-80 Hz (%) | 9 | 3 | 1 |
| Energia de 80-200 Hz (%) | 89 | 95 | 50 |
| Energia de 200-500 Hz (%) | 2 | 2 | 46 |
| Energia de 500-660 Hz (%) | 0 | 0 | 2 |
| Frequência dominante de S1 (Hz) | 126 | 57 | 137 |
| Frequência dominante de S2 (Hz) | 99 | 76 | 86 |

- **Still** concentra 95% da energia entre 80 e 200 Hz, com um pico em 120 Hz, o tom do sopro. A planura espectral baixa (0,047) indica um tom estreito.
- **VSD** espalha a energia: 46% fica entre 200 e 500 Hz, o centroide sobe para 202 Hz e a planura é 4,2 vezes a do Still. É o ruído de banda larga de um sopro áspero.
- **Normal** tem pouca energia na janela sistólica. O espectro dela é o do fundo, com a frequência dominante de 144 Hz sem significado (o nível é o da diástole). Na escala logarítmica de `espectros.png`, o Still fica uma ordem de grandeza ou mais acima do Normal, e o VSD fica de uma a duas ordens acima do Still.
- **S1 do Still** tem a frequência dominante em 57 Hz, abaixo dos 80 Hz que o artigo atribui a ruído ambiente e a oscilações tardias do primeiro som.

O espectrograma (`espectrogramas.png`, mesma escala de cor para os três) mostra o mesmo: dois estouros de banda larga e uma sístole escura no Normal, uma faixa estreita em torno de 120 Hz na sístole do Still, e energia espalhada de 0 a 500 Hz durante toda a sístole do VSD.

## Conferência com o artigo

O artigo (Tomas et al., 2011) descreve o espectro de 20 gravações de cada tipo. Aqui há uma de cada, então estas conferências mostram se os sinais reconstruídos são coerentes com o artigo, e não o provam.

| Painel | Afirmação do artigo | Critério | Medido | Atendido |
|---|---|---|---|---|
| b | Still tem ressonância entre 90 e 170 Hz | frequência dominante do sopro em 90 a 170 Hz | 120 Hz | sim |
| b | A frequência do tom de Still fica na maioria dos casos entre 110 e 130 Hz (média de 118,75 Hz) | frequência dominante do sopro em 110 a 130 Hz | 120 Hz | sim |
| b | A energia de Still fica abaixo de 200 Hz, em banda estreita | pelo menos 80% da energia do sopro abaixo de 200 Hz | 0.971 | sim |
| a | A energia do Normal fica abaixo de 200 Hz | pelo menos 80% da energia da janela sistólica abaixo de 200 Hz | 0.981 | sim |
| a | O Normal não tem sopro (sem ressonância) | sopro menos de 3 dB acima do nível da diástole | -0.821 dB | sim |
| c | A energia do VSD se distribui em banda larga de 80 a 500 Hz | pelo menos 80% da energia do sopro de 80 a 500 Hz | 0.964 | sim |
| c | O VSD é mais disperso em frequência que o Still | planura espectral do VSD maior que a do Still | 4.15x | sim |
| c | A frequência do maior pico do VSD passa de 200 Hz | frequência dominante do sopro acima de 200 Hz | 164 Hz | **não** |

Sete das oito afirmações se reproduzem. O tom do Still em 120 Hz coincide com a média de 118,75 Hz do artigo.

Sobre a que não se reproduz: o pico dominante do VSD fica em 164 Hz, e o centroide em 202 Hz. O artigo fala de uma estatística de 20 gravações. Com uma gravação, um pico em 164 Hz não a contradiz. O espectro do VSD tem vários picos entre 160 e 220 Hz (veja a escala linear de `espectros.png`), então qual deles é o maior é sensível a pequenos erros da extração.

A largura do tom do Still (18 Hz) não é comparável com a média de 40,75 Hz do artigo, porque o artigo mede em quadros de 20 ms, cuja resolução física é de cerca de 50 Hz (1 / 20 ms), embora avalie o espectro de 5 em 5 Hz, e aqui a janela de cerca de 170 ms resolve uns 8 Hz.

## Limites desta análise

- **Banda:** a figura só entrega conteúdo abaixo de uns 667 Hz. Nada acima disso é analisável.
- **Uma gravação por tipo:** as conclusões valem para estes três sinais.
- **Localização de S1 e S2:** depende da posição dos rótulos da figura. Os instantes foram conferidos contra a leitura visual da figura, mas se o layout mudar, a regra deve ser revista.
- **Níveis absolutos:** não comparáveis entre painéis (ver o aviso acima).
- **Extração:** o painel c é o mais denso, e a extração só recupera o envelope das oscilações mais rápidas (ver `docs/extracao_sinal.md`). Isso pode deslocar picos no espectro do VSD.

## Resposta audível

> **A preencher.** Esta seção depende de ouvir `outputs/audio/normal.wav`, `still.wav` e `vsd.wav`, o que o código não faz. O texto abaixo é o roteiro, com o que os números levam a esperar. As impressões da audição ainda não foram registradas.

Como ouvir: os arquivos duram 0,5 s e têm todos o mesmo pico (normalização por pico). Por isso o volume relativo entre eles não vale. Para comparar o volume entre eles, gere os arquivos com `normalizacao="escala"` (ver `docs/audio.md`), e lembre do aviso sobre níveis entre painéis acima. Use fones, porque os alto-falantes de notebook quase não reproduzem abaixo de 150 Hz, onde está a maior parte da energia.

| Sinal | O que os números levam a esperar | O que foi ouvido |
|---|---|---|
| Normal | Duas batidas curtas e graves ("tum-tá"), com silêncio entre elas. | _a preencher_ |
| Still | As duas batidas, com um zumbido grave e suave entre elas, de tom definido perto de 120 Hz. | _a preencher_ |
| VSD | As duas batidas dentro de um chiado contínuo e áspero, que ocupa todo o intervalo entre elas. | _a preencher_ |
