# Task 9: reamostragem e exportação dos .wav

Gerado por `python demo_audio.py`, que grava `outputs/audio/normal.wav`, `still.wav` e `vsd.wav`.

## Formato dos arquivos

| Parâmetro | Valor |
|---|---|
| Taxa de amostragem | 5000 amostras/s |
| Resolução | 16 bits por amostra, PCM com sinal |
| Canais | 1 (mono) |
| Duração | 0,5 s, ou seja, 2500 amostras |
| Tamanho | 5044 bytes (cabeçalho de 44 bytes mais 2 bytes por amostra) |

O arquivo é gravado com o módulo `wave` da biblioteca padrão. Os testes o releem com `wave` e com `scipy.io.wavfile` (leitor independente) e conferem os parâmetros acima.

## Cadeia de processamento

1. **Reamostragem:** o sinal tem uma amostra por coluna da figura (cerca de 664 por painel, a 1333 por segundo). Ele é interpolado para a grade uniforme de 5000 Hz, que começa em `t_ini` do painel e tem 2500 pontos. As pontas, que a ROI deixa de fora, repetem o valor da borda (de 1 a 3 amostras a cada lado).
2. **Remoção do DC:** subtrai a média.
3. **Normalização:** por padrão, o maior valor absoluto vira 0,95 do fundo de escala (-0,45 dBFS). Isso usa quase todo o intervalo de 16 bits em cada arquivo e deixa folga contra saturação.
4. **Quantização:** `round(x * 32767)`, com saturação em [-32768, 32767]. O erro é no máximo meio degrau (0,000015 do fundo de escala).

## Interpolação: linear ou cúbica

Comparação por ida e volta com sinais sintéticos de gabarito conhecido (média de 4 sinais por tipo, `src/synthetic.py`), depois de passar pelo pipeline de extração:

| sinal | método | correlação | erro relativo (NRMSE) | amplitude do pico (razão) |
|---|---|---|---|---|
| burst | linear | 0.981 | 0.193 | 1.01 |
| burst | cubica | 0.977 | 0.213 | 1.10 |
| sopro | linear | 0.970 | 0.242 | 1.01 |
| sopro | cubica | 0.965 | 0.270 | 1.09 |
| denso | linear | 0.780 | 0.674 | 1.01 |
| denso | cubica | 0.767 | 0.724 | 1.10 |

A linear erra menos e mantém a amplitude dos picos. A cúbica ultrapassa os picos em cerca de 10% e oscila ao redor deles. Por isso o padrão é `linear`, e `cubica` fica disponível como opção.

## Volume relativo entre os arquivos

Com a normalização por pico, os três arquivos têm o mesmo pico, e o volume relativo original se perde nos arquivos. Os níveis antes da normalização, em % da escala do gráfico, são devolvidos por `audio.exportar_wavs` e impressos pelo demo:

| arquivo | pico (%) | RMS (%) |
|---|---|---|
| normal.wav | 92,7 | 10,87 |
| still.wav | 54,9 | 8,51 |
| vsd.wav | 88,1 | 17,64 |

A Task 11 deve usar esses números ao comparar os sinais, não o nível dos .wav. Para gravar com volume relativo preservado, use `normalizacao="escala"`: 100% da escala do gráfico vira 0,95 do fundo de escala, com o mesmo ganho nos três arquivos. O arquivo still.wav ficaria então com pico de cerca de 0,52.

## Limites

- **Banda:** a figura dá 1333 colunas por segundo, então só o que está abaixo de uns 667 Hz é recuperável. Subir para 5 kHz interpola e não cria detalhe. O artigo grava a 8 kHz.
- **Painel c (VSD):** é o mais denso, e só o envelope da oscilação é recuperado (ver `docs/extracao_sinal.md`).
- **Onde estão os arquivos:** `python main.py` os grava em `outputs/audio/`, que o `.gitignore` não versiona. Os três arquivos entregues ficam versionados em `audio/` (cópia dos gerados), e `tests/test_audio_versionado.py` falha se o código mudar e a cópia não for atualizada.
