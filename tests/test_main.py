"""Testes da linha de comando (main.py)."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import main
from src import audio

RAIZ = Path(__file__).resolve().parent.parent


def test_main_gera_todos_os_arquivos(tmp_path):
    assert main.main(["--saida", str(tmp_path)]) == 0
    for nome in ("normal.wav", "still.wav", "vsd.wav"):
        w = audio.ler_wav(tmp_path / "audio" / nome)
        assert (w.taxa, w.bits, w.canais, w.n_amostras) == (5000, 16, 1, 2500)
    assert (tmp_path / "resumo.json").exists()
    assert len(list((tmp_path / "etapas").glob("*.png"))) == 25      # 8 por painel + as ROIs detectadas


def test_main_com_figura_inexistente_devolve_erro_e_explica(tmp_path, capsys):
    codigo = main.main(["--figura", str(tmp_path / "nada.png"), "--saida", str(tmp_path / "s")])
    assert codigo == 1
    assert "ERROR" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()


def test_main_com_imagem_sem_curva_devolve_erro(tmp_path, capsys):
    branca = tmp_path / "branca.png"
    cv2.imwrite(str(branca), np.full((867, 707, 3), 255, dtype=np.uint8))
    codigo = main.main(["--figura", str(branca), "--saida", str(tmp_path / "s")])
    assert codigo == 1
    assert "não foi possível extrair o sinal" in capsys.readouterr().err


def test_modo_silencioso_nao_imprime_etapas(tmp_path, capsys):
    assert main.main(["--silencioso", "--saida", str(tmp_path)]) == 0
    assert capsys.readouterr().err == ""
    main.main(["--saida", str(tmp_path)])
    assert "pronto em" in capsys.readouterr().err


def test_python_main_py_roda_do_zero_em_um_processo_novo(tmp_path):
    saida = tmp_path / "novo" / "dentro"          # a pasta não existe ainda
    r = subprocess.run([sys.executable, "main.py", "--saida", str(saida)], cwd=RAIZ,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "pronto em" in r.stderr
    assert sorted(p.name for p in (saida / "audio").iterdir()) == ["normal.wav", "still.wav", "vsd.wav"]


def test_parser_tem_os_padroes_do_projeto():
    args = main.criar_parser().parse_args([])
    assert Path(args.figura).name == "Figure1.png"
    assert Path(args.saida).name == "outputs"
    assert args.silencioso is False
