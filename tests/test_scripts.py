from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_record_and_process_files_exist():
    assert (ROOT / "run.py").is_file()
    assert (ROOT / "record.py").is_file()
    assert (ROOT / "process.py").is_file()
    assert (ROOT / "simulate.py").is_file()


def test_record_help():
    r = subprocess.run(
        [sys.executable, str(ROOT / "record.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "--table-height" in r.stdout
    assert "--out" in r.stdout


def test_process_help():
    r = subprocess.run(
        [sys.executable, str(ROOT / "process.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "--run" in r.stdout
    assert "--detector" in r.stdout
    assert "--no-cloud" in r.stdout
    assert "--cloud-stride" in r.stdout


def test_run_help():
    r = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "--table-height" in r.stdout
    assert "--duration" in r.stdout
    assert "--simulate" in r.stdout
