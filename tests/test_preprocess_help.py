import subprocess
from pathlib import Path

def test_preprocess_help():
    script = Path("scripts/preprocess.py")
    result = subprocess.run(["python", str(script), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
