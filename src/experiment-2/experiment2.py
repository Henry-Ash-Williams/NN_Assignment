import os
import sys
import subprocess
from subprocess import DEVNULL, CalledProcessError
from tqdm import tqdm

config = {
    "num_runs": 5,
    "dropout_rates": [0, 0.0625, 0.125, 0.1875, 0.25], 
}

def is_venv():
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

def run_train(dropout_rate, run):
    if not is_venv():
        print("Make sure this program is running in the correct virtual environment")
        sys.exit(-1)


    subprocess.run(
        [
            "python",
            "src/train.py",
            f"--dropout-rate={dropout_rate}",
            f"--experiment-name=dropout-{dropout_rate}",
            f"--run-id={run}",
            f"--run-name=E2-DO-{dropout_rate}-R{i + 1}",
            "--save=both",
            "--tag=E2P1",
            "--experiment-id=2",
            "--model=ImageClassifierWithDropout",
        ], 
        stdout=DEVNULL,
        stderr=DEVNULL
    ).check_returncode()

if __name__ == "__main__":
    for i in tqdm(range(config['num_runs']), position=0):
        for dropout in tqdm(config['dropout_rates'], position=1):
            try: 
                run_train(dropout, i)
            except CalledProcessError: 
                print(f"Failed, run {i + 1} / {config['num_runs']} dropout = {dropout}")
