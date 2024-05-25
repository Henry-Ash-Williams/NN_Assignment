import os
import sys
import subprocess
from subprocess import CalledProcessError

# config = {"num_runs": 5, "learning_rates": [1e-4, 0.0026, 0.0051, 0.0077, 0.01]}
config = {"num_runs": 5, "learning_rates": [1e-4]}


def is_venv():
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def run_training(lr, run):
    if not is_venv():
        print("Make sure this program is running in the correct virtual environment")
        sys.exit(-1)


    subprocess.run(
        [
            "python",
            "src/train.py",
            "--epochs=20",
            f"--learning-rate={lr}",
            f"--experiment-name=lr-{lr} without relu on fc layers using lr scheduler",
            f"--run-id={run}",
            f"--run-name=E1-LR-{lr}-NO_RELU-LR",
            "--save=both",
            "--tag=LR_Scheduler",
            "--use-lr-scheduler=true"
        ]
    ).check_returncode()



if __name__ == "__main__":
    for lr in config["learning_rates"]:
        for run in range(config["num_runs"]):
            print(f"Run {run + 1}/{config['num_runs']} with lr: {lr}")
            try: 
                run_training(lr, run)
            except CalledProcessError: 
                print("Process failed!")
