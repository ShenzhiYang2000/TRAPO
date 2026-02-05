import wandb
import pandas as pd
import matplotlib.pyplot as plt

def load_wandb_run(run_path):
  api = wandb.Api()
  run = api.from_path(run_path)

  history = pd.DataFrame(run.history())
  return history

def save_csv(history, csv_path):
  history.to_csv(csv_path, index=False)
  print(f'csv saved to {csv_path}')

if __name__ == "__main__":
  run_path = "/ossfs/workspace/aml0/484999/code/LUFFY/wandb/wandb/offline-run-20250821_135955-med8ay1m"

  # history = load_wandb_run(run_path)
  save_csv(history, "/ossfs/workspace/aml0/484999/code/LUFFY/wandb/wandb/csv/o1k_d1k_unsupervised.csv")

