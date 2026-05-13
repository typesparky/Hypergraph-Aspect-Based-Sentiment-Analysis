import os
import subprocess
import json
import pandas as pd

def run_experiment(dataset, embedding_path, overfit=True):
    cmd = [
        "/opt/anaconda3/envs/open_manus/bin/python", "train.py",
        "--dataset", dataset,
        "--embedding_path", embedding_path,
        "--epochs", "2" if overfit else "200",
        "--batch_size", "20"
    ]
    if overfit:
        cmd.append("--overfit_test")
    
    print(f"\n>>> Running Stage: {dataset} (Overfit: {overfit})")
    print(f">>> Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"!!! Error running {dataset}:")
        print(result.stderr)
        return None
    
    # Try to find the results file
    # results_{dataset}_seed42.json
    results_file = os.path.expanduser(f"~/Documents/Econometrics Courses/Thesis/results/results_{dataset}_seed42.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def main():
    data_dir = os.path.expanduser("~/Documents/Econometrics Courses/Thesis/data")
    haabsa_ref_dir = os.path.join(data_dir, "haabsa_ref")
    
    # Check for GloVe
    glove_path = os.path.join(data_dir, "glove.6B.300d.txt")
    if not os.path.exists(glove_path):
        print(f"WARNING: GloVe file not found at {glove_path}. Training will use random embeddings.")

    stages = [
        {
            "name": "haabsa16",
            "dataset": "haabsa16",
            "emb": glove_path,
            "overfit": True
        },
        {
            "name": "semeval14_rest",
            "dataset": "semeval14_rest",
            "emb": glove_path,
            "overfit": True
        },
        {
            "name": "semeval14_laptop",
            "dataset": "semeval14_laptop",
            "emb": glove_path,
            "overfit": True
        }
    ]
    
    all_results = []
    
    for stage in stages:
        res = run_experiment(stage["dataset"], stage["emb"], overfit=stage["overfit"])
        if res:
            all_results.append({
                "Stage": stage["name"],
                "Dataset": res["dataset"],
                "Test Acc": f"{res['test_accuracy']:.4f}",
                "Test F1": f"{res['test_f1_macro']:.4f}",
                "Epochs": res["best_epoch"]
            })
    
    if all_results:
        print("\n" + "="*60)
        print("PHASE 1: BASELINE REPRODUCTION SUMMARY")
        print("="*60)
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))
        print("="*60)
        
        # Save summary to file
        summary_path = os.path.expanduser("~/Documents/Econometrics Courses/Thesis/results/phase1_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
