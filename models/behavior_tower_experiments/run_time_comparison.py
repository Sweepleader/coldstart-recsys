import subprocess
import re
import sys
import os

def run_experiment(model_name, epochs=10, fusion_type=None):
    print(f"Running experiment for model: {model_name}" + (f" with fusion: {fusion_type}" if fusion_type else "") + "...")
    cmd = [
        sys.executable,
        "models/behavior_tower_experiments/train_gru_rec.py",
        "--model", model_name,
        "--epochs", str(epochs)
    ]
    if fusion_type:
        cmd.extend(["--fusion_type", fusion_type])
    
    # Run command and capture output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, # Merge stderr to stdout
        cwd="f:\\coldstart-recsys", 
        text=True,
        bufsize=1 # Line buffered
    )
    
    last_metrics = None
    
    # Real-time output processing
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip()) # Print to stdout so we can see it
            if "Epoch" in line and "HR@10" in line:
                last_metrics = line.strip()
                
    return last_metrics

def parse_metrics(metric_str):
    if not metric_str:
        return {}
    # Extract values
    hr = re.search(r"HR@10 ([\d\.]+)", metric_str)
    recall = re.search(r"Recall@10 ([\d\.]+)", metric_str)
    ndcg = re.search(r"NDCG@10 ([\d\.]+)", metric_str)
    
    return {
        "HR@10": float(hr.group(1)) if hr else 0.0,
        "Recall@10": float(recall.group(1)) if recall else 0.0,
        "NDCG@10": float(ndcg.group(1)) if ndcg else 0.0
    }

def main():
    epochs = 1
    print(f"Starting comparison experiment (Epochs={epochs})...")
    
    configs = [
        {"name": "SASRec", "model": "sasrec", "fusion": None},
        {"name": "TimeSASRec(Concat)", "model": "time_sasrec", "fusion": "concat"},
        {"name": "TimeSASRec(Add)", "model": "time_sasrec", "fusion": "add"},
        {"name": "TimeSASRec(Bias)", "model": "time_sasrec", "fusion": "attention_bias"},
    ]
    
    results = {}
    
    for config in configs:
        output = run_experiment(config["model"], epochs, config["fusion"])
        metrics = parse_metrics(output)
        results[config["name"]] = metrics
    
    # Print Comparison
    print("\n" + "="*80)
    print(f"{'Metric':<12} | {'SASRec':<10} | {'Concat':<10} | {'Add':<10} | {'Bias':<10}")
    print("-" * 80)
    
    metrics_list = ["HR@10", "Recall@10", "NDCG@10"]
    sasrec_metrics = results["SASRec"]
    
    for m in metrics_list:
        row = f"{m:<12} | "
        
        # SASRec baseline
        base_val = sasrec_metrics.get(m, 0.0)
        row += f"{base_val:.4f}     | "
        
        # Others
        for name in ["TimeSASRec(Concat)", "TimeSASRec(Add)", "TimeSASRec(Bias)"]:
            val = results[name].get(m, 0.0)
            diff = (val - base_val) / base_val * 100 if base_val > 0 else 0.0
            row += f"{val:.4f} ({diff:+.1f}%) | "
            
        print(row.strip(" |"))
        
    print("="*80)

if __name__ == "__main__":
    main()
