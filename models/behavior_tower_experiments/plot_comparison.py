import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_comparisons(result_dir):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(result_dir, "*.csv"))
    
    data = {}
    
    # Read each CSV file
    for file in csv_files:
        filename = os.path.basename(file)
        # Extract model name from filename (remove _50epochs.csv)
        model_name = filename.replace('_50epochs.csv', '')
        
        try:
            df = pd.read_csv(file)
            data[model_name] = df
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
            
    if not data:
        print("No valid CSV files found.")
        return

    # Create output directory for plots
    plot_dir = os.path.join(result_dir, "comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot settings
    metrics = {
        'avg_loss': 'Average Loss', 
        'recall@10': 'Recall@10', 
        'ndcg@10': 'NDCG@10'
    }
    
    # Generate plots for each metric
    for metric_col, metric_name in metrics.items():
        plt.figure(figsize=(12, 8))
        
        for model_name, df in data.items():
            if metric_col in df.columns:
                # Sort by epoch just in case
                df_sorted = df.sort_values('epoch')
                plt.plot(df_sorted['epoch'], df_sorted[metric_col], marker='o', label=model_name, linewidth=2, markersize=4)
            else:
                print(f"Warning: {metric_col} not found in {model_name}")
        
        plt.title(f'{metric_name} Comparison over Epochs', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = os.path.join(plot_dir, f'comparison_{metric_col}.png')
        plt.savefig(output_path, dpi=300)
        print(f"Saved {metric_name} plot to {output_path}")
        plt.close()

if __name__ == "__main__":
    result_dir = r"f:\coldstart-recsys\models\behavior_tower_experiments\result"
    plot_comparisons(result_dir)
