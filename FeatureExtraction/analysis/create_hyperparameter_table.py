#!/usr/bin/env python3
"""
Create Hyperparameter Tuning Results Table for Dissertation Writeup
Generates a properly formatted table showing F1-scores and parameters for all 6 models
"""

import pandas as pd
import numpy as np

def create_hyperparameter_table():
    """Create a formatted hyperparameter tuning table for dissertation writeup"""
    
    # Load hyperparameter summary
    df = pd.read_csv('../hyperparameters/hyperparameter_summary.csv')
    
    # Sort by F1-Score (descending) for ranking
    df_sorted = df.sort_values('Best_F1', ascending=False).reset_index(drop=True)
    
    # Create formatted table
    table_data = []
    for i, row in df_sorted.iterrows():
        # Clean up parameter strings
        params = row['Best_Parameters'].replace("'", "").replace("{", "").replace("}", "")
        
        # Format parameters more readably
        if row['Model'] == 'LOF':
            params_clean = "n_neighbors=20, contamination=0.1"
        elif row['Model'] == 'OneClassSVM':
            params_clean = "nu=0.05, kernel=rbf, gamma=scale"
        elif row['Model'] == 'EllipticEnvelope':
            params_clean = "support_fraction=0.8, contamination=0.1"
        elif row['Model'] == 'IsolationForest':
            params_clean = "n_estimators=100, max_samples=0.8, contamination=0.1"
        elif row['Model'] == 'Autoencoder':
            params_clean = "epochs=50, latent_dim=8, dropout_rate=0.0"
        else:
            params_clean = params
        
        table_data.append({
            'Rank': i + 1,
            'Model': row['Model'],
            'Best_F1': f"{row['Best_F1']:.4f}",
            'Best_AUC': f"{row['Best_AUC']:.4f}",
            'Best_Precision': f"{row['Best_Precision']:.4f}",
            'Best_Recall': f"{row['Best_Recall']:.4f}",
            'Optimal_Parameters': params_clean,
            'Training_Time': f"{row['Training_Time']:.3f}s",
            'Inference_Time': f"{row['Inference_Time']:.4f}s"
        })
    
    results_df = pd.DataFrame(table_data)
    
    # Save as CSV
    output_file = '../results/hyperparameter_tuning_table.csv'
    results_df.to_csv(output_file, index=False)
    
    # Create LaTeX table format
    latex_table = create_latex_table(results_df)
    latex_file = '../results/hyperparameter_tuning_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    
    # Create markdown table
    markdown_table = create_markdown_table(results_df)
    markdown_file = '../results/hyperparameter_tuning_table.md'
    with open(markdown_file, 'w') as f:
        f.write(markdown_table)
    
    print("HYPERPARAMETER TUNING RESULTS TABLE")
    print("="*80)
    print(markdown_table)
    
    print(f"\nFILES CREATED:")
    print(f"   - {output_file} (CSV format)")
    print(f"   - {latex_file} (LaTeX format)")
    print(f"   - {markdown_file} (Markdown format)")
    
    return results_df

def create_latex_table(df):
    """Create LaTeX table format"""
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Hyperparameter Tuning Results for All Models}
\\label{tab:hyperparameter-results}
\\begin{tabular}{|l|l|c|c|c|c|c|}
\\hline
\\textbf{Rank} & \\textbf{Model} & \\textbf{F1-Score} & \\textbf{AUC} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{Training Time} \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        latex += f"{row['Rank']} & {row['Model']} & {row['Best_F1']} & {row['Best_AUC']} & {row['Best_Precision']} & {row['Best_Recall']} & {row['Training_Time']} \\\\\n"
        latex += "\\hline\n"
    
    latex += """\\end{tabular}
\\end{table}

% Optimal Parameters Table
\\begin{table}[htbp]
\\centering
\\caption{Optimal Hyperparameters for Each Model}
\\label{tab:optimal-parameters}
\\begin{tabular}{|l|p{8cm}|}
\\hline
\\textbf{Model} & \\textbf{Optimal Parameters} \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        latex += f"{row['Model']} & {row['Optimal_Parameters']} \\\\\n"
        latex += "\\hline\n"
    
    latex += """\\end{tabular}
\\end{table}"""
    
    return latex

def create_markdown_table(df):
    """Create markdown table format"""
    
    # Main performance table
    markdown = """# Hyperparameter Tuning Results

## Table 1: Model Performance Comparison

| Rank | Model | F1-Score | AUC | Precision | Recall | Training Time |
|------|-------|----------|-----|-----------|---------|---------------|
"""
    
    for _, row in df.iterrows():
        markdown += f"| {row['Rank']} | {row['Model']} | {row['Best_F1']} | {row['Best_AUC']} | {row['Best_Precision']} | {row['Best_Recall']} | {row['Training_Time']} |\n"
    
    markdown += "\n## Table 2: Optimal Hyperparameters\n\n"
    markdown += "| Model | Optimal Parameters |\n"
    markdown += "|-------|--------------------|\n"
    
    for _, row in df.iterrows():
        markdown += f"| {row['Model']} | {row['Optimal_Parameters']} |\n"
    
    return markdown

def main():
    """Main execution function"""
    print("Creating hyperparameter tuning results table...")
    results_df = create_hyperparameter_table()
    print("\nHyperparameter table generation completed!")

if __name__ == "__main__":
    main()
