import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def search_metrics_files(parent_dir, metric_folder):
    search_items = ["mpjpe_ra_r", "cd_icp", "cd_icp_no_scale", "f10_icp", "f10_icp_no_scale", "f5_icp", "f5_icp_no_scale", "scale", "cd_right"]
    results = {}
    
    for root, dirs, files in os.walk(parent_dir):
        if 'metric.json' in files and metric_folder in root:
            path_parts = Path(root).parts
            
            try:
                metrics_idx = path_parts.index(metric_folder)
                if metrics_idx >= 2:
                    method_dir = path_parts[metrics_idx-1]
                    sequence_dir = path_parts[metrics_idx-2]
                    
                    json_path = os.path.join(root, 'metric.json')
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            
                        if method_dir not in results:
                            results[method_dir] = {}
                            
                        metrics_dict = {}
                        for item in search_items:
                            # Round to 2 decimal places
                            value = data.get(item, "N/A")
                            if isinstance(value, (int, float)):
                                metrics_dict[item] = round(value, 2)
                            else:
                                metrics_dict[item] = "N/A"
                            
                        results[method_dir][sequence_dir] = metrics_dict
                            
                    except json.JSONDecodeError:
                        print(f"Error reading JSON file: {json_path}")
                    except Exception as e:
                        print(f"Error processing file {json_path}: {str(e)}")
            
            except ValueError:
                continue
    
    return results, search_items

def calculate_averages(df):
    df_numeric = df.replace('N/A', np.nan)
    means = df_numeric.apply(pd.to_numeric, errors='coerce').mean()
    means = means.apply(lambda x: '{:.2f}'.format(round(x, 2)) if pd.notnull(x) else 'N/A')
    return means

def natural_sort_key(s):
    import re
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', s)]

def create_results_file(results, search_items, output_path):
    sorted_methods = sorted(results.keys(), key=natural_sort_key)
    
    with open(output_path, 'w') as f:
        for method in sorted_methods:
            f.write(f"Table for {method}\n")
            
            df = pd.DataFrame.from_dict(results[method], orient='index')
            df = df[search_items]
            
            sorted_index = sorted(df.index, key=natural_sort_key)
            df = df.reindex(sorted_index)
            
            # Format float values to 2 decimal places
            table_str = df.to_string(
                float_format=lambda x: '{:.2f}'.format(round(float(x), 2)) if isinstance(x, (float, int)) else str(x)
            )
            
            header_line = table_str.split('\n')[0]
            col_positions = {}
            current_pos = len(df.index.name or '') + 2
            
            for col in df.columns:
                col_pos = header_line.find(col, current_pos)
                col_positions[col] = col_pos
                current_pos = col_pos + len(col)
            
            f.write(table_str)
            f.write("\n" + "-"*len(table_str.split('\n')[0]) + "\n")
            
            averages = calculate_averages(df)
            avg_line = "Average" + " "*(len(df.index.name or '') + 2)
            
            for col in df.columns:
                spaces_needed = col_positions[col] - len(avg_line)
                avg_line += " "*spaces_needed + f"{averages[col]}"
            
            f.write(avg_line)
            f.write("\n\n" + "="*80 + "\n\n")

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Copy and rename model.obj files to export_meshes directory.')
    parser.add_argument(
        '--parent_dir',
        type=str,
        required=True,
        help='Parent directory containing sequence and method subdirectories'
    )
    parser.add_argument(
        '--metric_folder',
        type=str,
        required=True,
        help='metric folder name'
    )    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='output folder name'
    )      
    return parser.parse_args()
    
def main():
    # parent_dir = "outputs_Nov13_256_res_to_128"
    args = parse_args()
    parent_dir = args.parent_dir
    output_file = f"{args.out_dir}/{args.metric_folder}_results.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results, search_items = search_metrics_files(parent_dir, args.metric_folder)
    create_results_file(results, search_items, output_file)
    
    print(f"Results have been written to {output_file}")

if __name__ == "__main__":
    main()
