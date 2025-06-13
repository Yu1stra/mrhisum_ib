#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Base directory containing all the SMI result files
base_dir = Path('/home/jay/MR.HiSum/Summaries/IB/SL_module/multi/VsameA_cmib_SMI/mr')

# Function to parse SMI_result.txt file
def parse_smi_file(file_path):
    try:
        # Read the file and convert to dataframe
        data = pd.read_csv(file_path, delim_whitespace=True)
        return data
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

# Function to create a plot for a specific SMI result file
def plot_smi_file(file_path, output_dir):
    data = parse_smi_file(file_path)
    if data is None:
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot each SMI column
    if 'v_SMI' in data.columns:
        plt.plot(data['Epoch'], data['v_SMI'], label='v_SMI', marker='o', markersize=3)
    if 'a_SMI' in data.columns:
        plt.plot(data['Epoch'], data['a_SMI'], label='a_SMI', marker='s', markersize=3)
    if 'm_SMI' in data.columns:
        plt.plot(data['Epoch'], data['m_SMI'], label='m_SMI', marker='^', markersize=3)
    
    # Add title and labels
    relative_path = os.path.relpath(file_path, base_dir)
    plt.title(f"SMI Results - {relative_path}")
    plt.xlabel('Epoch')
    plt.ylabel('SMI Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    output_path = output_dir / f"{relative_path.replace('/', '_')}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    return output_path

# Function to find all SMI_result.txt files
def find_smi_files():
    smi_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'SMI_result.txt' and 'logs' in root:
                smi_files.append(os.path.join(root, file))
    return smi_files

# Create output directory for plots
output_dir = Path('/home/jay/MR.HiSum/Summaries/smi_plots')
os.makedirs(output_dir, exist_ok=True)

# Find all SMI result files
smi_files = find_smi_files()
print(f"Found {len(smi_files)} SMI result files")

# Create plots for each file
output_files = []
for file_path in smi_files:
    output_file = plot_smi_file(file_path, output_dir)
    if output_file:
        output_files.append(output_file)
        print(f"Created plot: {output_file}")

# Create a combined plot with one line for each v_SMI value
def create_combined_plot():
    plt.figure(figsize=(12, 8))
    
    for file_path in smi_files:
        data = parse_smi_file(file_path)
        if data is None:
            continue
        
        # Get directory parameters from path
        path_parts = file_path.split('/')
        vbeta = next((p for p in path_parts if p.startswith('vbeta_')), 'unknown')
        abeta = next((p for p in path_parts if p.startswith('abeta_')), 'unknown')
        mbeta = next((p for p in path_parts if p.startswith('mbeta_')), 'unknown')
        
        label = f"{vbeta}/{abeta}/{mbeta}"
        
        # Plot v_SMI
        if 'v_SMI' in data.columns:
            plt.plot(data['Epoch'], data['v_SMI'], label=f"v_SMI - {label}", alpha=0.7)
    
    plt.title("Combined v_SMI Results Across Different Configurations")
    plt.xlabel('Epoch')
    plt.ylabel('v_SMI Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / "combined_v_smi.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Created combined plot: {output_path}")
    plt.clf()

    # Similarly for a_SMI and m_SMI
    for smi_type in ['a_SMI', 'm_SMI']:
        plt.figure(figsize=(12, 8))
        
        for file_path in smi_files:
            data = parse_smi_file(file_path)
            if data is None or smi_type not in data.columns:
                continue
            
            # Get directory parameters from path
            path_parts = file_path.split('/')
            vbeta = next((p for p in path_parts if p.startswith('vbeta_')), 'unknown')
            abeta = next((p for p in path_parts if p.startswith('abeta_')), 'unknown')
            mbeta = next((p for p in path_parts if p.startswith('mbeta_')), 'unknown')
            
            label = f"{vbeta}/{abeta}/{mbeta}"
            
            plt.plot(data['Epoch'], data[smi_type], label=f"{smi_type} - beta = {vbeta}", alpha=0.7)
        
        plt.title(f"Combined {smi_type} Results Across Different Configurations")
        plt.xlabel('Epoch')
        plt.ylabel(f'{smi_type} Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the figure
        output_path = output_dir / f"combined_{smi_type.lower()}.png"
        plt.savefig(output_path, dpi=300)
        plt.show())
        print(f"Created combined plot: {output_path}")
        plt.clf()

# Create combined plots
create_combined_plot()

print(f"\nAll plots have been saved to {output_dir}")
