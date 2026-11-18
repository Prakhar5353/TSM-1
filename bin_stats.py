import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_swarm_diagnostics():
    # Updated glob pattern to match folders like 'global_output_vornoi_19_1sigma'
    json_files = glob.glob("global_output_vornoi_*sigma/global_phase2_summary.json")
    
    if not json_files:
        print("No 'global_phase2_summary.json' files found in any matching subdirectories.")
        return

    data_points = []
    
    for j_file in json_files:
        with open(j_file, 'r') as f:
            data = json.load(f)
            
        n_bins = data['target_seeds']
        target_events = data['target_events']
        agg_sig = data['final_aggregate_significance']
        pops = np.array(list(data['bin_populations'].values()))
        
        data_points.append({
            'n_bins': n_bins,
            'agg_sig': agg_sig,
            'fractional_pops': pops / target_events 
        })

    # Sort by number of superbins
    data_points.sort(key=lambda x: x['n_bins'])

    x_n_bins = [dp['n_bins'] for dp in data_points]
    y_agg_sig = [dp['agg_sig'] for dp in data_points]
    y_pop_distributions = [dp['fractional_pops'] for dp in data_points]

    # Create the Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Global Phase 2 Diagnostics: Box + Swarm Overlay', fontsize=16, fontweight='bold')

    # Plot 1: Aggregate Significance vs N_bins
    ax1.plot(x_n_bins, y_agg_sig, marker='o', linestyle='-', color='indigo', markersize=8, linewidth=2)
    ax1.set_title('Total Physics Sensitivity', fontsize=14)
    ax1.set_xlabel('Target Number of Superbins (N)', fontsize=12)
    ax1.set_ylabel('Final Aggregate Significance (\u03A3 S)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xticks(x_n_bins)

    # Plot 2: Box Plot with Swarm Overlay
    boxprops = dict(linewidth=2, color='gray', alpha=0.5)
    medianprops = dict(linewidth=2, color='crimson', alpha=0.8)
    whiskerprops = dict(linewidth=1.5, color='gray', alpha=0.5)
    capprops = dict(linewidth=1.5, color='gray', alpha=0.5)
    
    ax2.boxplot(y_pop_distributions, positions=x_n_bins, widths=0.5, 
                showfliers=False,
                boxprops=boxprops, medianprops=medianprops, 
                whiskerprops=whiskerprops, capprops=capprops, zorder=1)
    
    # Overlay the scatter (Swarm)
    for x_val, fractional_pops in zip(x_n_bins, y_pop_distributions):
        jitter = np.random.uniform(-0.15, 0.15, size=len(fractional_pops))
        x_jittered = x_val + jitter
        ax2.scatter(x_jittered, fractional_pops, color='teal', edgecolor='black', 
                    alpha=0.75, s=40, zorder=2)

    ax2.axhline(1.0, color='crimson', linestyle='--', linewidth=2, label='Perfect Target Equality (1.0)', zorder=0)
    
    ax2.set_title('Superbin Population Distributions', fontsize=14)
    ax2.set_xlabel('Target Number of Superbins (N)', fontsize=12)
    ax2.set_ylabel('Fraction of Target Events (Pop / Target)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.set_xticks(x_n_bins)
    
    plt.tight_layout()
    output_filename = 'superbin_diagnostics_swarm.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nSuccessfully processed {len(json_files)} runs.")
    print(f"Box+Swarm plot diagnostics saved to '{output_filename}'.")

if __name__ == "__main__":
    generate_swarm_diagnostics()