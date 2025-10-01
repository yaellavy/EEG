import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def clean_coeffs_2d(data):
    """Clean 2D coefficient data by removing NaN and infinite values"""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

def plot_stft_analysis(raw_data, ch_names, df, sfreq, valid_freqs, output_dir, 
                       figsize=(18, 12), window_seconds=2, overlap_ratio=0.5, figure_name="STFT_Analysis"):
    """
    Generate STFT plots for multiple EEG channels with task/rest annotations.
    """
    
    n_channels, n_samples = raw_data.shape
    n_plot_channels = min(n_channels, len(ch_names))
    
    # Calculate subplot grid
    if n_plot_channels <= 4:
        n_cols, n_rows = n_plot_channels, 1
    elif n_plot_channels <= 8:
        n_cols, n_rows = 4, 2
    elif n_plot_channels <= 12:
        n_cols, n_rows = 4, 3
    else:
        n_cols = 4
        n_rows = int(np.ceil(n_plot_channels / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_plot_channels == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Prepare event data
    start_of_part = np.round(df['Start_sec'].to_numpy(), 3)
    
    # STFT parameters
    window_length = int(window_seconds * sfreq)
    overlap = int(overlap_ratio * window_length)
    
    # First pass: compute all STFT data to find global range
    global_min, global_max = float('inf'), float('-inf')
    
    im = None  # Initialize 'im' to ensure it's in scope for the colorbar
    for i in range(n_plot_channels):
        freqs, times, stft_data = signal.stft(
            raw_data[i],
            fs=sfreq,
            window='boxcar',
            nperseg=window_length,
            noverlap=overlap,
            return_onesided=True )
        stft_data = clean_coeffs_2d(stft_data)
        stft_magnitude = np.abs(stft_data)#/len(freqs)
        stft_magnitude = 20 * np.log10(stft_magnitude + 1e-12)
        freq_mask = (freqs >= valid_freqs[0]) & (freqs <= valid_freqs[1])
        freqs_show = freqs[freq_mask]
        stft_show = stft_magnitude[freq_mask, :]
        
        global_min = stft_show.min()
        global_max = stft_show.max()
        ax = axes[i]
        
        # Create STFT plot, storing the last `im` object
        im = ax.imshow(stft_show, 
                       aspect='auto',
                       extent=[times[0], times[-1], freqs_show[0], freqs_show[-1]],
                       cmap='jet', 
                       interpolation='bilinear',
                       origin='lower',
                       vmin=global_min, 
                       vmax=global_max)
        
        # Add vertical lines at event boundaries
        for x in start_of_part:
            if x <= times[-1]:
                ax.axvline(x, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Add task/rest labels
        for idx, row in df.iterrows():
            start_time = row['Start_sec']
            section_type = row.get('Type', 'other')
            if section_type in ['task', 'rest'] and start_time <= times[-1]:
                label_time = start_time + 1
                y_pos = valid_freqs[1] * 0.9
                label = 'T' if section_type == 'task' else 'R'
                ax.text(label_time, y_pos, label,
                        color='black', fontsize=10, fontweight='bold',
                        ha='left', va='center')
        
        # Set subplot title and labels
        ax.set_title(f"{ch_names[i]}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(freqs_show[0], freqs_show[-1])
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("magnitude [dB]", rotation=270, labelpad=15)  # 270 makes it vertical

    # for i in range(n_plot_channels, len(axes)):
    #     axes[i].set_visible(False)
    
    # Add a single colorbar for all subplots 
    # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02)
    
    # Set main title
    # fig.suptitle(f'STFT Analysis', fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.4, hspace=0.3)
    
    # Save and show plot
    save_path = os.path.join(output_dir, f"{figure_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"STFT analysis plot saved to: {save_path}")
    # plt.show()
    
    return save_path