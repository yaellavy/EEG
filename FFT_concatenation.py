"""
FFT_concatenation.py
Module for generating concatenated FFT plots per segment type
Compatible with the main EEG analysis pipeline
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
import pandas as pd
import os


def plot_concatenated_fft_segments(raw_data, df, ch_names, sfreq, valid_freqs, output_dir, 
                                 selected_channels=None, apply_windowing=True,file_name='fft_concatenated_segments'):
    """
    Generate concatenated FFT plots per segment type for EEG analysis pipeline.
    
    Parameters:
    -----------
    raw_data : np.ndarray, shape (n_channels, n_samples)
        Raw EEG data
    df : pd.DataFrame
        Event dataframe with columns: Start_sec, Duration, Type, Section
    ch_names : list
        List of channel names
    sfreq : float
        Sampling frequency in Hz
    valid_freqs : list
        [min_freq, max_freq] frequency range for analysis
    output_dir : str
        Output directory for saving plots
    selected_channels : list, optional
        Specific channels to plot. If None, plots all channels
    apply_windowing : bool, default=True
        Whether to apply Hanning window to reduce spectral leakage
        
    Returns:
    --------
    str : Path to saved plot file
    dict : Summary statistics of the analysis
    """
    
    print("Generating concatenated FFT plots per segment...")
    
    # Validate inputs
    if not isinstance(valid_freqs, (list, tuple)) or len(valid_freqs) != 2:
        raise ValueError("valid_freqs must be [min_freq, max_freq]")
    
    valid_range = valid_freqs[1] - valid_freqs[0]
    
    # Filter channels if specified
    if selected_channels:
        channel_indices = [ch_names.index(ch) for ch in selected_channels if ch in ch_names]
        plot_ch_names = selected_channels
    else:
        channel_indices = list(range(len(ch_names)))
        plot_ch_names = ch_names
    
    if not channel_indices:
        print("Warning: No valid channels found for FFT analysis")
        return None, {}
    
    # Setup subplot configuration
    n_channels = len(channel_indices)
    n_cols = min(4, n_channels)
    n_rows = int(np.ceil(n_channels / n_cols))
    
    fig_fft, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle single subplot case
    if n_channels == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = list(axes)
    elif n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # fig_fft.suptitle("FFT Magnitude per Segment", fontsize=16, y=0.98)
    
    # Enhanced color scheme for segment types
    segment_colors = {
        'task': "#270D97",      # Vibrant red
        'rest': "#BC1919",      # Teal
        'other': "#0DA208",     # Blue
        'baseline': '#96CEB4',   # Light green
        'stimulus': '#FFEAA7',   # Light yellow
        'default': '#95A5A6'     # Gray
    }
    
    # Analysis statistics
    analysis_stats = {
        'total_segments': 0,
        'processed_segments': 0,
        'skipped_segments': 0,
        'channels_analyzed': plot_ch_names,
        'frequency_range': valid_freqs,
        'dominant_frequencies': {}
    }
    
    # Process each channel
    for plot_idx, ch_idx in enumerate(channel_indices):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        ax.set_title(f"{plot_ch_names[plot_idx]}", fontweight='bold', fontsize=12)
        ax.set_xlabel("frequencies (Hz)")
        ax.set_ylabel("FFT Magnitude")
        
        x_offset = 0
        segment_stats = []
        
        # Filter valid segments
        valid_segments = df[
            df['Duration'].notna() & 
            (df['Duration'] > 0) & 
            (df['Start_sec'].notna())
        ].copy()
        
        analysis_stats['total_segments'] = len(valid_segments)
        
        for idx, row in valid_segments.iterrows():
            try:
                segment_type = row.get('Type', 'other').lower()
                if segment_type=='other':
                    continue
                color = segment_colors.get(segment_type, segment_colors['default'])
                
                # Calculate segment boundaries
                start_sec = float(row['Start_sec'])
                duration = float(row['Duration'])
                start_idx = int(start_sec * sfreq)
                end_idx = int((start_sec + duration) * sfreq)
                
                # Validate segment bounds
                if end_idx > raw_data.shape[1] or start_idx < 0 or start_idx >= end_idx:
                    print(f"Warning: Segment {idx} out of bounds, skipping...")
                    analysis_stats['skipped_segments'] += 1
                    continue
                
                # Extract segment data
                segment = raw_data[ch_idx, start_idx:end_idx]
                
                # Clean data: remove non-finite values
                finite_mask = np.isfinite(segment)
                if not np.any(finite_mask):
                    print(f"Warning: Segment {idx} contains no finite values, skipping...")
                    analysis_stats['skipped_segments'] += 1
                    continue
                
                segment = segment[finite_mask]
                
                # Check minimum segment length
                if len(segment) < 32:  # Minimum for meaningful FFT
                    print(f"Warning: Segment {idx} too short ({len(segment)} samples), skipping...")
                    analysis_stats['skipped_segments'] += 1
                    continue
                
                # Apply windowing to reduce spectral leakage
                if apply_windowing and len(segment) > 1:
                    window = np.hanning(len(segment))
                    segment = segment * window
                
                # Compute FFT
                freqs = rfftfreq(len(segment), 1 / sfreq)
                fft_vals = np.abs(rfft(segment))/ len(segment)
                
                # Apply frequency filtering
                valid_mask = (freqs >= valid_freqs[0]) & (freqs <= valid_freqs[1])
                freqs_filtered = freqs[valid_mask]
                fft_vals_filtered = fft_vals[valid_mask]
                
                if len(fft_vals_filtered) == 0:
                    analysis_stats['skipped_segments'] += 1
                    continue
                
                # Plot FFT with frequency offset
                plot_freqs = freqs_filtered - valid_freqs[0] + x_offset
                ax.plot(plot_freqs, fft_vals_filtered, color=color, alpha=0.7, 
                       linewidth=1.2, label=f"{segment_type.title()}" if x_offset == 0 else "")
                
                # Add background shading for segment
                ax.axvspan(x_offset, x_offset + valid_range, 
                          color=color, alpha=0.15, zorder=0)
                
                # Find dominant frequency
                if len(fft_vals_filtered) > 0:
                    max_idx = np.argmax(fft_vals_filtered)
                    max_freq = freqs_filtered[max_idx]
                    max_magnitude = fft_vals_filtered[max_idx]
                    
                    # Calculate modulo frequency for cyclic analysis
                    max_freq_mod = (max_freq - valid_freqs[0]) % valid_range + valid_freqs[0]
                    
                    # Mark dominant frequency
                    ax.plot(plot_freqs[max_idx], max_magnitude, 'o', 
                           color='darkred', markersize=3, zorder=5)
                    ax.set_xticklabels([])
                    # ax.set_yscale('log')                # log scale for y-axis
                    # ax.set_ylim(bottom=1e-3)            # avoid log(0)
                    ax.grid(axis='y', alpha=0.3, which='both') 
                    # Add annotations
                    mid_x = x_offset + valid_range / 2
                    y_max = np.max(fft_vals_filtered) * 1.1
                    
                    # Segment type label
                    # ax.text(mid_x, y_max * 0.9, segment_type[0].upper(), 
                    #        color="black", ha='center', va='center', fontsize=10, 
                    #        fontweight='bold',
                    #        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", 
                    #                edgecolor=color, alpha=0.8))
                    
                    # Dominant frequency label
                    value = max_freq_mod + valid_freqs[0]
                    ax.text(mid_x, y_max * 0.7, f"{value:.1f}",
                           color="darkred", ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", 
                                   alpha=0.8))
                    
                    # Store statistics
                    segment_stats.append({
                        'type': segment_type,
                        'dominant_freq': max_freq_mod,
                        'max_magnitude': max_magnitude,
                        'segment_length': len(segment)
                    })
                
                # Update offset for next segment
                x_offset += valid_range
                analysis_stats['processed_segments'] += 1
                
            except Exception as e:
                print(f"Error processing segment {idx}: {str(e)}")
                analysis_stats['skipped_segments'] += 1
                continue
        
        # Finalize axis properties
        if segment_stats:
            # Set y-axis limits
            all_magnitudes = [s['max_magnitude'] for s in segment_stats]
            y_max = max(all_magnitudes) * 1.15
            ax.set_ylim(0, y_max)
            
            # Store channel statistics
            analysis_stats['dominant_frequencies'][plot_ch_names[plot_idx]] = segment_stats
        
        # Styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, x_offset if x_offset > 0 else valid_range)
        
        # Add legend to first subplot only
        # if plot_idx == 0 and segment_stats:
        unique_types = list(set([s['type'] for s in segment_stats]))
        legend_elements = [
            plt.Line2D([0], [0], color=segment_colors.get(t, segment_colors['default']), 
                lw=2, label=t.title()) for t in unique_types
            ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_channels, len(axes)):
        if idx < len(axes):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    fft_concat_path = os.path.join(output_dir, f"{file_name}.png")
    fig_fft.savefig(fft_concat_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    print(f"Saved concatenated FFT plot to: {fft_concat_path}")
    print(f"Analysis complete: {analysis_stats['processed_segments']}/{analysis_stats['total_segments']} segments processed")
    
    # Save analysis summary
    summary_path = save_fft_analysis_summary(output_dir, analysis_stats)
    
    # plt.show()
    
    return fft_concat_path, analysis_stats


def save_fft_analysis_summary(output_dir, stats):
    """Save detailed analysis summary to file"""
    try:
        summary_path = os.path.join(output_dir, "fft_concatenated_analysis_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("FFT Concatenated Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total segments in dataset: {stats['total_segments']}\n")
            f.write(f"Successfully processed: {stats['processed_segments']}\n")
            f.write(f"Skipped segments: {stats['skipped_segments']}\n")
            f.write(f"Success rate: {stats['processed_segments']/max(stats['total_segments'], 1)*100:.1f}%\n\n")
            
            f.write(f"Frequency range analyzed: {stats['frequency_range'][0]}-{stats['frequency_range'][1]} Hz\n")
            f.write(f"Channels analyzed: {', '.join(stats['channels_analyzed'])}\n\n")
            
            # Dominant frequency analysis per channel
            f.write("Dominant Frequencies by Channel:\n")
            f.write("-" * 30 + "\n")
            
            for channel, segments in stats['dominant_frequencies'].items():
                f.write(f"\n{channel}:\n")
                
                # Group by segment type
                by_type = {}
                for seg in segments:
                    seg_type = seg['type']
                    if seg_type not in by_type:
                        by_type[seg_type] = []
                    by_type[seg_type].append(seg['dominant_freq'])
                
                for seg_type, freqs in by_type.items():
                    avg_freq = np.mean(freqs)
                    std_freq = np.std(freqs)
                    f.write(f"  {seg_type.title()}: {avg_freq:.2f} ± {std_freq:.2f} Hz (n={len(freqs)})\n")
        
        print(f"Saved FFT analysis summary to: {summary_path}")
        return summary_path
        
    except Exception as e:
        print(f"Warning: Could not save analysis summary: {e}")
        return None
