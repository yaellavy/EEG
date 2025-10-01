import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

def clean_coeffs_2d(data):
    """Clean 2D coefficient data by removing NaN and infinite values"""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

def reclassify_other_as_rest(df, rest_duration_threshold=None):
    """
    Reclassify 'other' events as 'rest' if their duration is similar to rest events.
    
    Args:
        df: DataFrame with events
        rest_duration_threshold: If None, use median rest duration as threshold
    
    Returns:
        Modified DataFrame
    """
    df_modified = df.copy()
    
    # Get rest durations for threshold calculation
    rest_durations = df_modified[df_modified['Type'] == 'rest']['Duration'].tolist()
    
    if not rest_durations:
        print("No rest events found for threshold calculation")
        return df_modified
    
    if rest_duration_threshold is None:
        # Use median rest duration as threshold (with some tolerance)
        median_rest_duration = np.median(rest_durations)
        rest_duration_threshold = median_rest_duration * 0.8  # 80% of median as threshold
    
    print(f"Rest duration threshold: {rest_duration_threshold:.1f}s")
    
    # Reclassify 'other' events that meet the duration criterion
    other_mask = (df_modified['Type'] == 'other') & (df_modified['Duration'] >= rest_duration_threshold)
    reclassified_count = other_mask.sum()
    
    df_modified.loc[other_mask, 'Type'] = 'rest'
    
    print(f"Reclassified {reclassified_count} 'other' events as 'rest'")
    
    return df_modified

def extract_standard_cycles(df, target_pre_rest=10.0, target_post_rest=20.0, tolerance=1.0):
    """
    Extract standardized cycles: 10s pre-task rest + task + 20s post-task rest.
    Finds the best available rest segments before and after each task, allowing for overlap.

    Args:
        df: DataFrame with events (already reclassified)
        target_pre_rest: Target duration for pre-task rest (seconds). MUST be 10.0.
        target_post_rest: Target duration for post-task rest (seconds). MUST be 20.0.
        tolerance: Tolerance for the required rest duration (seconds). 
                   A rest period must be at least (target - tolerance) seconds.

    Returns:
        List of standardized cycles with exact timing
    """
    # Get all events sorted by start time
    df_sorted = df.sort_values('Start_sec').reset_index(drop=True)
    df_sorted = df_sorted.dropna(subset=['Duration']).copy() 
    total_duration = df_sorted['Start_sec'].iloc[-1] + df_sorted['Duration'].iloc[-1]

    print(f"Processing {len(df_sorted)} events for cycle extraction")
    print(f"Target cycle: {target_pre_rest}s pre-rest + task + {target_post_rest}s post-rest")
    print(f"Total recording duration: {total_duration:.1f}s")

    cycles = []
    rest_events = df_sorted[df_sorted['Type'] == 'rest'].copy()
    task_events = df_sorted[df_sorted['Type'] == 'task'].copy()
    try:
            rest_dur = rest_events['Duration'][0]
    except:
           rest_dur = rest_events['Duration'].iloc[0]
    if rest_dur < target_post_rest:
        target_post_rest = rest_dur
    if rest_events.empty or task_events.empty:
        raise ValueError("Need both 'rest' and 'task' events to form cycles.")

    for task_idx, task_row in task_events.iterrows():
        task_start = task_row['Start_sec']
        task_duration = task_row['Duration']
        task_end = task_start + task_duration

        print(f"\nProcessing Task at {task_start:.1f}s (duration: {task_duration:.1f}s)")

        # --- Find PRE-TASK REST (looking backwards from task_start) ---
        # We need a 10-second window ending exactly at task_start: [task_start - 10, task_start]
        required_pre_rest_start = task_start - target_pre_rest
        # Find any rest event that covers this entire required window
        pre_rest_found = False
        pre_rest_candidate = None

        for _, rest_row in rest_events.iterrows():
            rest_start = rest_row['Start_sec']
            rest_end = rest_start + rest_row['Duration']
            # Check if this rest event covers the entire required pre-rest window
            if rest_start <= required_pre_rest_start and rest_end >= task_start:
                pre_rest_candidate = rest_row
                pre_rest_found = True
                break

        if not pre_rest_found:
            print(f"  Pre-rest: No rest event found covering [{required_pre_rest_start:.1f}s, {task_start:.1f}s]. Skipping cycle.")
            continue
        else:
            print(f"  Pre-rest: Found candidate covering required window.")

        # --- Find POST-TASK REST (looking forwards from task_end) ---
        # We need a 20-second window starting exactly at task_end: [task_end, task_end + 20]
        required_post_rest_end = task_end + target_post_rest
        # Find any rest event that covers this entire required window
        post_rest_found = False
        post_rest_candidate = None

        for _, rest_row in rest_events.iterrows():
            rest_start = rest_row['Start_sec']
            rest_end = rest_start + rest_row['Duration']
            # Check if this rest event covers the entire required post-rest window
            if rest_start <= task_end and rest_end >= required_post_rest_end:
                post_rest_candidate = rest_row
                post_rest_found = True
                break

        if not post_rest_found:
            print(f"  Post-rest: No rest event found covering [{task_end:.1f}s, {required_post_rest_end:.1f}s]. Skipping cycle.")
            continue
        else:
            print(f"  Post-rest: Found candidate covering required window.")

        # --- Create the standardized cycle ---
        # For pre-rest, we use the exact required window from the candidate
        # For post-rest, we use the exact required window from the candidate
        cycle = {
            'cycle_start': required_pre_rest_start,
            'pre_rest_start': required_pre_rest_start,
            'pre_rest_end': task_start,
            'pre_rest_duration': target_pre_rest,  # Fixed at 10.0s
            'task_start': task_start,
            'task_end': task_end,
            'task_duration': task_duration,
            'post_rest_start': task_end,
            'post_rest_end': required_post_rest_end,
            'post_rest_duration': target_post_rest,  # Fixed at 20.0s
            'total_duration': target_pre_rest + task_duration + target_post_rest,
            'cycle_end': required_post_rest_end,
            'pre_rest_source': pre_rest_candidate['Section'],  # For debugging
            'post_rest_source': post_rest_candidate['Section']  # For debugging
        }

        cycles.append(cycle)
        print(f"  ✓ Cycle created: {target_pre_rest:.1f}s R + "
              f"{task_duration:.1f}s T + {target_post_rest:.1f}s R")
        print(f"     Pre-rest source: {pre_rest_candidate['Section']} "
              f"({pre_rest_candidate['Start_sec']:.1f}s - {pre_rest_candidate['Start_sec'] + pre_rest_candidate['Duration']:.1f}s)")
        print(f"     Post-rest source: {post_rest_candidate['Section']} "
              f"({post_rest_candidate['Start_sec']:.1f}s - {post_rest_candidate['Start_sec'] + post_rest_candidate['Duration']:.1f}s)")

    print(f"\nExtracted {len(cycles)} standardized cycles")
    return cycles

def compute_average_cycle_spectrogram(raw_data, df, ch_names, sfreq, valid_freqs, 
                                    output_dir, window_seconds=2, overlap_ratio=0.5,
                                    ignore_cycles=None, target_pre_rest=10.0, 
                                    target_post_rest=20.0):
    """
    Compute average spectrogram using standardized cycles: 10s rest + task + 20s rest
    
    Args:
        raw_data: EEG data array
        df: DataFrame with events
        ch_names: Channel names
        sfreq: Sampling frequency
        valid_freqs: [min_freq, max_freq]
        output_dir: Output directory
        window_seconds: STFT window duration
        overlap_ratio: STFT overlap ratio
        ignore_cycles: List of cycle indices to ignore
        target_pre_rest: Duration of pre-task rest (seconds)
        target_post_rest: Duration of post-task rest (seconds)
    """
    
    if ignore_cycles is None:
        ignore_cycles = []
    
    n_channels, n_samples = raw_data.shape
    
    # Step 1: Reclassify 'other' events as 'rest' if appropriate
    print("Step 1: Reclassifying 'other' events...")
    df_modified = reclassify_other_as_rest(df)
    
    # Step 2: Extract standardized cycles
    print("Step 2: Extracting standardized cycles...")
    cycles = extract_standard_cycles(df_modified, target_pre_rest, target_post_rest)
    
    if not cycles:
        raise ValueError("No valid standardized cycles found!")
    
    # Filter out ignored cycles
    valid_cycles = [cycle for i, cycle in enumerate(cycles) if i not in ignore_cycles]
    
    if not valid_cycles:
        raise ValueError("No valid cycles remaining after filtering!")
    
    print(f"Using {len(valid_cycles)} cycles for averaging")
    
    # STFT parameters
    window_length = int(window_seconds * sfreq)
    overlap = int(overlap_ratio * window_length)
    
    # Dictionary to store average spectrograms for each channel
    avg_spectrograms = {}
    
    # Calculate standard cycle duration
    standard_cycle_duration = target_pre_rest + valid_cycles[0]['task_duration'] + target_post_rest
    
    # Process each channel
    for ch_idx in range(n_channels):
        ch_name = ch_names[ch_idx]
        print(f"Processing channel: {ch_name}")
        
        cycle_spectrograms = []
        
        # Extract and compute spectrogram for each cycle
        for cycle_idx, cycle in enumerate(valid_cycles):
            # Extract cycle data with exact timing
            start_sample = int(cycle['cycle_start'] * sfreq)
            end_sample = int(cycle['cycle_end'] * sfreq)
            
            if start_sample < 0 or end_sample > n_samples:
                print(f"Warning: Cycle {cycle_idx} extends beyond data boundaries, skipping")
                continue
            
            cycle_data = raw_data[ch_idx, start_sample:end_sample]
            
            # Compute STFT for this cycle
            freqs, times, stft_data = signal.spectrogram(
                cycle_data,
                fs=sfreq,
                window='hann',
                nperseg=window_length,
                noverlap=overlap,
                scaling='density'
            )
            
            # Clean and convert to magnitude
            stft_data = clean_coeffs_2d(stft_data)
            stft_magnitude = np.sqrt(stft_data)
            stft_magnitude = 20 * np.log10(stft_magnitude + 1e-12)

            # Filter to valid frequency range
            freq_mask = (freqs >= valid_freqs[0]) & (freqs <= valid_freqs[1])
            freqs_filtered = freqs[freq_mask]
            stft_filtered = stft_magnitude[freq_mask, :]
            
            cycle_spectrograms.append(stft_filtered)
        
        if not cycle_spectrograms:
            print(f"Warning: No valid spectrograms for channel {ch_name}")
            continue
        
        # Find minimum time dimension across all cycles
        min_time_bins = min([spec.shape[1] for spec in cycle_spectrograms])
        
        # Truncate all spectrograms to same time dimension
        cycle_spectrograms_truncated = []
        for spec in cycle_spectrograms:
            cycle_spectrograms_truncated.append(spec[:, :min_time_bins])
        
        # Stack and compute average
        stacked_spectrograms = np.stack(cycle_spectrograms_truncated, axis=0)
        avg_spectrogram = np.mean(stacked_spectrograms, axis=0)
        
        # Create time axis for the average cycle
        cycle_duration = valid_cycles[0]['total_duration']
        avg_times = np.linspace(0, cycle_duration, min_time_bins)
        
        # Calculate separator positions and labels
        pre_rest_end_ratio = target_pre_rest / cycle_duration
        task_end_ratio = (target_pre_rest + valid_cycles[0]['task_duration']) / cycle_duration
        
        # Store results with separator information
        avg_spectrograms[ch_name] = {
            'freqs': freqs_filtered,
            'times': avg_times,
            'spectrogram': avg_spectrogram,
            'n_cycles': len(cycle_spectrograms_truncated),
            'cycle_duration': cycle_duration,
            'pre_rest_duration': target_pre_rest,
            'task_duration': valid_cycles[0]['task_duration'],
            'post_rest_duration': target_post_rest,
            'separator_ratios': [pre_rest_end_ratio, task_end_ratio],
            'section_labels': [
                {'label': 'R', 'start_ratio': 0, 'duration_ratio': pre_rest_end_ratio},
                {'label': 'T', 'start_ratio': pre_rest_end_ratio, 'duration_ratio': task_end_ratio - pre_rest_end_ratio},
                {'label': 'R', 'start_ratio': task_end_ratio, 'duration_ratio': 1 - task_end_ratio}
            ]
        }
    
    return avg_spectrograms, valid_cycles

def plot_average_cycle_spectrogram(avg_spectrograms, ch_names, valid_freqs, output_dir, 
                                 cycles=None, figsize=(18, 12)):
    """
    Plot the average cycle spectrograms with proper R/T labels and separators
    """
    
    n_channels = len(avg_spectrograms)
    
    if n_channels == 0:
        print("No spectrograms to plot!")
        return None
    
    # Calculate subplot grid
    if n_channels <= 4:
        n_cols, n_rows = n_channels, 1
    elif n_channels <= 8:
        n_cols, n_rows = 4, 2
    elif n_channels <= 12:
        n_cols, n_rows = 4, 3
    else:
        n_cols = 4
        n_rows = int(np.ceil(n_channels / n_cols))
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_channels == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    # Find global min/max for consistent color scaling
    global_min = float('inf')
    global_max = float('-inf')
    
    for ch_name in avg_spectrograms:
        spec_data = avg_spectrograms[ch_name]['spectrogram']
        global_min = min(global_min, spec_data.min())
        global_max = max(global_max, spec_data.max())
    
    # Plot each channel
    plot_idx = 0
    
    for ch_name in ch_names:
        if ch_name not in avg_spectrograms:
            continue
            
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        # Get spectrogram data
        spec_data = avg_spectrograms[ch_name]
        freqs = spec_data['freqs']
        times = spec_data['times']
        spectrogram = spec_data['spectrogram']
        separator_ratios = spec_data['separator_ratios']
        section_labels = spec_data['section_labels']
        
        # Create spectrogram plot
        im = ax.imshow(spectrogram,
                      aspect='auto',
                      extent=[times[0], times[-1], freqs[0], freqs[-1]],
                      cmap='jet',
                      interpolation='bilinear',
                      origin='lower',
                      vmin=global_min,
                      vmax=global_max)
        
        # Add separator lines
        cycle_duration = times[-1]
        for sep_ratio in separator_ratios:
            sep_time = sep_ratio * cycle_duration
            ax.axvline(x=sep_time, color='black', linewidth=3, linestyle='--', alpha=0.9)
        
        # Add section labels (R, T, R)
        y_pos = valid_freqs[1] * 0.9
        
        for section in section_labels:
            # Calculate label position (middle of section)
            section_start = section['start_ratio'] * cycle_duration
            section_duration = section['duration_ratio'] * cycle_duration
            label_time = section_start + section_duration / 2
            
            ax.text(label_time, y_pos, section['label'],
                    color='black', fontsize=14, fontweight='bold',
                    ha='center', va='center')
        
        # Add title and labels
        ax.set_title(f"{ch_name} ", fontsize=11, fontweight='bold', pad=20)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.grid(True, alpha=0.3, color='black', linewidth=0.5)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Power [dB]", rotation=270, labelpad=15, fontsize=9)

        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # Set main title
    total_duration = list(avg_spectrograms.values())[0]['cycle_duration']
    pre_rest = list(avg_spectrograms.values())[0]['pre_rest_duration']
    task_dur = list(avg_spectrograms.values())[0]['task_duration']
    post_rest = list(avg_spectrograms.values())[0]['post_rest_duration']
        
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.6)
    
    # Save plot
    save_path = os.path.join(output_dir, "STFT_Analysis_Average_frequency_Cycle.png")
    try:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Average standardized cycle spectrogram saved to: {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # Show plot
    try:
        # plt.show()
        pass
    except Exception as e:
        print(f"Display failed: {e}")
        print("Plot has been saved to file instead.")
    
    # Clean up
    plt.close(fig)
    
    return save_path