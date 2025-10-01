import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft, fftfreq
import os

def plot_frequency_band_ratio_analysis(raw_data, df, ch_names, sfreq, 
                                     output_dir=None, valid_freqs=[3, 30], durations=[30, 30],
                                     save_name="frequency_band_analysis", statistical_results=None, 
                                     figsize=(15, 8), ignore_cycles=None):
    """
    Create frequency band PSD analysis plot showing absolute power values for task vs rest periods.
    
    Parameters:
    -----------
    raw_data : np.ndarray
        Raw continuous EEG data with shape (n_channels, n_timepoints)
    df : pd.DataFrame
        DataFrame with event information containing 'Start_sec', 'Type', and 'Duration' columns
    ch_names : list
        List of channel names (up to 8 channels will be plotted)
    sfreq : float
        Sampling frequency in Hz
    output_dir : str, optional
        Directory to save the plot. If None, plot is not saved.
    save_name : str, optional
        Name for saved plot file (default: "frequency_band_analysis")
    statistical_results : dict, optional
        Statistical results from enhanced analysis for significance markers
    figsize : tuple, optional
        Figure size (default: (15, 8))
    ignore_cycles : list, optional
        List of cycle indices to ignore during analysis
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    band_powers : dict
        Dictionary containing the computed power values for each channel and band
    """
    
    if ignore_cycles is None:
        ignore_cycles = []
    
    enhanced_bands = {
        r'$\delta$': (valid_freqs[0], 4),     # Delta
        r'$\theta$': (4, 8),                  # Theta  
        r'$\alpha$': (8, 13),                 # Alpha
        r'$\beta$': (13, valid_freqs[1]),     # Beta
    }
    band_names = list(enhanced_bands.keys())

    # Add this for legend display:
    band_descriptions = {
        r'$\delta$': f'Delta: {valid_freqs[0]}-4 Hz',
        r'$\theta$': 'Theta: 4-8 Hz',
        r'$\alpha$': 'Alpha: 8-13 Hz', 
        r'$\beta$': f'Beta: 13-{valid_freqs[1]} Hz'
    }
    # Create figure with extra column for legend
    channels_amount=raw_data.shape[0]
    if channels_amount <4:
        rows=1
        cols=channels_amount+1
    else:
        rows=2
        if channels_amount%2==0:
            cols=channels_amount//2+1
        else:
            cols=(channels_amount+1)//2
    fig, axes = plt.subplots(rows,cols, figsize=figsize)
    if channels_amount > 4:
        axes_flat = axes[:, :min(4,channels_amount)].flatten()  # Use first 4 columns for data
        legend_ax = axes[0, min(4,channels_amount)]             # Use 5th column for legend
    else:
        axes_flat = axes[ :min(4,channels_amount)].flatten()
        legend_ax = axes[min(4,channels_amount)]             # Use 5th column for legend    # Store computed power values for return
    band_powers = {}
    
    # Process up to 8 channels
    n_channels_to_plot = min(8,channels_amount)
    
    # Extract rest and task segments from continuous data
    def extract_segments(event_type, duration_sec=1/2*(durations[0]+durations[1])):
        """Extract segments of specified duration from events of given type"""
        segments = []
        events = df[df['Type'] == event_type].copy()
        
        for idx, row in events.iterrows():
            if idx in ignore_cycles:
                continue
                
            start_time = row['Start_sec']
            start_sample = int(start_time * sfreq)
            end_sample = int((start_time + duration_sec) * sfreq)
            
            # Check bounds
            if start_sample >= 0 and end_sample < raw_data.shape[1]:
                segment = raw_data[:, start_sample:end_sample]
                segments.append(segment)
        
        return segments
    
    # Extract rest and task segments (10 seconds each for comparison)
    task_segments = extract_segments('task', duration_sec=durations[0])
    rest_segments = extract_segments('rest', duration_sec=durations[1])

    print(f"Extracted {len(rest_segments)} rest segments and {len(task_segments)} task segments")
    
    if len(rest_segments) == 0 or len(task_segments) == 0:
        raise ValueError("No rest or task segments found! Check your event classification.")
    
    for i in range(n_channels_to_plot):
        ax = axes_flat[i]
        ch = ch_names[i]
        
        # Compute average PSD for rest and task periods
        rest_psds = []
        task_psds = []
        
        # Process rest segments
        for segment in rest_segments:
            signal = segment[i, :]  # Get channel i
            # if len(signal) > 64:  # Minimum length check
                # freqs1, psd1 = welch(signal, fs=sfreq, nperseg=min(256, len(signal)))
            fft_vals = fft(signal)
            psd = np.abs(fft_vals)/ len(signal)
            freqs = fftfreq(len(signal), 1/sfreq)
            pos_mask = (freqs >= valid_freqs[0]) & (freqs <= valid_freqs[1])
            freqs_rest = freqs[pos_mask]
            psd = psd[pos_mask]
            rest_psds.append(psd)
        
        # Process task segments  
        for segment in task_segments:
            signal = segment[i, :]  # Get channel i
            fft_vals = fft(signal)
            psd = np.abs(fft_vals) / len(signal)
            freqs = fftfreq(len(signal), 1/sfreq)
            pos_mask = (freqs >= valid_freqs[0]) & (freqs <= valid_freqs[1])
            freqs_task = freqs[pos_mask]
            psd = psd[pos_mask]
            task_psds.append(psd)
        
        if len(rest_psds) == 0 or len(task_psds) == 0:
            print(f"Warning: No valid segments for channel {ch}")
            continue
        
        # # Average PSDs across segments
        rest_psds_minimum_len = min([psd.shape for psd in rest_psds])
        rest_psds_truncated = [psd[ :rest_psds_minimum_len[0]] for psd in rest_psds]
        rest_psd_mean = np.mean(rest_psds_truncated, axis=0)
        task_psds_minimum_len = min([psd.shape for psd in task_psds])
        task_psds_truncated = [psd[:task_psds_minimum_len[0]] for psd in task_psds]
        task_psd_mean = np.mean(task_psds_truncated, axis=0)
        # Extract power in each frequency band
        rest_band_powers = []
        task_band_powers = []
        
        for band in enhanced_bands.values():
            fmin, fmax = band
            band_mask = (freqs_task >= fmin) & (freqs_task < fmax)

            task_power = np.mean(task_psd_mean[band_mask[:task_psds_minimum_len[0]]]) if np.any(band_mask) else np.nan
            
            task_band_powers.append(task_power)
        
        for band in enhanced_bands.values():
            fmin, fmax = band
            band_mask = (freqs_rest >= fmin) & (freqs_rest < fmax)

            rest_power = np.mean(rest_psd_mean[band_mask[:rest_psds_minimum_len[0]]]) if np.any(band_mask) else np.nan
            
            rest_band_powers.append(rest_power)
        
        # Store absolute power values (no ratios)
        rest_powers = np.array(rest_band_powers)
        task_powers = np.array(task_band_powers)
        
        # Store powers for this channel
        band_powers[ch] = {
            'rest_powers': rest_powers,
            'task_powers': task_powers,
            'band_names': band_names
        }
        
        # Create side-by-side bar plot with absolute values
        x = np.arange(len(band_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, task_powers, width, label='Task', 
                      color='blue', alpha=0.8)
        bars2 = ax.bar(x + width/2, rest_powers, width, label='Rest', 
                      color='red', alpha=0.8)
        
        # Add significance markers if statistical results provided
        if statistical_results and ch in statistical_results:
            for j, band_name in enumerate(band_names):
                # full_name = full_band_names.get(band_name, band_name)
                full_name = band_descriptions.get(band_name, band_name)
                if full_name in statistical_results[ch]:
                    p_val = statistical_results[ch][full_name].get('p_corrected',
                                                                   statistical_results[ch][full_name]['p_value'])
                    # Add significance stars
                    y_pos = max(task_powers[j], rest_powers[j]) + 0.1 * max(max(task_powers), max(rest_powers))
                    if p_val < 0.001:
                        ax.text(j, y_pos, '***', ha='center', fontweight='bold', fontsize=12)
                    elif p_val < 0.01:
                        ax.text(j, y_pos, '**', ha='center', fontweight='bold', fontsize=12)
                    elif p_val < 0.05:
                        ax.text(j, y_pos, '*', ha='center', fontweight='bold', fontsize=12)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=10)
        ax.set_title(f"{ch}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Power Spectral Density", fontsize=10)
        # ax.set_ylim(bottom=0)  # Start from 0 for absolute power values
        ax.grid(axis='y', alpha=0.3, which='both')
        # ax.set_yscale('log')       
        # ax.set_ylim(bottom=1e-3) 
        # Add sample count info with colored lines
        y_start = 0.98
        line_length = 0.15
        ax.legend(fontsize=10)

    if channels_amount%2==0:
        axes_flat[channels_amount-1].axis('off')
    # Create frequency bands legend
    legend_ax.axis('off')
    freq_text = []

    for symbol, description in band_descriptions.items():
        freq_text.append(f'{symbol}: {description}')
    
    legend_text = 'Frequency Bands:\n\n' + '\n\n'.join(freq_text)

    legend_ax.text(0.05, 0.95, legend_text, transform=legend_ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Hide the bottom right subplot  
    if channels_amount>4:
        axes[1, 4].axis('off')
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved frequency band analysis plot to: {save_path}")
    # plt.show()

    return fig, band_powers

