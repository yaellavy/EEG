import os
import pandas as pd
import numpy as np
import mne
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import fft, fftfreq

# ==================== USER CONFIGURATION ====================
# Analysis settings
APPLY_FILTERING = True
FILTER_RAW_SIGNAL = True
IGNORE_CYCLES = []
# Frequency analysis parameters
VALID_FREQS = [1, 30]  # [min_freq, max_freq] in Hz
VALID_RANGE = VALID_FREQS[1] - VALID_FREQS[0]

# Channel mapping - alternative descriptive names
CHANNELS_NAMES = {
    0: 'Front, middle',
    1: 'Front, left', 
    2: 'Center',
    3: 'Front, right',
    4: 'Middle, below',
    5: 'Middle, left',
    6: 'Back, middle',
    7: 'Middle, right'
}

# Frequency bands definition
FREQUENCY_BANDS = {
    'δ (1-4 Hz)': (1, 4),
    'θ (4-8 Hz)': (4, 8),
    'α (8-13 Hz)': (8, 13),
    'β (13-30 Hz)': (13, 30)
}

# ==================== HARDCODED FILE PATHS ====================
# Define your 9 file pairs here
FILE_PAIRS = [
    #########################aritmetic
    # Format: (fif_path, timestamp_path, subject_name)
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nadav_4_9_2025\EXP_04_09_2025_190934_aritmetic\Raw\data_of_04_09_2025at07_15_46_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nadav_4_9_2025\EXP_04_09_2025_190934_aritmetic\Raw\arithmetic_timestamps_20250904_191543.csv",
    #  "Nadav"),

    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nisan_1_9_25\EXP_01_09_2025_164726_aritmetic\Raw\data_of_01_09_2025at04_53_48_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nisan_1_9_25\EXP_01_09_2025_164726_aritmetic\Raw\arithmetic_timestamps_20250901_165335.csv",
    #  "Nisan"),

    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\noa_1_9_2025\EXP_01_09_2025_224849_aritmetic\Raw\data_of_01_09_2025at10_55_01_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\noa_1_9_2025\EXP_01_09_2025_224849_aritmetic\Raw\arithmetic_timestamps_20250901_225458.csv",
    #  "Noa"),
    
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\ohad_1_9_2025\EXP_01_09_2025_210810_aritmetic\Raw\data_of_01_09_2025at09_14_23_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\ohad_1_9_2025\EXP_01_09_2025_210810_aritmetic\Raw\arithmetic_timestamps_20250901_211420.csv",
    #  "Ohad"),

    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\rian_4_9_2025\EXP_04_09_2025_172815_aritmetic\Raw\data_of_04_09_2025at05_34_27_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\rian_4_9_2025\EXP_04_09_2025_172815_aritmetic\Raw\arithmetic_timestamps_20250904_173424.csv",
    #  "Rian"),

    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\tamar_4_9_2025\EXP_04_09_2025_182515_aritmetic\Raw\data_of_04_09_2025at06_31_27_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\tamar_4_9_2025\EXP_04_09_2025_182515_aritmetic\Raw\arithmetic_timestamps_20250904_183124.csv",
    #  "Tamar"),

    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\vika_19_8_2025\EXP_19_08_2025_150654_aritmetic\Raw\data_of_19_08_2025at03_13_03_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\vika_19_8_2025\EXP_19_08_2025_150654_aritmetic\Raw\arithmetic_timestamps_20250819_151300.csv",
    #  "Vika"),

    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_lavi_center_19_08_2025_old\EXP_19_08_2025_122651_aritmetic\Raw\data_of_19_08_2025at12_31_30_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_lavi_center_19_08_2025_old\EXP_19_08_2025_122651_aritmetic\Raw\arithmetic_timestamps_20250819_123127.csv",
    #  "Yael Lavi"),

    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_reina_6_8_2025_old\yael_reina_aritmetic_2\EXP_06_08_2025_130117\Raw\data_of_06_08_2025at01_05_57_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_reina_6_8_2025_old\yael_reina_aritmetic_2\EXP_06_08_2025_130117\Raw\arithmetic_timestamps_20250806_130553.xlsx",
    #  "Yael Reina")
    # #########################nback
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nadav_4_9_2025\EXP_04_09_2025_191844_2back\Raw\data_of_04_09_2025at07_24_57_PM.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nadav_4_9_2025\EXP_04_09_2025_191844_2back\Raw\nback_timestamps_20250904_192454.csv",
        "Nadav"),
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nisan_1_9_25\EXP_01_09_2025_182141_2back\Raw\data_of_01_09_2025at06_27_53_PM.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nisan_1_9_25\EXP_01_09_2025_182141_2back\Raw\nback_timestamps_20250901_182750.csv",
        "Nisan"),
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\noa_1_9_2025\EXP_01_09_2025_230343_2back\Raw\data_of_01_09_2025at11_09_54_PM.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\noa_1_9_2025\EXP_01_09_2025_230343_2back\Raw\nback_timestamps_20250901_230952.csv",
        "Noa"),
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\ohad_1_9_2025\EXP_01_09_2025_211820_2back\Raw\data_of_01_09_2025at09_24_32_PM.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\ohad_1_9_2025\EXP_01_09_2025_211820_2back\Raw\nback_timestamps_20250901_212429.csv",
        "Ohad"),
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\rian_4_9_2025\EXP_04_09_2025_173827_2back\Raw\data_of_04_09_2025at05_44_41_PM.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\rian_4_9_2025\EXP_04_09_2025_173827_2back\Raw\nback_timestamps_20250904_174437.csv",
        "Rian"),
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\tamar_4_9_2025\EXP_04_09_2025_183609_2back\Raw\data_of_04_09_2025at06_42_21_PM.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\tamar_4_9_2025\EXP_04_09_2025_183609_2back\Raw\nback_timestamps_20250904_184218.csv",
        "Tamar"),
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\vika_19_8_2025\EXP_19_08_2025_143220_2back\Raw\data_of_19_08_2025at02_36_58_PM.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\vika_19_8_2025\EXP_19_08_2025_143220_2back\Raw\nback_timestamps_20250819_143655.csv",
        "Vika"),
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_lavi_center_19_08_2025_old\EXP_19_08_2025_114719_nback\Raw\data_of_19_08_2025at11_51_57_AM-fix.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_lavi_center_19_08_2025_old\EXP_19_08_2025_114719_nback\Raw\nback_timestamps_20250819_115155.csv",
        "Yael Lavi"),
    (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_reina_6_8_2025_old\yael_reina_nback_first\EXP_06_08_2025_144215\Raw\data_of_06_08_2025at02_46_55_PM.fif",
        r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_reina_6_8_2025_old\yael_reina_nback_first\EXP_06_08_2025_144215\Raw\nback_timestamps_20250806_144651.xlsx",
        "Yael Reina")
     #########################word
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nadav_4_9_2025\EXP_04_09_2025_192750_word\Raw\data_of_04_09_2025at07_34_03_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nadav_4_9_2025\EXP_04_09_2025_192750_word\Raw\word_timestamps_20250904_193359.csv",
    #     "Nadav"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nisan_1_9_25\EXP_01_09_2025_180825_word2\Raw\data_of_01_09_2025at06_14_37_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nisan_1_9_25\EXP_01_09_2025_180825_word2\Raw\word_timestamps_20250901_181434.csv",
    #     "Nisan"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\noa_1_9_2025\EXP_01_09_2025_231756_word\Raw\data_of_01_09_2025at11_24_08_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\noa_1_9_2025\EXP_01_09_2025_231756_word\Raw\word_timestamps_20250901_232405.csv",
    #     "Noa"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\ohad_1_9_2025\EXP_01_09_2025_214312_word\Raw\data_of_01_09_2025at09_49_25_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\ohad_1_9_2025\EXP_01_09_2025_214312_word\Raw\word_timestamps_20250901_214922.csv",
    #     "Ohad"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\rian_4_9_2025\EXP_04_09_2025_174805_word\Raw\data_of_04_09_2025at05_54_17_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\rian_4_9_2025\EXP_04_09_2025_174805_word\Raw\word_timestamps_20250904_175414.csv",
    #     "Rian"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\tamar_4_9_2025\EXP_04_09_2025_184620_word\Raw\data_of_04_09_2025at06_52_32_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\tamar_4_9_2025\EXP_04_09_2025_184620_word\Raw\word_timestamps_20250904_185229.csv",
    #     "Tamar"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\vika_19_8_2025\EXP_19_08_2025_145451_word\Raw\data_of_19_08_2025at03_01_01_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\vika_19_8_2025\EXP_19_08_2025_145451_word\Raw\word_timestamps_20250819_150058.csv",
    #     "Vika"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_lavi_center_19_08_2025_old\EXP_19_08_2025_121710_word\Raw\data_of_19_08_2025at12_21_49_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_lavi_center_19_08_2025_old\EXP_19_08_2025_121710_word\Raw\word_timestamps_20250819_122146.csv",
    #     "Yael Lavi"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_reina_6_8_2025_old\yael_reina_word\EXP_06_08_2025_145547\Raw\data_of_06_08_2025at03_00_26_PM.fif",
    #     r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_reina_6_8_2025_old\yael_reina_word\EXP_06_08_2025_145547\Raw\word_timestamps_20250806_150022.xlsx",
    #     "Yael Reina")
]
output_dir=r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\code\final_code"

# ==================== FREQUENCY ANALYSIS FUNCTION ====================
def plot_frequency_band_ratio_analysis(raw_data, df, ch_names, sfreq, 
                                     output_dir=None, valid_freqs=[3, 30], durations=[30, 30],
                                     save_name="frequency_band_analysis", statistical_results=None, 
                                     figsize=(15, 8), ignore_cycles=None):
    """
    Create frequency band PSD analysis plot showing absolute power values for task vs rest periods.
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
    channels_amount = raw_data.shape[0]
    if channels_amount < 4:
        rows = 1
        cols = channels_amount + 1
    else:
        rows = 2
        if channels_amount % 2 == 0:
            cols = channels_amount // 2 + 1
        else:
            cols = (channels_amount + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows > 1:
        axes_flat = axes[:, :min(4, channels_amount)].flatten()
        legend_ax = axes[0, min(4, channels_amount)]
    else:
        axes_flat = axes[:min(4, channels_amount)].flatten()
        legend_ax = axes[min(4, channels_amount)]
    
    # Store computed power values for return
    band_powers = {}
    
    # Process up to 8 channels
    n_channels_to_plot = min(8, channels_amount)
    
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
        
        # Average PSDs across segments
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
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=10)
        ax.set_title(f"{ch}", fontsize=10, fontweight='bold', pad=5)  # Reduced padding
        ax.set_ylabel("Power", fontsize=9)
        ax.grid(axis='y', alpha=0.3, which='both')
        ax.legend(fontsize=8)

    # Hide empty subplots
    for i in range(n_channels_to_plot, len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Create frequency bands legend
    legend_ax.axis('off')
    freq_text = []

    for symbol, description in band_descriptions.items():
        freq_text.append(f'{symbol}: {description}')
    
    legend_text = 'Frequency Bands:\n\n' + '\n\n'.join(freq_text)

    legend_ax.text(0.05, 0.95, legend_text, transform=legend_ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Hide any remaining empty subplots
    if rows > 1 and cols > channels_amount:
        axes[1, cols-1].axis('off')
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"Saved frequency band analysis plot to: {save_path}")
    
    return fig, band_powers

# ==================== DATA LOADING FUNCTIONS ====================
def load_raw_data(filepath):
    """
    Load raw EEG data from FIF or CSV files.
    """
    file_extension = os.path.splitext(filepath)[-1].lower()
    
    if file_extension == '.fif':
        # Load MNE FIF format
        raw = mne.io.read_raw_fif(filepath, preload=True)
        sampling_freq = raw.info['sfreq']
        return raw, sampling_freq
        
    elif file_extension == '.csv':
        # Load CSV format with time column
        df_csv = pd.read_csv(filepath)
        
        # Validate required columns
        if 'Time (s)' not in df_csv.columns:
            raise ValueError("CSV must contain a 'Time (s)' column.")
            
        # Extract time and data
        times = df_csv['Time (s)'].values
        data = df_csv.drop(columns=['Time (s)']).values.T  # Shape: [channels, samples]
        channel_names = list(df_csv.columns[1:])
        
        # Calculate sampling frequency from time differences
        sampling_freq = 1 / np.mean(np.diff(times))
        
        # Create MNE Raw object
        info = mne.create_info(ch_names=channel_names, sfreq=sampling_freq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        return raw, sampling_freq
    else:
        raise ValueError("Unsupported file format. Use .fif or .csv")


def load_timestamp_data(filepath):
    """
    Load timestamp/event data from CSV or Excel files.
    """
    file_extension = os.path.splitext(filepath)[-1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(filepath)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(filepath).iloc[:, :2]
    else:
        raise ValueError("Unsupported timestamp file format. Use .csv, .xlsx, or .xls")


def classify_event_type(section_name):
    """
    Classify event sections into task, rest, or other categories.
    """
    section_lower = section_name.lower()
    
    # Define rest patterns
    rest_patterns = ["rest", "rest_start"]
    if section_lower in rest_patterns:
        return "rest"
    
    short_rest_patterns = ["initial rest", "initial_rest", "final rest"]
    if section_lower in short_rest_patterns:
        return "other"
    
    # Define task patterns
    task_patterns = [
        "tap index finger", "arithmetic task", "word task", 
        "task_start", "nback task"
    ]
    if section_lower in task_patterns:
        return "task"
    
    # Check for arithmetic expressions (e.g., "2 + 3?")
    if re.match(r"^\d+\s*\+\s*\d+\??$", section_name.strip()):
        return "task"
    
    return "other"


def create_combined_frequency_analysis_plot(all_results, output_dir):
    """
    Create a combined plot with 9 subjects x 3 channels each.
    Only the first subplot gets axis labels and legend.
    """
    # Create figure with appropriate size
    fig = plt.figure(figsize=(16, 20))
    plt.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.005)

    
    # Create 9x4 grid (9 rows, 4 columns - last column for legend)
    gs = GridSpec(9, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # Channel names for the 3 selected channels
    selected_channels = ['Front, middle', 'Front, left', 'Front, right']
    
    # Frequency bands for legend
    enhanced_bands = {
        r'$\delta$': '1-4 Hz',
        r'$\theta$': '4-8 Hz', 
        r'$\alpha$': '8-13 Hz',
        r'$\beta$': '13-30 Hz'
    }
    
    # Get list of subjects (ensure we have exactly 9)
    subject_names = list(all_results.keys())[:9]
    
    print(f"Creating plot for {len(subject_names)} subjects: {subject_names}")
    
    # Process each subject
    for subj_idx, subject_name in enumerate(subject_names):
        band_powers = all_results[subject_name]
        
        # Add subject name as row title  , 0.95 - (subj_idx * 0.105)
        fig.text(0.02, 0.95 - (subj_idx * 0.115), subject_name, 
                ha='left', va='center', fontsize=12, fontweight='bold',
                rotation=90)
        
        # Plot each channel for this subject
        for ch_idx, ch_name in enumerate(selected_channels):
            if ch_name not in band_powers:
                continue
                
            # Create subplot
            ax = fig.add_subplot(gs[subj_idx, ch_idx])
            
            # Get power data
            ch_data = band_powers[ch_name]
            rest_powers = ch_data['rest_powers']
            task_powers = ch_data['task_powers']
            band_names = ch_data['band_names']
            
            # Create bar plot
            x = np.arange(len(band_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, task_powers, width, label='Task', 
                          color='blue', alpha=0.8)
            bars2 = ax.bar(x + width/2, rest_powers, width, label='Rest', 
                          color='red', alpha=0.8)
            
            # Only add labels and legend to the first subplot
            if subj_idx == 0 and ch_idx == 0:
                ax.set_ylabel("Power Spectral Density", fontsize=11)
                ax.legend(fontsize=10, loc='upper right')
            ax.set_xticks(x)
            ax.set_xticklabels(band_names, fontsize=10)
            # else:
            #     # Remove labels for other subplots
            #     ax.set_xticks(x)
            #     ax.set_xticklabels([])  # No x-axis labels
            #     ax.set_ylabel('')       # No y-axis label
            
            # Add channel name as column title only for first row
            if subj_idx == 0:
                ax.set_title(ch_name, fontsize=11, fontweight='bold', pad=10)
            
            ax.grid(axis='y', alpha=0.3)
            
            # Remove spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Add frequency bands legend in top-right corner
    legend_ax = fig.add_subplot(gs[0, 3])
    legend_ax.axis('off')
    
    legend_text = 'Frequency Bands:\n\n'
    for symbol, freq_range in enhanced_bands.items():
        legend_text += f'{symbol}: {freq_range}\n\n'
    
    legend_ax.text(0.1, 0.9, legend_text, transform=legend_ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    
    # Save the combined plot
    if output_dir:
        save_path = os.path.join(output_dir, "combined_9_subjects_frequency_analysis.png")
        fig.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Saved combined analysis plot to: {save_path}")

    plt.show()
    return fig


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    
    print("=== EEG Analysis Pipeline for 9 Records ===")
    
    # Use hardcoded file pairs
    file_pairs = FILE_PAIRS
    
    # Validate that we have exactly 9 pairs
    if len(file_pairs) != 9:
        print(f"Warning: Expected 9 records, but got {len(file_pairs)}. Continuing with available records...")
    
    print(f"\nProcessing {len(file_pairs)} records:")
    for i, (fif_path, timestamp_path, subject_name) in enumerate(file_pairs):
        print(f"{i+1}. {subject_name}")
    
    # Set output directory to the first file's directory
    # if os.path.exists(file_pairs[0][0]):
    #     output_dir = os.path.dirname(file_pairs[0][0])
    # else:
    #     output_dir = os.getcwd()  # Current directory as fallback
    
    # Store results for all subjects
    all_results = {}
    successful_count = 0
    
    # Process each record
    for i, (raw_filepath, timestamp_path, subject_name) in enumerate(file_pairs):
        print(f"\n{'='*50}")
        print(f"Processing {i+1}/{len(file_pairs)}: {subject_name}")
        print(f"{'='*50}")
        
        # Check if files exist
        if not os.path.exists(raw_filepath):
            print(f"Error: FIF file not found: {raw_filepath}")
            continue
            
        if not os.path.exists(timestamp_path):
            print(f"Error: Timestamp file not found: {timestamp_path}")
            continue
        
        try:
            # Load and preprocess raw data
            print(f"Loading raw EEG data from: {os.path.basename(raw_filepath)}")
            raw, sfreq = load_raw_data(raw_filepath)
            raw.set_montage('standard_1020')
            
            # Remove reference channels if present
            reference_channels = ["M1", "M2"]
            channels_to_drop = [ch for ch in reference_channels if ch in raw.ch_names]
            if channels_to_drop:
                raw.drop_channels(channels_to_drop)
            
            # Set channel names
            ch_names = list(CHANNELS_NAMES.values())
            print(f"Sampling frequency: {sfreq} Hz")
            
            # Apply filtering if enabled
            if APPLY_FILTERING:
                print(f"Applying bandpass filter {VALID_FREQS[0]}-{VALID_FREQS[1]} Hz and notch filter at 50, 60 Hz...")
                raw.notch_filter([50, 60])
                raw.filter(l_freq=VALID_FREQS[0], h_freq=VALID_FREQS[1])
            
            # Load and process timestamp data
            print(f"Loading timestamp data from: {os.path.basename(timestamp_path)}")
            df = load_timestamp_data(timestamp_path)
            
            # Process timestamps
            df['StartDatetime'] = pd.to_datetime(df['StartTimestamp'])
            df['Start_sec'] = (df['StartDatetime'] - df['StartDatetime'].iloc[0]).dt.total_seconds()
            
            # Classify event types
            df["Type"] = df["Section"].apply(classify_event_type)
            df["Duration"] = df["Start_sec"].shift(-1) - df["Start_sec"]
            
            # Get duration statistics
            rest_durations = df[df["Type"] == "rest"]["Duration"].tolist()
            task_durations = df[df["Type"] == "task"]["Duration"].tolist()
            other_durations = df[df["Type"] == "other"]["Duration"].tolist()

            if not rest_durations or not task_durations:
                print(f"Warning: Missing rest or task entries for {subject_name}")
                continue
            
            # Calculate durations
            rest_duration = min(rest_durations)
            task_duration = min(task_durations)
            other_duration = min(other_durations) if other_durations else 0
            durations = [task_duration, rest_duration]
            
            if other_duration == 0:
                min_duration_raw = min(rest_duration, task_duration)
            else:
                min_duration_raw = min(rest_duration, task_duration, other_duration)
            
            print(f"Event classification complete:")
            print(f"- Rest events: {len(rest_durations)} (min duration: {rest_duration:.1f}s)")
            print(f"- Task events: {len(task_durations)} (min duration: {task_duration:.1f}s)")
            
            # Extract raw data and select channels [0,1,3] (Front middle, left, right)
            raw_data = raw.get_data()
            raw_data = raw_data[[0, 1, 3], :]  # Select specific channels
            rows_to_keep = [0, 1, 3]
            ch_names_selected = [ch_names[i] for i in rows_to_keep]
            
            print(f"Selected channels: {ch_names_selected}")
            
            # Run frequency analysis
            print("Running frequency band analysis...")
            fig, band_powers = plot_frequency_band_ratio_analysis(
                raw_data=raw_data, 
                df=df, 
                ch_names=ch_names_selected,
                sfreq=sfreq, 
                output_dir=None,  # Don't save individual plots
                valid_freqs=VALID_FREQS, 
                durations=durations,
                save_name=f"frequency_analysis_{subject_name}"
            )
            plt.close(fig)  # Close individual plot
            
            # Store results
            all_results[subject_name] = band_powers
            successful_count += 1
            print(f"Successfully processed {subject_name}")
            
        except Exception as e:
            print(f"Error processing {subject_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create combined plot
    print(f"\n{'='*50}")
    print("Creating combined analysis plot...")
    print(f"{'='*50}")
    
    if all_results:
        print(f"Successfully processed {successful_count} subjects: {list(all_results.keys())}")
        create_combined_frequency_analysis_plot(all_results, output_dir)
        print(f"\nAnalysis complete! Processed {successful_count} subjects successfully.")
    else:
        print("No successful analyses to plot!")