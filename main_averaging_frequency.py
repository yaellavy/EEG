import os
import tkinter as tk
import pandas as pd
import numpy as np
import mne
import re
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox

import averaging_frquency
import averaging_time
import averaging_time_for_STFT
import bining
import cleaning_data
import FFT_concatenation
import original_data
import STFT

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
    4: 'middle, a little below',
    5: 'middle, Left',
    6: 'Behind, middle',
    7: 'middle, right'
}

# Define the file paths and tester names
FILE_DATA = [
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
    ###########################################word
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nadav_4_9_2025\EXP_04_09_2025_192750_word\Raw\data_of_04_09_2025at07_34_03_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nadav_4_9_2025\EXP_04_09_2025_192750_word\Raw\word_timestamps_20250904_193359.csv",
    #  "Nadav"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nisan_1_9_25\EXP_01_09_2025_180825_word2\Raw\data_of_01_09_2025at06_14_37_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\nisan_1_9_25\EXP_01_09_2025_180825_word2\Raw\word_timestamps_20250901_181434.csv",
    #  "Nisan"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\noa_1_9_2025\EXP_01_09_2025_231756_word\Raw\data_of_01_09_2025at11_24_08_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\noa_1_9_2025\EXP_01_09_2025_231756_word\Raw\word_timestamps_20250901_232405.csv",
    #  "Noa"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\ohad_1_9_2025\EXP_01_09_2025_214312_word\Raw\data_of_01_09_2025at09_49_25_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\ohad_1_9_2025\EXP_01_09_2025_214312_word\Raw\word_timestamps_20250901_214922.csv",
    #  "Ohad"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\rian_4_9_2025\EXP_04_09_2025_174805_word\Raw\data_of_04_09_2025at05_54_17_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\rian_4_9_2025\EXP_04_09_2025_174805_word\Raw\word_timestamps_20250904_175414.csv",
    #  "Rian"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\tamar_4_9_2025\EXP_04_09_2025_184620_word\Raw\data_of_04_09_2025at06_52_32_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\tamar_4_9_2025\EXP_04_09_2025_184620_word\Raw\word_timestamps_20250904_185229.csv",
    #  "Tamar"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\vika_19_8_2025\EXP_19_08_2025_145451_word\Raw\data_of_19_08_2025at03_01_01_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\vika_19_8_2025\EXP_19_08_2025_145451_word\Raw\word_timestamps_20250819_150058.csv",
    #  "Vika"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_lavi_center_19_08_2025_old\EXP_19_08_2025_121710_word\Raw\data_of_19_08_2025at12_21_49_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_lavi_center_19_08_2025_old\EXP_19_08_2025_121710_word\Raw\word_timestamps_20250819_122146.csv",
    #  "Yael Lavi"),
    # (r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_reina_6_8_2025_old\yael_reina_word\EXP_06_08_2025_145547\Raw\data_of_06_08_2025at03_00_26_PM.fif",
    #  r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\measurements\yael_reina_6_8_2025_old\yael_reina_word\EXP_06_08_2025_145547\Raw\word_timestamps_20250806_150022.xlsx",
    #  "Yael Reina")
]

# ==================== DATA LOADING FUNCTIONS ====================
def load_raw_data(filepath):
    """
    Load raw EEG data from FIF or CSV files.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        tuple: (raw_data_object, sampling_frequency)
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
    
    Args:
        filepath (str): Path to the timestamp file
        
    Returns:
        DataFrame: Event data with timestamps
    """
    file_extension = os.path.splitext(filepath)[-1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(filepath)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported timestamp file format. Use .csv, .xlsx, or .xls")


# ==================== EVENT CLASSIFICATION ====================
def classify_event_type(section_name):
    """
    Classify event sections into task, rest, or other categories.
    
    Args:
        section_name (str): Name of the event section
        
    Returns:
        str: 'task', 'rest', or 'other'
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


def process_single_subject(raw_filepath, timestamp_path, tester_name, output_dir):
    """
    Process a single subject's data and return the average spectrograms.
    
    Args:
        raw_filepath (str): Path to raw EEG file
        timestamp_path (str): Path to timestamp file
        tester_name (str): Name of the tester
        output_dir (str): Output directory
        
    Returns:
        dict: Average spectrograms for each channel
    """
    print(f"\n{'='*60}")
    print(f"Processing subject: {tester_name}")
    print(f"{'='*60}")
    
    try:
        # Load raw data
        raw, sfreq = load_raw_data(raw_filepath)
        raw.set_montage('standard_1020')
        
        # Remove reference channels if present
        reference_channels = ["M1", "M2"]
        channels_to_drop = [ch for ch in reference_channels if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)
        
        # Set channel names (using descriptive names)
        ch_names = list(CHANNELS_NAMES.values())
        
        # Create unfiltered copy for later analysis
        unfiltered_raw = raw.copy()
        
        # Apply filtering if enabled
        if APPLY_FILTERING:
            raw.notch_filter([50, 60])
            raw.filter(l_freq=VALID_FREQS[0], h_freq=VALID_FREQS[1])
        
        # Load timestamp data
        df = load_timestamp_data(timestamp_path)
        
        # Process timestamps
        df['StartDatetime'] = pd.to_datetime(df['StartTimestamp'])
        df['Start_sec'] = (
            df['StartDatetime'] - df['StartDatetime'].iloc[0]
        ).dt.total_seconds()
        
        # Classify event types
        df["Type"] = df["Section"].apply(classify_event_type)
        
        # Calculate event durations
        df["Duration"] = df["Start_sec"].shift(-1) - df["Start_sec"]
        
        # Get duration statistics for validation
        rest_durations = df[df["Type"] == "rest"]["Duration"].tolist()
        task_durations = df[df["Type"] == "task"]["Duration"].tolist()
        other_durations = df[df["Type"] == "other"]["Duration"].tolist()

        if not rest_durations or not task_durations:
            print(f"Warning: Missing rest or task entries for {tester_name}")
            return None
        
        # Calculate minimum duration for labeling
        rest_duration = min(rest_durations)
        task_duration = min(task_durations)
        other_duration = min(other_durations) if other_durations else 0
        
        if other_duration == 0:
            min_duration_raw = min(rest_duration, task_duration)
        else:
            min_duration_raw = min(rest_duration, task_duration, other_duration)
        
        # Get analysis signal
        raw_data = raw.get_data()
        raw_data = raw_data[[0, 1, 3], :]  # Select specific channels
        rows_to_keep = [0, 1, 3]
        ch_names = [ch_names[i] for i in rows_to_keep]
        
        # Compute average spectrograms
        avg_spectrograms_dB, cycles = averaging_frquency.compute_average_cycle_spectrogram(
            raw_data, df, ch_names, sfreq, valid_freqs=VALID_FREQS, 
            output_dir=output_dir, window_seconds=2, overlap_ratio=0.5, 
            ignore_cycles=IGNORE_CYCLES, target_pre_rest=10.0, target_post_rest=20.0)
        
        print(f"Successfully processed {tester_name}")
        return avg_spectrograms_dB
        
    except Exception as e:
        print(f"Error processing {tester_name}: {str(e)}")
        return None


def create_multi_subject_plot(all_spectrograms, tester_names, output_dir):
    """
    Create a 3x9 subplot grid showing all subjects and channels.
    Each subplot has its own colorbar and tester names are shown as titles in the left margin.
    
    Args:
        all_spectrograms (list): List of spectrogram dictionaries for each subject
        tester_names (list): List of tester names
        output_dir (str): Output directory
    """
    # Channel names for the three selected channels
    selected_channels = ['Front, middle', 'Front, left', 'Front, right']
    
    # Create figure with 3 columns (channels) and 9 rows (testers)
    fig = plt.figure(figsize=(24, 40))
    
    # Create GridSpec for precise subplot positioning
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(9, 3, figure=fig, left=0.05, right=0.99, top=0.93, bottom=0.01,
                  wspace=0.25, hspace=0.25)

    # Set column titles (channel names) at the top
    fig.text(0.15, 0.95, selected_channels[0], ha='center', va='center', 
             fontsize=18, fontweight='bold')
    fig.text(0.5, 0.95, selected_channels[1], ha='center', va='center', 
             fontsize=18, fontweight='bold')
    fig.text(0.83, 0.95, selected_channels[2], ha='center', va='center', 
             fontsize=18, fontweight='bold')
    
    # Plot each subject's data
    for row_idx, (spectrograms, tester_name) in enumerate(zip(all_spectrograms, tester_names)):
        # Add tester name as title in left margin
        fig.text(0.02, 0.9 - (row_idx * 0.107), tester_name, ha='left', va='center',
                 fontsize=12, fontweight='bold', rotation=90)
        
        if spectrograms is None:
            # If processing failed, show empty plots
            for col_idx in range(3):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, 'Processing Failed', 
                       ha='center', va='center', 
                       transform=ax.transAxes,
                       fontsize=12, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
            continue
        
        # Plot each channel for this subject
        for col_idx, ch_name in enumerate(selected_channels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            if ch_name in spectrograms:
                # Get spectrogram data
                spec_data = spectrograms[ch_name]
                freqs = spec_data['freqs']
                times = spec_data['times']
                spectrogram = spec_data['spectrogram']
                separator_ratios = spec_data['separator_ratios']
                section_labels = spec_data['section_labels']
                
                # Create spectrogram plot with individual scaling
                im = ax.imshow(spectrogram,
                              aspect='auto',
                              extent=[times[0], times[-1], freqs[0], freqs[-1]],
                              cmap='jet',
                              interpolation='bilinear',
                              origin='lower')
                
                # Add individual colorbar for this subplot
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                cbar.set_label("Power [dB]", rotation=270, labelpad=15, fontsize=9)
                cbar.ax.tick_params(labelsize=8)
                
                # Add separator lines
                cycle_duration = times[-1]
                for sep_ratio in separator_ratios:
                    sep_time = sep_ratio * cycle_duration
                    ax.axvline(x=sep_time, color='black', linewidth=2, linestyle='--', alpha=0.8)
                
                # Add section labels (R, T, R)
                y_pos = VALID_FREQS[1] * 0.9
                
                for section in section_labels:
                    # Calculate label position (middle of section)
                    section_start = section['start_ratio'] * cycle_duration
                    section_duration = section['duration_ratio'] * cycle_duration
                    label_time = section_start + section_duration / 2
                    
                    ax.text(label_time, y_pos, section['label'],
                            color='black', fontsize=12, fontweight='bold',
                            ha='center', va='center')
                
                # Set labels and grid
                # if row_idx == 8:  # Bottom row
                ax.set_xlabel("Time (s)", fontsize=11)
                # else:
                #     ax.set_xlabel("")
                #     ax.set_xticks([])
                
                # if col_idx == 0:  # Left column
                ax.set_ylabel("Frequency (Hz)", fontsize=11)
                # else:
                #     ax.set_ylabel("")
                #     ax.set_yticks([])
                
                ax.grid(True, alpha=0.3, color='black', linewidth=0.5)
                ax.tick_params(labelsize=9)
                
            else:
                # Channel not found
                ax.text(0.5, 0.5, f'{ch_name}\nNot Found', 
                       ha='center', va='center', 
                       transform=ax.transAxes,
                       fontsize=10, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Set main title
    # fig.suptitle('EEG Spectrogram Analysis: All Subjects and Channels', 
    #              fontsize=22, fontweight='bold', y=0.96)
    
    # Save plot
    save_path = os.path.join(output_dir, "Multi_Subject_Spectrogram_Analysis_Individual_Colorbars.png")
    try:
        fig.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"\nMulti-subject analysis plot saved to: {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # Show plot
    try:
        plt.show()
    except Exception as e:
        print(f"Display failed: {e}")
        print("Plot has been saved to file instead.")
    
    # Clean up
    plt.close(fig)
    
    return save_path


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    
    # Initialize tkinter (hidden root window)
    root = tk.Tk()
    root.withdraw()
    
    print("=== Multi-Subject EEG Analysis Pipeline ===")
    
    # Set output directory (use desktop or current directory)
    output_dir =r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Desktop\שנה ד\מעבדה נוירו\פרויקט\eeg\code\final_code"
    # os.path.join(os.path.expanduser("~"), "Desktop", "EEG_Multi_Subject_Analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Output directory: {output_dir}")
    
    # Process all subjects
    all_spectrograms = []
    tester_names = []
    
    for raw_filepath, timestamp_path, tester_name in FILE_DATA:
        print(f"\nProcessing: {tester_name}")
        
        # Check if files exist
        if not os.path.exists(raw_filepath):
            print(f"Raw file not found: {raw_filepath}")
            all_spectrograms.append(None)
        elif not os.path.exists(timestamp_path):
            print(f"Timestamp file not found: {timestamp_path}")
            all_spectrograms.append(None)
        else:
            # Process the subject
            spectrograms = process_single_subject(raw_filepath, timestamp_path, tester_name, output_dir)
            all_spectrograms.append(spectrograms)
        
        tester_names.append(tester_name)
    
    # Create the multi-subject plot
    print(f"\n{'='*60}")
    print("Creating multi-subject comparison plot...")
    print(f"{'='*60}")
    
    create_multi_subject_plot(all_spectrograms, tester_names, output_dir)
    
    print("\n=== Analysis Complete! ===")
    print(f"Results saved to: {output_dir}")