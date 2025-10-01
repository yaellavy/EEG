import os
import tkinter as tk
import pandas as pd
import numpy as np
import mne
import re
import matplotlib.pyplot as plt
# from scipy.fft import rfft, rfftfreq, fft, fftfreq
from tkinter import filedialog, messagebox
# from scipy import signal

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
# ARTIFACT_RECOGNITION = True
FILTER_RAW_SIGNAL = True
IGNORE_CYCLES=[]
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
# Standard 10-20 system names (commented out)
# CHANNELS_NAMES = {0: 'Fz', 1: 'C3', 2: 'Cz', 3: 'C4', 4: 'Pz', 5: 'PO7', 6: 'Oz', 7: 'PO8'}


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


# ==================== GUI FUNCTIONS ====================
def select_channels(available_channels):
    """
    Create a GUI dialog for selecting channels for individual plotting.
    
    Args:
        available_channels (list): List of available channel names
        
    Returns:
        list: Selected channel names
    """
    # Create and configure main selection window
    selection_window = tk.Toplevel()
    selection_window.title("Select Channels for Individual Plots")
    selection_window.geometry("400x500")
    selection_window.transient()
    selection_window.grab_set()
    
    # Center the window on screen
    selection_window.update_idletasks()
    screen_x = (selection_window.winfo_screenwidth() // 2) - (400 // 2)
    screen_y = (selection_window.winfo_screenheight() // 2) - (500 // 2)
    selection_window.geometry(f"400x500+{screen_x}+{screen_y}")
    
    # Add title label
    title_label = tk.Label(
        selection_window,
        text="Select channels to create individual plots:",
        font=("Arial", 12, "bold")
    )
    title_label.pack(pady=10)
    
    # Create checkbox frame
    checkbox_frame = tk.Frame(selection_window)
    checkbox_frame.pack(pady=10, padx=20, fill='both', expand=True)
    
    # Create checkboxes for each channel
    channel_vars = {}
    for i, channel in enumerate(available_channels):
        var = tk.BooleanVar()
        checkbox = tk.Checkbutton(
            checkbox_frame,
            text=channel,
            variable=var,
            font=("Arial", 10)
        )
        checkbox.pack(anchor='w', pady=2)
        channel_vars[channel] = var
    
    # Create button frame
    button_frame = tk.Frame(selection_window)
    button_frame.pack(pady=20)
    
    # Store selected channels
    selected_channels = []
    
    # Button callback functions
    def select_all():
        """Select all available channels."""
        for var in channel_vars.values():
            var.set(True)
    
    def deselect_all():
        """Deselect all channels."""
        for var in channel_vars.values():
            var.set(False)
    
    def confirm_selection():
        """Confirm channel selection and close dialog."""
        nonlocal selected_channels
        selected_channels = [ch for ch, var in channel_vars.items() if var.get()]
        
        if not selected_channels:
            messagebox.showwarning("No Selection", "Please select at least one channel.")
            return
            
        selection_window.destroy()
    
    def cancel_selection():
        """Cancel selection and close dialog."""
        nonlocal selected_channels
        selected_channels = []
        selection_window.destroy()
    
    # Create control buttons
    tk.Button(button_frame, text="Select All", command=select_all).pack(side='left', padx=5)
    tk.Button(button_frame, text="Deselect All", command=deselect_all).pack(side='left', padx=5)
    tk.Button(button_frame, text="OK", command=confirm_selection, 
              bg='green', fg='white').pack(side='left', padx=5)
    tk.Button(button_frame, text="Cancel", command=cancel_selection, 
              bg='red', fg='white').pack(side='left', padx=5)
    
    # Wait for user interaction
    selection_window.wait_window()
    
    return selected_channels


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
    
    short_rest_patterns = [ "initial rest", "initial_rest", "final rest"]
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


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    
    # Initialize tkinter (hidden root window)
    root = tk.Tk()
    root.withdraw()
    
    # ===== FILE SELECTION =====
    print("=== EEG Analysis Pipeline ===")
    print("Step 1: File Selection")
    
    # Select raw EEG data file
    raw_filepath = filedialog.askopenfilename(
        title="Select your raw EEG file (.fif or .csv)",
        filetypes=[
            ("FIF Files", "*.fif"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
    )
    
    if not raw_filepath:
        raise ValueError("No raw EEG file selected!")
    
    # Select timestamp/event file
    timestamp_path = filedialog.askopenfilename(
        title="Select your fingertapping/timestamp file (CSV or Excel)",
        filetypes=[
            ("All Supported", "*.csv;*.xlsx;*.xls"),
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx"),
            ("Excel Files (Legacy)", "*.xls"),
            ("All Files", "*.*")
        ]
    )
    
    if not timestamp_path:
        raise ValueError("No timestamp file selected!")
    
    # Set output directory to same location as raw data
    output_dir = os.path.dirname(raw_filepath)
    
    # ===== RAW DATA LOADING & PREPROCESSING =====
    print("\nStep 2: Loading and Preprocessing Raw Data")
    print(f"Loading raw EEG data from: {raw_filepath}")
    
    # Load raw data
    raw, sfreq = load_raw_data(raw_filepath)
    raw.set_montage('standard_1020')
    
    # Remove reference channels if present
    reference_channels = ["M1", "M2"]
    channels_to_drop = [ch for ch in reference_channels if ch in raw.ch_names]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
        print(f"Dropped reference channels: {channels_to_drop}")
    
    # Set channel names (using descriptive names)
    ch_names = list(CHANNELS_NAMES.values())
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Channels: {ch_names}")
    
    # Create unfiltered copy for later analysis
    unfiltered_raw = raw.copy()
    
    # Apply filtering if enabled
    if APPLY_FILTERING:
        print(f"Applying bandpass filter {VALID_FREQS[0]}-{VALID_FREQS[1]} Hz and notch filter at 50, 60 Hz...")
        raw.notch_filter([50, 60])
        raw.filter(l_freq=VALID_FREQS[0], h_freq=VALID_FREQS[1])
    
    # ===== TIMESTAMP DATA PROCESSING =====
    print("\nStep 3: Processing Timestamp Data")
    print(f"Loading timestamp data from: {timestamp_path}")
    
    # Load timestamp data
    df = load_timestamp_data(timestamp_path)
    
    # Process timestamps
    df['StartDatetime'] = pd.to_datetime(df['StartTimestamp'])
    df['Start_sec'] = (
        df['StartDatetime'] - df['StartDatetime'].iloc[0]
    ).dt.total_seconds()
    
    print(f"Loaded {len(df)} events from {timestamp_path}")
    
    # Classify event types
    df["Type"] = df["Section"].apply(classify_event_type)
    
    # Calculate event durations
    df["Duration"] = df["Start_sec"].shift(-1) - df["Start_sec"]
    
    # Get duration statistics for validation
    rest_durations = df[df["Type"] == "rest"]["Duration"].tolist()
    task_durations = df[df["Type"] == "task"]["Duration"].tolist()
    other_durations = df[df["Type"] == "other"]["Duration"].tolist()

    if not rest_durations or not task_durations:
        raise ValueError("Missing rest or task entries in the data.")
    
    # Calculate minimum duration for labeling
    rest_duration = min(rest_durations)
    task_duration = min(task_durations)
    other_duration = min(other_durations) if other_durations else 0
    durations = [task_duration, rest_duration]
    if other_duration == 0:
        min_duration_raw = min(rest_duration, task_duration)
    else:
        min_duration_raw = min(rest_duration, task_duration, other_duration)
    min_duration = min(rest_duration, task_duration)
    
    print(f"Event classification complete:")
    print(f"- Rest events: {len(rest_durations)} (min duration: {rest_duration:.1f}s)")
    print(f"- Task events: {len(task_durations)} (min duration: {task_duration:.1f}s)")
    
    # ===== CHANNEL SELECTION FOR INDIVIDUAL PLOTS =====
    print("\nStep 4: Channel Selection")
    print(f"Available channels: {ch_names}")
    
    # selected_channels = select_channels(ch_names)
    selected_channels=['Front, middle','Front, left','Front, right',]#FIXME
    if selected_channels:
        print(f"Selected channels for individual plots: {selected_channels}")
    else:
        print("No channels selected for individual plots.")
    
    print("\nSetup complete! Ready for analysis...")

    # analysis signal
    raw_data = raw.get_data()
    raw_data=raw_data[[0,1,3],:]#FIXME
    rows_to_keep = [0, 1, 3]#FIXME
    ch_names = [ch_names[i] for i in rows_to_keep]#FIXME
    #original data
    raw_data, df = original_data.plot_individual_raw_data(raw_data, ch_names, df, sfreq, selected_channels, output_dir,
                             filter_raw_signal=False, min_duration=min_duration_raw, ignore_cycles=IGNORE_CYCLES,file_name='Raw_Data_Original')     
    #filter the data
    raw_data, df = original_data.plot_individual_raw_data(raw_data, ch_names, df, sfreq, selected_channels, output_dir,
                             filter_raw_signal=True, min_duration=min_duration_raw, ignore_cycles=IGNORE_CYCLES,file_name='Raw_Data_Filtered')     
    #averaging on cycle time
    raw_data_cycle, df_cycle = averaging_time.compute_avg_events(raw_data, df, sfreq, ch_names, include_time=True, ignore_cycles=IGNORE_CYCLES)
    #averaging on cycle time for STFT
    raw_data_cycle_for_STFT, df_cycle_for_STFT = averaging_time_for_STFT.compute_avg_events(raw_data, df, sfreq, ch_names, include_time=True, ignore_cycles=IGNORE_CYCLES)
   # averaging after frequency transformation
    avg_spectrograms_dB, cycles = averaging_frquency.compute_average_cycle_spectrogram(
        raw_data, df, ch_names, sfreq, valid_freqs=VALID_FREQS, 
        output_dir=output_dir, window_seconds=2, overlap_ratio=0.5, 
        ignore_cycles=IGNORE_CYCLES, target_pre_rest=10.0, target_post_rest=20.0)
    save_path = averaging_frquency.plot_average_cycle_spectrogram(
        avg_spectrograms_dB, ch_names, valid_freqs=VALID_FREQS, output_dir=output_dir, 
        cycles=cycles, figsize=(18, 12))
    #FFT analysis 
    fft_concat_path, analysis_stats=FFT_concatenation.plot_concatenated_fft_segments(raw_data, df, ch_names, sfreq, VALID_FREQS, output_dir, 
                                 selected_channels=selected_channels, apply_windowing=False,file_name='fft_concatenated_segments')
    #FFT analysis on average cycle
    fft_concat_path_average, analysis_stats_average=FFT_concatenation.plot_concatenated_fft_segments(raw_data_cycle,df_cycle,  
                                ch_names, sfreq, VALID_FREQS, output_dir, 
                                 selected_channels=selected_channels, apply_windowing=False,file_name='fft_concatenated_segments_average')
    #STFT analysis
    save_path = STFT.plot_stft_analysis(
    raw_data=raw_data,ch_names=ch_names,df=df,sfreq=sfreq,valid_freqs=VALID_FREQS,output_dir=output_dir,   
    figsize=(20, 12), window_seconds=2,overlap_ratio=0.5,figure_name="STFT_Analysis")
    #STFT analysis on average time cycle
    save_path_average_time = STFT.plot_stft_analysis(
    raw_data=raw_data_cycle_for_STFT,ch_names=ch_names,df=df_cycle_for_STFT,sfreq=sfreq,valid_freqs=VALID_FREQS,output_dir=output_dir,   
    figsize=(20, 12), window_seconds=2,overlap_ratio=0.5,figure_name="STFT_Analysis_Average_Time_Cycle")
    #bining frequencies
    fig, ratios = bining.plot_frequency_band_ratio_analysis(raw_data=raw_data,df=df,ch_names=ch_names,
    sfreq=sfreq,output_dir=output_dir,valid_freqs=VALID_FREQS,durations=durations,save_name="frequency_analysis_from_raw")
 
