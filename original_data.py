import os
import tkinter as tk
import pandas as pd
import numpy as np
import mne
import re
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, fft, fftfreq
from tkinter import filedialog, messagebox
from scipy import signal
import cleaning_data
def remove_ignored_cycles(raw_data, df, sfreq, ignore_cycles=None, min_duration=30):
    """
    Remove cycles (time segments) from raw_data based on df indices to ignore.

    Parameters
    ----------
    raw_data : np.ndarray
        EEG data [n_channels x n_samples].
    df : pd.DataFrame
        Events table with Start_sec and Duration (if available).
    sfreq : float
        Sampling frequency (Hz).
    ignore_cycles : list[int] or None
        Indices of df rows (cycles) to ignore.
    min_duration : float
        Default duration to use if Duration not in df.

    Returns
    -------
    raw_data_clean : np.ndarray
        Cleaned data with ignored cycles removed.
    df_clean : pd.DataFrame
        Events dataframe with ignored cycles removed.
    """
    if ignore_cycles is None or len(ignore_cycles) == 0:
        return raw_data, df

    mask = np.ones(raw_data.shape[1], dtype=bool)  # keep everything initially

    for idx in ignore_cycles:
        if idx not in df.index:
            continue
        start_time = df.loc[idx, "Start_sec"]
        duration = df.loc[idx, "Duration"] if "Duration" in df.columns else min_duration
        start_sample = int(start_time * sfreq)
        end_sample = int((start_time + duration) * sfreq)
        mask[start_sample:end_sample] = False  # remove these samples

    raw_data_clean = raw_data[:, mask]
    df_clean = df.drop(index=ignore_cycles)

    return raw_data_clean, df_clean

def plot_individual_raw_data(raw_data, ch_names, df, sfreq, selected_channels, output_dir,
                             filter_raw_signal=True, min_duration=30, ignore_cycles=None, file_name='Raw_Data'):
    raw_data = np.asarray(raw_data)

    if filter_raw_signal:
        mask = np.abs(raw_data) <= 250
        raw_data_masked = np.where(mask, raw_data, np.nan)
        raw_data_masked = cleaning_data.cleaning_data(raw_data_masked)
        raw_data = raw_data_masked

    # Remove ignored cycles if requested
    raw_data, df = remove_ignored_cycles(raw_data, df, sfreq, ignore_cycles, min_duration)

    n_channels, n_samples = raw_data.shape
    time_axis = np.arange(n_samples) / sfreq

    fig, axes = plt.subplots(len(selected_channels), 1, figsize=(15, 4 * len(selected_channels)))
    plt.subplots_adjust(hspace=0.4)  # Add this line for vertical spacing
    if len(selected_channels) == 1:
        axes = [axes]

    for ax, channel in zip(axes, selected_channels):
        if channel not in ch_names:
            continue
        ch_idx = ch_names.index(channel)
        ax.plot(time_axis, raw_data[ch_idx, :], label=channel, color='blue')
        ax.set_title(f'{channel}')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(True, alpha=0.3)

        for idx, row in df.iterrows():
            event_time = row['Start_sec']
            event_type = row.get('Type', '')
            ax.axvline(event_time, color='red', linestyle='--', linewidth=1)
            ymin, ymax = ax.get_ylim()
            ax.text(event_time, 100, f"#{idx}", color='black', rotation=90,
                    fontsize=8, verticalalignment='top')
            ax.text(event_time + min_duration / 2, 200, event_type[0].upper(),
                    color='black', ha='center', fontsize=8)

        ax.axhline(y=250, color='green', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=-250, color='green', linestyle='--', linewidth=1, alpha=0.7)
    # plt.suptitle("Clean EEG Data", fontsize=16)
    # plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    save_path = os.path.join(output_dir, f"{file_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    return raw_data, df



def plot_raw_data_with_events(raw_data, df, ch_names, sfreq, output_dir, min_duration, 
                               filter_raw_signal=True, ignore_cycles=None,file_name='Raw_Data'):
    raw_data = np.asarray(raw_data)

    if filter_raw_signal:
        mask = np.abs(raw_data) <= 250
        raw_data_masked = np.where(mask, raw_data, np.nan)
        raw_data_masked = cleaning_data.cleaning_data(raw_data_masked)
        raw_data = raw_data_masked

    # Remove ignored cycles
    raw_data, df = remove_ignored_cycles(raw_data, df, sfreq, ignore_cycles, min_duration)

    n_channels, n_samples = raw_data.shape
    time_axis = np.arange(n_samples) / sfreq

    nrows, ncols = 3, 4
    fig_obs, axes_obs = plt.subplots(nrows, ncols, figsize=(18, 10))
    axes_obs = axes_obs.flatten()
    # fig_obs.subplots_adjust(hspace=0.5)
    fig_obs.subplots_adjust(hspace=4, wspace=0.3)  # Add wspace for horizontal spacing
    # fig_obs.tight_layout(pad=4.0)
    fig_obs.subplots_adjust(hspace=1, wspace=0.3)  # Changed from hspace=4


    for i in range(min(n_channels, nrows * ncols)):
        ax = axes_obs[i]
        ax.plot(time_axis, raw_data[i, :], label=ch_names[i])
        ax.set_title(ch_names[i])
        if i == 0:
            ax.legend()

    for j in range(i + 1, nrows * ncols):
        fig_obs.delaxes(axes_obs[j])

    for idx, row in df.iterrows():
        event_time = row['Start_sec']
        event_type = row.get('Type', '')
        for ax in axes_obs:
            ax.axvline(event_time, color='red', linestyle='--', linewidth=1)
            ymin, ymax = ax.get_ylim()
            ax.text(event_time, 100, f"#{idx}", color='black', rotation=90, fontsize=8)
            ax.text(event_time + min_duration / 2, 200, event_type[0].upper(),
                    color='black', ha='center', fontsize=8)
            ax.axhline(y=250, color='green', linestyle='--', linewidth=1)
            ax.axhline(y=-250, color='green', linestyle='--', linewidth=1)

    raw_data_concat_path = os.path.join(output_dir, f"{file_name}.png")
    fig_obs.savefig(raw_data_concat_path, dpi=300)
    # plt.show()

    return raw_data_concat_path, raw_data, df
