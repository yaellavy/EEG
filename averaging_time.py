import numpy as np
import pandas as pd

def compute_avg_events(raw, df_events, sfreq, ch_names, include_time=True, ignore_cycles=None):
    """
    Compute averaged task and averaged rest segments from NumPy array,
    and return the data with channels as rows.

    Args:
        raw (np.ndarray): EEG data [n_channels x n_samples]
        df_events (pd.DataFrame): events with 'Section', 'Type', 'Start_sec', 'Duration'
        sfreq (float): sampling frequency
        ch_names (list[str]): channel names
        include_time (bool): whether to include a 'Time (s)' column in the output
        ignore_cycles (list[int] or None): indices of cycles to skip (0-based)

    Returns:
        avg_data_df (pd.DataFrame): averaged data with channels as rows
        df_avg_events (pd.DataFrame): events corresponding to the averaged segments
    """

    # Remove initial/final rests
    df_filtered = df_events[
        ~(df_events["Section"].str.lower().str.contains("initial rest|final rest"))
    ].sort_values("Start_sec").reset_index(drop=True)

    # Separate task and rest events
    task_events = df_filtered[df_filtered["Type"] == "task"].reset_index(drop=True)
    rest_events = df_filtered[df_filtered["Type"] == "rest"].reset_index(drop=True)

    if len(task_events) == 0 or len(rest_events) == 0:
        raise ValueError("No task or rest events found after filtering initial/final rests.")

    print(f"Found {len(task_events)} task events and {len(rest_events)} rest events.")

    # Handle ignored cycles
    if ignore_cycles:
        # Convert ignore_cycles to set for faster lookup
        ignore_set = set(ignore_cycles)
        
        # Filter task events
        task_events = task_events[~task_events.index.isin(ignore_set)].reset_index(drop=True)
        
        # Filter rest events
        rest_events = rest_events[~rest_events.index.isin(ignore_set)].reset_index(drop=True)
        
        print(f"Ignored cycles: {sorted(ignore_set)}")
        print(f"Remaining task events: {len(task_events)}, rest events: {len(rest_events)}")

    if len(task_events) == 0 or len(rest_events) == 0:
        raise ValueError("All events were ignored. Nothing to average.")

    # Extract task segments
    task_segments = []
    for _, event in task_events.iterrows():
        start_sample = int(event["Start_sec"] * sfreq)
        end_sample = int((event["Start_sec"] + event["Duration"]) * sfreq)
        seg = raw[:, start_sample:end_sample]  # [n_channels x n_samples]
        task_segments.append(seg)

    # Extract rest segments
    rest_segments = []
    for _, event in rest_events.iterrows():
        start_sample = int(event["Start_sec"] * sfreq)
        end_sample = int((event["Start_sec"] + event["Duration"]) * sfreq)
        seg = raw[:, start_sample:end_sample]  # [n_channels x n_samples]
        rest_segments.append(seg)

    # Align task segments to shortest length
    min_task_len = min(seg.shape[1] for seg in task_segments)
    aligned_tasks = [seg[:, :min_task_len] for seg in task_segments]

    # Align rest segments to shortest length
    min_rest_len = min(seg.shape[1] for seg in rest_segments)
    aligned_rests = [seg[:, :min_rest_len] for seg in rest_segments]

    # Average across task segments -> (n_channels x n_samples)
    avg_task = np.stack(aligned_tasks, axis=0).mean(axis=0)

    # Average across rest segments -> (n_channels x n_samples)
    avg_rest = np.stack(aligned_rests, axis=0).mean(axis=0)

    # Create DataFrame with channels as rows
    avg_data = np.concatenate([avg_task, avg_rest], axis=1)  # [n_channels x (task_len + rest_len)]
    
    # Create DataFrame with channels as rows and time as columns
    avg_data_df = pd.DataFrame(avg_data, index=ch_names)

    # Add time information if needed
    if include_time:
        # Create time array for the concatenated data
        total_samples = avg_task.shape[1] + avg_rest.shape[1]
        times = np.arange(total_samples) / sfreq
        
        # You can either add time as a separate row or handle it differently
        # For now, we'll just print the time information
        print(f"Time range: 0 to {times[-1]:.2f} seconds")
        print(f"Task portion: 0 to {avg_task.shape[1] / sfreq:.2f} seconds")
        print(f"Rest portion: {avg_task.shape[1] / sfreq:.2f} to {times[-1]:.2f} seconds")

    # Create reduced events DataFrame with one task and one rest
    df_avg_events = pd.DataFrame({
        'Section': ['Averaged Task', 'Averaged Rest'],
        'Type': ['task', 'rest'],
        'Start_sec': [0, avg_task.shape[1] / sfreq],  # Start times in seconds
        'Duration': [avg_task.shape[1] / sfreq, avg_rest.shape[1] / sfreq]
    })

    print(f"Averaging complete:")
    print(f"- Task duration: {avg_task.shape[1] / sfreq:.2f}s ({avg_task.shape[1]} samples)")
    print(f"- Rest duration: {avg_rest.shape[1] / sfreq:.2f}s ({avg_rest.shape[1]} samples)")
    print(f"- Output shape: {avg_data_df.shape} (channels x timepoints)")
    avg_data_array = avg_data_df.values
    avg_data_array = avg_data_df.to_numpy()

    return avg_data_array, df_avg_events