import numpy as np
import pandas as pd

def compute_avg_events(raw, df_events, sfreq, ch_names, include_time=True, ignore_cycles=None, 
                      target_pre_rest=10.0, target_post_rest=20.0):
    """
    Compute averaged standardized cycles (10s pre-rest + task + 20s post-rest) from NumPy array,
    and return the data with channels as rows.

    Args:
        raw (np.ndarray): EEG data [n_channels x n_samples]
        df_events (pd.DataFrame): events with 'Section', 'Type', 'Start_sec', 'Duration'
        sfreq (float): sampling frequency
        ch_names (list[str]): channel names
        include_time (bool): whether to include a 'Time (s)' column in the output
        ignore_cycles (list[int] or None): indices of cycles to skip (0-based)
        target_pre_rest (float): duration of pre-task rest in seconds (default: 10.0)
        target_post_rest (float): duration of post-task rest in seconds (default: 20.0)

    Returns:
        avg_data_array (np.ndarray): averaged cycle data [n_channels x n_samples]
        df_avg_events (pd.DataFrame): events corresponding to the averaged cycle segments
    """

    # First, reclassify 'other' events as 'rest' if appropriate
    def reclassify_other_as_rest(df, rest_duration_threshold=None):
        df_modified = df.copy()
        rest_durations = df_modified[df_modified['Type'] == 'rest']['Duration'].tolist()
        
        if not rest_durations:
            return df_modified
            
        if rest_duration_threshold is None:
            median_rest_duration = np.median(rest_durations)
            rest_duration_threshold = median_rest_duration * 0.8
            
        other_mask = (df_modified['Type'] == 'other') & (df_modified['Duration'] >= rest_duration_threshold)
        df_modified.loc[other_mask, 'Type'] = 'rest'
        
        return df_modified

    df_modified = reclassify_other_as_rest(df_events)

    # Extract standardized cycles (10s pre-rest + task + 20s post-rest)
    def extract_standard_cycles(df, target_pre_rest=10.0, target_post_rest=20.0):
        df_sorted = df.sort_values('Start_sec').reset_index(drop=True)
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

            # Find PRE-TASK REST (10s ending at task_start)
            required_pre_rest_start = task_start - target_pre_rest
            pre_rest_found = False
            for _, rest_row in rest_events.iterrows():
                rest_start = rest_row['Start_sec']
                rest_end = rest_start + rest_row['Duration']
                if rest_start <= required_pre_rest_start and rest_end >= task_start:
                    pre_rest_found = True
                    break

            if not pre_rest_found:
                continue

            # Find POST-TASK REST (20s starting at task_end)
            required_post_rest_end = task_end + target_post_rest
            post_rest_found = False
            for _, rest_row in rest_events.iterrows():
                rest_start = rest_row['Start_sec']
                rest_end = rest_start + rest_row['Duration']
                if rest_start <= task_end and rest_end >= required_post_rest_end:
                    post_rest_found = True
                    break

            if not post_rest_found:
                continue

            # Create the standardized cycle
            cycle = {
                'cycle_start': required_pre_rest_start,
                'pre_rest_start': required_pre_rest_start,
                'pre_rest_end': task_start,
                'pre_rest_duration': target_pre_rest,
                'task_start': task_start,
                'task_end': task_end,
                'task_duration': task_duration,
                'post_rest_start': task_end,
                'post_rest_end': required_post_rest_end,
                'post_rest_duration': target_post_rest,
                'total_duration': target_pre_rest + task_duration + target_post_rest,
                'cycle_end': required_post_rest_end,
                'cycle_index': task_idx
            }
            cycles.append(cycle)

        return cycles

    # Extract all valid cycles
    cycles = extract_standard_cycles(df_modified, target_pre_rest, target_post_rest)
    
    if not cycles:
        raise ValueError("No valid standardized cycles found!")

    print(f"Found {len(cycles)} standardized cycles (10s pre-rest + task + 20s post-rest)")

    # Handle ignored cycles
    if ignore_cycles:
        ignore_set = set(ignore_cycles)
        cycles = [cycle for i, cycle in enumerate(cycles) if i not in ignore_set]
        print(f"Ignored cycles: {sorted(ignore_set)}")
        print(f"Remaining cycles: {len(cycles)}")

    if not cycles:
        raise ValueError("All cycles were ignored. Nothing to average.")

    # Extract complete cycle segments
    cycle_segments = []
    for cycle in cycles:
        start_sample = int(cycle['cycle_start'] * sfreq)
        end_sample = int(cycle['cycle_end'] * sfreq)
        seg = raw[:, start_sample:end_sample]  # [n_channels x n_samples]
        cycle_segments.append(seg)

    # Align all cycles to shortest length
    min_cycle_len = min(seg.shape[1] for seg in cycle_segments)
    aligned_cycles = [seg[:, :min_cycle_len] for seg in cycle_segments]

    # Average across all cycles -> (n_channels x n_samples)
    avg_cycle = np.stack(aligned_cycles, axis=0).mean(axis=0)

    # Create DataFrame with channels as rows
    avg_data_df = pd.DataFrame(avg_cycle, index=ch_names)

    # Calculate time information for the averaged cycle
    cycle_duration = min_cycle_len / sfreq
    pre_rest_duration = target_pre_rest
    task_duration = cycles[0]['task_duration']  # Use first cycle's task duration
    post_rest_duration = target_post_rest

    # Create events DataFrame for the averaged cycle
    df_avg_events = pd.DataFrame({
        'Section': ['Averaged Pre-Rest', 'Averaged Task', 'Averaged Post-Rest'],
        'Type': ['rest', 'task', 'rest'],
        'Start_sec': [0, pre_rest_duration, pre_rest_duration + task_duration],
        'Duration': [pre_rest_duration, task_duration, post_rest_duration]
    })

    if include_time:
        times = np.arange(avg_cycle.shape[1]) / sfreq
        print(f"Time range: 0 to {times[-1]:.2f} seconds")
        print(f"Pre-rest: 0 to {pre_rest_duration:.2f}s")
        print(f"Task: {pre_rest_duration:.2f} to {pre_rest_duration + task_duration:.2f}s")
        print(f"Post-rest: {pre_rest_duration + task_duration:.2f} to {times[-1]:.2f}s")

    print(f"Averaging complete:")
    print(f"- Pre-rest duration: {pre_rest_duration:.2f}s")
    print(f"- Task duration: {task_duration:.2f}s") 
    print(f"- Post-rest duration: {post_rest_duration:.2f}s")
    print(f"- Total cycle duration: {cycle_duration:.2f}s ({min_cycle_len} samples)")
    print(f"- Output shape: {avg_data_df.shape} (channels x timepoints)")
    
    avg_data_array = avg_data_df.to_numpy()

    return avg_data_array, df_avg_events