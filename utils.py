import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from scipy import io

sys.path.append('..')

# Set publication quality parameters globally
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'lines.linewidth': 2
})

def extract_capacity(cycles, discharge_indices):
    """
    Extracts the capacity value for each discharge cycle.
    """
    capacity_list = []
    for index in discharge_indices:
        # data structure: cycles[0, index][3][0,0][6] is the capacity field
        cap = cycles[0,index][3][0,0][6]
        if np.size(cap) > 0:
            capacity_list.append(float(cap.flatten()[0]))
        else:
            capacity_list.append(np.nan)
    return np.array(capacity_list)

def capacity_vectorizer(discharge_indices, cycles):
    """
    Pads the capacity scalar to match the dimensions of the time vector.
    """
    for index in discharge_indices:
        # Get number of time steps (index 5 contains time)
        N = (cycles[0,index][3][0,0][5]).shape[1]
        capacity_scalar = cycles[0,index][3][0,0][6]
        
        # Pad with the single scalar value
        cycles[0,index][3][0,0][6] = np.pad(capacity_scalar.flatten().tolist(), (0, N-1), 'constant')
        
    return cycles

def get_indices(cycles, is_charge=True):
    """Returns a list of indices for either charge or discharge cycles."""
    label = 'charge' if is_charge else 'discharge'
    index_list = [i for i in range(cycles.shape[1]) if cycles[0,i][0] == np.array([label])]
    return index_list

# --- Visualization Functions ---

def cycle_plotter(cycles, discharge_indices, cycle_indices):
    """
    Plots Voltage, Current, and Temp for specific cycles.
    """
    features = ['Voltage Measured (V)', 'Current Measured (A)', 'Temperature (C)', 
                'Current Load/Charge (A)', 'Voltage Load/Charge (V)']  

    cmap = plt.get_cmap('tab10')

    # Ensure directories exist
    for path in ['Figures', 'Figures/PDF', 'Figures/PNG']:
        if not os.path.exists(path):
            os.makedirs(path)

    for i, label in enumerate(features):
        plt.figure(figsize=(10, 6))
        color_idx = 0
        
        for discharge_cycle_number, discharge_index in enumerate(discharge_indices):
            if (discharge_cycle_number + 1) in cycle_indices:
                data_struct = cycles[0,discharge_index][3][0,0]

                try:
                    y = np.array(data_struct[i]).flatten()
                    x = np.array(data_struct[5]).flatten()
                    
                    if len(y) != len(x):
                        print(f"Warning: dimension mismatch in Cycle {discharge_cycle_number+1}. Skipping.")
                        continue
                        
                    color = cmap(color_idx % 10)
                    plt.plot(x, y, linestyle='-', label=f'Cycle {discharge_cycle_number+1}', color=color, alpha=0.8)
                    color_idx += 1
                except Exception as e:
                    print(f"Error plotting Cycle {discharge_cycle_number+1}: {e}")
                    continue

        plt.ylabel(label)
        plt.xlabel('Time (s)')   
        plt.title(label) 
        plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='best')
        plt.tight_layout()
        
        clean_label = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        plt.savefig(f'Figures/PDF/{clean_label}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'Figures/PNG/{clean_label}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_capacity_vs_cycle(data_dict=None, cycles=None, discharge_indices=None, label='Battery'):
    """
    Plots capacity vs cycle number. Handles both single battery and dictionary of batteries.
    """
    # Ensure directories exist
    for path in ['Figures', 'Figures/PDF', 'Figures/PNG']:
        if not os.path.exists(path):
            os.makedirs(path)

    # Normalize input
    if data_dict is not None:
        batteries_to_plot = data_dict
    elif cycles is not None and discharge_indices is not None:
        batteries_to_plot = {label: (cycles, discharge_indices)}
    else:
        print("Error: Must provide either data_dict or (cycles, discharge_indices)")
        return

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    color_idx = 0
    
    for batt_label, (batt_cycles, batt_dis_indices) in batteries_to_plot.items():
        capacities = []
        cycle_numbers = []
        
        for i, discharge_index in enumerate(batt_dis_indices):
            try:
                cap = (batt_cycles[0,discharge_index][3][0,0][6]).flatten().tolist()[0]
                capacities.append(cap)
                cycle_numbers.append(i + 1)
            except Exception:
                continue
        
        color = cmap(color_idx % 10)
        plt.plot(cycle_numbers, capacities, marker='o', markersize=4, linestyle='-', 
                 color=color, label=batt_label)
        color_idx += 1
        
    plt.ylabel('Capacity (Ah)')
    plt.xlabel('Cycle Number')
    plt.title('Battery Capacity Degradation')
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='best')
    plt.tight_layout()
    
    # Save figure
    filename_base = 'Capacity_vs_Cycle'
    if len(batteries_to_plot) == 1:
        clean_label = list(batteries_to_plot.keys())[0].replace(" ", "_")
        filename_base = f'Capacity_vs_Cycle_{clean_label}'
        
    plt.savefig(f'Figures/PDF/{filename_base}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'Figures/PNG/{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- Feature Extraction Functions ---

def extract_feature_1_2_6_7(indices, cycles, l_threshold=250, r_threshold=750, peak_width=3):
    """
    Returns (time_of_max_temp, max_temp) for each cycle.
    """
    max_temp_time_list = []
    max_temp_list = []
    
    for index in indices:
        times = cycles[0,index][3][0,0][5].flatten()
        temps = cycles[0,index][3][0,0][2].flatten()
        
        # Filter data within time thresholds
        # Using boolean masking is cleaner and faster than generators
        mask = (times > l_threshold) & (times < r_threshold)
        
        # Fallback if no data in window
        if not np.any(mask): 
            # If empty, just search the whole thing or a default range
            # Here we just take the last 10 points to avoid crashing
            mask = np.arange(len(times)) > 0 

        times_window = times[mask]
        temps_window = temps[mask]

        decreasing_count = 0
        max_temp_index = -1
        
        # Peak detection logic
        for i, temp in enumerate(temps_window):
            if i == 0: continue
            if temp < temps_window[i-1]:
                decreasing_count += 1
                if decreasing_count == peak_width:
                    max_temp_index = i - (peak_width - 1)
                    break
            else:
                decreasing_count = 0
        
        if max_temp_index == -1:
            max_temp_time_list.append(np.nan)
            max_temp_list.append(np.nan)
        else:
            max_temp_time_list.append(times_window[max_temp_index])
            max_temp_list.append(temps_window[max_temp_index])
            
    return max_temp_time_list, max_temp_list

def extract_feature_3(discharge_indices, cycles, l_threshold=250, r_threshold=750, peak_width=3):
    """
    Calculates the slope of temperature rise: (Max_Temp - Initial_Temp) / Time_to_Max
    """
    max_temp_times, max_temps = extract_feature_1_2_6_7(discharge_indices, cycles, l_threshold, r_threshold, peak_width)
    
    initial_temps = []
    for index in discharge_indices:
        initial_temp = cycles[0,index][3][0,0][2].flatten()[0]
        initial_temps.append(initial_temp)
        
    slopes = (np.array(max_temps) - np.array(initial_temps)) / np.array(max_temp_times)
    return slopes.tolist()

def extract_feature_4(dataset, indices, threshold=500, voltage_cutoff=3):
    """
    Time for voltage to drop below 3V during discharge (after threshold time).
    """
    feature_4_list = []
    for index in indices:
        v = (dataset[0,index][3][0,0][0]).flatten()
        t = (dataset[0,index][3][0,0][5]).flatten()
        
        # Filter out early noise/relaxation
        mask = t > threshold
        v_active = v[mask]
        t_active = t[mask]
        
        # Find indices where voltage < cutoff
        drop_indices = np.where(v_active < voltage_cutoff)[0]
        
        if len(drop_indices) > 0:
            feature_4_list.append(t_active[drop_indices[0]])
        else:
            feature_4_list.append(np.nan)
            
    return feature_4_list

def extract_feature_5(dataset, indices, start_time=100, end_time=500):
    """
    Slope of voltage discharge curve between start_time and end_time.
    """
    feature_5_list = []

    for index in indices:
        v = (dataset[0,index][3][0,0][0]).flatten()
        t = (dataset[0,index][3][0,0][5]).flatten()
        
        # Find indices closest to requested timestamps
        start_idx = (np.abs(t - start_time)).argmin()
        end_idx = (np.abs(t - end_time)).argmin()
        
        dv = v[end_idx] - v[start_idx]
        dt = t[end_idx] - t[start_idx]
        
        if dt != 0:
            feature_5_list.append(dv / dt)
        else:
            feature_5_list.append(np.nan)
  
    return feature_5_list

def extract_label(dataset, indices):
    """Extracts the label (capacity) for the given indices."""
    labels = []
    for index in indices:
        label = (dataset[0,index][3][0,0][6]).flatten().tolist()
        labels.append(label[0])
    return labels

def remaining_cycles(dataset, indices, threshold=0.7):
    """
    Calculates Remaining Useful Life (RUL) based on a capacity threshold.
    """
    initial_capacity = (dataset[0,1][3][0,0][6]).flatten()[0]
    cutoff = initial_capacity * threshold
    
    # Find the critical cycle index where capacity drops below cutoff
    critical_cycle = len(indices) - 1 # Default to end if never reached
    
    for i, idx in enumerate(indices):
        cap = (dataset[0,idx][3][0,0][6]).flatten()[0]
        if cap < cutoff:
            critical_cycle = i - 1
            break

    remaining_cycles_list = [(critical_cycle - j) for j in range(len(indices))]
    return remaining_cycles_list

def extract_voltage_drop_interval(dataset, indices, v_upper=4.1, v_lower=3.9):
    """
    Calculates the time duration for voltage to drop from v_upper to v_lower.
    Useful for partial discharge analysis.
    """
    duration_list = []
    for index in indices:
        voltages = (dataset[0,index][3][0,0][0]).flatten()
        times = (dataset[0,index][3][0,0][5]).flatten()
        
        try:
            # Find first index where voltage drops below v_upper
            idx_start = np.argmax(voltages < v_upper)
            
            # If argmax returns 0, check if it's actually valid (voltage[0] < upper)
            # or if it never dropped (all > upper)
            if voltages[idx_start] >= v_upper:
                duration_list.append(np.nan)
                continue

            # Find first index where voltage drops below v_lower, starting search after idx_start
            idx_end_offset = np.argmax(voltages[idx_start:] < v_lower)
            idx_end = idx_start + idx_end_offset
            
            if voltages[idx_end] >= v_lower:
                duration_list.append(np.nan)
                continue
            
            time_diff = times[idx_end] - times[idx_start]
            duration_list.append(time_diff)
            
        except Exception:
            duration_list.append(np.nan)
            
    return duration_list

def extract_voltage_slope(dataset, indices, start_soc_v=4.0, end_soc_v=3.8):
    """
    Calculates the slope (dV/dt) within a specific voltage window.
    """
    slope_list = []
    
    for index in indices:
        try:
            voltages = (dataset[0,index][3][0,0][0]).flatten()
            times = (dataset[0,index][3][0,0][5]).flatten()
            
            # Find indices closest to voltage points
            idx_start = (np.abs(voltages - start_soc_v)).argmin()
            
            # Find end index, ensuring it's chronologically after start
            idx_end_offset = (np.abs(voltages[idx_start:] - end_soc_v)).argmin()
            idx_end = idx_start + idx_end_offset

            v_start, t_start = voltages[idx_start], times[idx_start]
            v_end, t_end = voltages[idx_end], times[idx_end]
            
            if t_end - t_start == 0:
                slope = np.nan
            else:
                slope = (v_end - v_start) / (t_end - t_start)
                
            slope_list.append(slope)
            
        except Exception:
            slope_list.append(np.nan)
            
    return slope_list

def extract_segment_features(dataset, indices, start_soc_v=4.0, end_soc_v=3.8):
    """
    Extracts multiple features (slope, duration, variance, current) from a voltage segment.
    """
    features = {
        'slope': [],
        'duration': [],
        'voltage_variance': [],
        'mean_current': []
    }
    
    for index in indices:
        try:
            voltages = (dataset[0,index][3][0,0][0]).flatten()
            times = (dataset[0,index][3][0,0][5]).flatten()
            currents = (dataset[0,index][3][0,0][1]).flatten()
            
            idx_start = (np.abs(voltages - start_soc_v)).argmin()
            idx_end_offset = (np.abs(voltages[idx_start:] - end_soc_v)).argmin()
            idx_end = idx_start + idx_end_offset
            
            # Extract segment
            seg_v = voltages[idx_start:idx_end+1]
            seg_t = times[idx_start:idx_end+1]
            seg_i = currents[idx_start:idx_end+1]
            
            duration = seg_t[-1] - seg_t[0] if len(seg_t) > 1 else np.nan
            
            if duration > 0 and len(seg_v) > 1:
                slope = (seg_v[-1] - seg_v[0]) / duration
                variance = np.var(seg_v)
                mean_current = np.mean(np.abs(seg_i))
                
                features['slope'].append(slope)
                features['duration'].append(duration)
                features['voltage_variance'].append(variance)
                features['mean_current'].append(mean_current)
            else:
                for key in features: features[key].append(np.nan)
            
        except Exception:
            for key in features: features[key].append(np.nan)
    
    return features

def extract_multi_window_features(dataset, indices, windows=[(4.2, 4.0), (4.0, 3.8), (3.8, 3.6), (3.6, 3.4)]):
    """
    Extracts slope features for multiple voltage windows.
    """
    features = {}
    for w in windows:
        v_start, v_end = w
        feat_name = f'slope_{str(v_start).replace(".","")}_{str(v_end).replace(".","")}'
        features[feat_name] = extract_voltage_slope(dataset, indices, start_soc_v=v_start, end_soc_v=v_end)
    return features

def generate_universal_segment_dataset(battery_ids, datapath='raw_files/', 
                                     windows=None, num_segments=10, 
                                     ambient_temp=None, mode='uniform'):
    """
    Creates a 'Long Format' dataset where Start_V and End_V are explicit features.
    """
    all_dfs = []
    
    for battery_id in battery_ids:
        # Construct path based on NASA dataset folder structure
        bat_num = int(battery_id[1:])
        if battery_id in ['B0005', 'B0006', 'B0007', 'B0018']: subfolder = 'Battery5-18'
        elif 25 <= bat_num <= 44: subfolder = 'Battery25-44'
        elif 45 <= bat_num <= 48: subfolder = 'Battery45-48'
        elif 49 <= bat_num <= 52: subfolder = 'Battery49-52'
        elif 53 <= bat_num <= 56: subfolder = 'Battery53-56'
        else: subfolder = None

        path = None
        if subfolder:
            path = os.path.join(datapath, subfolder, f'{battery_id}.mat')
        
        # Fallback search if path construction failed or file missing
        if not path or not os.path.exists(path):
            for root, _, files in os.walk(datapath):
                if f'{battery_id}.mat' in files:
                    path = os.path.join(root, f'{battery_id}.mat')
                    break
        
        if not path:
            print(f"Could not find file for {battery_id}")
            continue
             
        try:
            dict_data = io.loadmat(path)
            raw_cycles = np.vstack(dict_data[battery_id][0,0])
            dis_indices = get_indices(raw_cycles, is_charge=False)
            capacity = extract_capacity(raw_cycles, dis_indices) 

            current_windows = windows
            # Auto-generate windows if needed
            if current_windows is None:
                if len(dis_indices) > 0:
                    first_cycle_idx = dis_indices[0]
                    v_curve = raw_cycles[0, first_cycle_idx][3][0,0][0].flatten()
                    v_max, v_min = np.max(v_curve), np.min(v_curve)
                    
                    if mode == 'random':
                        current_windows = []
                        for _ in range(num_segments):
                            # Retry loop to find valid window with min gap
                            for attempt in range(10):
                                pts = np.random.uniform(v_min, v_max, 2)
                                if abs(pts[0] - pts[1]) > 0.1:
                                    current_windows.append((max(pts), min(pts)))
                                    break
                    else: 
                        # Uniform mode
                        v_levels = np.linspace(v_max, v_min, num_segments + 1)
                        current_windows = [(v_levels[i], v_levels[i+1]) for i in range(len(v_levels)-1)]
                else:
                    current_windows = [(4.2, 3.0)] # Fallback
            
            # Process each window
            for v_start, v_end in current_windows:
                segment_features = extract_segment_features(raw_cycles, dis_indices, start_soc_v=v_start, end_soc_v=v_end)
                
                min_len = min(len(segment_features['slope']), len(capacity))
                capacity_subset = capacity[:min_len]

                data_dict = {
                    'slope': segment_features['slope'][:min_len],
                    'duration': segment_features['duration'][:min_len],
                    'mean_current': segment_features['mean_current'][:min_len],
                    'start_v': v_start, 
                    'end_v': v_end,     
                    'capacity': capacity_subset,
                    'battery_id': battery_id,
                    'cycle_num': range(min_len)
                }
                
                if ambient_temp is not None:
                     data_dict['ambient_temp'] = ambient_temp
                     
                df_window = pd.DataFrame(data_dict)
                all_dfs.append(df_window.dropna())
                
        except Exception as e:
            print(f"Skipping {battery_id}: {e}")
            
    if not all_dfs:
         return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def create_n_step_target(df, steps=20, scale_factor=1000, smooth_window=5):
    """
    Creates target variable: future capacity change rate (n-step lookahead).
    """
    df_list = []

    for b_id in df['battery_id'].unique():
        temp_df = df[df['battery_id'] == b_id].copy()
        temp_df.sort_values('cycle_num', inplace=True)
        
        if smooth_window > 1:
             smoothed_capacity = temp_df['capacity'].rolling(window=smooth_window, center=True, min_periods=1).mean()
             future_cap = smoothed_capacity.shift(-steps)
             current_cap = smoothed_capacity
        else:
             future_cap = temp_df['capacity'].shift(-steps)
             current_cap = temp_df['capacity']

        temp_df['target_rate'] = ((future_cap - current_cap) / steps) * scale_factor
        df_list.append(temp_df.dropna(subset=['target_rate']))

    return pd.concat(df_list)