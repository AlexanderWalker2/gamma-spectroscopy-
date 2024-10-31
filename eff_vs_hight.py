import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import glob

Iso_peaks = {
    'co_60': 1173.2,
    'na': 510.99
}

def load_files(file_path):
    print(f"Processing file: {file_path}")
    numbers = []
    within_data_section = False
    measured_time = None
    mca_cal_value = 1

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "$DATA:":
                within_data_section = True
                continue
            if line.strip() == "$ROI:":
                within_data_section = False
            if within_data_section:
                try:
                    num = float(line.split()[0])
                    numbers.append(num)
                except ValueError:
                    continue
            if line.strip() == "$MCA_CAL:":
                for _ in range(3):
                    cal_line = file.readline()
                    if cal_line:
                        parts = cal_line.split()
                        if len(parts) >= 2:
                            try:
                                mca_cal_value = float(parts[1])
                            except ValueError:
                                pass
            if line.strip() == "$MEAS_TIM:":
                cal_line = file.readline()
                if cal_line:
                    parts = cal_line.split()
                    try:
                        measured_time = float(parts[1])
                    except ValueError:
                        pass

    counts = np.array(numbers[501:])
    bins = np.arange(1, 1 + len(counts))
    return counts, bins, measured_time, mca_cal_value

def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def process_file(file_path, background_counts, background_time):
    counts, bins, measured_time, mca_cal_value = load_files(file_path)
    scale_factor = measured_time / background_time
    counts_filtered = counts - background_counts * scale_factor
    peak_indices, peak_dict = sc.signal.find_peaks(counts_filtered, prominence=100, height=1, width=10)
    main_peak_idx = peak_indices[np.argmax(peak_dict['peak_heights'])]
    popt, _ = sc.optimize.curve_fit(
        gaussian, bins[main_peak_idx - 100:main_peak_idx + 100], counts_filtered[main_peak_idx - 100:main_peak_idx + 100],
        p0=[counts_filtered[main_peak_idx], main_peak_idx, 1]
    )
    mca = Iso_peaks[file_path[:2]] / popt[1]
    popt[1:] = popt[1:] * mca
    sum_counts = np.sum(counts_filtered[int(popt[1] / mca - 50):int(popt[1] / mca + 50)])
    efficiency = sum_counts / (4.2e5 * measured_time)
    return popt, mca, efficiency, counts_filtered, bins

def plotting(all_peaks, efficiencies, heights):
    num_plots = len(all_peaks)
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14), sharex=True)
    axes = axes.ravel()

    for idx, (counts_filtered, bins, peak, mca, file_name) in enumerate(all_peaks):
        energy = bins * mca
        axes[idx].bar(energy, counts_filtered, color='gray', alpha=0.3)
        peak_energy = peak[1]
        axes[idx].plot(energy[int(peak_energy) - 100:int(peak_energy) + 100], gaussian(energy[int(peak_energy) - 100:int(peak_energy) + 100], *peak), color='orange')
        axes[idx].set_title(f"Peak for {file_name}")
        axes[idx].set_xlim(peak_energy - 50, peak_energy + 50)
        axes[idx].set_xlabel("Energy [keV]")
        axes[idx].set_ylabel("Counts")

    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.figure(figsize=(8, 6))
    
    plt.scatter(heights, efficiencies, marker='x', color='blue')
    plt.xlabel("Height [cm]")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    background_file = glob.glob(r'**/background.spe')[0]
    background_counts, _, background_time, _ = load_files(background_file)

    all_peaks = []
    efficiencies = []
    heights = []

    for file in glob.glob(r'na_4kv_10mins\*.spe', recursive=True):
        if 'background' in file.lower():
            continue
        popt, mca, efficiency, counts_filtered, bins = process_file(file, background_counts, background_time)
        all_peaks.append((counts_filtered, bins, popt, mca, file))
        efficiencies.append(efficiency)
        height = int(file.split('_')[-1].replace('cm.Spe', ''))
        heights.append(height)

    plotting(all_peaks, efficiencies, heights)
