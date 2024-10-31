import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import glob, os

Iso_peaks = {
    'co_60' : 1173.2,
    'na' : 510.99
}

def load_files(file_path):
    print(f"Processing file: {file_path}")
    numbers = []
    within_data_section = False
    measured_time = None
    
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
                for _ in range(1):
                    cal_line = file.readline()
                    if cal_line:
                        parts = cal_line.split()
                        try:
                            measured_time = float(parts[1])
                        except ValueError:
                            pass

    counts = np.array(numbers[501:])
    bins = np.array(range(1,1+np.size(counts)))

    print(f'counts prossesed - {np.sum(counts)}, number of bins - {np.size(bins)}')

    return counts, bins, measured_time

def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def inv_gauss(x, amp, mean, std_dev):
    return ((-2*np.log(x/amp))**(1/2)*std_dev+mean)

def fwnm_eq(n, amp, mean, std_dev):
    return (2*np.log(1/n))**(1/2)*std_dev+mean, -(2*np.log(1/n))**(1/2)*std_dev+mean

def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def prosessing(file_path):
    scale_factor = measured_time/background_array[2]
    counts_filterd = counts - background_array[0]*scale_factor

    width = 100
    activity = 4.2e5
    emitted = activity*measured_time

    peak_indices, peak_dict = sc.signal.find_peaks(counts_filterd, prominence = 100, height=1, width=10)
    peak_heights = peak_dict['peak_heights']

    peaks = [peak_indices[np.argmax(peak_heights)]]
    num = np.size(peaks)
    [print(f'peaks found at {c}') for c in peaks]

    param_array = np.zeros((num,3))
    covariance_array = np.zeros((num,4))
    fitted_gaussians = np.zeros((num,width*2))
    fwhm_array = np.zeros((num,1))
    fwtm_array = np.zeros((num,1))

    c = 0 #used to be a for loop. for multiple peaks
    initial_guess = [counts_filterd[peaks[c]],peaks[c], 1]
    popt, pcov = sc.optimize.curve_fit(gaussian, bins[peaks[c]-width:peaks[c]+width], counts_filterd[peaks[c]-width:peaks[c]+width], p0=initial_guess)
    mca = Iso_peaks[file_path[:2]]/popt[1]
    print(f'mca value - {mca}')
    popt[1:] = popt[1:]*mca
    param_array[c] = popt
    covariance_array[c] = [np.linalg.cond(pcov), *np.diag(pcov)]
    fwhm_array[c] = 2 * (fwnm_eq(1/2, *popt)[0]-popt[1])
    fwtm_array[c] = 2 * (fwnm_eq(1/10, *popt)[0]-popt[1])
    fitted_gaussians[c] = gaussian(bins[peaks[c]-width:peaks[c]+width], *popt)
    sum_counts = sum(counts_filterd[int(popt[1]/mca-fwhm_array[c]/mca):int(popt[1]/mca+fwhm_array[c]/mca)])
    eff = sum_counts / emitted
    print(f"Peak {c+1}:")
    print(f"  Amplitude = {popt[0]:.2f}, Mean = {popt[1]:.2f} keV, Std Dev = {popt[2]:.2f} keV")
    print(f"  FWHM = {fwhm_array[c][0]:.2f} keV")
    print(f"  Covariance Condition Number = {covariance_array[c][0]:.2e}")
    print(f"  Covariance Diagonal (Variance of Amplitude, Mean, Std Dev) = {covariance_array[c][1]:.2e}, {covariance_array[c][2]:.2e}, {covariance_array[c][3]:.2e}")
    print(f'  Sum of counts at peak - {sum_counts:.2f}')
    print(f'  Efficiency at peak - {eff:.6f}')
    print(f'  Energy resolution - {(fwhm_array[c]/popt[1]*100)[0]:.3f}')
    print(f'  Peak shape fwhm / fwtm - {(fwtm_array[c]/fwhm_array[c])[0]:.3f}')
    #print(f'  Relative efficency - {popt[1]}')
    print("--------------------------------------------------------------")
    peaks_array.append([popt, fwhm_array[c][0], eff, (fwhm_array[c]/popt[1]*100)[0], (fwtm_array[c]/fwhm_array[c])[0]])
    return  param_array, fwhm_array, fwtm_array, peaks, num, counts_filterd, mca


def plotting():
    d = 100
    first_peak = peaks[0]
    energy = bins * mca

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main energy spectrum plot
    ax1.bar(energy, counts, edgecolor='grey', alpha=0.3)
    ax1.bar(energy, counts_filterd, edgecolor='b', alpha=0.7)
    for c in range(num):
        ax1.plot(energy[peaks[c] - width:peaks[c] + width], gaussian(energy[peaks[c] - width:peaks[c] + width], *param_array[c]), c='tab:orange')
    ax1.set_xlabel(r'Energy [$KeV$]')
    ax1.set_ylabel('Counts')
    ax1.set_title('Full Energy Spectrum')
    
    # Zoomed-in plot around the first peak
    ax2.bar(energy[first_peak - d:first_peak + d], counts[first_peak - d:first_peak + d], edgecolor='grey', alpha=0.3)
    
    for c in range(num):
        # Add FWHM and FWTM lines for first peak
        ax2.plot([param_array[c][1] - fwhm_array[c] / 2, param_array[c][1] - fwhm_array[c] / 2], 
                 [0, gaussian(param_array[c][1] - fwhm_array[c] / 2, *param_array[c])[0]], linestyle='--', c='r')
        ax2.plot([param_array[c][1] + fwhm_array[c] / 2, param_array[c][1] + fwhm_array[c] / 2], 
                 [0, gaussian(param_array[c][1] - fwhm_array[c] / 2, *param_array[c])[0]], linestyle='--', c='r')
        ax2.plot([param_array[c][1] - fwtm_array[c] / 2, param_array[c][1] - fwtm_array[c] / 2], 
                 [0, gaussian(param_array[c][1] - fwtm_array[c] / 2, *param_array[c])[0]], linestyle='--', c='b')
        ax2.plot([param_array[c][1] + fwtm_array[c] / 2, param_array[c][1] + fwtm_array[c] / 2], 
                 [0, gaussian(param_array[c][1] - fwtm_array[c] / 2, *param_array[c])[0]], linestyle='--', c='b')
        # Plot Gaussian fit for the first peak in zoomed-in section
        ax2.plot(energy[peaks[c] - width:peaks[c] + width], gaussian(energy[peaks[c] - width:peaks[c] + width], *param_array[c]), c='tab:orange')
    
    ax2.set_xlabel(r'Energy [$KeV$]')
    ax2.set_ylabel('Counts')
    ax2.set_title('Zoomed on First Peak')
    ax2.set_xlim(energy[first_peak - d], energy[first_peak + d])

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    global counts, bins, mca, param_array, fwhm_array, peaks, num, width, measured_time, counts_filterd
    global peaks_array
    peaks_array = []
    width = 100
    background_array = load_files(glob.glob(r'**/background.spe')[0])
    print(background_array)
    for file in glob.glob(r'**/*.spe', recursive=True):
        #print(f"Processing file: {file}")
        counts, bins, measured_time = load_files(file)
        param_array, fwhm_array, fwtm_array, peaks, num, counts_filterd, mca = prosessing(file)
        plotting()
        print('\n\n')