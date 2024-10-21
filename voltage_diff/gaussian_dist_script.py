import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import glob, os

def load_files(file_path):
    numbers = []
    within_data_section = False
    mca_cal_value = None
    
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
    counts = np.array(numbers[1:])
    mca = mca_cal_value
    bins = np.array(range(1,1+np.size(counts)))

    print(f'counts prossesed - {np.sum(counts)}, mca value - {mca}, number of bins - {np.size(bins)}')

    return counts, bins, mca



def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def prosessing():
    energy = bins*mca
    start_val = 400
    peak_indices, peak_dict = sc.signal.find_peaks(counts[energy > 400], prominence = 100, height=1, width=10)
    peak_heights = peak_dict['peak_heights']
    peak_indices=peak_indices + int(start_val/mca)
    peaks = [peak_indices[np.argmax(peak_heights)], peak_indices[np.argpartition(peak_heights,-2)[-2]]]
    num = np.size(peaks)

    [print(f'peaks found at {c}') for c in peaks]

    width = 100

    def gaussian(x, amp, mean, std_dev):
        return amp * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    param_array = np.zeros((num,3))
    covariance_array = np.zeros((num,4))
    fitted_gaussians = np.zeros((num,width*2))
    fwhm_array = np.zeros((num,1))

    
    for c in range(0, num):
        print(f'for peak, {peaks[c]}')
        initial_guess = [counts[peaks[c]],peaks[c], 10]
        popt, pcov = sc.optimize.curve_fit(gaussian, bins[peaks[c]-width:peaks[c]+width], counts[peaks[c]-width:peaks[c]+width], p0=initial_guess)

        popt[1:] = popt[1:]*mca
        param_array[c] = popt
        covariance_array[c] = [np.linalg.cond(pcov), *np.diag(pcov)]
        fitted_gaussians[c] = gaussian(bins[peaks[c]-width:peaks[c]+width], *popt)
        fwhm_array[c] = 2 * np.sqrt(2 * np.log(2)) * popt[2]

        print(f"Peak {c+1}:")
        print(f"  Amplitude = {popt[0]:.2f}, Mean = {popt[1]:.2f} keV, Std Dev = {popt[2]:.2f} keV")
        print(f"  FWHM = {fwhm_array[c][0]:.2f} keV")
        print(f"  Covariance Condition Number = {covariance_array[c][0]:.2e}")
        print(f"  Covariance Diagonal (Variance of Amplitude, Mean, Std Dev) = {covariance_array[c][1]:.2e}, {covariance_array[c][2]:.2e}, {covariance_array[c][3]:.2e}")
        print("--------------------------------------------------------------")

    return  param_array, fwhm_array, peaks, num


def plotting():
    print('plotting')
    d=100
    energy = bins*mca

    plt.bar(energy[min(peaks)-d:max(peaks)+d], counts[min(peaks)-d:max(peaks)+d], edgecolor='grey', alpha = 0.3)
    [plt.plot([param_array[c][1]-fwhm_array[c]/2,param_array[c][1]-fwhm_array[c]/2], [0, gaussian(param_array[c][1]-fwhm_array[c]/2, *param_array[c])[0]],linestyle = '--', c = 'g') for c in range(0,num)]
    [plt.plot([param_array[c][1]+fwhm_array[c]/2,param_array[c][1]+fwhm_array[c]/2], [0, gaussian(param_array[c][1]-fwhm_array[c]/2, *param_array[c])[0]],linestyle = '--', c = 'g') for c in range(0,num)]
    [plt.plot(energy[peaks[c]-width:peaks[c]+width], gaussian(energy[peaks[c]-width:peaks[c]+width], *param_array[c]), c='tab:orange') for c in range(0, num)]

    plt.xlabel(r'Energy [$KeV$]')
    plt.ylabel('counts')
    plt.show()


if __name__ == "__main__":
    global counts, bins, mca, param_array, fwhm_array, peaks, num, width
    width = 100

    for file in glob.glob(r'**/*.spe', recursive=True):
        print(f"Processing file: {file}")
        counts, bins, mca = load_files(file)
        param_array, fwhm_array, peaks, num = prosessing()
        plotting()