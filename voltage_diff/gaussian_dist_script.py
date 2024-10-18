import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import glob, os

def load_files(filepath):
    global counts, bins
    counts = np.loadtxt(filepath)
    bins = np.array(range(1, 1 + np.size(counts)))

def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def prosessing(width):
    global param_array, fwhm_array, peaks, num, energy
    peaks = np.array(sc.signal.find_peaks(counts, prominence=1e4)[0])
    num = np.size(peaks)

    param_array = np.zeros((num, 3))
    covariance_array = np.zeros((num, 4))
    fitted_gaussians = np.zeros((num, width * 2))
    fwhm_array = np.zeros((num, 1))

    for c in range(0, num):
        initial_guess = [counts[peaks[c]], peaks[c], 1]
        popt, pcov = sc.optimize.curve_fit(
            gaussian,
            bins[peaks[c] - width:peaks[c] + width],
            counts[peaks[c] - width:peaks[c] + width],
            p0=initial_guess
        )

        popt[1:] = popt[1:] * 4.882741E-001
        param_array[c] = popt
        covariance_array[c] = [np.linalg.cond(pcov), *np.diag(pcov)]
        fitted_gaussians[c] = gaussian(bins[peaks[c] - width:peaks[c] + width], *popt)
        fwhm_array[c] = 2 * np.sqrt(2 * np.log(2)) * popt[2]

        print(f"Peak {c + 1}:")
        print(f"  Amplitude = {popt[0]:.2f}, Mean = {popt[1]:.2f} keV, Std Dev = {popt[2]:.2f} keV")
        print(f"  FWHM = {fwhm_array[c][0]:.2f} keV")
        print(f"  Covariance Condition Number = {covariance_array[c][0]:.2e}")
        print(f"  Covariance Diagonal (Variance of Amplitude, Mean, Std Dev) = {covariance_array[c][1]:.2e}, {covariance_array[c][2]:.2e}, {covariance_array[c][3]:.2e}")
        print("--------------------------------------------------------------")

    energy = bins * 4.882741E-001

def plotting(width):
    d = 100
    plt.bar(energy[min(peaks) - d:max(peaks) + d], counts[min(peaks) - d:max(peaks) + d], edgecolor='grey', alpha=0.3)

    for c in range(0, num):
        plt.plot([param_array[c][1] - fwhm_array[c] / 2, param_array[c][1] - fwhm_array[c] / 2],
                 [0, gaussian(param_array[c][1] - fwhm_array[c] / 2, *param_array[c])[0]], linestyle='--', c='g')
        plt.plot([param_array[c][1] + fwhm_array[c] / 2, param_array[c][1] + fwhm_array[c] / 2],
                 [0, gaussian(param_array[c][1] - fwhm_array[c] / 2, *param_array[c])[0]], linestyle='--', c='g')
        plt.plot(energy[peaks[c] - width:peaks[c] + width],
                 gaussian(energy[peaks[c] - width:peaks[c] + width], *param_array[c]), c='tab:orange')

    plt.xlabel(r'Energy [$KeV$]')
    plt.ylabel('Counts')
    plt.show()

if __name__ == "__main__":
    width = 100

    x = str(glob.glob(r'**/*.spe', recursive=True)[0])
    print(x)

    np.loadtxt(x)


    for file in glob.glob(r'**/*.sple', recursive=True):
        print(f"Processing file: {file}")
        load_files(file)
        processing(width)
        plotting(width)
