import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import glob, os

def load_files(file_path):
    numbers = []
    within_data_section = False
    mca_cal_value = None
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

    counts = np.array(numbers[1:])
    mca = mca_cal_value
    bins = np.array(range(1,1+np.size(counts)))

    print(f'counts prossesed - {np.sum(counts)}, mca value - {mca}, number of bins - {np.size(bins)}')

    return counts, bins, mca, measured_time

def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def inv_gauss(x, amp, mean, std_dev):
    return ((-2*np.log(x/amp))**(1/2)*std_dev+mean)

def fwnm_eq(n, amp, mean, std_dev):
    return (2*np.log(1/n))**(1/2)*std_dev+mean, -(2*np.log(1/n))**(1/2)*std_dev+mean

def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def prosessing():
    scale_factor = measured_time/background_array[3]
    counts_filterd = counts - background_array[0]*scale_factor
    
    energy = bins*mca
    start_val = 400
    width = 100
    activity = 4.2e5
    emitted = activity*measured_time

    peak_indices, peak_dict = sc.signal.find_peaks(counts_filterd[energy > 400], prominence = 100, height=1, width=10)
    peak_heights = peak_dict['peak_heights']
    peak_indices=peak_indices + int(start_val/mca)
    peaks = [peak_indices[np.argmax(peak_heights)], peak_indices[np.argpartition(peak_heights,-2)[-2]]]
    num = np.size(peaks)

    [print(f'peaks found at {c}') for c in peaks]

    param_array = np.zeros((num,3))
    covariance_array = np.zeros((num,4))
    fitted_gaussians = np.zeros((num,width*2))
    fwhm_array = np.zeros((num,1))
    fwtm_array = np.zeros((num,1))

    for c in range(0, num):
        initial_guess = [counts_filterd[peaks[c]],peaks[c], 1]
        popt, pcov = sc.optimize.curve_fit(gaussian, bins[peaks[c]-width:peaks[c]+width], counts_filterd[peaks[c]-width:peaks[c]+width], p0=initial_guess)
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
        print(f'  Peak shape fwhm / fwtm - {(fwhm_array[c]/fwtm_array[c])[0]:.3f}')
        #print(f'  Relative efficency - {popt[1]}')
        print("--------------------------------------------------------------")

    return  param_array, fwhm_array, fwtm_array, peaks, num, counts_filterd


def plotting():
    print('plotting')
    d=100
    first_peak=peaks[0]
    second_peak=peaks[1]
    energy=bins*mca
    fig=plt.figure(figsize=(12,14))
    ax1=fig.add_subplot(3,1,1)
    ax2=fig.add_subplot(3,2,3)
    ax3=fig.add_subplot(3,2,4)
    ax1.bar(energy,counts,edgecolor='grey',alpha=0.3)
    ax1.bar(energy,counts_filterd,edgecolor='b')
    [ax1.plot(energy[peaks[c]-width:peaks[c]+width],gaussian(energy[peaks[c]-width:peaks[c]+width],*param_array[c]),c='tab:orange')for c in range(0,num)]
    ax1.set_xlabel(r'Energy [$KeV$]')
    ax1.set_ylabel('Counts')
    ax1.set_title('Full Energy Spectrum')
    ax2.bar(energy[first_peak-d:first_peak+d],counts[first_peak-d:first_peak+d],edgecolor='grey',alpha=0.3)
    [ax2.plot([param_array[c][1]-fwhm_array[c]/2,param_array[c][1]-fwhm_array[c]/2],[0,gaussian(param_array[c][1]-fwhm_array[c]/2,*param_array[c])[0]],linestyle='--',c='r')for c in range(0,num)]
    [ax2.plot([param_array[c][1]+fwhm_array[c]/2,param_array[c][1]+fwhm_array[c]/2],[0,gaussian(param_array[c][1]-fwhm_array[c]/2,*param_array[c])[0]],linestyle='--',c='r')for c in range(0,num)]
    [ax2.plot([param_array[c][1]-fwtm_array[c]/2,param_array[c][1]-fwtm_array[c]/2],[0,gaussian(param_array[c][1]-fwtm_array[c]/2,*param_array[c])[0]],linestyle='--',c='b')for c in range(0,num)]
    [ax2.plot([param_array[c][1]+fwtm_array[c]/2,param_array[c][1]+fwtm_array[c]/2],[0,gaussian(param_array[c][1]-fwtm_array[c]/2,*param_array[c])[0]],linestyle='--',c='b')for c in range(0,num)]
    [ax2.plot(energy[peaks[c]-width:peaks[c]+width],gaussian(energy[peaks[c]-width:peaks[c]+width],*param_array[c]),c='tab:orange')for c in range(0,num)]
    ax2.set_xlabel(r'Energy [$KeV$]')
    ax2.set_ylabel('Counts')
    ax2.set_title('Zoomed on First Peak')
    ax2.set_xlim(energy[first_peak-d],energy[first_peak+d])
    ax3.bar(energy[second_peak-d:second_peak+d],counts[second_peak-d:second_peak+d],edgecolor='grey',alpha=0.3)
    [ax3.plot([param_array[c][1]-fwhm_array[c]/2,param_array[c][1]-fwhm_array[c]/2],[0,gaussian(param_array[c][1]-fwhm_array[c]/2,*param_array[c])[0]],linestyle='--',c='r')for c in range(0,num)]
    [ax3.plot([param_array[c][1]+fwhm_array[c]/2,param_array[c][1]+fwhm_array[c]/2],[0,gaussian(param_array[c][1]-fwhm_array[c]/2,*param_array[c])[0]],linestyle='--',c='r')for c in range(0,num)]
    [ax3.plot([param_array[c][1]-fwtm_array[c]/2,param_array[c][1]-fwtm_array[c]/2],[0,gaussian(param_array[c][1]-fwtm_array[c]/2,*param_array[c])[0]],linestyle='--',c='b')for c in range(0,num)]
    [ax3.plot([param_array[c][1]+fwtm_array[c]/2,param_array[c][1]+fwtm_array[c]/2],[0,gaussian(param_array[c][1]-fwtm_array[c]/2,*param_array[c])[0]],linestyle='--',c='b')for c in range(0,num)]
    [ax3.plot(energy[peaks[c]-width:peaks[c]+width],gaussian(energy[peaks[c]-width:peaks[c]+width],*param_array[c]),c='tab:orange')for c in range(0,num)]
    ax3.set_xlabel(r'Energy [$KeV$]')
    ax3.set_ylabel('Counts')
    ax3.set_title('Zoomed on Second Peak')
    ax3.set_xlim(energy[second_peak-d],energy[second_peak+d])
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    global counts, bins, mca, param_array, fwhm_array, peaks, num, width, measured_time, counts_filterd
    width = 100
    background_array = load_files(glob.glob(r'**/background.spe')[0])
    for file in glob.glob(r'**/*.spe', recursive=True):
        print(f"Processing file: {file}")
        counts, bins, mca, measured_time = load_files(file)
        param_array, fwhm_array, fwtm_array, peaks, num, counts_filterd = prosessing()
        plotting()
        print('\n\n')