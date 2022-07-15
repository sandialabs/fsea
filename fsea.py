
# packages in call order
from config import *
import pandas as pd
from scipy import signal
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cv2


def import_data(filepath):

    data = pd.read_parquet(filepath)
    waveform = data.data # amplitude values
    time_series = data.time # unix time
    time = time_series.to_numpy()

    return waveform, time


def generate_spectrogram(waveform, samples_per_second, window, overlap):

    """
    Compute a spectrogram from the short-time Fourier transform of the
    waveform.

    Args:
        waveform (1D float array): the infrasound amplitudes.
        samples_per_second (int): the number of samples per second in the
            waveform.
        window (int): the time in seconds over which to take the Fourier
            transforms.
        overlap (float, [0,1)): the fraction of samples shared between
            adjacent windows.

    Returns:
        freq (1D float array): the frequency (Hz) value of each row in the
            spectrogram.
        time_offsets (1D float array): the time (seconds) value of each column
            in the spectogram starting at 0.
        spec (2D float array): the spectrogram with dimensions freq x time
            (input unit ** 2 / Hz).
    """

    # calculate spectrogram function inputs and ensure ints
    fs = int(samples_per_second)
    nperseg = int(window*fs)
    noverlap = int(window*fs*overlap)

    # calculate the spectrogram
    freq, time_offsets, spec = signal.spectrogram(waveform,
                                                  fs=fs,
                                                  nperseg=nperseg,
                                                  noverlap=noverlap,
                                                  return_onesided=True,
                                                  scaling='density',
                                                  window='hann',
                                                  detrend='constant'
                                                  )

    return freq, time_offsets, spec


def correct_smartphone_response(freq, spec):

    """
    Correct for the frequency-dependent response of the Samsung S8 microphone
    using measurements from [22].

    Args:
        freq (1D float array): the frequency (Hz) value of each row in the
            spectrogram.
        spec (2D float array): the spectrogram (input unit ** 2 / Hz).

    Returns:
        spec_corrected (2D float array): the spectrogram (input unit ** 2 / Hz)
            corrected for the frequency-dependent response of the microphone.
    """

    # phone gain scaling from ref [22]
    gain_freq = np.array([0.5, 1, 2, 4, 8, 16, 32, 63, 125, 250, 500, 1000])
    gain_val = np.array([19.66, 41.45, 60.64, 76.61, 88.65, 96.76, 101.7,
                         103.9, 104.7, 104.9, 105.0, 105.0])

    # fit a polynomial to discrete phone gain values in order to resample
    order = 6
    z = np.polyfit(np.log10(gain_freq), gain_val, order)
    p = np.poly1d(z)
    freq[0] = 10**-5  # replace DC with approximately zero to avoid log10 error
    interp_gain_vals = p(np.log10(freq))

    # convert from gain to sensitivity and apply correction
    interp_sens_vals = (1/(10**(interp_gain_vals/10)))**(1/2)
    spec_corrected = (spec.transpose() * (interp_sens_vals)**2).transpose()

    return spec_corrected


def convert_counts_to_pa(waveform):

    """
    Convert the counts provided by the microphone array digitizer to units of
    Pascals (Pa).

    Args:
        waveform (1D float array): the infrasound amplitudes in counts.

    Returns:
        waveform_corrected (1D float array): the infrasound amplitudes in Pa.
    """

    V_per_count = 1.58997E-6 #V/count
    V_per_Pa = 0.2 #V/Pa

    waveform_corrected = waveform * V_per_count * (1/V_per_Pa)

    return waveform_corrected


def convert_to_sound_pressure_level(spec):

    """
    Convert from Pascals (Pa) to Sound Pressure Level (SPL).

    Args:
        spec (2D float array): the spectrogram (Pa ** 2 / Hz).

    Returns:
        spec_spl (2D float array): the input spectrogram scaled to SPL.
    """

    ref = 20*10**-6 # 20 microPa
    spec_spl = 10*np.log10(spec/ref**2)

    return spec_spl


def generate_time_vector(time, time_offsets):

    """
    Generate dates and times for each time column in the spectrogram.

    Args:
        time (1D float array): Unix timestamps coresponding to original
            waveform amplitudes.
        time_offsets (1D float array): time values, starting at zero, in
            seconds for each column in the spectrogram.

    Returns:
        spec_time (UTC datetime list): datetimes for each column in the
            spectrogram.
    """

    start_timestamp = int(time[0]) # only need the first value of "time"
    dt = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
    spec_t = [dt + timedelta(seconds=int(offset)) for offset in time_offsets]

    return spec_t

def generate_spectrogram_indices(time, time_lims, freq, freq_lims):

    """
    Identify time and frequency indices coresponding to user defined values.

    Args:
        time (UTC datetime list): datetimes for each column in the spectrogram.
        time_lims (naive datetime list, n=2): the desired lower and upper time
            limits.
        freq (1D float array): frequencies (Hz) for each row in the
            spectrogram.
        freq_lims (float list, n=2): the desired lower and upper freq limits.

    Returns:
        time_idx (int list, n=2): lower and upper indices coresponding to the
            desired time range.
        freq_idx (int list, n=2): lower and upper indices coresponding to the
            desired frequency range.
    """

    time_idx = []
    for lim in time_lims:
        lim = pytz.utc.localize(lim)
        difs = [abs((t-lim).total_seconds()) for t in time]
        min_ind = np.argmin(np.asarray(difs))
        min_val = time[min_ind]
        time_idx.append((min_ind, min_val))

    freq_idx = []
    for lim in freq_lims:
        min_ind = np.argmin(np.abs(freq - lim))
        min_val = freq[min_ind]
        freq_idx.append((min_ind, min_val))

    return time_idx, freq_idx


def crop_spectrogram(spec, time, time_idx, freq, freq_idx):

    """
    Crop the spectrogram and associated time/frequency containers using a set
    of indices.

    Args:
        spec (2D float array): the spectrogram to crop with dimensions freq x
            time.
        time (datetime list): datetimes for each column in the spectrogram.
        time_idx (int list, n=2): lower and upper indices coresponding to the
            desired time range.
        freq (1D float array): frequencies for each row in the spectrogram.
        freq_idx (int list, n=2): lower and upper indices coresponding to the
            desired frequency range.

    Returns:
        cut_spec (2D float array): the cropped spectrogram.
        cut_time (datetime list): datetimes for each column in the cropped
            spectrogram.
        cut_freq (1D float array): frequencies for each row in the cropped
            spectrogram.
    """

    cut_time = time[time_idx[0][0]:time_idx[1][0]]
    cut_freq = freq[freq_idx[0][0]:freq_idx[1][0]]
    cut_spec = spec[freq_idx[0][0]:freq_idx[1][0],
                    time_idx[0][0]:time_idx[1][0]]

    return cut_spec, cut_time, cut_freq

def plot_spectrogram(spec, time, time_idx, freq, freq_idx, spl_lims,
                     points_to_plot=(), day_switch=(False)):

    """
    Plot and label the spectrogram.

    Args:
        spec (2D float array): the spectrogram to plot with dimensions freq x
            time.
        time (datetime list): datetimes for each column in the spectrogram.
        time_idx (int list, n=2): lower and upper indices coresponding to the
            desired time range.
        freq (1D float array): frequencies for each row in the spectrogram.
        freq_idx (int list, n=2): lower and upper indices coresponding to the
            desired frequency range.
        spl_lims (float list, n=2): lower and upper sound pressure level
            values.
        points_to_plot (tuple of two lists): list of datetimes and list of
            coresponding frequencies to plot.
        day_switch (bool): if false (default), generates appropriate x-label
            for a single day; if true, generates appropriate x-label for
            multiple days.

    Returns:
        An image of the spectrogram.

    """

    fig, ax = plt.subplots(figsize=(7.2,4), dpi=600)

    # overlay points if avalible
    if any(points_to_plot):
        ax.scatter(points_to_plot[0], points_to_plot[1], alpha=0.2, c='r', marker='.')

    # plot image
    im = ax.imshow(spec,
                   origin='lower',
                   cmap='viridis',
                   aspect='auto',
                   extent = [mdates.date2num(time_idx[0][1]),
                             mdates.date2num(time_idx[1][1]),
                             freq_idx[0][1],
                             freq_idx[1][1]],
                   vmin=spl_lims[0],
                   vmax=spl_lims[1])

    # update plot style
    ax.xaxis_date()

    if day_switch == True:
        date_format = mdates.DateFormatter('%m-%d')
        plt.xlabel('Dates', fontsize=8)
    else:
        date_format = mdates.DateFormatter('%H:%M')
        plt.xlabel('Time (UTC) on ' +
                   str(time[0].year) +
                   '-' + str(time[0].month) +
                   '-' + str(time[0].day),
                   fontsize=8)

    ax.xaxis.set_major_formatter(date_format)
    plt.ylabel('Frequency (Hz)', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    cbar = plt.colorbar(im)
    cbar.set_label(label='SPL (dB)', fontsize=8)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(6)

    plt.show()


def generate_horizontal_filter_kernel(total_height, mid_height, mid_val=(2),
                                      edge_val=(-1)):

    """
    Generate a square, horizontally symmetric matrix containing two user
    specified values in three regions.
    Example:  total_height = 4, mid_height = 2, mid_val = 1, edge_val = 0
    [[0,0,0,0],
    [1,1,1,1],
    [1,1,1,1],
    [0,0,0,0]]

    Args:
        total_height (int): the number of rows and columns, n, of the kernel.
        mid_height (int): the number of rows in the middle kernel section.
        mid_val (float): the value of the elements in the middle kernel
            section.
        edge_val (float): the value of the elements in the outer kernel
            sections.

    Returns:
        kernel (2D array): the n x n horizontal filter kernel matrix.
    """

    kernel = np.full((total_height, total_height), edge_val)

    dif = float(total_height) - float(mid_height)
    if dif % 2 == 0:
        kernel[int(dif/2): int(dif/2) + mid_height] = mid_val
        return kernel
    else:
        print('Total height and mid height must both be odd or even.')
        return


def apply_horizontal_line_filter(spec, total, mid):

    """
    Amplify horizontal features in an image by generating and convolving a
    horizontal line kernel with that image.

    Args:
        spec (2D float array): the spectrogram (or image) to filter.
        total (int): the number of rows and columns in the kernel matrix.
        mid (int): the number of rows in the middle section of the kernel.

    Return:
        filtered_spec (2D float array):  the filtered spectrogram.
    """

    kern = generate_horizontal_filter_kernel(total, mid)
    norm_kern = kern/(np.sum(kern))
    filtered_spec = cv2.filter2D(spec, -1, norm_kern)

    return filtered_spec


def weight_by_mf(freq, spec, span, gear_ratio, blades):

    """
    For each frequency row in the spectrogram, (1) assume it coresponds to the
    blade passing frequency (BPF), (2) use the fixed relationship between the
    motor frequency (MF) and BPF to calculate the MF row in the spectrogram,
    and (3) weight the assumed BPF row by the MF row.  Pixels with higher
    values are most likley to corespond to the true BPF.

    Args:
        freq (1D float array): frequencies for each row in the spectrogram.
        spec (2D float array): the spectrogram to weight with dimensions freq
            x time.
        span (int): the number of frequency bins to either side of the central
            bin to integrate (total number of integrated points = 2 * span + 1)
        gear_ratio (float): the gear ratio of the fan gearbox.
        blades (int): the number of blades on the fan.

    Return:
        bpf_results (2D float array): the spectrogram weighted by the combined
            BPF/MF values.

    """

    bpf_results = np.zeros_like(spec)

    for bpf_idx, bpf_guess in enumerate(freq):
        mf_guess = bpf_guess * gear_ratio / blades
        values = np.abs(freq - mf_guess)
        mf_idx = np.argmin(values)

        if (bpf_idx - span >= 0) and (mf_idx + span <= (freq.size - 1)):
            bpf_slice = np.sum(spec[bpf_idx - span:bpf_idx + span, :], axis=0)
            mf_slice = np.sum(spec[mf_idx - span:mf_idx + span, :], axis=0)

            bpf_results[bpf_idx, :] = np.multiply(bpf_slice, mf_slice)

    return bpf_results


def weight_by_harmonic(freq, spec, span, harmonics):

    """
    For each frequency row in the spectrogram, (1) assume it coresponds to the
    fundamental harmonic of the blade passing frequency, (2) use the fixed
    relationship between the higher harmonics and the fundamental to calculate
    the higher harmonic row in the spectrogram, and (3) weight the assumed
    fundamental row by the higher harmonic row.  Pixels with higher values are
    most likley to corespond to the true fundamental.

    Args:
        freq (1D float array): frequencies for each row in the spectrogram.
        spec (2D float array): the spectrogram to weight with dimensions freq
            x time.
        span (int): the number of frequency bins to either side of the central
            bin to integrate (total number of integrated points = 2 * span + 1)
        harmonics (int list): integers corresponding to the harmonics of
            interest.

    Return:
        fund_results (2D float array): the spectrogram weighted by the combined
            harmonic values.
    """

    fund_results = np.zeros([len(harmonics), spec.shape[0], spec.shape[1]])

    for harmonic_idx, harmonic in enumerate(harmonics):
        for fund_idx, fund_guess in enumerate(freq):
            multiple_guess = harmonic * fund_guess
            values = np.abs(freq - multiple_guess)
            idxs = np.argmin(values)


            if (fund_idx - span >= 0) and (idxs + span <= (freq.size - 1)):
                harmonic_slice = np.sum(spec[int(idxs)-span:int(idxs)+span, :],
                                        axis=0)
                fund_results[harmonic_idx, fund_idx, :] = harmonic_slice

    fund_results = np.prod(fund_results, axis=0)

    return fund_results


def find_peaks(spectrogram, time, freq, fan_number_guess):

    """
    Identify the most prominent peaks within each spectrum of the spectrogram.

    Args:
        spectrogram (2D float array): the spectrogram in which to identify the
        most prominent peaks.
        time (datetime list): datetimes for each column in the spectrogram.
        freq (1D float array): frequencies for each row in the spectrogram.
        fan_number_guess (int):  the number of fans believed to be operational.

    Returns:
        peak_points (tuple of two lists):  list of datetimes and list of
            corresponding frequencies of most prominent peaks in the
            spectrogram.
    """

    fan_freq = []
    fan_time = []

    for time_idx, spectrum in enumerate(spectrogram.transpose()):

        peaks = signal.find_peaks(spectrum,
                                  height=(0),
                                  distance=(2),
                                  width=(2),
                                  prominence=(0))

        # extract relevant output from peak finding algorithm
        peaks_idx = peaks[0]
        peaks_prominence = peaks[1]['prominences']
        peaks_data = [z for z in zip(peaks_idx, peaks_prominence)]

        # get peaks in order from most prominent to least
        peaks_data.sort(key = lambda y: y[1], reverse=(True))

        # get the most promient peaks up to the number of fans
        selected_peaks = peaks_data[:fan_number_guess]

        # get only the indices for the selected peaks
        selected_peaks_idx = [elm[0] for elm in selected_peaks]

        # store results
        fan_freq.extend(freq[selected_peaks_idx])

        # get coresponding timing information
        time_snipet = [time[time_idx]] * len(selected_peaks_idx)
        fan_time.extend(time_snipet)

    peak_points = (fan_time, fan_freq)

    return peak_points


def compare_ground_truth(filepath, peak_points, fan_number_guess, max_fan_bpf,
                         total_fans, time_lims):

    """
    Import and compare the fan speed ground truth to the extracted fan speed
    values.

    Args:
        peak_points (tuple of two lists):  list of datetimes and list of
            corresponding frequencies of most prominent peaks in the
            spectrogram.
        fan_number_guess (int):  the number of fans believed to be operational.
        max_fan_bpf (float): the maximum possible blade passing frequency
            assuming a motor speed of 1800 RPM.
        total_fans (int): the maximum number of fans able to run
            simultaneously.
        time_lims (naive datetime list, n=2): the desired lower and upper time
            limits.

    Returns:
        A plot comparing extracted fan speed against fan speed ground truth.
    """

    # rearrange the fan times and freqs to match ground truth
    fan_freqs = peak_points[1]
    fan_freqs = np.reshape(fan_freqs, [-1, fan_number_guess])
    fan_percent_ave = fan_freqs / max_fan_bpf * 100

    fan_times = peak_points[0][::fan_number_guess]

    measured = pd.DataFrame(data=fan_percent_ave, index=fan_times)
    measured['Fan_Ave'] = measured.sum(axis=1) / total_fans

    # get ground truth
    gt = pd.read_parquet(filepath)

    # make mask
    mask = gt.index.intersection(measured.index)

    # down select ground truth to times that were measured
    gt_cut = gt.loc[mask, :]

    # plot results
    fig, ax = plt.subplots(figsize=(3.5,3.5), dpi=600)
    ax.plot(measured.index, measured['Fan_Ave'], alpha=0.8, c='r')
    ax.plot(gt_cut.index, gt_cut['Combined'], alpha=0.8, c='b')
    date_format = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_format)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlim([time_lims[0], time_lims[1]])
    plt.xlabel('Time (UTC) on ' + str(time_lims[1].year) +
               '-' + str(time_lims[1].month) +
               '-' + str(time_lims[1].day),
               fontsize=8)
    plt.ylabel('Active Cooling Tower Capacity (%)', fontsize=8)
    plt.legend(['Calculated', 'Ground Truth'], fontsize=8, loc=4)


def main():
    # prepare data for initial plotting
    waveform, time = import_data(filepath)
    time_lims = [datetime.fromisoformat(start), datetime.fromisoformat(stop)]

    waveform = convert_counts_to_pa(waveform)
    freq, time_offsets, spec = generate_spectrogram(waveform, samples_per_second, window, overlap)
    spec_spl = convert_to_sound_pressure_level(spec)
    spec_time = generate_time_vector(time, time_offsets)

    # prepare and plot spectrogram
    time_idx, freq_idx = generate_spectrogram_indices(spec_time, time_lims, freq, freq_lims)
    cut_spec, cut_time, cut_freq = crop_spectrogram(spec_spl, spec_time, time_idx, freq, freq_idx)
    plot_spectrogram(cut_spec, cut_time, time_idx, cut_freq, freq_idx, spl_lims)

    # filter spectrogram 2x
    filt_spec = apply_horizontal_line_filter(cut_spec, total_filter_height, mid_filter_height)
    filt_spec = apply_horizontal_line_filter(filt_spec, total_filter_height, mid_filter_height)

    # identify bpf/mf and harmonics
    bpf_spec = weight_by_mf(cut_freq, filt_spec, span, gear_ratio, blades)
    fund_spec = weight_by_harmonic(cut_freq, bpf_spec, span, harmonics)

    # find the signal and compare to ground truth
    peak_points = find_peaks(fund_spec, cut_time, cut_freq, fan_number_guess)
    compare_ground_truth(gt_filepath, peak_points, fan_number_guess, max_fan_bpf, total_fans, time_lims)

    # plot resulting points

    time_idx, freq_idx = generate_spectrogram_indices(spec_time, time_lims, freq, freq_lims)
    plot_spectrogram(cut_spec, cut_time, time_idx, cut_freq, freq_idx, spl_lims, peak_points)

if __name__ == '__main__':
    main()
