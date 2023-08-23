import os
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

def sin_fn(x):
    return np.where(x == 0, 1.0, np.sin(x) / x)

def blackman_window(N):
    n = np.arange(N)
    return (0.42 - 0.5*np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1)))

def low_pass_filter(fc, N, rate):
    n = np.arange(N)
    # Calculate the filter coefficients
    h = 2 * fc/rate * sin_fn(2*np.pi*fc*(n-(N-1)/2)/rate)
    h = h * blackman_window(N)
    return h

def high_pass_filter(fc, N, rate):
    # low-pass filter
    lp_filter = low_pass_filter(fc, N, rate)
    # Create high-pass filter
    hp_filter = -lp_filter
    hp_filter[N // 2] += 1
    return hp_filter

def band_pass_filter(f1, f2, N, rate):
    return low_pass_filter(f2, N, rate) - low_pass_filter(f1, N, rate)

def apply_echo(signal, delay, alpha):
    echo_sig = np.zeros_like(signal)
    echo_sig[delay:] = signal[:-delay] * alpha
    return signal + echo_sig

def downsample_signal(signal, new_rate, old_rate):
    downsample_ratio = int(old_rate / new_rate)
    return signal[::downsample_ratio]

def conv1(signal, filter):
    conv_sig = np.zeros_like(signal)
    # avoid edge effects
    N = len(filter)
    padded_sig = np.pad(signal, (N // 2, N // 2), mode="constant")
    for i in range(len(signal)):
        conv_sig[i] = np.sum(padded_sig[i:i + N] * filter)
    return conv_sig

def save_spectrum(signal, title, filename):
    spectrum = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1 / rate)
    plt.figure()
    plt.plot(freq, np.abs(spectrum))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.savefig('./output/' + filename)
    plt.clf()

def save_filter_shape(filter_kernel, title, filename):
    plt.figure()
    plt.plot(filter_kernel)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.savefig('./output/' + filename)
    plt.clf()



if __name__ == "__main__":
    # Load input signal
    rate, data = wavfile.read("./HW2_Mix.wav")
    save_spectrum(data, "Input signal spectrum", "input.png")

    # Implement 3 different FIR filters
    N = 5500
    lowpass_cutoff = 350
    bandpass_cutoff = [350, 750]
    highpass_cutoff = 750

    # lowpass filter
    lowpass_filter_kernel = low_pass_filter(lowpass_cutoff, N, rate)
    save_spectrum(lowpass_filter_kernel, "Low-pass filter spectrum", f"LowPass_filter_spectrum.png")
    save_filter_shape(lowpass_filter_kernel, "Low-pass filter shape", f"LowPass_filter_shape.png")
    lowpass_output = conv1(data, lowpass_filter_kernel) * 2000
    wavfile.write(f'./output/LowPass_{lowpass_cutoff}_{N}.wav', rate, lowpass_output.astype(np.int16))
    save_spectrum(lowpass_output, "Low-pass output signal spectrum", f"output_by_LowPass.png")

    # bandpass filter
    bandpass_filter_kernel = band_pass_filter(bandpass_cutoff[0], bandpass_cutoff[1], N, rate)
    save_spectrum(bandpass_filter_kernel, "Band-pass filter spectrum", f"BandPass_filter_spectrum.png")
    save_filter_shape(bandpass_filter_kernel, "Band-pass filter shape", f"BandPass_filter_shape.png")
    bandpass_output = conv1(data, bandpass_filter_kernel) * 2000
    wavfile.write(f'./output/BandPass_{bandpass_cutoff[0]}_{bandpass_cutoff[1]}_{N}.wav', rate, bandpass_output.astype(np.int16))
    save_spectrum(bandpass_output, "Band-pass output signal spectrum", f"output_by_Bandpass.png")

    # highpass filter
    highpass_filter_kernel = high_pass_filter(highpass_cutoff, N, rate)
    save_spectrum(highpass_filter_kernel, "High-pass filter spectrum", f"HighPass_filter_spectrum.png")
    save_filter_shape(highpass_filter_kernel, "High-pass filter shape", f"HighPass_filter_shape.png")
    highpass_output = conv1(data, highpass_filter_kernel) * 5000
    wavfile.write(f'./output/HighPass_{highpass_cutoff}_{N}.wav', rate, highpass_output.astype(np.int16))
    save_spectrum(highpass_output, "High-pass output signal spectrum", f"output_by_HighPass.png")

    # Reduce the sampling rates
    new_sample_rate = 2000

    lowpass_output_downsampled = downsample_signal(lowpass_output, new_sample_rate, rate)
    wavfile.write(f'./output/LowPass_{lowpass_cutoff}_{N}_2kHz.wav', 2000, lowpass_output_downsampled.astype(np.int16))

    bandpass_output_downsampled = downsample_signal(bandpass_output, new_sample_rate, rate)
    wavfile.write(f'./output/BandPass_{bandpass_cutoff[0]}_{bandpass_cutoff[1]}_{N}_2kHz.wav', 2000, bandpass_output_downsampled.astype(np.int16))

    highpass_output_downsampled = downsample_signal(highpass_output, new_sample_rate, rate)
    wavfile.write(f'./output/HighPass_{highpass_cutoff}_{N}_2kHz.wav', 2000, highpass_output_downsampled.astype(np.int16))

    echo_delay = int(rate * 0.5)  # 0.5 seconds delay
    echo_alpha = 0.5

    echo_one = apply_echo(lowpass_output, echo_delay, echo_alpha)
    wavfile.write('./output/Echo_one.wav', rate, echo_one.astype(np.int16))

    echo_multiple = echo_one.copy()
    for i in range(2, 5):
        echo_multiple = apply_echo(echo_multiple, echo_delay * i, echo_alpha)
    wavfile.write('./output/Echo_multiple.wav', rate, echo_multiple.astype(np.int16))