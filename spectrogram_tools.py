import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import pywt

def gabor_transform(x, fs, T_window, sigma_window, hop=1, over_sampling=1, fft_mode="onesided"):
    """Calculate the Gabor transform of signal x with sampling frequency fs, 
    using scipy.signal.ShortTimeFFT (STFT) with a Gaussian window of length T_window 
    with standard deviation sigma_window, both expressed in seconds.

    # Arguments:
        x: np.ndarray
            The signal to be transformed, should be a 1D vector.

        fs: float
            The sampling frequency of x, expressed in Hertz.

        T_window: float
            The length of the moving window, expressed in seconds.

        sigma_window: float
            The standard deviation of the Gaussian window, expressed in seconds.

        hop: int
            The interval at which the STFT is applied, expressed in frames/samples.
            For example, if x.size is 1000 and hop is 10, the output will have 100 columns (i.e. 100 samples along time axis).
            Default is 1.
        
        oversampling: int
            Used to resample x and the window to a higher frequency before the FFT, to obtain a better image quality.
            Default is 1.

        fft_mode: 'twosided', 'centered', 'onesided', 'onesided2X'
            Passed along to STFT, check scipy.signal.ShortTimeFFT documentation for details.
            Default is 'onesided'.

    # Returns:
        stft: [nf x nx] np.ndarray of complex values
            The Gabor transform of the input signal x.
            nf is (int(oversampling*T_window*fs) + oversampling*x.size//2 + 1) if fft_mode="onesided", 
            otherwise it is (int(oversampling*T_window*fs) + oversampling*x.size + 1).
            nx is x.size//hop
        
        delta_f: float
            The increment along the frequency axis (axis 0).
        
        delta_t: float
            The increment along the time axis (axis 0).
    """

    n_window = int(fs*T_window)
    n_sigma = int(fs*sigma_window)
    window = gaussian(n_window, std=n_sigma)
    GT = ShortTimeFFT(window, hop, fs, mfft=int(over_sampling*fs*T_window), fft_mode=fft_mode)

    return GT.stft(x), GT.delta_f, GT.delta_t

def gabor_spectrogram(x, fs, T_window, sigma_window, 
                      hop=1, over_sampling=1, 
                      freq_bounds=[1, 100], freq_scale="linear", 
                      plot=True, colorbar=False, plot_hw_ratio=0.4, plot_scale="linear", clip=True):
    """Calculate the Gabor transform-based spectrogram of signal x with sampling frequency fs, 
    using scipy.signal.ShortTimeFFT (STFT) with a Gaussian window of length T_window 
    with standard deviation sigma_window, both expressed in seconds.
    The result can be visualized with a customizable plot.

    # Arguments:
        x: np.ndarray
            The signal to be transformed, should be a 1D vector.

        fs: float
            The sampling frequency of x, expressed in Hertz.

        T_window: float
            The length of the moving window, expressed in seconds.

        sigma_window: float
            The standard deviation of the Gaussian window, expressed in seconds.

        hop: int
            The interval at which the STFT is applied, expressed in frames/samples.
            For example, if x.size is 1000 and hop is 10, the output will have 100 columns (i.e. 100 samples along time axis).
            Default is 1.
        
        oversampling: int
            Used to resample x and the window to a higher frequency before the FFT, to obtain a better image quality.
            Default is 1.

        freq_bounds: array-like of length 2
            The lower and upper bound for the plotted frequencies.
            Slight deviation is possible due to the rounding of floats.
            Default is [1, 100].
        
        freq_scale: 'linear', 'log10', 'log2', 'log'
            The scaling of the frequency axis in the plot. Default is 'linear'.
        
        plot: bool
            Wether to plot the spectrogram or not. Default is True.
        
        colorbar: bool
            Wether to plot the colorbar or not. Default is False.
        
        plot_hw_ratio: float
            The ratio of the height to the width of the plot. Default is 0.4.
        
        plot_scale: 'linear', 'log10', 'log2', 'log'
            The scaling of the plotted values. 
            If the scaling is logarithmic, then log10(spec), log2(spec) or log(spec) is plotted instead of spec.
            Default is 'linear'.
        
        clip: bool
            The Gabor transform results in a lengthening of the original signal along the time axis.
            At the start and end the window mainly overlaps with zeros, with which the signal was padded.
            clip states wether these regions should be clipped of the plot or not.
            Default is true.

    # Returns:
        spec: [nf x nx] np.ndarray
            The norm of the Gabor transform of the input signal x, squared.
            nf is (int(oversampling*T_window*fs) + oversampling*x.size//2 + 1) if fft_mode="onesided"
            nx is x.size//hop
        
        delta_f: float
            The increment along the frequency axis (axis 0).
        
        delta_t: float
            The increment along the time axis (axis 0).
    """
    
    # Calculate spectrogram
    X, delta_f, delta_t = gabor_transform(x, fs, T_window, sigma_window, hop=hop, over_sampling=over_sampling)
    spec = np.abs(X)**2

    if plot:
        # Update maximum and minimum frequency plotted
        max_freq_plot = min(freq_bounds[1], fs//2)
        max_freq_idx_plot = int(max_freq_plot*over_sampling*T_window)

        min_freq_plot = max(freq_bounds[0], delta_f)
        min_freq_idx_plot = int(np.ceil(min_freq_plot/delta_f))

        # Reconstruct time and frequency axis
        n_half_window = int(((fs*T_window)//(2*hop)) - 1)
        n_sample = spec.shape[1] - 2*n_half_window

        t_axis = np.linspace(0, (n_sample-1)*delta_t, num=n_sample)
        freq_axis = np.linspace(min_freq_plot, max_freq_plot, num=(max_freq_idx_plot-min_freq_idx_plot+1))

        # Start plot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Set plotted value scale
        if plot_scale == "log10":
            spec_plot = np.log10(spec)
            ax.set_title(f"Gabor transform ({T_window:.4g}, {sigma_window:.4g}) spectrogram (log10)")
        elif plot_scale == "log2":
            spec_plot = np.log2(spec)
            ax.set_title(f"Gabor transform ({T_window:.4g}, {sigma_window:.4g}) spectrogram (log2)")
        elif plot_scale == "log":
            spec_plot = np.log(spec)
            ax.set_title(f"Gabor transform ({T_window:.4g}, {sigma_window:.4g}) spectrogram (log)")
        else:
            spec_plot = spec
            ax.set_title(f"Gabor transform ({T_window:.4g}, {sigma_window:.4g}) spectrogram")

        # Clip off the half windows before and after the actual data, and the actually plot
        if clip:
            mappable = ax.pcolormesh(t_axis, freq_axis, spec_plot[min_freq_idx_plot:max_freq_idx_plot+1, n_half_window:n_half_window+n_sample])
        else:
            mappable = ax.pcolormesh(t_axis, freq_axis, spec_plot[min_freq_idx_plot:max_freq_idx_plot+1])

        # Set y-axis scale
        if freq_scale == "log10":
            ax.set_yscale("log")
        elif freq_scale == "log2":
            ax.set_yscale("log", base=2)
        elif freq_scale == "log":
            ax.set_yscale("log", base=np.e)

        # Make it look nice
        ax.set_box_aspect(plot_hw_ratio)
        
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        if colorbar:
            plt.colorbar(mappable=mappable, ax=ax)

        fig.tight_layout()
        plt.show()

    return spec, delta_f, delta_t

def wavelet_transform(x, fs, wavelet, precision=12, downsample=True, n_freqs=512, freq_bounds=[1, 100], freq_scale="log10"):
    """Calculate the wavelet transform of signal x with sampling frequency fs, 
    using pywt.cwt with wavelet a the mother wavelet.
    The result can be visualized with a customizable plot.

    # Arguments:
        x: np.ndarray
            The signal to be transformed, should be a 1D vector.

        fs: float
            The sampling frequency of x, expressed in Hertz.

        wavelet: str
            Passed along to pywt.cwt, check the PyWavelets documentation for details.
        
        precision: int
            Passed along to pywt.cwt, check the PyWavelets documentation for details.
            Default is 12.

        downsample: bool
            Wether to downsample the signal to the Nyquist frequency 
            defined by the upper bound in freq_bounds before applying the transform.
            Default is True.

        n_freqs: int
            Number of frequencies to calculate the transform for (length of resulting frequency axis).
            Default is 512.

        freq_bounds: array-like of length 2
            The lower and upper bound for the plotted frequencies.
            Slight deviation is possible due to the rounding of floats.
            Default is [1, 100].
        
        freq_scale: 'linear', 'log10', 'log2', 'log'
            The scaling of the frequency axis in the plot. Default is 'log10'.

    # Returns:
        coefs: [n_freqs x nx] np.ndarray of complex values
            The continuous wavelet transform of the input signal x.
            nx is x.size
        
        freq_axis: [n_freqs] np.ndarray of floats floats
            The frequency axis accompanying spec.
    """
    
    # Downsample to nyquist frequency of interest
    if downsample:
        f_nyquist = 2*freq_bounds[1]
        sub_sample_factor = np.floor(fs/f_nyquist)
        idxs_to_keep = (np.arange(x.size) % sub_sample_factor) == 0
        x = x[idxs_to_keep]
        fs = fs/sub_sample_factor

    # Define desired frequency axis of the transformed signal
    if freq_scale == "linear":
        freqs_of_interest = np.linspace(freq_bounds[0]/fs, freq_bounds[1]/fs, num=n_freqs) # linear, frequencies to evaluate CWT for
    elif freq_scale == "log10":
        start = np.log10(freq_bounds[0]/fs)
        end = np.log10(freq_bounds[1]/fs)
        freqs_of_interest = np.logspace(start, end, num=n_freqs) # logarithmic, base 10
    elif freq_scale == "log2":
        start = np.log2(freq_bounds[0]/fs)
        end = np.log2(freq_bounds[1]/fs)
        freqs_of_interest = np.logspace(start, end, num=n_freqs, base=2) # logarithmic, base 2
    elif freq_scale == "log":
        start = np.log(freq_bounds[0]/fs)
        end = np.log(freq_bounds[1]/fs)
        freqs_of_interest = np.logspace(start, end, num=n_freqs, base=np.e) # logarithmic, base 2

    # Actually compute CWT
    scales = pywt.frequency2scale(wavelet, freqs_of_interest, precision=precision) # transform frequencies to corresponding scales for a given wavelet
    coefs, freq_axis = pywt.cwt(x, scales, wavelet, sampling_period=1/fs, method="fft", precision=precision) # evaluate CWT for given wavelet and scales

    return coefs, freq_axis
    
def wavelet_spectrogram(x, fs, wavelet, 
                        precision=12, downsample=True, n_freqs=512, 
                        freq_bounds=[1, 100], freq_scale="log10", 
                        plot=True, colorbar=False, plot_hw_ratio=0.4, plot_scale="linear"):
    """Calculate the wavelet transform-based spectrogram of signal x with sampling frequency fs, 
    using pywt.cwt with wavelet a the mother wavelet.
    The result can be visualized with a customizable plot.

    # Arguments:
        x: np.ndarray
            The signal to be transformed, should be a 1D vector.

        fs: float
            The sampling frequency of x, expressed in Hertz.

        wavelet: str
            Passed along to pywt.cwt, check the PyWavelets documentation for details.
        
        precision: int
            Passed along to pywt.cwt, check the PyWavelets documentation for details.
            Default is 12.

        downsample: bool
            Wether to downsample the signal to the Nyquist frequency 
            defined by the upper bound in freq_bounds before applying the transform.
            Default is True.

        n_freqs: int
            Number of frequencies to calculate the transform for (length of resulting frequency axis).
            Default is 512.

        freq_bounds: array-like of length 2
            The lower and upper bound for the plotted frequencies.
            Slight deviation is possible due to the rounding of floats.
            Default is [1, 100].
        
        freq_scale: 'linear', 'log10', 'log2', 'log'
            The scaling of the frequency axis in the plot. Default is 'log10'.
        
        plot: bool
            Wether to plot the spectrogram or not. Default is True.
        
        colorbar: bool
            Wether to plot the colorbar or not. Default is False.
        
        plot_hw_ratio: float
            The ratio of the height to the width of the plot. Default is 0.4.
        
        plot_scale: 'linear', 'log10', 'log2', 'log'
            The scaling of the plotted values. 
            If the scaling is logarithmic, then log10(spec), log2(spec) or log(spec) is plotted instead of spec.
            Default is 'linear'.

    # Returns:
        spec: [n_freqs x nx] np.ndarray
            The norm of the wavelet transform of the input signal x, squared.
            nx is x.size
        
        freq_axis: [n_freqs] np.ndarray of floats floats
            The frequency axis accompanying spec.
    """
    
    # Compute transform
    coefs, freq_axis = wavelet_transform(x, fs, wavelet, freq_bounds=freq_bounds, freq_scale=freq_scale, 
                                         n_freqs=n_freqs, precision=precision, downsample=downsample)

    # Update fs if necessary
    if downsample:
        f_nyquist = 2*freq_bounds[1]
        sub_sample_factor = np.floor(fs/f_nyquist)
        fs = fs/sub_sample_factor

    # Compute spectrogram
    spec = np.abs(np.sqrt(1/fs)*coefs)**2

    # Plot
    if plot:
        # Reconstruct time axis
        n_samples = spec.shape[1]
        t_axis = np.linspace(0, (n_samples-1)/fs, num=n_samples)

        # Start plot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Set plotted value scale
        if plot_scale == "log10":
            ax.set_title("Wavelet transform (" + wavelet + ") spectrogram (log10)")
            spec_plot = np.log10(spec)
        if plot_scale == "log2":
            ax.set_title("Wavelet transform (" + wavelet + ") spectrogram (log2)")
            spec_plot = np.log2(spec)
        elif plot_scale == "log":
            ax.set_title("Wavelet transform (" + wavelet + ") spectrogram (log)")
            spec_plot = np.log(spec)
        else:
            ax.set_title("Wavelet transform (" + wavelet + ") spectrogram")
            spec_plot = spec

        # Actually plot
        mappable = ax.pcolormesh(t_axis, freq_axis, spec_plot)

        # Set y-axis scale
        if freq_scale == "log10":
            ax.set_yscale("log")
        elif freq_scale == "log2":
            ax.set_yscale("log", base=2)
        elif freq_scale == "log":
            ax.set_yscale("log", base=np.e)

        # Make it look nice
        ax.set_box_aspect(plot_hw_ratio)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        if colorbar:
            plt.colorbar(mappable=mappable, ax=ax)

        fig.tight_layout()
        plt.show()

    return spec, freq_axis