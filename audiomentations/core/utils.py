import imp
import os
import io
import random
from subprocess import Popen, PIPE
from pathlib import Path
from typing import List, Union

import math
import numpy as np

SUPPORTED_EXTENSIONS = (
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
)


def find_audio_files(
    root_path,
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) in a directory.
    Also traverses subdirectories by default.
    """
    file_paths = []

    for root, dirs, filenames in os.walk(root_path, followlinks=follow_symlinks):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.lower().endswith(filename_endings):
                file_paths.append(Path(file_path))
        if not traverse_subdirectories:
            # prevent descending into subfolders
            break

    return file_paths


def find_audio_files_in_paths(
    paths: Union[List[Path], List[str], Path, str],
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) contained in the list or in its directories.
    Also traverses subdirectories by default.
    """

    file_paths = []

    if isinstance(paths, (list, tuple, set)):
        paths = list(paths)
    else:
        paths = [paths]

    for p in paths:
        if str(p).lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = Path(os.path.abspath(p))
            file_paths.append(file_path)
        elif os.path.isdir(p):
            file_paths += find_audio_files(
                p,
                filename_endings=filename_endings,
                traverse_subdirectories=traverse_subdirectories,
                follow_symlinks=follow_symlinks,
            )
    return file_paths


def calculate_rms(samples):
    """Given a numpy array of audio samples, return its Root Mean Square (RMS)."""
    return np.sqrt(np.mean(np.square(samples)))


def calculate_rms_without_silence(samples, sample_rate):
    """
    This function returns the rms of a given noise whose silent periods have been removed. This ensures
    that the rms of the noise is not underestimated. Is most useful for short non-stationary noises.
    """

    window = int(0.025 * sample_rate)

    if samples.shape[-1] < window:
        return calculate_rms(samples)

    rms_all_windows = np.zeros(samples.shape[-1] // window)
    current_time = 0

    while current_time < samples.shape[-1] - window:
        rms_all_windows[current_time // window] += calculate_rms(
            samples[current_time : current_time + window]
        )
        current_time += window

    rms_threshold = np.max(rms_all_windows) / 25

    # The segments with a too low rms are identified and discarded
    rms_all_windows = rms_all_windows[rms_all_windows > rms_threshold]
    # Beware that each window must have the same number of samples so that this calculation of the rms is valid.
    return calculate_rms(rms_all_windows)


def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)


def is_waveform_multichannel(samples):
    """
    Return bool that answers the question: Is the given ndarray a multichannel waveform or not?
    :param samples: numpy ndarray
    :return:
    """
    return len(samples.shape) > 1


def is_spectrogram_multichannel(spectrogram):
    """
    Return bool that answers the question: Is the given ndarray a multichannel spectrogram?
    :param samples: numpy ndarray
    :return:
    """
    return len(spectrogram.shape) > 2 and spectrogram.shape[-1] > 1


def convert_float_samples_to_int16(y):
    """Convert floating-point numpy array of audio samples to int16."""
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError("input samples not floating-point")
    return (y * np.iinfo(np.int16).max).astype(np.int16)


def convert_frequency_to_mel(f: float) -> float:
    """
    Convert f hertz to mels
    https://en.wikipedia.org/wiki/Mel_scale#Formula
    """
    return 2595.0 * math.log10(1.0 + f / 700.0)


def convert_mel_to_frequency(m: Union[float, np.array]) -> Union[float, np.array]:
    """
    Convert m mels to hertz
    https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
    """
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def find_time_shift(a, b, max_shift=5000):
    """
    Find time shift between two signals a and b.
    :param a: numpy array. First signal. Must be mono.
    :param b: numpy array. Second signal. Must be mono.
    :param max_shift: maximum possible shift. This will help to speed up the algorithm
    """
    start_slice = min(a.shape[-1], b.shape[-1])
    if max_shift:
        start_slice = min(start_slice, 2 * max_shift)

    a = a[:start_slice]
    b = b[:start_slice]
    
    return np.argmax(np.abs(np.convolve(a, b[::-1]))) - a.shape[0]


def apply_ffmpeg_commands(samples, sample_rate, commands):
    """
    Apply a list of ffmpeg commands to the given audio samples.
    Everything is done in memory avoiding the need to write to disk.
    The input samples are assumed to be in the range [-1, 1].
    Currently only 16 bit PCM WAV files are supported as input, 
    othervise, the input samples are converted to 16 bit PCM WAV.
    :param samples: numpy array. Audio samples.
    :param sample_rate: int. Sampling rate of the audio samples.
    :param commands: list of strings. List of ffmpeg commands.
        For example to apply simple compressor with threshold 15dB: ["-af", "acompressor=threshold=-15dB"]
    :return: numpy array. Audio samples after applying the ffmpeg commands.
    """
    import soundfile as sf
    
    if len(samples.shape) == 2:
        samples = samples.T

    cmd = [
        "ffmpeg", 
        "-i", 'pipe:0',
        *commands,
        "-f", "wav",
        "-"
    ]

    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    b = io.BytesIO()
    b.name = "toffmpeg.wav"
    sf.write(b, samples, samplerate=sample_rate, format='WAV')
    b.seek(0)

    data, out = p.communicate(b.read())

    if b"Error" in out:
        raise Exception(out)

    reconstructed = np.fromstring(data[data.find(b"data")+8:], np.int16)
    reconstructed = reconstructed / 32767
    
    if samples.dtype == np.float32:
        reconstructed = reconstructed.astype(np.float32)

    if len(samples.shape) == 2:
        reconstructed = reconstructed.reshape((reconstructed.shape[0] // 2, 2)).T

    return reconstructed


def random_log(a, b):
    """
    Pick a random number between a and b in logarithmic scale
    param a: float. Lower bound
    param b: float. Upper bound
    """
    return math.exp(random.uniform(math.log(a), math.log(b)))


def random_log_int(a, b):
    """
    Pick a random integer between a and b in logarithmic scale
    param a: float. Lower bound
    param b: float. Upper bound
    """
    c = random_log(a, b)
    return max(a, min(b, round(c)))


def weights_to_probabilities(w):
    """
    converts the weights to probabilites
    """
    norm = sum(w)
    return [x / norm for x in w]