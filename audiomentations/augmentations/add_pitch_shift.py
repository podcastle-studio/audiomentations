import torchaudio
import random

from torch import Tensor
from typing import Optional
from torch_pitch_shift import get_fast_shifts, semitones_to_ratio

from audiomentations.core.transforms_interface import BaseWaveformTransform


def pitch_shift(
    input: torch.Tensor,
    shift,
    sample_rate: int,
    bins_per_octave=12,
    n_fft=None,
    hop_length=None

) -> torch.Tensor:
    """
    Shift the pitch of a batch of waveforms by a given amount.
    Parameters
    ----------
    input: torch.Tensor [shape=(batch_size, channels, samples)]
        Input audio clips of shape (batch_size, channels, samples)
    shift: float OR Fraction
        `float`: Amount to pitch-shift in # of bins. (1 bin == 1 semitone if `bins_per_octave` == 12)
        `Fraction`: A `fractions.Fraction` object indicating the shift ratio. Usually an element in `get_fast_shifts()`.
    sample_rate: int
        The sample rate of the input audio clips.
    bins_per_octave: int [optional]
        Number of bins per octave. Default is 12.
    n_fft: int [optional]
        Size of FFT. Default is `sample_rate // 64`.
    hop_length: int [optional]
        Size of hop length. Default is `n_fft // 32`.
    Returns
    -------
    output: torch.Tensor [shape=(batch_size, channels, samples)]
        The pitch-shifted batch of audio clips
    """
    if not n_fft:
        n_fft = sample_rate // 64
    if not hop_length:
        hop_length = n_fft // 32
    if not isinstance(shift, Fraction):
        shift = 2.0 ** (float(shift) / bins_per_octave)
    resampler = T.Resample(sample_rate, int(sample_rate / shift)).to(input.device)
    output = input
    output = output.reshape(1, output.shape[2])

    v011 = version.parse(torchaudio.__version__) >= version.parse("0.11.0")
    output = torch.stft(output, n_fft, hop_length, return_complex=v011)[None, ...]
    stretcher = T.TimeStretch(
        fixed_rate=float(1 / shift), n_freq=output.shape[2], hop_length=hop_length
    ).to(input.device)
    output = stretcher(output)
    output = torch.istft(output[0], n_fft, hop_length)
    output = resampler(output)
    del resampler, stretcher
    if output.shape[1] >= input.shape[2]:
        output = output[:, : (input.shape[2])]
    else:
        output = pad(output, pad=(0, input.shape[2] - output.shape[1], 0, 0))

    return output


class PitchShift(BaseWaveformTransform):
    """
    Pitch-shift sounds up or down without changing the tempo.
    """
    supports_multichannel = True

    def __init__(
        self,
        min_transpose_semitones: float = -4.0,
        max_transpose_semitones: float = 4.0,
        p: float = 0.5,
        sample_rate: int = 22050
    ):
        """
        :param sample_rate:
        :param min_transpose_semitones: Minimum pitch shift transposition in semitones (default -4.0)
        :param max_transpose_semitones: Maximum pitch shift transposition in semitones (default +4.0)
        :param p:
        """
        super().__init__(
            p=p
        )

        if min_transpose_semitones > max_transpose_semitones:
            raise ValueError("max_transpose_semitones must be > min_transpose_semitones")
        if not sample_rate:
            raise ValueError("sample_rate is invalid.")
        self._sample_rate = sample_rate
        self._fast_shifts = get_fast_shifts(
            sample_rate,
            lambda x: x >= semitones_to_ratio(min_transpose_semitones)
                      and x <= semitones_to_ratio(max_transpose_semitones)
                      and x != 1,
        )
        if not len(self._fast_shifts):
            raise ValueError(
                "No fast pitch-shift ratios could be computed for the given sample rate and transpose range."
            )

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None
    ):
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        self.parameters["transpositions"] = random.choices(self._fast_shifts, k=1)

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None
    ):
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """

        assert sample_rate == self._sample_rate
        random_num = random.randint(0, samples.shape[1] - sample_rate)
        samples_new = pitch_shift(torch.unsqueeze(samples[:, random_num: random_num + sample_rate], 0),
                                  self.parameters["transpositions"][0],
                                  sample_rate
                                  )
        samples[:, random_num: random_num + sample_rate] = samples_new[:, :]
        return samples
