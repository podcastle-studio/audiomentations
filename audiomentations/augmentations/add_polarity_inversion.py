from torch import Tensor
from typing import Optional

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict


class PolarityInversion(BaseWaveformTransform):
    """
    Flip the audio samples upside-down, reversing their polarity. In other words, multiply the
    waveform by -1, so negative values become positive, and vice versa. The result will sound
    the same compared to the original when played back in isolation. However, when mixed with
    other audio sources, the result may be different. This waveform inversion technique
    is sometimes used for audio cancellation or obtaining the difference between two waveforms.
    However, in the context of audio data augmentation, this transform can be useful when
    training phase-aware machine learning models.
    """

    supports_multichannel = True

    def __init__(
        self,
        p: float = 0.5
    ):
        super().__init__(p)

    def apply(self, samples, sample_rate):
        new_samples = samples * (-1)
        return new_samples