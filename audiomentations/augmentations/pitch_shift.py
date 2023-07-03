import random
import warnings

import librosa
import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class PitchShift(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    supports_multichannel = True

    def __init__(
        self, min_semitones: float = 3.0, max_semitones: float = 4, p: float = 0.5
    ):
        """
        :param min_semitones: Minimum semitones to shift. Negative number means shift down.
        :param max_semitones: Maximum semitones to shift. Positive number means shift up.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_semitones >= -12
        assert max_semitones <= 12
        assert min_semitones <= max_semitones
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["num_semitones"] = random.uniform(
                self.min_semitones, self.max_semitones
            )

    
    def apply(self, samples, sample_rate):

        random_num = random.randint(0, samples.shape - sample_rate)
        samples_torch = np.copy(samples)
        
        samples_new = librosa.effects.pitch_shift(samples_torch[random_num: random_num + sample_rate],
                                n_steps =self.parameters["num_semitones"],
                                sr=sample_rate
                                )
        samples_torch[random_num: random_num + sample_rate] = samples_new
        return samples_torch
