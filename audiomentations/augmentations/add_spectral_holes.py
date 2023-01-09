import random

import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform


class AddSpectralHoles(BaseWaveformTransform):

    def __init__(self, p=0.5, min_proportion=0.05, max_proportion=0.2):
        """
        :param p:
        """
        super().__init__(p)
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            freq_proportion = random.uniform(self.min_proportion, self.max_proportion)
            self.parameters["proportion"] = freq_proportion

    def apply(self, samples, sample_rate):
        fourier = np.fft.rfft(samples)
        indices = random.sample(list(range(len(fourier))), int(len(fourier) * self.parameters["proportion"]))
        fourier[indices] = 0
        new_samples = np.fft.irfft(fourier)

        assert samples.shape == new_samples.shape
        return new_samples
