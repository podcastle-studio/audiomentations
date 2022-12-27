import random

import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform


class AddDCComponent(BaseWaveformTransform):

    def __init__(self, p=0.5, min_amplitude_shift_prportion=0.5, max_amplitude_shift_prportion=1):
        """
        :param p:
        """
        super().__init__(p)
        self.min_amplitude_shift_proportion = min_amplitude_shift_prportion
        self.max_amplitude_shift_proportion = max_amplitude_shift_prportion

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            amplitude_shift_type = np.random.choice(["positive", "negative"])
            self.parameters["amplitude_shift_type"] = amplitude_shift_type
            self.parameters["shift_proportion"] = random.uniform(
                self.min_amplitude_shift_proportion, self.max_amplitude_shift_proportion
            )

    def apply(self, samples, sample_rate):
        if self.parameters["amplitude_type"] == "positive":
            amplitude_gain = max(0, 1 - samples.max())
        else:
            amplitude_gain = min(0, -(1 + samples.min()))
        shift_proportion = amplitude_gain * self.parameters["shift_proportion"]
        new_samples = samples + shift_proportion
        return new_samples
