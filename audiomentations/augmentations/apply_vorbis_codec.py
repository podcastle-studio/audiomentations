import random

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import apply_ffmpeg_codec, random_log


class ApplyVorbisCodec(BaseWaveformTransform):
    """
    Apply OGG/Vorbis Codec. 
    OGG/Vorbis encode and decode the audio signal.
    """

    supports_multichannel = True

    def __init__(self,
                 min_compression=-1,
                 max_compression=10,
                 p=0.5):
        """
        :param min_compression, int, minimum compression. This corresponds to ``-C`` option of ``sox`` command.
        :param max_compression, int, maximum compression. This corresponds to ``-C`` option of ``sox`` command.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_compression = min_compression
        self.max_compression = max_compression
        assert self.min_compression < self.max_compression

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['compression'] = random_log(
                self.min_compression, self.max_compression
            )

    def apply(self, samples, sample_rate):
        ffmpeg_codec = [
            '-c:a', 'libvorbis', '-q:a', '10', '-f', 'ogg'
        ]
        
        compressed_samples = apply_ffmpeg_codec(samples, sample_rate, ffmpeg_codec)
        assert compressed_samples.shape == samples.shape
        return compressed_samples
