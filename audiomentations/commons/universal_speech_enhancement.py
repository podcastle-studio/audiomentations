from audiomentations.core.composition import SomeOf, OneOf, Compose

from audiomentations import (

    PolarityInversion,
    PitchShift,

    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
    BandLimitWithTwoPhaseResample,

    ApplyMP3Codec,
    ApplyVorbisCodec,
    ApplyULawCodec,

    Overdrive,
    ClippingDistortion,

    Compressor,
    DestroyLevels,
    NoiseGate,
    SimpleCompressor,
    SimpleExpansor,
    Tremolo,

    BandStopFilter,
    SevenBandParametricEQ,
    TwoPoleAllPassFilter,

    AddBackgroundNoise,
    AddShortNoises,

    AddDCComponent,

    Phaser,
    ApplyImpulseResponse,
    ShortDelay,

    AddGaussianNoise
)


def universal_speech_enhancement(
        environmental_noises_path,
        background_noises_path,
        short_noises_path,
        impulse_responses_path,
        simulated_impulse_responses_path = None
    ):
    # Implementation of the universal speech enhancement augmentation from https://arxiv.org/pdf/2206.03065.pdf
    # The weights are taken from the paper

    # Band Limiting
    band_limiting = OneOf([
        BandPassFilter(min_center_freq=600, p=1),
        HighPassFilter(p=1),
        LowPassFilter(min_cutoff_freq=400, p=1),
        BandLimitWithTwoPhaseResample(p=1),
    ], weights=[5, 5, 20, 30]),

    # Codec
    codec = OneOf([
        ApplyVorbisCodec(p=1),
        ApplyULawCodec(p=1)
    ], weights=[3, 3]),

    # Distortion
    distortion = OneOf([
        Overdrive(p=1),
        ClippingDistortion(p=1)
    ], weights=[5, 8]),

    # Loudness and dynamics
    loudness_dynamics = OneOf([
        Compressor(p=1, max_makeup=3),
        DestroyLevels(p=1),
        NoiseGate(p=1),
        SimpleCompressor(p=1),
        SimpleExpansor(p=1),
        Tremolo(p=1)
    ], weights=[10, 20, 10, 3, 2, 2]),

    # Equalization
    equalization = OneOf([
        BandStopFilter(p=1),
        SevenBandParametricEQ(p=1),
        TwoPoleAllPassFilter(p=1)
    ]),

    # Recorded noise. Contains only real world noise
    recorded_noise = SomeOf(
        num_transforms=([1, 2, 3], [0.2, 0.5, 0.3]),
        transforms=[
            AddBackgroundNoise(environmental_noises_path, p=1),
            AddBackgroundNoise(background_noises_path, p=1),
            AddShortNoises(short_noises_path, noise_rms='relative_to_whole_input', p=1),
        ]
    ),

    # Reverb and delay. Contains both real world and simulated impulse responses
    reverb_delay_augmentations = [
        Phaser(p=1),
        ApplyImpulseResponse(impulse_responses_path, p=1),
        ShortDelay(p=1)
    ]
    reverb_delay_weights = [1, 120, 3]
    if simulated_impulse_responses_path is not None:
        reverb_delay_augmentations.append(ApplyImpulseResponse(simulated_impulse_responses_path, p=1))
        reverb_delay_weights.append(30)
    reverb_delay = OneOf(reverb_delay_augmentations, weights=reverb_delay_weights),

    # Synthetic noise.
    synthetic_noise = OneOf([
        AddGaussianNoise(max_amplitude=0.5, p=1),
        AddDCComponent(p=1)
    ], weights=[15, 1]),

    augment = SomeOf(
        num_transforms=([1, 2, 3, 4, 5], [0.35, 0.45, 0.15, 0.04, 0.01]),
        weights=[1, 1, 1, 1, 1, 4, 1, 1],
        transforms=[
            band_limiting,
            codec,
            distortion,
            loudness_dynamics,
            equalization,
            recorded_noise,
            reverb_delay,
            synthetic_noise
        ]
    )

    return Compose([PitchShift(p=0.2), Pitch_PolarityInversion(p=0.5), augment])
