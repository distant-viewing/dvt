# -*- coding: utf-8 -*-
"""Audio annotation objects.

This module provides audio aggregators. We generally use frame numbers
as the reference point, with the frames per second and audio sampling rate
values making it possible to translate from the audio samples to visual frames.
"""

from os.path import join

from matplotlib.pyplot import (
    close, pcolormesh, savefig, plot, xlabel, ylabel, ylim
)
from matplotlib import use
from numpy import (
    arange, int64, log10, mean as np_mean, sqrt, transpose, vstack
)
from scipy.signal import spectrogram

from ..abstract import Aggregator
from ..utils import _check_data_exists, _check_out_dir


class SpectrogramAggregator(Aggregator):
    """Compute a spectrogram on the audio input.

    A spectrogram shows how the spectrum of frequencies varies with time. This
    aggregator optionaly produces two types of output: png files visualizing
    the audio track, and an array of numbers describing the spectrogram.

    Attributes:
        breaks (list): An increasing list of break points given by frame
            numbers. When given N+1 breaks, the annotator will produce N
            outputs.
        spectrogram (bool): Should the numeric spectrogram be returned.
            Defaults to False.
        output_dir (str): Directory pointing to where to store the output
            spectrogram. Set to None (the default), to suppress the creation
            of output PNG files.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "spectrogram"

    def __init__(self, **kwargs):
        self.breaks = kwargs['breaks']
        self.spectrogram = kwargs.get('spectrogram', False)
        self.output_dir = kwargs.get('output_dir', None)

        super().__init__(**kwargs)

    def aggregate(self, ldframe, **kwargs):
        """Run a collection of annotators over the input material.

        If output_dir is not none, produces PNG files of the spectrograms for
        each group in the desired output location. If spectrogram is set to
        True, will return the numeric spectrograms. Otherwise returns an
        empty output.
        """
        _check_data_exists(ldframe, ["meta", "audio", "audiometa"])

        if self.output_dir is not None:
            _check_out_dir(self.output_dir)
            use("template")

        dta = ldframe['audio']['data'].values
        rate = ldframe['audiometa']['rate'].values[0]

        saved_times = []
        saved_specs = []

        for stime, audio, i in _audio_chunks(
            self.breaks, dta, rate, ldframe['meta']['fps']
        ):

            frequencies, times, spec = spectrogram(audio, fs=rate)

            if self.output_dir is not None:
                opath = join(self.output_dir, "frame-{0:06d}.png".format(i))

                pcolormesh(times + int(stime), frequencies, 10 * log10(spec))
                xlabel("Time (seconds)")
                ylabel("Frequency")
                savefig(opath)
                close()

            if self.spectrogram:
                saved_times.extend(times + stime)
                saved_specs.extend([transpose(spec)])
                print(spec.shape)

        if self.spectrogram:
            return {
                'times': saved_times,
                'spectrogram': vstack(saved_specs)
            }

        return None


class PowerToneAggregator(Aggregator):
    """Computes the RMS of power and optionally plot the tone of chunks.

    The RMS of the tone gives a rough measurement of how loud the input audio
    track is. The tone PNG files visualize the sound wave over time.

    Attributes:
        breaks (list): An increasing list of break points given by frame
            numbers. When given N+1 breaks, the annotator will produce N
            outputs.
        output_dir (str): Directory pointing to where to store the output
            spectrogram. Set to None (the default), to suppress the creation
            of output PNG files.
        name (str): A description of the aggregator. Used as a key in the
            output data.
    """

    name = "power"

    def __init__(self, **kwargs):
        self.breaks = kwargs['breaks']
        self.output_dir = kwargs.get('output_dir')

        super().__init__(**kwargs)

    def aggregate(self, ldframe, **kwargs):
        """Run a collection of annotators over the input material.

        If output_dir is not none, produces PNG files of the tone for
        each group in the desired output location. Then returns the RMS power
        calculated for each batch of the audio.
        """
        _check_data_exists(ldframe, ["meta", "audio", "audiometa"])

        if self.output_dir is not None:
            _check_out_dir(self.output_dir)
            use("template")

        dta = ldframe['audio']['data'].values
        rate = ldframe['audiometa']['rate'].values[0]

        output = {'frame_start': [], 'frame_end': [], 'rms': []}
        for stime, audio, i in _audio_chunks(
            self.breaks, dta, rate, ldframe['meta']['fps']
        ):

            if self.output_dir is not None:
                opath = join(self.output_dir, "frame-{0:06d}.png".format(i))
                time_array = arange(0, audio.shape[0], 1)
                time_array = time_array / rate

                plot(time_array + stime, audio, color='k')
                ylim([-32768, 32767])
                xlabel("Time (seconds)")
                ylabel("Amplitude")
                savefig(opath)
                close()

            output['frame_start'].append(self.breaks[i])
            output['frame_end'].append(self.breaks[i+1])
            output['rms'].append(sqrt(np_mean(int64(audio)**2)))

        return output


def _audio_chunks(breaks, data, rate, fps):
    for i in range(len(breaks) - 1):
        time_start = float(breaks[i] / fps)
        time_end = float(breaks[i + 1] / fps)
        index_start = int(time_start * rate)
        index_end = int(time_end * rate)

        yield time_start, data[index_start:index_end], i
