# -*- coding: utf-8 -*-
"""Audio annotation objects.
"""

from os.path import join

from matplotlib.pyplot import close, pcolormesh, savefig, plot, xlabel, ylabel
from matplotlib import use
from numpy import (
    arange, int64, log10, mean as np_mean, sqrt, transpose, vstack
)
from scipy.signal import spectrogram

from .abstract import AudioAnnotator
from .utils import _check_data_exists, _check_out_dir


class SpectrogramAnnotator(AudioAnnotator):
    """Simple
    """

    name = "spectrogram"

    def __init__(self, **kwargs):
        self.spectrogram = kwargs.get('spectrogram', False)
        self.breaks = kwargs.get('breaks')
        self.output_dir = _check_out_dir(kwargs.get('output_dir'))

        super().__init__()

    def annotate(self, rate, data, ldframe):
        """Run a collection of annotators over the input material.
        """
        _check_data_exists(ldframe, ["meta"])

        if self.output_dir is not None:
            use("template")

        dta = data
        if len(dta.shape) == 2:
            dta = (dta[:, 0] + dta[:, 1]) // 2

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


class PowerToneAnnotator(AudioAnnotator):
    """Simple
    """

    name = "power"

    def __init__(self, **kwargs):
        self.breaks = kwargs.get('breaks')
        self.output_dir = _check_out_dir(kwargs.get('output_dir'))

        super().__init__()

    def annotate(self, rate, data, ldframe):
        """Run a collection of annotators over the input material.
        """
        _check_data_exists(ldframe, ["meta"])

        if self.output_dir is not None:
            use("template")

        dta = data
        if len(dta.shape) == 2:
            dta = (dta[:, 0] + dta[:, 1]) // 2

        output = {'start_frame': [], 'end_frame': [], 'rms': []}
        for stime, audio, i in _audio_chunks(
            self.breaks, dta, rate, ldframe['meta']['fps']
        ):

            if self.output_dir is not None:
                opath = join(self.output_dir, "frame-{0:06d}.png".format(i))
                time_array = arange(0, audio.shape[0], 1)
                time_array = time_array / rate

                plot(time_array + stime, audio, color='k')
                xlabel("Time (seconds)")
                ylabel("Amplitude")
                savefig(opath)
                close()

            output['start_frame'].append(self.breaks[i])
            output['end_frame'].append(self.breaks[i+1])
            output['rms'].append(sqrt(np_mean(int64(audio)**2)))

        return output


def _audio_chunks(breaks, data, rate, fps):
    for i in range(len(breaks) - 1):
        time_start = float(breaks[i] / fps)
        time_end = float(breaks[i + 1] / fps)
        index_start = int(time_start * rate)
        index_end = int(time_end * rate)

        yield time_start, data[index_start:index_end], i
