import numpy as np
import biosppy.signals.ecg as ecg

def test_median(x_sample):
    # testing plot
    # remove nan value in nparray
    x_sample = x_sample[np.logical_not(np.isnan(x_sample))]
    signal_processed = ecg.ecg(signal=x_sample, sampling_rate=300)
    # extract rpeaks indices
    rpeak_indices = signal_processed[2]
    templates = signal_processed[4]
    # take the median of templates along row dimension
    template_median = np.median(templates, axis=0)

    template_median = np.tile(template_median, 100)
    print(template_median.shape)
    ecg.ecg(signal=template_median, sampling_rate=300)
    return