"""Spec Augment module for preprocessing i.e., data augmentation"""
# JJ: Edited from espnet/transform/spec_augment.py. This is different from
# espnet/utils/spec_augment.py, which was already there from git clone. The
# change has been made to apply freq_mask, and time_mask in forward
# loop. This is a stopgap but later will be moved to LoadInputsAndTargets in espnet/utils/io_utils.py

import random
import logging

import numpy
from PIL import Image
from PIL.Image import BICUBIC

from espnet.transform.functional import FuncTrans


def time_warp(x, max_time_warp=80, inplace=False, mode="PIL"):
    """time warp for spec augment

    move random center frame by the random width ~ uniform(-window, window)
    :param numpy.ndarray x: spectrogram (time, freq)
    :param int max_time_warp: maximum time frames to warp
    :param bool inplace: overwrite x with the result
    :param str mode: "PIL" (default, fast, not differentiable) or "sparse_image_warp" (slow, differentiable)
    :returns numpy.ndarray: time warped spectrogram (time, freq)
    """
    window = max_time_warp
    if mode == "PIL":
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return numpy.concatenate((left, right), 0)
    elif mode == "sparse_image_warp":
        import torch

        from espnet.utils import spec_augment

        # TODO(karita): make this differentiable again
        return spec_augment.time_warp(torch.from_numpy(x), window).numpy()
    else:
        raise NotImplementedError("unknown resize mode: " + mode + ", choose one from (PIL, sparse_image_warp).")


class TimeWarp(FuncTrans):
    _func = time_warp
    __doc__ = time_warp.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)


# JJ: default values were fixed by following conf/specaug.yaml
def freq_mask(spec, F=30, n_mask=2, replace_with_zero=False, inplace=True):
    """freq mask for spec agument

    :param numpy.ndarray or Pytorch Tensor spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()

    v = cloned.shape[1] # v is the number of mel channels
    fs = numpy.random.randint(0, F, size=n_mask)

    for f in fs:
        # avoid randrange error
        if v - f <= 0:
            logging.warning("F show set <= # fbank channels. Currently, F is {0} while # fbank channels is {1}".format(F, v))
            continue

        f_zero = random.randrange(0, v - f) # random.randrange doesn't include v - f
        if replace_with_zero:
            cloned[:, f_zero:f_zero + f] = 0
        else:
            cloned[:, f_zero:f_zero + f] = cloned.mean()
    return cloned


class FreqMask(FuncTrans):
    _func = freq_mask
    __doc__ = freq_mask.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)


# JJ: default values were fixed by following conf/specaug.yaml
def time_mask(spec, T=40, n_mask=2, replace_with_zero=False, inplace=True):
    """freq mask for spec agument

    :param numpy.ndarray or Pytorch Tensor spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()
    tau = cloned.shape[0] # tau is the length of the spectrogram (i.e., # frames)
    ts = numpy.random.randint(0, T, size=2)
    for t in ts:
        # avoid randrange error
        if tau - t <= 0:
            continue

        t_zero = random.randrange(0, tau - t) # random.randrange doesn't include tau - t
        if replace_with_zero:
            cloned[t_zero:t_zero + t] = 0
        else:
            cloned[t_zero:t_zero + t] = cloned.mean()
    return cloned


class TimeMask(FuncTrans):
    _func = time_mask
    __doc__ = time_mask.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)


def spec_augment(x, resize_mode="PIL", max_time_warp=80,
                 max_freq_width=27, n_freq_mask=2,
                 max_time_width=100, n_time_mask=2, inplace=True, replace_with_zero=True):
    """spec agument

    apply random time warping and time/freq masking
    default setting is based on LD (Librispeech double) in Table 2 https://arxiv.org/pdf/1904.08779.pdf

    :param numpy.ndarray x: (time, freq)
    :param str resize_mode: "PIL" (fast, nondifferentiable) or "sparse_image_warp" (slow, differentiable)
    :param int max_time_warp: maximum frames to warp the center frame in spectrogram (W)
    :param int freq_mask_width: maximum width of the random freq mask (F)
    :param int n_freq_mask: the number of the random freq mask (m_F)
    :param int time_mask_width: maximum width of the random time mask (T)
    :param int n_time_mask: the number of the random time mask (m_T)
    :param bool inplace: overwrite intermediate array
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert isinstance(x, numpy.ndarray)
    assert x.ndim == 2
    x = time_warp(x, max_time_warp, inplace=inplace, mode=resize_mode)
    x = freq_mask(x, max_freq_width, n_freq_mask, inplace=inplace, replace_with_zero=replace_with_zero)
    x = time_mask(x, max_time_width, n_time_mask, inplace=inplace, replace_with_zero=replace_with_zero)
    return x


class SpecAugment(FuncTrans):
    _func = spec_augment
    __doc__ = spec_augment.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)
