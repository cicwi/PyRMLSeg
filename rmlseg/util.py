# -*- coding: utf-8 -*-
"""
Utility functions for the use of the package functionality.

@author: Nicola VIGANÒ
"""

import numpy as np
import time as tm

from . import rmlseg
from . import tomo


def reconstruct_simple_2D(rec_vol_shape, projs, angles):
    """This function computes a simple FBP reconstruction of the projections
    from the given projection data and corresponding angles, for the given
    recontruction shape.

    :param rec_vol_shape: The output reconstruction shape (np.array_like)
    :param projs: The projection data (np.array_like)
    :param angles: Corresponding angles of the projections (np.array_like)
    :returns: The reconstructed image
    :rtype: np.array_like

    """
    prnt_str = "Reconstructing.."
    print(prnt_str, end="", flush=True)
    c_in = tm.time()

    with tomo.Projector2D(rec_vol_shape, angles) as p:
        # Reconstructing
        rec_vol = p.fbp(projs)

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    return rec_vol


def segment_denoise(rec_vol, rhos):
    """This function computes the segmentation of the denoised image.

    :param rec_vol: The reconstruction (np.array_like)
    :param rhos: The segmentation target levels (np.array_like)
    :returns: The segmented image
    :rtype: np.array_like

    """
    prnt_str = "Solving with denoising.."
    print(prnt_str, end="", flush=True)
    c_in = tm.time()

    rec_vol_den = rmlseg.denoise(rec_vol, iterations=100, lambda_tv=1, lambda_smooth=1e-1)
    seg_vol_den = rmlseg.segment_simple(rec_vol_den, rhos)

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    return seg_vol_den, rec_vol_den


def _renorm(x, rhos):
    return (x - np.min(rhos)) / (np.max(rhos) - np.min(rhos))


def _unrenorm(x, rhos):
    return x * (np.max(rhos) - np.min(rhos)) + np.min(rhos)


def segment_levelset(rec_vol, rhos):
    """This function computes the segmentation of the relaxed levelset based
    regularization.

    :param rec_vol: The reconstruction (np.array_like)
    :param rhos: The segmentation target levels (np.array_like)
    :returns: The segmented image
    :rtype: np.array_like

    """
    prnt_str = "Solving with levelsets.."
    print(prnt_str, end="", flush=True)
    c_in = tm.time()

    levels = _renorm(rhos, rhos)
    rec_vol_rescaled = _renorm(rec_vol, rhos)

    seg_conv_tv2 = rmlseg.regularize_levelsets(rec_vol_rescaled, levels, iterations=100, lambda_tv=1)

    seg_conv_tv2 = _unrenorm(seg_conv_tv2, rhos)

    seg_seg_tv2 = rmlseg.segment_simple(seg_conv_tv2, rhos)

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    return seg_seg_tv2, seg_conv_tv2


def estimate_local_rre(seg_vol, rhos, projs, angles):
    """This function estimates the local Reconstructed Residual Error (RRE) for
    the given segmentation.

    :param seg_vol: The segmentation (np.array_like)
    :param rhos: The segmentation target levels (np.array_like)
    :param projs: The projection data (np.array_like)
    :param angles: Corresponding angles of the projections (np.array_like)
    :returns: The corresponding RRE
    :rtype: np.array_like

    """
    prnt_str = "Estimating local RRE.."
    print(prnt_str, end="", flush=True)
    c_in = tm.time()

    pos = np.argsort(rhos)
    rec_rho = seg_vol.copy()
    for ii, rho in zip(pos, rhos):
        rec_rho[seg_vol == ii] = rho

    with tomo.Projector2D(seg_vol.shape, angles) as p:
        fp_img = p.fp(rec_rho)
        rre = p.fbp(fp_img - projs)

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    return np.abs(rre)


def refine_rre(seg_vol, rhos, rre):
    """This function refines the segmentation using the RRE.

    :param seg_vol: The segmentation (np.array_like)
    :param rhos: The segmentation target levels (np.array_like)
    :param rre: The pixel-wise RRE (np.array_like)
    :returns: The segmented image
    :rtype: np.array_like

    """
    prnt_str = "Refining segmentation using local RRE.."
    print(prnt_str, end="", flush=True)
    c_in = tm.time()

    refined = rmlseg.refine_rre(seg_vol, rhos, rre, iterations=100, lambda_tv=1)

    levels = _renorm(rhos, rhos)
    refined_seg = rmlseg.segment_simple(refined, levels)

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    return refined_seg, refined
