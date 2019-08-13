# -*- coding: utf-8 -*-
"""
Example script, demonstrating the use of the functionlity offered by this
package.

@author: Henri DER SARKISSIAN, Nicola VIGANÒ
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mim

from numpy import random as rnd

import os

import rmlseg

def load_phantom(img_name, levels_nb=1):
    # Create the inputs
    data_path = os.path.join('examples', 'images', img_name)
    phantom = mim.imread(data_path)
    phantom = np.asarray(phantom, dtype=np.float32)
    phantom = np.ascontiguousarray(phantom)

    max_val = phantom.max()
    max_dim = phantom.shape[1]
    size_regions = np.floor(max_dim / levels_nb).astype(np.int)
    levels = np.arange(1, levels_nb+1) / levels_nb
    for ii in range(levels_nb):
        phantom[:, (size_regions*ii):(size_regions*(ii+1)-1)] *= levels[ii]

    return phantom, np.concatenate(([0], levels * max_val))

def create_projections(phantom, proj_nb=30):
    # Setup projection parameters
    angles = np.linspace(5, 185, proj_nb, False) * np.pi / 180

    # Simulate projection data
    with rmlseg.tomo.Projector2D(phantom.shape, angles) as p:
        projs = p.fp(phantom)
    return projs, angles

def add_noise(projs, photon_int=1e5, background_std = 1e1, zigler = 1e4):

    projs = rmlseg.tomo.astra.add_noise_to_sino(projs, photon_int)

    if background_std is not None:
        projs = projs + np.abs(rnd.normal(0, background_std, projs.shape))

    # Add ziglers
    for ii in range(10):
        ii_r = rnd.randint(0, projs.size)
        projs[np.unravel_index(ii_r, projs.shape)] += np.abs(rnd.normal(0, zigler))

    return projs

ph, rhos = load_phantom('cylinders.pgm')
projs, angles = create_projections(ph)
projs = add_noise(projs)

# Reconstruct using FBP
rec_vol = rmlseg.util.reconstruct_simple_2D(ph.shape, projs, angles)

# Segment using denoising as a reference
seg_vol_den, _ = rmlseg.util.segment_denoise(rec_vol, rhos)

# Initial segmentation using relaxed multi-levelset approach
seg_vol_rls, _ = rmlseg.util.segment_levelset(rec_vol, rhos)

# Estimation of the pixel-wise RRE in the levelset segmentation
local_rre = rmlseg.util.estimate_local_rre(seg_vol_rls, rhos, projs, angles)

# Refinement of the levelset segmentation, based on the RRE
seg_vol_ref, _ = rmlseg.util.refine_rre(seg_vol_rls, rhos, local_rre)

# Plotting the results
f, axs = plt.subplots(1, 5, sharex=True, sharey=True)
axs[0].imshow(rec_vol)
axs[0].set_title('FBP')
axs[1].imshow(seg_vol_den)
axs[1].set_title('Denoise')
axs[2].imshow(seg_vol_rls)
axs[2].set_title('Relaxed levelset')
axs[3].imshow(local_rre)
axs[3].set_title('RRE')
axs[4].imshow(seg_vol_ref)
axs[4].set_title('Refined')
plt.show()
