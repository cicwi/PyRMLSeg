# -*- coding: utf-8 -*-
"""
Basic tomography operations, including a projector class.

@author: Nicola VIGANÒ
"""

import numpy as np
import scipy as sp

import astra


class Projector2D(object):
    def __init__(self, vol_shape, angles, beam_shape="parallel"):
        if len(vol_shape) > 2:
            raise ValueError("Only 2D volumes")

        self.proj_id = []
        self.dispose_projector()

        self.vol_geom = astra.create_vol_geom(vol_shape)
        self.proj_geom = astra.create_proj_geom(beam_shape, 1, np.max(vol_shape), angles)

    def initialize_projector(self):
        self.proj_id = astra.create_projector("linear", self.proj_geom, self.vol_geom)
        self.W = astra.OpTomo(self.proj_id)

    def dispose_projector(self):
        astra.projector.delete(self.proj_id)
        self.proj_id = []
        self.W = []

    def __enter__(self):
        self.initialize_projector()
        return self

    def __exit__(self, *args):
        self.dispose_projector()

    def fp(self, volume):
        return self.W.FP(volume)

    def bp(self, projections):
        return self.W.BP(projections)

    def fbp(self, projections, iterations=50, opts={}):
        return self.W.reconstruct("FBP_CUDA", projections, iterations=iterations, extraOptions=opts)

    def sirt(self, projections, iterations=50, opts={}):
        return self.W.reconstruct("SIRT_CUDA", projections, iterations=iterations, extraOptions=opts)

    def cgls(self, projections, iterations=50, opts={}):
        return self.W.reconstruct("CGLS_CUDA", projections, iterations=iterations, extraOptions=opts)

    def get_matrix(self):
        return sp.sparse.csc_matrix(astra.matrix.get(astra.projector.matrix(self.proj_id)))
