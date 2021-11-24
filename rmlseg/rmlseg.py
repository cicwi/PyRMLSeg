# -*- coding: utf-8 -*-

"""
Main module, where all the main functions can be found.

@author: Henri DER SARKISSIAN, Nicola VIGANÒ
"""

import numpy as np

from tqdm import tqdm


def _close_to_0(x, tol=np.finfo(np.float32).eps):
    return np.abs(x) < tol


def _gradientN(x):
    num_dims = len(x.shape)
    d = np.empty(np.concatenate(([num_dims], x.shape)), dtype=x.dtype)
    for ii in range(num_dims):
        pad_widths = [(0, 0)] * num_dims
        pad_widths[ii] = (0, 1)
        x_tmp = np.pad(x, pad_widths, mode='constant')
        d[ii, ...] = np.diff(x_tmp, n=1, axis=ii)
    return d


def _divergenceN(x):
    num_dims = x.shape[0]
    d = np.empty(x.shape, dtype=x.dtype)
    for ii in range(num_dims):
        pad_widths = [(0, 0)] * num_dims
        pad_widths[ii] = (1, 0)
        x_tmp = np.pad(x[ii, ...], pad_widths, mode='constant')
        d[ii, ...] = np.diff(x_tmp, n=1, axis=ii)
    return np.sum(d, axis=0)


def _laplacianN(x):
    num_dims = len(x.shape)
    d = np.empty(np.concatenate(([num_dims], x.shape)), dtype=x.dtype)
    for ii in range(num_dims):
        pad_widths = [(0, 0)] * num_dims
        pad_widths[ii] = (1, 1)
        x_tmp = np.pad(x, pad_widths, mode='edge')
        d[ii, ...] = np.diff(x_tmp, n=2, axis=ii)
    return np.sum(d, axis=0)


def denoise(
        img, iterations=50, lambda_tv=1e-2, lambda_smooth=1e-2, img_max=255.0,
        data_type = np.float32):
    """This function denoises the input image, based on the given TV and
    smoothnes constraint weights.

    It returns a denoised image.

    :param img: The image (np.array_like)
    :param iterations: Number of iterations (int)
    :param lambda_tv: Weight of the TV regularization (float, default: 1e-2)
    :param lambda_smooth: Weight of the smoothing regularization (float, default: 1e-2)
    :param img_max: renormalization value (float, default: 255.0)
    :param data_type: Expected data type (np.dtype, default: np.float32)
    :returns: a denoised image
    :rtype: np.array_like

    """
    img = np.array(img, dtype=data_type) / img_max

    sigma_tv = 0.5
    sigma_smooth = 1.0 / (4.0 * len(img.shape))

    tau = 1 / (1 + (2 * lambda_tv + 4 * lambda_smooth) * len(img.shape))

    x = img
    xe = x

    qimg = np.zeros_like(img)
    qtv = np.zeros(np.concatenate(([len(img.shape)], img.shape)))
    ql = np.zeros_like(img)

    for ii in range(iterations):

        qimg += img - xe
        qimg /= np.fmax(1, np.abs(qimg))

        qtv += _gradientN(xe) * sigma_tv
        qtv /= np.fmax(1, np.sqrt(np.sum(qtv ** 2, axis=0)))

        ql += _laplacianN(xe) * sigma_smooth
        ql /= np.fmax(1, np.abs(ql))

        xn = x + tau * (qimg + lambda_tv * _divergenceN(qtv) - lambda_smooth * _laplacianN(ql))

        xe = xn + (xn - x)
        x = xn

    x *= img_max

    return x


def regularize_levelsets(
        img, rhos, iterations=50, lambda_tv=1e-1, lambda_smooth=None,
        weight_norm_p=2, dataterm_norm_p=1, lower_limit=None, upper_limit=None,
        data_type=np.float32):
    """This function computes the regularization of the input image, based on
    the given expected level values and regularization weights.

    It returns the regularized image.

    :param img: The image (np.array_like)
    :param rhos: Expected levels (np.array_like)
    :param iterations: Number of iterations (int)
    :param lambda_tv: Weight of the TV regularization (float, default: 1e-1)
    :param lambda_smooth: Weight of the smoothing regularization (float, default: None)
    :param weight_norm_p: l_p norm of the weights (int, default: 2)
    :param dataterm_norm_p: l_p norm of the data term (int, default: 1)
    :param lower_limit: Lower limit of the image, used for clipping (float, default: None)
    :param upper_limit: Upper limit of the image, used for clipping (float, default: None)
    :param data_type: Expected data type (np.dtype, default: np.float32)
    :returns: a regularized image
    :rtype: np.array_like

    """
    rhos = np.array(rhos, dtype=data_type)
    img = np.array(img, dtype=data_type)

    rhos_shape = np.concatenate((rhos.shape, [1] * len(img.shape)))
    rhos_exp = np.reshape(rhos, rhos_shape)
    W_prime = np.expand_dims(img, axis=0) - rhos_exp
    W_prime = np.abs(W_prime) ** weight_norm_p

    W_second = 1 / (W_prime + _close_to_0(W_prime))
    W_rhos = W_second / np.sum(W_second, axis=0)

    sigma_rhos = 1 / (W_rhos + _close_to_0(W_rhos))
    sigma1_rhos = 1 / (1 + sigma_rhos)

    x = img.copy()
    # x = np.zeros_like(img)
    # x[:] = rhos[np.argmax(W_rhos, axis=0)]
    xe = x

    tau = np.sum(W_rhos + _close_to_0(W_rhos), axis=0)

    if lambda_tv is not None:
        sigma_tv = 0.5
        tau += 2 * lambda_tv * len(img.shape)

        qtv = np.zeros(np.concatenate(([len(img.shape)], img.shape)), dtype=data_type)

    if lambda_smooth is not None:
        sigma_smooth = 1.0 / (4.0 * len(img.shape))
        tau += 4 * lambda_smooth * len(img.shape)

        ql = np.zeros_like(img, dtype=data_type)

    tau = 1 / tau

    q_rhos = np.zeros(np.concatenate(([rhos.size], img.shape)), dtype=data_type)

    for ii in tqdm(range(iterations)):

        q_rhos -= xe - rhos_exp
        if dataterm_norm_p == 1:
            q_rhos /= np.fmax(1, np.abs(q_rhos))
        elif dataterm_norm_p == 12:
            q_rhos /= np.fmax(1, np.sqrt(np.sum(q_rhos ** 2, axis=0)))
        elif dataterm_norm_p == 2:
            q_rhos *= sigma1_rhos

        x_upd = np.sum(W_rhos * q_rhos, axis=0)

        if lambda_tv is not None:
            qtv += _gradientN(xe) * sigma_tv
            qtv /= np.fmax(1, np.sqrt(np.sum(qtv ** 2, axis=0)))

            x_upd -= lambda_tv * _divergenceN(qtv)

        if lambda_smooth is not None:
            ql += _laplacianN(xe) * sigma_smooth
            ql /= np.fmax(1, np.abs(ql))

            x_upd += lambda_smooth * _laplacianN(ql)

        xn = x - tau * x_upd

        if lower_limit or upper_limit:
            xn = np.clip(xn, lower_limit, upper_limit)

        xe = xn + (xn - x)
        x = xn

    return x


def refine_rre(
        img, rhos, local_rre, iterations=50, lambda_tv=1.0, weight_norm_p=1,
        dataterm_norm_p=1, data_type=np.float32):
    """This function computes the refinement of the segmented image, based on
    the given locally reconstructed residual.

    It returns the refined image.

    :param img: The image (np.array_like)
    :param rhos: Expected levels (np.array_like)
    :param local_rre: Locally reconstructed residual error (np.array_like)
    :param iterations: Number of iterations (int)
    :param lambda_tv: Weight of the TV regularization (float, default: 1.0)
    :param weight_norm_p: l_p norm of the weights (int, default: 1)
    :param dataterm_norm_p: l_p norm of the data term (int, default: 1)
    :param data_type: Expected data type (np.dtype, default: np.float32)
    :returns: a refined regularized image
    :rtype: np.array_like

    """

    sigma_tv = 0.5
    local_rre_norm = np.fmax(local_rre.copy().astype(data_type), 0) / rhos.max()
    local_confidence = np.fmax(1 - local_rre_norm, 0) ** weight_norm_p

    sigma = local_confidence.copy()
    sigma = 1 / (sigma + _close_to_0(sigma))
    sigma1 = 1 / (1 + sigma)

    tau = local_confidence + _close_to_0(local_confidence)
    tau = 1 / (tau + (2 * lambda_tv) * len(img.shape))

    x = np.array(img, dtype=data_type)
    xe = x

    qa = np.zeros_like(img, dtype=data_type)
    qtv = np.zeros(np.concatenate(([len(img.shape)], img.shape)), dtype=data_type)

    for ii in tqdm(range(iterations)):
        qa += img - xe
        if (dataterm_norm_p == 1):
            qa /= np.fmax(1, np.abs(qa))
        elif (dataterm_norm_p == 2):
            qa *= sigma1

        qtv += _gradientN(xe) * sigma_tv
        qtv /= np.fmax(1, np.sqrt(np.sum(qtv ** 2, axis=0)))

        xn = x + tau * (local_confidence * qa  + lambda_tv * _divergenceN(qtv))

        xe = xn + (xn - x)
        x = xn

    return x


def estimate_rhos(p, projs, img, rhos0=None, iterations=100, dataterm_norm_p=1):
    """This function estimates the levelset values from the current
    segmentation and the available projections.

    It returns the etimated rhos.

    :param p: Projector from tomo module
    :param projs: The object projections (np.array_like)
    :param img: The segmented image (np.array_like)
    :param rhos0: Initial estimation of the rhos (np.array_like, default: None)
    :param iterations: Number of iterations (int)
    :param dataterm_norm_p: l_p norm of the data term (int, default: 1)
    :returns: estimated rhos
    :rtype: np.array_like

    """
    num_rhos = len(rhos0 or np.unique(img))
    p_x = np.empty(np.concatenate(([num_rhos], projs.shape)), dtype=projs.dtype)
    for ii in range(num_rhos):
        p_x[ii, ...] = p.fp(img == ii)

    sigma = np.sum(p_x, axis=0)
    sigma = 1 / (sigma + _close_to_0(sigma))
    sigma1 = 1 / (1 + sigma)
    tau = 1 / np.sum(np.reshape(p_x, (num_rhos, -1)), axis=1)

    rhos = np.array(rhos0 or np.sum(projs * sigma * p_x) * tau)
    rhos_e = rhos

    q = np.zeros_like(projs)

    rhos_e_shape = np.concatenate((rhos_e.shape, np.ones((len(img.shape)), dtype=np.intp)))

    for ii in tqdm(range(iterations)):
        res = projs - np.sum(p_x * np.reshape(rhos_e, rhos_e_shape), axis=0)
        q += res * sigma
        if dataterm_norm_p == 1:
            q /= np.fmax(1, np.abs(q))
        elif dataterm_norm_p == 2:
            q *= sigma1

        upd = np.sum(np.reshape(q * p_x, (num_rhos, -1)), axis=1) * tau
        rhos_n = rhos + upd

        rhos_e = rhos_n + (rhos_n - rhos)
        rhos = rhos_n

    return rhos

def segment_simple(img, rhos):
    """This function computes the simple segmentation of the input image, based
    on the given expected level values.

    It returns the regularized image.

    :param img: The image (np.array_like)
    :param rhos: Expected levels (np.array_like)
    :returns: The segmented image
    :rtype: np.array_like

    """
    rhos = np.array(rhos)
    pos = np.argsort(rhos)
    rhos = rhos[pos]
    thr = rhos[0:-1] + np.diff(rhos) / 2

    x = np.zeros_like(img, dtype=np.int)
    for ii, t in enumerate(thr):
        x[img > t] = pos[ii+1]

    return x

def segment_levelset(img, rhos, *args, **kwds):
    """This function computes the simple segmentation of the input image, based
    on the given expected level values and regularization weights.

    It returns the segmented image.

    :param img: The image (np.array_like)
    :param rhos: Expected levels (np.array_like)
    :param iterations: Number of iterations (int)
    :param lambda_tv: Weight of the TV regularization (float, default: 1e-2)
    :param lambda_smooth: Weight of the smoothing regularization (float, default: None)
    :param weight_norm_p: l_p norm of the weights (int, default: 2)
    :param dataterm_norm_p: l_p norm of the data term (int, default: 1)
    :param lower_limit: Lower limit of the image, used for clipping (float, default: None)
    :param upper_limit: Upper limit of the image, used for clipping (float, default: None)
    :returns: a segmented image
    :rtype: np.array_like

    """
    img_ls = regularize_levelsets(img, rhos, *args, **kwds)
    return segment_simple(img_ls, rhos)


