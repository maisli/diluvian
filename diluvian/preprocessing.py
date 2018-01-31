# -*- coding: utf-8 -*-
"""Volume preprocessing for seed generation and data augmentation."""


from __future__ import division

import logging

import numpy as np
from scipy import ndimage
from six.moves import range as xrange

from .config import CONFIG
from .util import (
        get_color_shader,
        WrappedViewer,
)
import pdb


def make_prewitt(size):
    """Construct a separable Prewitt gradient convolution of a given size.

    Adapted from SciPy's ndimage ``prewitt``.

    Parameters
    ----------
    size : int
        1-D size of the filter (should be odd).
    """
    def prewitt(input, axis=-1, output=None, mode='reflect', cval=0.0):
        input = np.asarray(input)
        if axis < 0:
            axis += input.ndim
        if type(output) is not np.ndarray:
            output = np.zeros_like(input)

        kernel = list(range(1, size // 2 + 1))
        kernel = [-x for x in reversed(kernel)] + [0] + kernel
        smooth = np.ones(size, dtype=np.int32)
        smooth = smooth / np.abs(kernel).sum()
        smooth = list(smooth)

        ndimage.correlate1d(input, kernel, axis, output, mode, cval, 0)
        axes = [ii for ii in range(input.ndim) if ii != axis]
        for ii in axes:
            ndimage.correlate1d(output, smooth, ii, output, mode, cval, 0)
        return output

    return prewitt


def intensity_distance_seeds(image_data, resolution, axis=0, erosion_radius=16, 
        min_sep=24, visualize=False):
    """Create seed locations maximally distant from a Sobel filter.

    Parameters
    ----------
    image_data : ndarray
    resolution : ndarray
    axis : int, optional
        Axis along which to slices volume to generate seeds in 2D. If
        None volume is processed in 3D.
    erosion_radius : int, optional
        L_infinity norm radius of the structuring element for eroding
        components.
    min_sep : int, optional
        L_infinity minimum separation of seeds in nanometers.

    Returns
    -------
    list of ndarray
    """
    # Late import as this is the only function using Scikit.
    from skimage import morphology

    structure = np.ones(np.floor_divide(erosion_radius, resolution) * 2 + 1)

    if axis is None:
        def slices():
            yield [slice(None), slice(None), slice(None)]
    else:
        structure = structure[axis]

        def slices():
            for i in xrange(image_data.shape[axis]):
                s = map(slice, [None] * 3)
                s[axis] = i
                yield s

    sobel = np.zeros_like(image_data)
    thresh = np.zeros_like(image_data)
    transform = np.zeros_like(image_data)
    skmax = np.zeros_like(image_data)
    for s in slices():
        image_slice = image_data[s]
        if axis is not None and not np.any(image_slice):
            logging.debug('Skipping blank slice.')
            continue
        logging.debug('Running Sobel filter on image shape %s', image_data.shape)
        sobel[s] = ndimage.generic_gradient_magnitude(image_slice, make_prewitt(int((24 / resolution).max() * 2 + 1)))
        # sobel = ndimage.grey_dilation(sobel, size=(5,5,3))
        logging.debug('Running distance transform on image shape %s', image_data.shape)

        # For low res images the sobel histogram is unimodal. For now just
        # threshold the histogram at the mean.
        thresh[s] = sobel[s] < np.mean(sobel[s])
        thresh[s] = ndimage.binary_erosion(thresh[s], structure=structure)
        transform[s] = ndimage.distance_transform_cdt(thresh[s])
        # Remove missing sections from distance transform.
        transform[s][image_slice == 0] = 0
        logging.debug('Finding local maxima of image shape %s', image_data.shape)
        skmax[s] = morphology.thin(morphology.extrema.local_maxima(transform[s]))

    if visualize:
        viewer = WrappedViewer()
        viewer.add(image_data, name='Image')
        viewer.add(sobel, name='Filtered')
        viewer.add(thresh.astype(np.float), name='Thresholded')
        viewer.add(transform.astype(np.float), name='Distance')
        viewer.add(skmax, name='Seeds', shader=get_color_shader(0, normalized=False))
        viewer.print_view_prompt()

    mask = np.zeros(np.floor_divide(min_sep, resolution) + 1)
    mask[0, 0, 0] = 1
    seeds = np.transpose(np.nonzero(skmax))
    for seed in seeds:
        if skmax[tuple(seed)]:
            lim = np.minimum(mask.shape, skmax.shape - seed)
            skmax[map(slice, seed, seed + lim)] = mask[map(slice, lim)]

    seeds = np.transpose(np.nonzero(skmax))

    return seeds


def grid_seeds(image_data, _, grid_step_spacing=1):
    """Create seed locations in a volume on a uniform grid.

    Parameters
    ----------
    image_data : ndarray

    Returns
    -------
    list of ndarray
    """
    seeds = []
    shape = image_data.shape
    grid_size = CONFIG.model.move_step * grid_step_spacing
    for x in range(grid_size[0], shape[0], grid_size[0]):
        for y in range(grid_size[1], shape[1], grid_size[1]):
            for z in range(grid_size[2], shape[2], grid_size[2]):
                seeds.append(np.array([x, y, z], dtype=np.int32))

    return seeds


def distance_transform_seeds(image_data):
    """Create seed locations maximally distant from thresholded raw image.

    Parameters
    ----------
    image_data : ndarray

    Returns
    -------
    list of ndarray
    """
    from skimage.morphology import extrema
    from skimage import filters
    
    seeds = []
    if image_data.dtype == np.bool:
        #assuming membrane zero and cells one labeled
        thresh = image_data
    else: 
        # otsu thresholding
        thresh = image_data < filters.threshold_otsu(image_data)
    
    transform = ndimage.distance_transform_cdt(thresh)
    skmax = extrema.local_maxima(transform)
    seeds = np.transpose(np.nonzero(skmax))
    
    return seeds


def membrane_seeds(image_data):
    """Create seed locations on membrane by thresholding and binary erosion.

    Parameters
    ----------
    image_data : ndarray

    Returns
    -------
    list of ndarray
    """
    from skimage import measure
    from skimage import filters
    seeds = []
    if image_data.dtype == np.bool:
        thresh = image_data
        thresh[0,:,:] = ndimage.binary_erosion(thresh[0,:,:])
    else:
        thresh = image_data > filters.threshold_otsu(image_data)
        thresh[0,:,:] = ndimage.binary_erosion(thresh[0,:,:])
        #filter small connected components
        components = measure.label(thresh, background=0)
        unique, counts = np.unique(components, return_counts=True)
        idx = counts < 20
        thresh[components == unique[idx]] = 0
    
    seeds = np.transpose(np.nonzero(thresh))
    
    return seeds


def few_membrane_seeds(image_data):
    """Create seed locations on membrane by thresholding and binary erosion.

    Parameters
    ----------
    image_data : ndarray

    Returns
    -------
    list of ndarray
    """
    from skimage import measure
    from skimage import filters
    
    seeds = []
    if image_data.dtype == np.bool:
        thresh = image_data
        thresh[0,:,:] = ndimage.binary_erosion(thresh[0,:,:])
    else:
        thresh = image_data > filters.threshold_otsu(image_data)
        thresh[0,:,:] = ndimage.binary_erosion(thresh[0,:,:])
        #filter small connected components
        components = measure.label(thresh, background=0)
        unique, counts = np.unique(components, return_counts=True)
        idx = counts < 20
        thresh[components == unique[idx]] = 0
    
    all_seeds = np.transpose(np.nonzero(thresh))
    idx = np.random.choice(len(all_seeds),15, replace=True)
    seeds = all_seeds[idx]
    
    return seeds



def local_minima_seeds(image_data):
    """Create seed locations which are local minimas of the original image.


    Parameters
    ----------
    image_data : ndarray

    Returns
    -------
    list of ndarray
    """
    from skimage.morphology import extrema

    seeds = []
    if image_data.dtype == np.bool:
        
        return distance_transform_seeds(image_data)
    else:
        skmax = extrema.local_minima(image_data)
        seeds = np.transpose(np.nonzero(skmax))
    
        return seeds


def neuron_seeds(image_data, seed_num):
    seeds = []

    thresh = ndimage.binary_erosion(image_data > 0) 
    seeds = np.transpose(np.nonzero(thresh))
    if len(seeds) < 10000:
        seeds = np.transpose(np.nonzero(image_data))
    
    idx = np.random.choice(len(seeds), seed_num, replace=True)
    seeds = seeds[idx]

    return seeds


def neuron_dt_seeds(image_data, seed_num):
    
    from skimage.feature import peak_local_max
    
    seeds = []
    thresh = image_data > 0
    transform = ndimage.distance_transform_cdt(thresh)
    seeds = peak_local_max(transform, exclude_border=0, min_distance=20)
    #skmax = extrema.local_maxima(transform)

    if seed_num < len(seeds):
        idx = np.random.choice(len(seeds), seed_num, replace=True)
        seeds = seeds[idx]
    
    return seeds


def cell_interior_seeds(image_data, mask_data):
    """Create seed locations as connected components wihtin the cells.

    Parameters
    ----------
    image_data : ndarray

    Returns
    -------
    list of ndarray
    """
    from skimage import filters
    seeds = []
    
    if image_data.dtype == np.bool:
        thresh = np.logical_not(image_data)
    else: 
        # otsu thresholding
        thresh = image_data > filters.threshold_otsu(image_data)
    
    thresh[0,:,:] = np.logical_or(ndimage.binary_dilation(thresh[0,:,:]), 
            np.logical_not(mask_data))
    interior = np.logical_not(thresh)
    seeds = np.transpose(np.nonzero(interior))
    
    return seeds


# Note that these must be added separately to the CLI.
SEED_GENERATORS = {
    'grid': grid_seeds,
    'sobel': intensity_distance_seeds,
    'membrane': membrane_seeds,
    'distance_transform': distance_transform_seeds,
    'local_minima': local_minima_seeds,
    'cell_interior': cell_interior_seeds,
    'few_membrane': few_membrane_seeds,
    'neuron': neuron_seeds,
    'neuron_dt': neuron_dt_seeds
}
