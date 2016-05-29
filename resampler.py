#!/usr/bin/env python3

from osgeo import gdal
from osgeo import gdalconst
import rhealpix_dggs.dggs as dggs
import numpy as np
import h5py
from itertools import chain
import sys

def cell_name(cell):
    return '/'.join(str(cell))

# Take out the numba check if you're not cool enough to run numba :P
from numba import jit, int16, int32
@jit(int16[:, :](int16[:, :], int16, int32, int32, int32, int32, int32), nopython=True)
def make_cell_data(band_data, missing_value, resolution_gap, bottom, top, left, right):
    data = np.zeros((3 ** resolution_gap, 3 ** resolution_gap), dtype=np.int16)
    w = (right - left) / (3 ** resolution_gap)
    h = (bottom - top) / (3 ** resolution_gap)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            # Numba doesn't do bounds checks, so we have to do this fancy stuff
            l = max(0, int(left+w*x))
            r = max(0, int(left+w*(x+1)))
            t = max(0, int(top+h*y))
            b = max(0, int(top+h*(y+1)))
            l_idx = min(l, band_data.shape[0]-1)
            r_idx = min(r+1, band_data.shape[0])
            t_idx = min(t, band_data.shape[1]-1)
            b_idx = min(b+1, band_data.shape[1])
            if l_idx < r_idx and t_idx < b_idx:
                data[x, y] = 0
                norm_subslice = band_data[l_idx:r_idx, t_idx:b_idx]
                subslice = norm_subslice.flatten()
                valid = subslice != missing_value
                if valid.any():
                    value = subslice[valid].mean()
                    data[x,y] = int(value)
    return data

def fromFile(filename, hdf5_file, band_num, max_resolution, resolution_gap):
    """ Reads a geotiff file and converts the data into a hdf5 rhealpix file """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    numBands = dataset.RasterCount
    transform = dataset.GetGeoTransform()
    rddgs = dggs.RHEALPixDGGS()
    for resolution in range(0,20):
        cells = rddgs.cells_from_region(
            resolution,
            (transform[0], transform[3]),
            (transform[0] + width * transform[1], transform[3] + height * transform[5]),
            plane=False
        )
        if len(cells) > 1:
            outer_res = resolution - 1
            break

    print("Processing band ", band_num, "/", numBands, "...")
    band = dataset.GetRasterBand(band_num)
    missing_val = band.GetNoDataValue()
    masked_data = np.ma.masked_equal(band.ReadAsArray(), missing_val)
    # nans play nice with numba
    band_data = band.ReadAsArray()
    for resolution in range(outer_res, max_resolution + 1):
        print("Processing resolution ", resolution, "/", max_resolution, "...")
        cells = rddgs.cells_from_region(
            resolution,
            (transform[0], transform[3]),
            (transform[0] + width * transform[1], transform[3] + height * transform[5]),
            plane=False
        )
        for cell in chain(*cells):
            north_west, north_east, south_east, south_west = cell.vertices(plane=False)

            # Get clamped bounds in image (row/col) coordinates
            left, top = north_west
            right, bottom = south_east
            left   = int((left   - transform[0]) / transform[1])
            right  = int((right  - transform[0]) / transform[1])
            top    = int((top    - transform[3]) / transform[5])
            bottom = int((bottom - transform[3]) / transform[5])
            l = max(0, left)
            r = max(0, right)
            t = max(0, top)
            b = max(0, bottom)

            pixel_value = masked_data[l:r+1,t:b+1].mean()
            if pixel_value is np.ma.masked:
                continue

            data = make_cell_data(band_data, np.int16(missing_val), resolution_gap, bottom, top, left, right)

            # Write the HDF5 group. This is much faster than writing inline,
            # and lets us use numba.
            group = hdf5_file.create_group(cell_name(cell))
            group.attrs['bounds'] = np.array([
                north_west, north_east, south_east, south_west, north_west
            ])
            group.attrs['centre'] = np.array(cell.centroid(plane=False))
            group['pixel'] = pixel_value
            group['data'] = data


def run_code(tif_name, hdf5_name, band_num, max_res, res_gap):
    with h5py.File(hdf5_name, "w") as hdf5_file:
        fromFile(tif_name, hdf5_file, int(band_num), int(max_res), int(res_gap))
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("-------------------------------------------------------------")
        print("Usage: resampler.py input.tif output.hdf5 band max_resolution resolution_gap")
        print("-------------------------------------------------------------")
        print("Example: resampler.py test.tif result.hdf5 2 7 5")
    else:
        run_code(*sys.argv[1:])
