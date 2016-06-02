#!/usr/bin/env python3

from osgeo import gdal
from osgeo import gdalconst
import pytz
import rhealpix_dggs.dggs as dggs
import numpy as np
import h5py
from itertools import chain
import re
import sys


# For parsing AGDC filenames
AGDC_RE = re.compile(
    r'^(?P<sat_id>[^_]+)_(?P<sensor_id>[^1234567890]+)_(?P<prod_code>[^_1234567890]+)_'
    r'(?P<lon>[-0123456789]+)_(?P<lat>[^_]+)_(?P<year>\d{4})-(?P<month>\d{2})-'
    r'(?P<day>\d{2})T(?P<hour>\d+)-(?P<minute>\d+)-(?P<second>\d+(\.\d+)?)'
    r'\.tif$'
)

def cell_name(cell):
    return '/'.join(str(cell))

# Take out the numba check if you're not cool enough to run numba :P
from numba import jit, int16, int32
@jit(int16[:, :](int16[:, :], int16, int32, int32, int32, int32, int32), nopython=True)
def make_cell_data(band_data, missing_value, resolution_gap, bottom, top, left, right):
    data = np.zeros((3 ** resolution_gap, 3 ** resolution_gap), dtype=np.int16)
    w = (right - left) / (3 ** resolution_gap)
    h = (bottom - top) / (3 ** resolution_gap)
    rng = np.arange(3 ** resolution_gap)

    # Numba doesn't do bounds checks or support .clamp(), so we have to do this
    # fancy stuff
    band_width = band_data.shape[1]

    lefts = (left + w*rng).astype(np.int64)
    lefts[lefts < 0] = 0
    # See below for explanation of *_invalid_mask
    horiz_invalid_mask = lefts >= band_width
    lefts[horiz_invalid_mask] = 0

    rights = (left + w*(rng+1)).astype(np.int64)
    rights[rights < 0] = 0
    rights[rights > band_width] = band_width
    rights[horiz_invalid_mask] = 0

    band_height = band_data.shape[1]

    tops = (top + h*rng).astype(np.int64)
    tops[tops < 0] = 0
    # Idea of *_invalid_mask is that if the top for a window is outside of the
    # image, then we'll always get garbage. We set those tops to 0 so that they
    # are cancelled by the "tops < bots" check below.
    vert_invalid_mask = tops >= band_height
    tops[vert_invalid_mask] = 0

    bots = (top + h*(rng+1)).astype(np.int64)
    bots[bots < 0] = 0
    bots[bots > band_height] = band_height
    # Also need to invalidate bots corresponding to tops outside the image
    bots[vert_invalid_mask] = 0

    # If we don't do this check then we end up with an exception once we call
    # flatten() on an empty array :(
    valid_y, = (tops < bots).nonzero()
    valid_x, = (lefts < rights).nonzero()

    for y in valid_y:
        t_idx = tops[y]
        b_idx = bots[y]
        sub_data = band_data[:, t_idx:b_idx]
        for x in valid_x:
            l_idx = lefts[x]
            r_idx = rights[x]
            norm_subslice = sub_data[l_idx:r_idx, :]
            flat_subslice = norm_subslice.flatten()
            subslice = flat_subslice[flat_subslice != missing_value]
            if subslice.size:
                value = subslice.mean()
                data[x, y] = int(value)

    return data


def parse_agdc_fn(fn):
    """Parses a filename in the format used by the AGDC Landsat archive. For
    example, ``LS7_ETM_NBAR_149_-036_2012-02-10T23-50-47.650696.tif`` is a
    Landsat 7 observation in GeoTIFF format taken on Februrary 10 2012 at
    11:50pm (amongst other things).

    >>> fn = 'LS7_ETM_NBAR_149_-036_2012-02-10T23-50-47.650696.tif'
    >>> sorted_results = sorted(parse_agdc_fn(fn).items())
    >>> print('\\n'.join('{}: {}'.format(k, v) for k, v in sorted_results))
    datetime: 2012-02-10 23:50:47.650696+00:00
    lat: -36.0
    lon: 149.0
    prod_code: NBAR
    sat_id: LS7
    sensor_id: ETM

    :param string fn: Filename for observation.
    :return: Dictionary of extracted metdata from filename.
    """
    match = AGDC_RE.match(fn)
    if match is None:
        raise ValueError('Invalid AGDC filename: "{}"'.format(fn))
    info = match.groupdict()
    raw_sec = float(info['second'])
    int_sec = int(raw_sec)
    microsecond = int(1e6 * (raw_sec % 1))
    dt = pytz.datetime.datetime(
        year=int(info['year']), month=int(info['month']), day=int(info['day']),
        hour=int(info['hour']), minute=int(info['minute']),
        second=int_sec, microsecond=microsecond, tzinfo=pytz.utc
    )
    return {
        'lat': float(info['lat']), 'lon': float(info['lon']), 'datetime': dt,
        'prod_code': info['prod_code'], 'sensor_id': info['sensor_id'],
        'sat_id': info['sat_id']
    }


def add_meta(metadata, group):
    """Add metadata, probably AGDC-related (e.g. acquisition time, satellite
    ID), to an HDF5 group."""
    for k, v in metadata.items():
        if isinstance(v, pytz.datetime.datetime):
            enc_v = v.isoformat()
        else:
            # Hope for the best!
            enc_v = v
        group.attrs[k] = enc_v
    return group


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

    tif_meta = parse_agdc_fn(filename)
    add_meta(tif_meta, hdf5_file)

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

            assert right > left and bottom > top
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
