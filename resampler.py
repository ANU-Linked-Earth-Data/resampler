#!/usr/bin/env python3

from osgeo import gdal
from osgeo import gdalconst
import pytz
from rhealpix_dggs import dggs
import numpy as np
import h5py

from argparse import ArgumentParser
from itertools import chain
import re
from time import time


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
    data = missing_value * np.ones(
        (3 ** resolution_gap, 3 ** resolution_gap), dtype=np.int16
    )
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
        sub_data = band_data[t_idx:b_idx, :]
        for x in valid_x:
            l_idx = lefts[x]
            r_idx = rights[x]
            norm_subslice = sub_data[:, l_idx:r_idx]
            flat_subslice = norm_subslice.flatten()
            subslice = flat_subslice[flat_subslice != missing_value]
            if subslice.size:
                value = subslice.mean()
                data[y, x] = int(value)

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


def apply_transform(transform, col, row):
    """Take a transform from GDAL's ``GetGeoTransform`` and apply it to
    translate a column and row in an image into a longitude and latitude"""
    lon, lat = gdal.ApplyGeoTransform(transform, col, row)
    return lon, lat


def invert_transform(transform, lon, lat):
    """Like apply_transform, but inverts the transform first so that it returns
    pixel coordinates from latitude and longitude."""
    all_good, actual_transform = gdal.InvGeoTransform(transform)
    assert all_good, 'InvGeoTransform failed (somehow?)'
    # Yup, (col, row) in pixel coordinates
    px, py = gdal.ApplyGeoTransform(actual_transform, lon, lat)
    return px, py


def from_file(filename, hdf5_file, band_num, max_resolution, resolution_gap):
    """ Reads a geotiff file and converts the data into a hdf5 rhealpix file """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    numBands = dataset.RasterCount
    transform = dataset.GetGeoTransform()
    rddgs = dggs.RHEALPixDGGS()
    for resolution in range(0,20):
        upper_left = apply_transform(transform, 0, 0)
        lower_right = apply_transform(transform, width, height)
        cells = rddgs.cells_from_region(
            resolution, upper_left, lower_right, plane=False
        )
        if len(cells) > 1:
            outer_res = resolution - 1
            break

    try:
        tif_meta = parse_agdc_fn(filename)
        add_meta(tif_meta, hdf5_file)
    except ValueError:
        print("Can't read metadata from filename. Is it in the AGDC format?")
        tif_meta = None

    print("Processing band ", band_num, "/", numBands, "...")
    band = dataset.GetRasterBand(band_num)
    missing_val = band.GetNoDataValue()
    masked_data = np.ma.masked_equal(band.ReadAsArray(), missing_val)
    # nans play nice with numba
    band_data = band.ReadAsArray()
    for resolution in range(outer_res, max_resolution + 1):
        print("Processing resolution ", resolution, "/", max_resolution, "...")
        upper_left = apply_transform(transform, 0, 0)
        lower_right = apply_transform(transform, width, height)
        cells = rddgs.cells_from_region(
            resolution, upper_left, lower_right, plane=False
        )
        for cell in chain(*cells):
            north_west, north_east, south_east, south_west = cell.vertices(plane=False)

            # Get clamped bounds in image (row/col) coordinates
            left, top = north_west
            right, bottom = south_east
            left, top = map(int, invert_transform(transform, left, top))
            right, bottom = map(int, invert_transform(transform, right, bottom))
            l = max(0, left)
            r = max(0, right)
            t = max(0, top)
            b = max(0, bottom)

            pixel_value = masked_data[t:b+1,l:r+1].mean()
            if pixel_value is np.ma.masked:
                continue

            assert right > left and bottom > top
            data = make_cell_data(band_data, np.int16(missing_val), resolution_gap, bottom, top, left, right)

            # Write the HDF5 group. This is much faster than writing inline,
            # and lets us use numba.
            group = hdf5_file.create_group(cell_name(cell))
            if tif_meta is not None:
                add_meta(tif_meta, group)
            group.attrs['bounds'] = np.array([
                north_west, north_east, south_east, south_west, north_west
            ])
            group.attrs['centre'] = np.array(cell.centroid(plane=False))
            group.attrs['missing_value'] = np.int16(missing_val)
            group['pixel'] = pixel_value
            group.create_dataset('data', data=data, compression='szip')


parser = ArgumentParser()
parser.add_argument('input', type=str, help='path to input GeoTIFF')
parser.add_argument('output', type=str, help='path to output HDF5 file')
parser.add_argument(
    '--band', type=int, default=2, help='band from GeoTIFF to resample'
)
parser.add_argument(
    '--max-res', type=int, dest='max_res', default=6,
    help='maximum DGGS depth to resample at'
)
parser.add_argument(
    '--res-gap', type=int, dest='res_gap', default=5,
    help='number of DGGS levels to go down when generating tile data'
)


if __name__ == "__main__":
    args = parser.parse_args()
    print('Reading from %s and writing to %s' % (args.input, args.output))
    print(
        'Resampling band %i to depth %i with gap %i (so %i pixels per tile)'
        % (args.band, args.max_res, args.res_gap, 9 ** args.res_gap)
    )

    start_time = time()
    with h5py.File(args.output, "w") as hdf5_file:
        from_file(
            args.input, hdf5_file, args.band, args.max_res, args.res_gap
        )

    elapsed = time() - start_time
    print("Done! Took %.2fs" % elapsed)
