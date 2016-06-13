#!/usr/bin/env python3

from osgeo import gdal
from osgeo import gdalconst
from osgeo import osr
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

rhealpix_proj4_string = "+proj=rhealpix +I +lon_0=0 +a=1 +ellps=WGS84 +npole=0 +spole=0 +wktext"
def reproject_dataset (dataset, cell, resolution_gap):
    """ Based on https://jgomezdans.github.io/gdal_notes/reprojection.html """

    sourceProj = osr.SpatialReference()
    error_code = sourceProj.ImportFromWkt(dataset.GetProjection())
    assert error_code == 0, "Dataset doesn't have a projection"

    destProj = osr.SpatialReference()
    error_code = destProj.ImportFromProj4(rhealpix_proj4_string)
    assert error_code == 0, "Couldn't create rHEALPix projection"

    tx = osr.CoordinateTransformation (sourceProj, destProj)
    geo_t = dataset.GetGeoTransform ()
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    (ulx, uly, ulz ) = tx.TransformPoint( geo_t[0], geo_t[3])
    (lrx, lry, lrz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size,geo_t[3] + geo_t[5]*y_size )

    # Calculate the new geotransform
    north_west, _, south_east, _ = cell.vertices(plane=False)
    left, top = north_west
    right, bottom = south_east
    left, top, _ = tx.TransformPoint(left, top)
    right, bottom, _ = tx.TransformPoint(right, bottom)
    num_pixels = 3 ** resolution_gap
    new_geo = ( left, (right - left) / num_pixels, 0, \
                top, 0, (bottom - top) / num_pixels )
    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', num_pixels, num_pixels, dataset.RasterCount, dataset.GetRasterBand(1).DataType)
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(destProj.ExportToWkt())

    # Perform the projection/resampling
    error_code = gdal.ReprojectImage(dataset, dest, sourceProj.ExportToWkt(), destProj.ExportToWkt(), gdal.GRA_Bilinear)
    assert error_code == 0, "Reprojection failed"

    return dest

def from_file(filename, hdf5_file, max_resolution, resolution_gap):
    """ Reads a geotiff file and converts the data into a hdf5 rhealpix file """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
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

    for resolution in range(outer_res, max_resolution + 1):
        print("Processing resolution ", resolution, "/", max_resolution, "...")
        upper_left = apply_transform(transform, 0, 0)
        lower_right = apply_transform(transform, width, height)
        cells = rddgs.cells_from_region(
            resolution, upper_left, lower_right, plane=False
        )
        for cell in chain(*cells):
            north_west, north_east, south_east, south_west = cell.vertices(plane=False)

            data = reproject_dataset(dataset, cell, resolution_gap).ReadAsArray()
            if not np.any(data):
                continue

            pixel_value = np.array([np.mean(x[np.nonzero(x)]) for x in data])

            # Write the HDF5 group. This is much faster than writing inline,
            # and lets us use numba.
            group = hdf5_file.create_group(cell_name(cell))
            if tif_meta is not None:
                add_meta(tif_meta, group)
            group.attrs['bounds'] = np.array([
                north_west, north_east, south_east, south_west, north_west
            ])
            group.attrs['centre'] = np.array(cell.centroid(plane=False))
            group.attrs['missing_value'] = 0 # Value used by gdal.ReprojectImage()
            group['pixel'] = pixel_value
            group.create_dataset('data', data=data, compression='szip')


parser = ArgumentParser()
parser.add_argument('input', type=str, help='path to input GeoTIFF')
parser.add_argument('output', type=str, help='path to output HDF5 file')
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
        'Resampling to depth %i with gap %i (so %i pixels per tile)'
        % (args.max_res, args.res_gap, 9 ** args.res_gap)
    )

    start_time = time()
    with h5py.File(args.output, "w") as hdf5_file:
        from_file(
            args.input, hdf5_file, args.max_res, args.res_gap
        )

    elapsed = time() - start_time
    print("Done! Took %.2fs" % elapsed)
