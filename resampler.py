#!/usr/bin/env python3

from osgeo import gdal, gdalconst, osr
import pytz
from dateutil.parser import parse as parse_date
from rhealpix_dggs import dggs
import numpy as np
from scipy.misc import toimage
import h5py

from argparse import ArgumentParser, ArgumentTypeError, FileType
from itertools import chain
from io import BytesIO
from os.path import basename, splitext, extsep
import re
import sys
from time import time

# For parsing AGDC filenames
AGDC_RE = re.compile(
    r'^(?P<sat_id>[^_]+)_(?P<sensor_id>ETM|OLI_TIRS)_(?P<prod_code>[^_]+)_'
    r'(?P<lon>[^_]+)_(?P<lat>[^_]+)_(?P<year>\d+)-(?P<month>\d+)-'
    r'(?P<day>\d+)T(?P<hour>\d+)-(?P<minute>\d+)-(?P<second>\d+(\.\d+)?)'
    r'\.tif$')


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
    dt = pytz.datetime.datetime(year=int(info['year']),
                                month=int(info['month']),
                                day=int(info['day']),
                                hour=int(info['hour']),
                                minute=int(info['minute']),
                                second=int_sec,
                                microsecond=microsecond,
                                tzinfo=pytz.utc)
    rv = {
        'lat': float(info['lat']),
        'lon': float(info['lon']),
        'datetime': dt,
        'prod_code': info['prod_code'],
        'sensor_id': info['sensor_id'],
        'sat_id': info['sat_id']
    }
    return rv


def pixel_to_long_lat(geotransform, dataset_projection, col, row):
    """ Given a pixel position as a column/row, calculates its position in the
        dataset's reference system, then converts it to a latitude and
        longitude in the WGS_84 system """
    tx = osr.CoordinateTransformation(dataset_projection, wgs_84_projection)
    lon, lat, height = tx.TransformPoint(*gdal.ApplyGeoTransform(geotransform,
                                                                 col, row))
    return lon, lat


rhealpix_proj4_string = "+proj=rhealpix +I +lon_0=0 +a=1 +ellps=WGS84" \
                        " +npole=0 +spole=0 +wktext"

rhealpix_projection = osr.SpatialReference()
rhealpix_projection.ImportFromProj4(rhealpix_proj4_string)

wgs_84_projection = osr.SpatialReference()
wgs_84_projection.ImportFromEPSG(4326)


def reproject_dataset(dataset, dataset_projection, cell, resolution_gap):
    """ Based on https://jgomezdans.github.io/gdal_notes/reprojection.html """
    data_to_rhealpix = osr.CoordinateTransformation(dataset_projection,
                                                    rhealpix_projection)
    lonlat_to_rhealpix = osr.CoordinateTransformation(wgs_84_projection,
                                                      rhealpix_projection)

    geo_t = dataset.GetGeoTransform()
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    (ulx, uly, ulz) = data_to_rhealpix.TransformPoint(geo_t[0], geo_t[3])
    (lrx, lry, lrz) = data_to_rhealpix.TransformPoint(
        geo_t[0] + geo_t[1] * x_size, geo_t[3] + geo_t[5] * y_size)

    # Calculate the new geotransform
    north_west, _, south_east, _ = cell.vertices(plane=False)
    left, top = north_west
    right, bottom = south_east
    left, top, _ = lonlat_to_rhealpix.TransformPoint(left, top)
    right, bottom, _ = lonlat_to_rhealpix.TransformPoint(right, bottom)
    num_pixels = 3**resolution_gap
    new_geo = (left, (right - left) / num_pixels, 0, top, 0,
               (bottom - top) / num_pixels)
    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', num_pixels, num_pixels, dataset.RasterCount,
                          dataset.GetRasterBand(1).DataType)
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(rhealpix_projection.ExportToWkt())

    # Perform the projection/resampling
    error_code = gdal.ReprojectImage(
        dataset, dest, dataset_projection.ExportToWkt(),
        rhealpix_projection.ExportToWkt(), gdal.GRA_Bilinear)
    assert error_code == 0, "Reprojection failed"

    array = dest.ReadAsArray()
    assert 2 <= array.ndim <= 3
    if array.ndim == 2:
        # Insert an extra axis for the band. Downstream code screws up
        # otherwise.
        array = array[np.newaxis, :]

    assert array.ndim == 3
    assert array.shape[1:] == (num_pixels, num_pixels)

    return array


def open_dataset(filename):
    """ Reads a geotiff or a HDF4 file and returns a gdal dataset """
    _, extension = splitext(filename)
    extension = extension.lstrip(extsep)
    if extension == "tif":
        return gdal.Open(filename, gdalconst.GA_ReadOnly)
    elif extension == "hdf":
        # Yay gdal can open MODIS hdf files! :D
        dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
        meta = dataset.GetMetadata()
        assert '_FillValue' in meta
        fill_value = meta['_FillValue']
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(float(fill_value))

        # But it doesn't read in the georeferencing system properly ...
        from pyhdf.SD import SD, SDC
        hdf = SD(filename, SDC.READ)
        latitudes = hdf.select('latitude')[:]
        longitudes = hdf.select('longitude')[:]

        left = longitudes[0]
        top = latitudes[0]

        x_spacing = np.mean([longitudes[i + 1] - longitudes[i]
                             for i in range(len(longitudes) - 1)])
        y_spacing = np.mean([latitudes[i + 1] - latitudes[i]
                             for i in range(len(latitudes) - 1)])

        left -= x_spacing / 2
        top -= y_spacing / 2

        geotransform = (left, x_spacing, 0, top, 0, y_spacing)

        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(wgs_84_projection.ExportToWkt())

        return dataset
    else:
        assert False, "Invalid file extension " + extension \
            + ", expected 'tif' or 'hdf'"


def time_format(timestamp):
    """Imitate Java's DateTimeFormatter.ISO_INSTANT style for datetimes. It's
    not exactly ISO_INSTANT, because we're truncating to second precision."""
    as_utc = timestamp.astimezone(pytz.UTC)
    return as_utc.strftime('%Y-%m-%dT%H:%M:%SZ')


def png_buffer(array):
    """Convert an array to PNG, handling transparency in an
    at-least-partially-sane manner."""
    assert array.ndim == 2

    im = toimage(array)
    alpha = toimage(array != 0)
    im.putalpha(alpha)

    # Return format is a buffer of PNG-encoded data
    fp = BytesIO()
    im.save(fp, format='png')

    return fp.getbuffer()


def from_file(filename, dataset, hdf5_file, max_resolution, resolution_gap,
              ds_name, timestamp):
    """ Converts a gdal dataset into a hdf5 rhealpix file """
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    rddgs = dggs.RHEALPixDGGS()

    dataset_projection = osr.SpatialReference()
    error_code = dataset_projection.ImportFromWkt(dataset.GetProjection())
    assert error_code == 0, "Dataset doesn't have a projection"

    upper_left = pixel_to_long_lat(geotransform, dataset_projection, 0, 0)
    lower_right = pixel_to_long_lat(geotransform, dataset_projection, width,
                                    height)

    print(geotransform, upper_left, lower_right)

    try:
        bounding_cell = rddgs.cell_from_region(upper_left,
                                               lower_right,
                                               plane=False)
        outer_res = bounding_cell.resolution
    except AttributeError:
        # dggs library produces this error, maybe when even top-level cells are
        # too small?
        outer_res = 0

    # Time suffix will be appended to each "pixel" and "png_band_<n>" record
    time_suffix = '@' + time_format(timestamp)

    # Will return this later
    num_bands = None

    for resolution in range(outer_res, max_resolution + 1):
        print("Processing resolution ", resolution, "/", max_resolution, "...")

        cells = rddgs.cells_from_region(resolution,
                                        upper_left,
                                        lower_right,
                                        plane=False)
        for cell in chain(*cells):
            north_west, north_east, south_east, south_west = cell.vertices(
                plane=False)
            if cell.region() != "equatorial":
                continue  # Yucky polar cells, ignore for now, maybe fix later

            data = reproject_dataset(dataset, dataset_projection, cell,
                                     resolution_gap)
            if not np.any(data):
                continue

            pixel_value = np.array([(np.mean(x[np.nonzero(x)])
                                     if np.any(x[np.nonzero(x)]) else 0)
                                    for x in data])
            assert pixel_value.ndim == 1
            num_bands = pixel_value.size

            # Write the HDF5 group in one go. This is faster than manipulating
            # the dataset directly.
            cell_group = hdf5_file.create_group(cell_name(cell))
            cell_group.attrs['bounds'] = np.array([
                north_west, north_east, south_east, south_west, north_west
            ])
            cell_group.attrs['centre'] = np.array(cell.centroid(plane=False))

            ds_group = cell_group.create_group(ds_name)
            ds_group['pixel' + time_suffix] = pixel_value

            if len(data.shape) == 2:
                data = np.array([data])

            for band_num in range(data.shape[0]):
                # Write out each band as a separate PNG
                band_data = data[band_num]
                out_bytes = png_buffer(band_data)

                png_ds_name = ('png_band_%i' % band_num) + time_suffix
                # H5T_OPAQUE (maps to np.void in h5py) doesn't work in JHDF5,
                # so we use an unsigned byte array for this (actually binary)
                # data.
                ds_group[png_ds_name] = np.frombuffer(out_bytes, dtype='uint8')

    return num_bands


def timestamp_arg(str_value):
    """argparse argument type that turns a supplied string value into a
    zone-aware datetime."""
    try:
        rv = parse_date(str_value)
    except ValueError:
        raise ArgumentTypeError("Couldn't parse date string '%s'" % str_value)
    if rv.tzinfo is None:
        raise ArgumentTypeError("Date string '%s' has no timezone" % str_value)
    return rv


parser = ArgumentParser()
parser.add_argument('input',
                    type=str,
                    help='path to input GeoTIFF or MODIS HDF4 file')
parser.add_argument('output', type=str, help='path to output HDF5 file')
parser.add_argument('--max-res',
                    type=int,
                    dest='max_res',
                    default=6,
                    help='maximum DGGS depth to resample at')
parser.add_argument('--ds-name',
                    type=str,
                    dest='ds_name',
                    default=None,
                    help='internal name for the new dataset')
parser.add_argument('--timestamp',
                    type=timestamp_arg,
                    dest='timestamp',
                    default=None,
                    help='override default timestamp')
parser.add_argument('--attributes',
                    type=FileType('r'),
                    default=None,
                    help='.ttl file containing qb:Attributes for the dataset')
parser.add_argument(
    '--res-gap',
    type=int,
    dest='res_gap',
    default=5,
    help='number of DGGS levels to go down when generating tile data')

if __name__ == "__main__":
    args = parser.parse_args()
    print('Reading from %s and writing to %s' % (args.input, args.output))
    print('Resampling to depth %i with gap %i (so %i pixels per tile)' %
          (args.max_res, args.res_gap, 9**args.res_gap))

    dataset = open_dataset(args.input)
    ds_name = args.ds_name
    timestamp = args.timestamp
    try:
        tif_meta = parse_agdc_fn(basename(args.input))
        timestamp = tif_meta['datetime']
        if ds_name is None:
            ds_name = '{sat_id}_{sensor_id}_{prod_code}'.format(**tif_meta)
    except ValueError:
        print("Could not parse filename. Is it in the AGDC format?",
              file=sys.stderr)
        if ds_name is None:
            ds_name = 'unknown'
        if timestamp is None:
            # give dataset a stupid timestamp just to spite the user
            timestamp = pytz.datetime.datetime(1923, 6, 4, tzinfo=pytz.UTC)

    print('Using dataset name "%s" and timestamp "%s"' % (ds_name, timestamp))

    start_time = time()
    with h5py.File(args.output, "w") as hdf5_file:
        # from_file() creates the general DGGS structure (but not top-level
        # metadata)
        num_bands = from_file(args.input, dataset, hdf5_file, args.max_res,
                              args.res_gap, ds_name, timestamp)

        # this adds metadata into the top level of the file
        if args.attributes is not None:
            attr_ttl = args.attributes.read()
        else:
            attr_ttl = ''
        name_pre = '/products/' + ds_name
        # Was using np.frombuffer(attr_ttl.encode('utf-8'), dtype='uint8'), but
        # I think it makes more sense to store this (non-binary) data as a
        # H5T_STRING.
        hdf5_file[name_pre + '/meta'] = attr_ttl
        hdf5_file[name_pre + '/numbands'] = num_bands
        hdf5_file[name_pre + '/tilesize'] = 3**args.res_gap

    elapsed = time() - start_time
    print("Done! Took %.2fs" % elapsed)
