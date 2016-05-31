# DGGS resampler script

This repo has some scripts for messing with satellite data and DGGSes. In
particular, `resampler.py` can take a GeoTIFF and stick it into a custom HDF5
format which partitions data into a different HDF5 dataset for each grid cell.

To run the resampler, do the following:

```sh
sudo apt-get install build-essential libedit-dev llvm-3.7 python-virtualenv \
    python3-dev libgdal-dev
virtualenv env -p "$(which python3)"
. env/bin/activate
# numba needs LLVM 3.7
LLVM_CONFIG="$(which llvm-config-3.7)" pip install -r requirements.txt
./resampler.py some-geotiff-you-downloaded.tif result.h5 2 7 5
```

The Python 3 requirement is important! Also, you might get some issues
installing GDAL. If so, try something like:

```sh
pip install --global-option=build_ext --global-option="-I/usr/include/gdal" \
    GDAL==1.11.2
```

You can replace `1.11.2` with a more recent release in the `1.11.X` series if
one is available.

For whatever reason, the `display` script didn't work for me initially (in a
virtualenv on Ubuntu 16.04). I fixed the issue by doing `sudo apt install
tk-dev` and updating `display.py` to fall back to the Tk matplotlib renderer if
no other interactive renderer is enabled.
