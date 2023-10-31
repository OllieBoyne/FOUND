## `trimesh`, `shapely` issues

If you are having any issues running evaluation because `shapely` cannot find `geos_c.h` or similar, you may find the steps below help you:

1) Download [GEOS](https://libgeos.org/usage/download/)
2) Unpack and install

```cmd
tar xvfj geos-3.11.1.tar.bz2
cd geos-3.11.1 mkdir _build 
cd _build # Set up the build
 cmake \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=/usr/local \
make
ctest
make install DESTDIR=<...>
```

3) Set the following environment variables:

```cmd
export GEOS_LIBRARY_PATH=<DESTDIR>/lib
export GEOS_INCLUDE_PATH=<DESTDIR>/include
export LD_LIBRARY_PATH=<DESTDIR>/lib 
```

With `<DESTDIR>` being the path to the directory you installed GEOS to.

4) `pip install shapely`


You may also need to define those environment variables whenver you run `eval.py`