#!/usr/bin/env python
from numpy import *

def image_open(name):
    from PIL import Image
    from sys import exit
    import numpy
    im   = Image.open(name)
    data = numpy.fromstring(im.tostring(), 'uint8')
    w, h = im.size
    mode = im.mode
    if mode == 'L': data = data.reshape((h, w))
    elif mode == 'RGB': data = data.reshape((3, h, w))
    else:
        print 'open_image: chanel must L or RGB'
        exit()

    return data

def volume_open(name, nx, ny, nz, nbyte):
    from numpy  import zeros
    from sys    import exit
    from struct import unpack
    try:
        data = open(name, 'rb')
    except IOError:
        print 'open_volume error: can not open the file'
        exit()

    # read file until the end
    buf = []
    while 1:
        # read n byte
        b = data.read(nbyte)
        # if the end stop
        if b == '': break
        # convert unicode hexa ton ascii
        b   = unpack('%iB' % nbyte, b)
        # value code on several byte so merge them
        val = 1
        for n in xrange(nbyte): val *= b[n]
        # save
        buf.append(val)
    # check if the size is ok
    if len(buf) != (nx * ny * nz):
        print 'open_volume error: check sum failed'
        exit()
    # convert and normalize to numpy array
    
    buf = array(buf, 'f')
    buf = buf.reshape(nz, ny, nx)
    buf = buf / buf.max()

    return buf

def volume_slice(vol, z):
    from numpy import array
    return vol[z]

def slice_export(slice, name):
    from PIL import Image
    ny, nx = slice.shape
    slice = slice * 255
    slice = slice.astype('uint8')
    pilImage = Image.frombuffer('L', (nx, ny), slice, 'raw', 'L', 0, 1)
    pilImage.save(name)
