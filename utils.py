#!/usr/bin/env python
#
# This file is part of FIREwire
# 
# FIREwire is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIREwire is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FIREwire.  If not, see <http://www.gnu.org/licenses/>.
#
# FIREwire Copyright (C) 2008 - 2010 Julien Bert 

# open image
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

# open a raw volume
def volume_open(name, nx, ny, nz, nbyte):
    from numpy  import zeros
    from sys    import exit
    try:
        data = open(name, 'rb').read()
    except IOError:
        print 'open_volume error: can not open the file'
        exit()
    if nbyte == 1:   buf = fromstring(data, 'uint8').astype(float)
    elif nbyte == 2: buf = fromstring(data, 'uint16').astype(float)
    buf *= 1.0 / max(buf)
    buf  = buf.reshape(nz, ny, nx)

    return buf

# get a slice from a volume
def volume_slice(vol, pos=0, axe='z'):
    if   axe == 'z': return vol[pos]
    elif axe == 'x': return vol[:, :, pos]
    elif axe == 'y': return vol[:, pos, :]

def image_write(slice, name):
    from PIL import Image
    ny, nx = slice.shape
    slice = slice * 255
    slice = slice.astype('uint8')
    pilImage = Image.frombuffer('L', (nx, ny), slice, 'raw', 'L', 0, 1)
    pilImage.save(name)

# barrier function
def wait():
    raw_input('WAITING [Enter]')
    return
