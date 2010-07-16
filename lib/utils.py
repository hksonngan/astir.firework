#!/usr/bin/env python
#
# This file is part of FIREwork
# 
# FIREwork is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIREwork is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
#
# FIREwork Copyright (C) 2008 - 2010 Julien Bert 

# ==== Image ================================
# ===========================================

# open image
def image_open(name):
    from PIL import Image
    from sys import exit
    import numpy
    im   = Image.open(name)
    w, h = im.size
    mode = im.mode
    if mode == 'RGB' or mode == 'RGBA':
        im = im.convert('L')
    data = numpy.fromstring(im.tostring(), 'uint8')
    data = data.reshape((h, w))
    data = image_int2float(data)

    return data

def image_write(slice, name):
    from PIL import Image
    slice /= slice.max()
    ny, nx = slice.shape
    slice = slice * 255
    slice = slice.astype('uint8')
    pilImage = Image.frombuffer('L', (nx, ny), slice, 'raw', 'L', 0, 1)
    pilImage.save(name)

# get the 1D projection of an image
def image_1D_projection(im, axe = 'x'):
    if   axe == 'x': return im.sum(axis = 1)
    elif axe == 'y': return im.sum(axis = 0)
    
# get the 1D slice of an image
def image_1D_slice(im, x1, y1, x2, y2):
    from numpy import array
    
    # line based on DDA algorithm
    length = 0
    length = abs(x2 - x1)
    if abs(y2 - y1) > length: length = abs(y2 - y1)
    xinc = float(x2 - x1) / float(length)
    yinc = float(y2 - y1) / float(length)
    x    = x1 + 0.5
    y    = y1 + 0.5
    vec  = [] 
    for i in xrange(length):
        vec.append(im[int(y), int(x)])
        x += xinc
        y += yinc
    
    return array(vec, 'float32')

# some info to images
def image_infos(im):
    sh = im.shape
    if len(sh) == 2:
        print 'size: %ix%i min %f max %f mean %f std %f' % (sh[0], sh[1], im.min(), im.max(), im.mean(), im.std())
    if len(sh) == 3:
        print 'size: %ix%ix%i min %f max %f mean %f std %f' % (sh[0], sh[1], sh[2], im.min(), im.max(), im.mean(), im.std())


# normalize image ave=0, std=1
def image_normalize(im):
    ave = im.mean()
    std = im.std()
    im  = (im - ave) / std

    return im

# convert int to float
def image_int2float(im):
    im  = im.astype('float32')
    im /= 255.0
    
    return im

# compute fft of image
def image_fft(im):
    from numpy import fft
    l, w = im.shape
    if l != w:
        print 'Image must be square !'
        return -1
    if w % 2 == 1: imf = fft.fft2(im)
    else:          imf = fft.fft2(im, s=(w+1, w+1))
    imf  = fft.fftshift(imf)

    return imf

# compute power spectrum of image
def image_pows(im):
    imf   = image_fft(im)
    imf   = imf * imf.conj()
    imf   = imf.real
    l, w  = im.shape
    imf  /= (float(l * w))
    
    return imf

# compute dB values
def image_atodB(im):
    from numpy import log10
    return 20 * log10(1+im)

# compute logscale image
def image_logscale(im):
    from numpy import log10
    return log10(1 + im)

# compute periodogram
def image_periodogram(im):
    imf   = image_fft(im)
    imf   = abs(imf)**2
    l, w  = im.shape
    imf  /= float(l * w)
    
    return imf

# create image noise (gauss model)
def image_noise(l, w, sigma):
    from numpy import zeros
    from random import gauss

    mu = 0.5
    im = zeros((l, w), 'float32')
    for i in xrange(w):
        for j in xrange(l):
            im[j, i] = gauss(mu, sigma)

    im -= im.min()
    im /= im.max()
    
    return im
            
# compute RAPS, radial averaging power spectrum
def image_raps(im):
    from numpy import zeros, array

    lo, wo = im.shape
    im     = image_pows(im)
    l, w   = im.shape
    c      = (w - 1) // 2
    rmax   = c
    val    = zeros((rmax + 1), 'float32')
    ct     = zeros((rmax + 1), 'float32')
    val[0] = im[c, c] # fundamental
    ct[0]  = 1.0
    for i in xrange(c - rmax, c + rmax + 1):
        for j in xrange(c - rmax, c + rmax + 1):
            r = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5)
            if r >= rmax: continue
            ir    = int(r)
            cir   = ir + 1
            frac  = r  - ir
            cfrac = 1  - frac
            val[ir] += (cfrac * im[j, i])
            if cfrac != 0.0: ct[ir] += 1.0
            if cir <= rmax:
                val[cir] += (frac * im[j, i])
                if frac != 0.0: ct[cir] += 1.0

    val /= ct
    
    freq  = range(0, wo // 2 + 1)
    freq  = array(freq, 'float32')
    freq /= float(wo)
    
    return val, freq

# ==== Volume ===============================
# ===========================================

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

# ==== Misc =================================
# ===========================================

# barrier function
def wait():
    raw_input('WAITING [Enter]')
    return

# convert engineer prefix
def prefix_SI(mem):
    from math import log
    pref   = ['', 'k', 'M', 'G', 'T']
    iemem  = int(log(mem) // log(1e3))
    mem   /= (1e3 ** iemem)

    return '%5.2f %sB' % (mem, pref[iemem])

# ==== List-Mode ============================
# ===========================================
    
# Open list-mode subset
def listmode_open_subset(filename, N_start, N_stop):
    from numpy import zeros
    f      = open(filename, 'r')
    nlines = N_stop - N_start
    lm_id1 = zeros((nlines), 'int32')
    lm_id2 = zeros((nlines), 'int32')
    for n in xrange(N_start):
        buf = f.readline()
    for n in xrange(nlines):
        id1, id2 = f.readline().split()
        lm_id1[n] = int(id1)
        lm_id2[n] = int(id2)
    f.close()

    return lm_id1, lm_id2

# Nb events ti list-mode file
def listmode_nb_events(filename):
    f = open(filename, 'r')
    n = 0
    while 1:
        buf = f.readline()
        if buf == '': break
        n += 1
        
    return n

# Open Sensibility Matrix
def listmode_open_SM(filename):
    from numpy import array
    f    = open(filename, 'r')
    s    = 0
    S    = []
    while 1:
        s = f.readline()
        if s == '': break
        S.append(float(s))
    f.close()
    SM = array(S, 'float32')
    del S

    return SM
