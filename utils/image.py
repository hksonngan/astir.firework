#!/usr/bin/env pythonfx
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
# FIREwork Copyright (C) 2008 - 2011 Julien Bert 

# ==== Image ================================
# ===========================================

# open image
def image_open(name):
    from PIL     import Image
    from sys     import exit
    from os.path import splitext
    import numpy

    filename, ext = splitext(name)
    if ext == '.png' or ext == '.tif' or ext == '.bmp' or ext == '.jpg':
        im   = Image.open(name)
        w, h = im.size
        mode = im.mode
        if mode == 'RGB' or mode == 'RGBA' or mode == 'LA':
            im = im.convert('L')
        data = numpy.fromstring(im.tostring(), 'uint8')
        data = data.reshape((h, w))
        data = image_int2float(data)
    elif ext == '.im':
        f    = open(name, 'rb')
        data = numpy.fromfile(f, 'float32')
        f.close()
        ny   = data[0]
        nx   = data[1]
        data = data[2:]
        data = data.reshape(ny, nx)

    return data

# export image
def image_write(im, name):
    from PIL     import Image
    from os.path import splitext
    from numpy   import array
    
    filename, ext = splitext(name)
    slice = im.copy()

    if ext == '.png' or ext == '.tif' or ext == '.bmp' or ext == '.jpg':
        slice -= slice.min()
        slice /= slice.max()
        ny, nx = slice.shape
        slice = slice * 255
        slice = slice.astype('uint8')
        pilImage = Image.frombuffer('L', (nx, ny), slice, 'raw', 'L', 0, 1)
        pilImage.save(name)
    elif ext == '.im':
        # FIREwork image format
        ny, nx = slice.shape
        slice  = slice.reshape((ny*nx))
        slice  = slice.tolist()
        slice.insert(0, nx)
        slice.insert(0, ny)
        slice  = array(slice, 'float32')
        slice.tofile(name)

# export image with map color
def image_write_mapcolor(im, name, color='jet'):
    from numpy import array, zeros, take, ones
    from PIL   import Image
    ny, nx = im.shape
    npix   = ny * nx
    map    = im.copy()
    map    = map.astype('float32')
    map   -= map.min()
    map   /= map.max()
    map   *= 255
    map    = map.astype('uint8')
    map    = map.reshape(npix)

    lutr   = zeros((256), 'uint8')
    lutg   = zeros((256), 'uint8')
    lutb   = zeros((256), 'uint8')
    if color == 'jet':
        up  = array(range(0, 255,  3), 'uint8')
        dw  = array(range(255, 0, -3), 'uint8')
        stp = 85
        lutr[stp:2*stp]   = up
        lutr[2*stp:]      = 255
        lutg[0:stp]       = up
        lutg[stp:2*stp]   = 255
        lutg[2*stp:3*stp] = dw
        lutb[0:stp]       = 255
        lutb[stp:2*stp]   = dw
    elif color == 'hot':
        up  = array(range(0, 255,  3), 'uint8')
        stp = 85
        lutr[0:stp]       = up
        lutr[stp:]        = 255
        lutg[stp:2*stp]   = up
        lutg[2*stp:]      = 255
        lutb[2*stp:3*stp] = up
        lutb[3*stp:]      = 255
    elif color == 'pet':
        up2 = array(range(0, 255, 4), 'uint8') #  64
        up3 = array(range(0, 255, 8), 'uint8') #  32
        dw  = array(range(255, 0, -8), 'uint8') #  32
        lutr[0:64]   = 0
        lutg[0:64]   = 0
        lutb[0:64]   = up2
        lutr[64:128]   = up2
        lutg[64:128]   = 0
        lutb[64:128]   = 255
        lutr[128:160] = 255
        lutg[128:160] = 0
        lutb[128:160] = dw
        lutr[160:224] = 255
        lutg[160:224] = up2
        lutb[160:224] = 0
        lutr[224:256] = 255
        lutg[224:256] = 255
        lutb[224:256] = up3
        
    else: # hsv kind default
        up  = array(range(0, 255,  5), 'uint8')
        dw  = array(range(255, 0, -5), 'uint8')
        stp = 51
        lutr[0:stp]       = dw
        lutr[3*stp:4*stp] = up
        lutr[4*stp:]      = 255
        lutg[0:2*stp]     = 255
        lutg[2*stp:3*stp] = dw
        lutb[stp:2*stp]   = up
        lutb[2*stp:4*stp] = 255
        lutb[4*stp:5*stp] = dw
        
    matr  = take(lutr, map)
    matg  = take(lutg, map)
    matb  = take(lutb, map)
    mata  = ones((npix), 'uint8') * 255
    newim = zeros((npix*4), 'uint8')
    newim[0:4*npix:4] = matr
    newim[1:4*npix:4] = matg
    newim[2:4*npix:4] = matb
    newim[3:4*npix:4] = mata

    pilImage = Image.frombuffer('RGBA', (nx, ny), newim, 'raw', 'RGBA', 0, 1)
    pilImage.save(name)
    
# get the 1D projection of an image
def image_projection(im, axe = 'x'):
    if   axe == 'y': return im.sum(axis = 1)
    elif axe == 'x': return im.sum(axis = 0)
    
# get the 1D slice of an image
def image_slice(im, x1, y1, x2, y2):
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
        print 'Image must be square!'
        return -1
    imf = fft.fft2(im)
    imf = fft.fftshift(imf)

    return imf

# compute ifft of image
def image_ifft(imf):
    from numpy import fft
    l, w = imf.shape
    if l!= w:
        print 'Image must be square!'
        return -1
    im = fft.ifft2(imf)
    im = abs(im)
    im = im.astype('float32')

    return im

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

# create image noise (gauss or poisson model)
def image_noise(ny, nx, sigma, model='gauss'):
    from numpy import zeros
    from random import gauss

    if model=='gauss':
        im = zeros((ny, nx), 'float32')
        mu = 0.0
        for i in xrange(ny):
            for j in xrange(nx):
                im[i, j] = gauss(mu, sigma)

        return im
    
    elif model=='poisson':
        '''
        from numpy.random import poisson
        v = poisson(lam=1000, size=(nx*ny)) / 1000.0
        v = 10 * sigma * (v - 1)
        v = v.astype('float32')
        v = v.reshape((ny, nx))
        '''
        from random import random
        from math   import log

        im = zeros((ny, nx), 'float32')
        for i in xrange(ny):
            for j in xrange(nx):
                im[i, j] = sigma * -log(random())
        
        return im
            
# Compute RAPS, radial averaging power spectrum
def image_raps(im):
    from numpy import array

    lo, wo = im.shape
    im     = image_pows(im)
    l, w   = im.shape
    c      = (w - 1) // 2
    val    = image_ra(im)
    
    freq  = range(0, c + 1) # should be wo // 2 + 1 coefficient need to fix!!
    freq  = array(freq, 'float32')
    freq /= float(wo)
    
    return val, freq

# Compute RA, radial average of image
def image_ra(im):
    from numpy import zeros

    l, w   = im.shape
    c      = (w - 1) // 2
    rmax   = c
    val    = zeros((rmax + 1), 'float32')
    ct     = zeros((rmax + 1), 'float32')
    val[0] = im[c, c] # central value
    ct[0]  = 1.0
    for i in xrange(c - rmax, c + rmax + 1):
        di = i - c
        if abs(di) > rmax: continue
        for j in xrange(c - rmax, c + rmax + 1):
            dj = j - c
            r = (di*di + dj*dj)**(0.5)
            ir = int(r)
            if ir > rmax: continue
            cir   = ir + 1
            frac  = r  - ir
            cfrac = 1  - frac

            val[ir]  += (cfrac * im[j, i])
            ct[ir]   += cfrac
            if cir <= rmax:
                val[cir] += (frac  * im[j, i])
                ct[cir]  += frac
            '''
            val[ir] += (cfrac * im[j, i])
            if cfrac != 0.0: ct[ir] += cfrac
            if cir <= rmax:
                val[cir] += (frac * im[j, i])
                if frac != 0.0: ct[cir] += frac
            '''

    val /= ct
    
    return val

# Compute FRC curve (Fourier Ring Correlation)
def image_frc(im1, im2):
    from numpy import zeros, array
    
    if im1.shape != im2.shape:
        print 'Images must have the same size!'
        return -1

    wo, ho = im1.shape
    #im1    = image_normalize(im1)
    #im2    = image_normalize(im2)
    imf1   = image_fft(im1)
    imf2   = image_fft(im2)
    imf2c  = imf2.conj()
    w, h   = imf1.shape
    c      = (w - 1) // 2
    rmax   = c
    fsc    = zeros((rmax + 1), 'float32')
    nf1    = zeros((rmax + 1), 'float32')
    nf2    = zeros((rmax + 1), 'float32')
    # center
    fsc[0] = abs(imf1[c, c] * imf2c[c, c])
    nf1[0] = abs(imf1[c, c])**2
    nf2[0] = abs(imf2[c, c])**2
    # over all rings
    for i in xrange(c-rmax, c+rmax+1):
        for j in xrange(c-rmax, c+rmax+1):
            r = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5)
            ir    = int(r)
            if ir > rmax: continue
            cir   = ir + 1
            frac  = r - ir
            cfrac = 1 - frac
            ifsc  = imf1[i, j] * imf2c[i, j]
            inf1  = abs(imf1[i, j])**2
            inf2  = abs(imf2[i, j])**2
            fsc[ir] += (cfrac * ifsc)
            nf1[ir] += (cfrac * inf1)
            nf2[ir] += (cfrac * inf2)
            if cir <= rmax:
                fsc[cir] += (frac * ifsc)
                nf1[cir] += (frac * inf1)
                nf2[cir] += (frac * inf2)
    
    fsc = fsc / (nf1 * nf2)**0.5

    freq  = range(rmax + 1)  # should be 0, wo // 2 + 1) need to fix!!
    freq  = array(freq, 'float32')
    freq /= float(wo)

    return fsc, freq
            
# Low pass filter
def image_lp_filter(im, fc, order):
    from numpy import zeros, array
    order *= 2
    wo, ho = im.shape
    imf    = image_fft(im)
    w, h   = imf.shape
    c      = (w - 1) // 2
    H      = zeros((w, h), 'float32')
    for i in xrange(h):
        for j in xrange(w):
            r       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f       = r / (w-1)                          # fequency
            H[i, j] = 1 / (1 + (f / fc)**order)**0.5     # filter

    imf *= H
    im   = image_ifft(imf, wo)
            
    profil  = image_1D_slice(H, c, c, w, c)
    freq    = range(0, wo // 2 + 1)
    freq    = array(freq, 'float32')
    freq   /= float(wo)

    return im, profil, freq

# High pass filter
def image_hp_filter(im, fc, order):
    from numpy import zeros, array
    order *= 2
    wo, ho = im.shape
    imf    = image_fft(im)
    w, h   = imf.shape
    c      = (w - 1) // 2
    H      = zeros((w, h), 'float32')
    for i in xrange(h):
        for j in xrange(w):
            r       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f       = r / (w-1)                          # fequency
            # build like a low pass filter with fc = 0.5 - fc and mirror also f (0.5 - f)
            H[i, j] = 1 / (1 + ((0.5-f) / (0.5-fc))**order)**0.5 

    imf *= H
    im   = image_ifft(imf, wo)
            
    profil  = image_1D_slice(H, c, c, w, c)
    freq    = range(0, wo // 2 + 1)
    freq    = array(freq, 'float32')
    freq   /= float(wo)

    return im, profil, freq

# Band pass filter
def image_bp_filter(im, fl, fh, order):
    from numpy import zeros, array
    order *= 2
    wo, ho = im.shape
    imf    = image_fft(im)
    w, h   = imf.shape
    c      = (w - 1) // 2
    H      = zeros((w, h), 'float32')
    for i in xrange(h):
        for j in xrange(w):
            r       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f       = r / (w-1)                          # fequency
            # low pass filter
            a1      = 1 / (1 + (f / fh)**order)**0.5 
            # high pass filter
            a2      = 1 / (1 + ((0.5-f) / (0.5-fl))**order)**0.5
            # band pass filter
            H[i, j] = a1 * a2

    imf *= H
    im   = image_ifft(imf, wo)
            
    profil  = image_1D_slice(H, c, c, w, c)
    freq    = range(0, wo // 2 + 1)
    freq    = array(freq, 'float32')
    freq   /= float(wo)

    return im, profil, freq

# rotate image with 90 deg
def image_rot90(im):
    from numpy import rot90
    return rot90(im)

# flip left to rigth an image
def image_flip_lr(im):
    from numpy import fliplr
    return fliplr(im)

# flip up to down an image
def image_flip_ud(im):
    from numpy import flipud
    return flipud(im)

# Compute ZNCC between 2 images
def image_zncc(i1, i2):
    from numpy import sqrt, sum

    im1   = i1.copy()
    im2   = i2.copy()
    im1  -= im1.mean()
    im2  -= im2.mean()
    s1    = sqrt(sum(im1*im1))
    s2    = sqrt(sum(im2*im2))
    
    return sum(im1 * im2) / (s1 * s2)

# Compute ZNCC between 2 images under a mask
def image_zncc_mask(i1, i2, mask):
    from numpy import sqrt, sum

    im1  = i1.copy()
    im2  = i2.copy()
    v1   = image_pick_undermask(im1, mask)
    v2   = image_pick_undermask(im2, mask)
    b1   = v1 - v1.mean()
    b2   = v2 - v2.mean()
    s1   = sqrt(sum(b1*b1))
    s2   = sqrt(sum(b2*b2))

    return sum(b1 * b2) / (s1 * s2)

# Compute SNR based on ZNCC
def image_snr_from_zncc(signal, noise):
    from math import sqrt
    
    ccc = image_zncc(signal, noise)
    snr = sqrt(ccc / (1 - ccc))

    return snr

# Compute SNR based on ZNCC under a mask
def image_snr_from_zncc_mask(signal, noise, mask):
    from math import sqrt

    ccc = image_zncc_mask(signal, noise, mask)
    snr = sqrt(ccc / (1 - ccc))

    return snr

# Create a 2D mask circle
def image_mask_circle(ny, nx, rad):
    from numpy import zeros, sqrt
    
    cy = ny // 2
    cx = nx // 2
    m  = zeros((ny, nx), 'float32')
    for y in xrange(ny):
        for x in xrange(nx):
            r = ((y-cy)*(y-cy) + (x-cx)*(x-cx))**(0.5)
            if r > rad: continue

            m[y, x] = 1.0

    return m

# Create a 2D mask square
def image_mask_square(ny, nx, c):
    from numpy import zeros
    
    cy = ny // 2
    cx = nx // 2
    m  = zeros((ny, nx), 'float32')
    for y in xrange(ny):
        for x in xrange(nx):
            if abs(y-cy) > c or abs(x-cx) > c: continue
            m[y, x] = 1.0

    return m

# Create a 2D mask with edge of a square
def image_mask_edge_square(ny, nx, c):
    m1 = image_mask_square(ny, nx, c)
    m2 = image_mask_square(ny, nx, max((c-1), 0))

    return m1 - m2

# Create a 2D mire based on edge square
def image_mire_edge_square(ny, nx, step):
    from numpy import zeros

    im = zeros((ny, nx), 'float32')
    n  = min(ny, nx)
    hn = n // 2
    im[ny//2, nx//2] = 1.0
    for i in xrange(step, hn, step):
        im += image_mask_edge_square(ny, nx, i)

    return im

# Get statistics values from a circle ROI on an image
def image_stats_ROI_circle(im, cx, cy, rad):
    from numpy import array, zeros

    val    = []
    ny, nx = im.shape
    ROI    = zeros((ny, nx), 'float32')
    for y in xrange(ny):
        for x in xrange(nx):
            r = ((y-cy)*(y-cy) + (x-cx)*(x-cx))**(0.5)
            if r > rad: continue
            val.append(im[y, x])
            ROI[y, x] = 1.0

    val = array(val, 'float32')

    return ROI, val.min(), val.max(), val.mean(), val.std()

# Get statiscitcs values under a specified mask
def image_stats_mask(im, mask):
    from numpy import zeros

    npix   = mask.sum()
    val    = zeros((npix), 'float32')
    ny, nx = mask.shape
    ct     = 0
    for y in xrange(ny):
        for x in xrange(nx):
            if mask[y, x] == 1.0:
                val[ct] = im[y, x]
                ct     += 1

    return val.min(), val.max(), val.mean(), val.std(), val.sum(), npix

# Get values under a mask
def image_pick_undermask(im, mask):
    from numpy import zeros

    npix   = mask.sum()
    val    = zeros((npix), 'float32')
    ny, nx = mask.shape
    ct     = 0
    for y in xrange(ny):
        for x in xrange(nx):
            if mask[y, x] == 1.0:
                val[ct] = im[y, x]
                ct     += 1

    return val

# Compute image centroid
def image_centroid(im):
    ny, nx = im.shape

    M00, M01, M10 = 0, 0, 0
    for y in xrange(ny):
        for x in xrange(nx):
            i    = im[y, x]
            M00 += i
            M01 += (y * i)
            M10 += (x * i)

    return M10 / float(M00), M01 / float(M00)

# Stitch two images in one
def image_stitch(im1, im2):
    from numpy import zeros

    ny1, nx1 = im1.shape
    ny2, nx2 = im2.shape
    res = zeros((max(ny1, ny2), nx1+nx2), 'float32')
    res[0:ny1, 0:nx1] = im1
    res[0:ny2, nx1:nx1+nx2] = im2

    return res

# Threshold an image (up version)
def image_threshold_up(im, th, val):
    from numpy import where

    ind = where(im >= th)
    im[ind] = val

    return im

# Threshold an image (down version)
def image_threshold_down(im, th, val):
    from numpy import where

    ind = where(im <= th)
    im[ind] = val

    return im

# Lanczos 2D interpolation
def image_interpolation_Lanczos(im, n):
    from numpy import zeros
    
    n      = int(n)
    ny, nx = im.shape
    res    = zeros((ny*n, nx*n), 'float32')
    H      = filter_build_2d_Lanczos(nx*n, a=2)
    # resample the image
    for i in xrange(ny):
        for j in xrange(nx):
            res[n*i:n*(i+1), n*j:n*(j+1)] = im[i, j]
    # interpolation
    return image_ifft(image_fft(res) * H)

