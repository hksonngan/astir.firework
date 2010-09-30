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
    
    freq  = range(0, rmax + 1) # should be wo // 2 + 1 coefficient need to fix!!
    freq  = array(freq, 'float32')
    freq /= float(wo)
    
    return val, freq

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
            if r >= rmax: continue
            ir    = int(r)
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

    freq  = range(len(fsc))  # should be 0, wo // 2 + 1) need to fix!!
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
def image_zncc(im1, im2):
    from numpy import sqrt, sum
    
    im1  -= im1.mean()
    im2  -= im2.mean()
    s1    = sqrt(sum(im1*im1))
    s2    = sqrt(sum(im2*im2))
    
    return sum(im1 * im2) / (s1 * s2)

# Compute SNR based on ZNCC
def image_snr_from_zncc(signal, noise):
    from math import sqrt
    
    ccc = image_zncc(signal, noise)
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

    return val.min(), val.max(), val.mean(), val.std()

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

# ==== Volume ===============================
# ===========================================

# write a raw volume (bin)
def volume_raw_write(vol, name):
    nz, ny, nx = vol.shape
    vol        = vol.reshape((nz*ny*nx))
    vol.tofile(name)

# write a volume in firework format
def volume_write(vol, name):
    from numpy import array
    nz, ny, nx = vol.shape
    vol        = vol.reshape((nz*ny*nx))
    vol        = vol.tolist()
    vol.insert(0, nx)
    vol.insert(0, ny)
    vol.insert(0, nz)
    vol        = array(vol, 'float32')
    vol.tofile(name)
    
# open a raw volume (datatype = 'uint8', 'uint16', etc.)
def volume_raw_open(name, nz, ny, nx, datatype):
    import numpy
    
    data = open(name, 'rb').read()
    vol  = numpy.fromstring(data, datatype)
    vol  = vol.reshape(nz, ny, nx)

    return vol

# open a volume (firework format)
def volume_open(name):
    from numpy import fromfile
    f   = open(name, 'rb')
    vol = fromfile(f, 'float32')
    f.close()
    nz  = vol[0]
    ny  = vol[1]
    nx  = vol[2]
    vol = vol[3:]
    vol = vol.reshape(nz, ny, nx)

    return vol

# get a slice from a volume
def volume_slice(vol, pos=0, axe='z'):
    from numpy import matrix
    if   axe == 'z': return vol[pos]
    elif axe == 'x':
        # exception: the slice must be rotate (with transpose)
        return matrix(vol[:, :, pos]).T.A
    elif axe == 'y': return vol[:, pos, :]

# get the projection from a volume
def volume_projection(vol, axe='z'):
    if   axe == 'z': axis = 0
    elif axe == 'y': axis = 1
    elif axe == 'x': axis = 2
    
    return vol.sum(axis=axis)

# export volume as multipage tiff file (unfortunately in separate file)
def volume_export_tiff(vol, name):
    from PIL  import Image
    from time import sleep
    import os, sys
    vol  = vol.copy()
    vol /= vol.max()
    vol *= 255
    vol  = vol.astype('uint8')
    nz, ny, nx = vol.shape
    for z in xrange(nz):
        slice = volume_slice(vol, z, 'z')
        pilImage = Image.frombuffer('L', (nx, ny), slice, 'raw', 'L', 0, 1)
        pilImage.save('%03i_tmp.tiff' % z)
    try:
        os.system('convert *_tmp.tiff -adjoin %s' % name)
        os.system('rm -f *_tmp.tiff')
    except:
        print 'Imagemagick must install !!'
        sys.exit()

# Compute Maximum Intensity Projection (MIP)
def volume_mip(vol, axe='z'):
    if axe == 'z':   axis = 0
    elif axe == 'y': axis = 1
    elif axe == 'x': axis = 2

    return vol.max(axis=axis)

# Compute Minimum Intensity Projection (MiIP)
def volume_miip(vol, axe='z'):
    if axe == 'z':   axis = 0
    elif axe == 'y': axis = 1
    elif axe == 'x': axis = 2

    return vol.min(axis=axis)

# Display volume as a mosaic
def volume_mosaic(vol, axe='z', norm=False):
    from numpy import zeros
    
    z, y, x = vol.shape
    if axe == 'z':
        wi  = int(z**0.5 + 0.5)
        if z%wi == 0: hi = z // wi
        else:         hi = (z // wi) + 1
        print wi, hi
        mos = zeros((hi * y, wi * x), 'float32')
        zi  = 0
        for i in xrange(hi):
            for j in xrange(wi):
                im  = volume_slice(vol, zi, 'z')
                if norm: im = image_normalize(im)
                mos[i*y:(i+1)*y, j*x:(j+1)*x] = im
                zi += 1
                if zi >= z: break
            if zi >= z: break
    elif axe == 'x':
        wi  = int(x**0.5 + 0.5)
        hi  = int(x / wi + 0.5)
        mos = zeros((hi * z, wi * y), 'float32')
        xi  = 0
        for i in xrange(hi):
            for j in xrange(wi):
                im = volume_slice(vol, xi, 'x')
                mos[i*z:(i+1)*z, j*y:(j+1)*y] = im
                xi += 1
                if xi >= x: break
            if xi >= x: break
    else:
        wi  = int(y**0.5 + 0.5)
        hi  = int(y / wi + 0.5)
        mos = zeros((hi * z, wi * x), 'float32')
        yi  = 0
        for i in xrange(hi):
            for j in xrange(wi):
                im = volume_slice(vol, yi, 'y')
                mos[i*z:(i+1)*z, j*x:(j+1)*x] = im
                yi += 1
                if yi >= y: break
            if yi >= y: break

    return mos

# compute volume fft
def volume_fft(vol):
    from numpy import fft
    z, y, x = vol.shape
    if z != y or z != x or x != y:
        print 'Volume must be square!'
        return -1
    volf = fft.fftn(vol)
    volf = fft.fftshift(volf)

    return volf

# compute volume ifft
def volume_ifft(volf):
    from numpy import fft
    z, y, x = volf.shape
    if z != y or z != x or x != y:    
        print 'Volume must be square!'
        return -1
    vol = fft.ifftn(volf)

    return abs(vol)

# Low pass filter
def volume_lp_filter(vol, fc, order):
    from kernel import kernel_matrix_lp_H
    from numpy import zeros

    zo, yo, xo  = vol.shape
    volf        = volume_fft(vol)
    z, y, x     = volf.shape
    H           = zeros((z, y, x), 'float32')
    kernel_matrix_lp_H(H, fc, order)
    volf       *= H
    vol         = volume_ifft(volf, xo)
    vol         = vol.astype('float32')
            
    #profil  = image_1D_slice(H, c, c, w, c)
    #freq    = range(0, wo // 2 + 1)
    #freq    = array(freq, 'float32')
    #freq   /= float(wo)

    return vol    #, profil, freq

# create box mask
def volume_mask_box(nz, ny, nx, w, h, d):
    from numpy import zeros
    vol  = zeros((nz, ny, nx), 'float32')
    cx   = nx // 2
    cy   = ny // 2
    cz   = nz // 2
    hw   = w // 2
    hh   = h // 2
    hd   = d // 2
    for z in xrange(nz):
        for y in xrange(ny):
            for x in xrange(nx):
                iz = abs(z-cz)
                iy = abs(y-cy)
                ix = abs(x-cx)
                if iz > hd: continue
                if iy > hh: continue
                if ix > hw: continue
                vol[z, y, x] = 1.0

    return vol

# create cylinder mask
def volume_mask_cylinder(nz, ny, nx, dc, rad):
    from numpy import zeros
    vol = zeros((nz, ny, nx), 'float32')
    cx  = nx // 2
    cy  = ny // 2
    cz  = nz // 2
    h   = dc // 2
    for z in xrange(nz):
        for y in xrange(ny):
            for x in xrange(nx):
                rxy = ((x-cx)*(x-cx) + (y-cy)*(y-cy))**(0.5)
                if rxy > rad: continue
                if abs(z-cz) > h: continue

                vol[z, y, x] = 1.0

    return vol

# pack a non-isovolume to a cube
def volume_pack_cube(vol):
    from numpy import zeros
    oz, oy, ox = vol.shape
    type       = vol.dtype
    c = max(oz, oy, ox)
    padx = (c-ox) // 2
    pady = (c-oy) // 2
    padz = (c-oz) // 2
    newvol = zeros((c, c, c), type)
    for z in xrange(oz):
        for y in xrange(oy):
            for x in xrange(ox):
                newvol[z+padz, y+pady, x+padx] = vol[z, y, x]

    return newvol

# unpack volume from a cube according its sizes
def volume_unpack_cube(vol, oz, oy, ox):
    from numpy import zeros
    nz, ny, nx = vol.shape
    center = nz // 2
    nzh    = oz // 2
    nyh    = oy // 2
    nxh    = ox // 2
    x1, x2 = center - nxh, center + nxh + 1
    y1, y2 = center - nyh, center + nyh + 1
    z1, z2 = center - nzh, center + nzh + 1
    newvol = vol[z1:z2, y1:y2, x1:x2]

    return newvol

# pack a volume inside a new one at the center position
def volume_pack_center(vol, newz, newy, newx):
    from numpy import zeros
    oz, oy, ox = vol.shape
    type       = vol.dtype
    padx = (newx-ox) // 2
    pady = (newy-oy) // 2
    padz = (newz-oz) // 2
    newvol = zeros((newz, newy, newx), type)
    for z in xrange(oz):
        for y in xrange(oy):
            for x in xrange(ox):
                newvol[z+padz, y+pady, x+padx] = vol[z, y, x]

    return newvol

# rotate the volume by a quarter turn
def volume_rotate(vol, axis='x'):
    from numpy import zeros
    nz, ny, nx = vol.shape
    if   axis == 'x':
        newvol = zeros((ny, nz, nx), vol.dtype)
        for z in xrange(nz): newvol[:, z, :] = volume_slice(vol, z, 'z')
    elif axis == 'y':
        newvol = zeros((nx, ny, nz), vol.dtype)
        for x in xrange(nx): newvol[x, :, :] = volume_slice(vol, x, 'x')
    elif axis == 'z':
        newvol = zeros((nz, nx, ny), vol.dtype)
        for y in xrange(ny): newvol[:, :, y] = volume_slice(vol, y, 'y')

    return newvol

# some info on volume
def volume_infos(vol):
    sh = vol.shape
    print 'size: %ix%ix%i min %f max %f mean %f std %f' % (sh[0], sh[1], sh[2], vol.min(), vol.max(), vol.mean(), vol.std())


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

    return '%5.2f %s' % (mem, pref[iemem])

# convert time format to nice format
def time_format(t):
    time_r  = int(t)
    time_h  = time_r // 3600
    time_m  = (time_r % 3600) // 60
    time_s  = (time_r % 3600)  % 60
    time_ms = int((t - time_r) * 1000.0)
    txt     = ' %03i ms' % time_ms
    if time_s != 0: txt = ' %02i s' % time_s + txt
    if time_m != 0: txt = ' %02i m' % time_m + txt
    if time_h != 0: txt = ' %02i h' % time_h + txt
    return txt

# plot
def plot(x, y):
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()

# plot points distribution
def plot_dist(x, y):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8,8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)
    
    # now determine nice limits by hand:
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    
    binwidth = (xmax - xmin) / 100.0
    binheight = (ymax - ymin) / 100.0
    
    #xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    #lim = ( int(xymax/binwidth) + 1) * binwidth

    #axScatter.set_xlim( (-lim, lim) )
    axScatter.set_xlim((xmin, xmax))
    #axScatter.set_ylim( (-lim, lim) )
    axScatter.set_ylim(ymin, ymax)
    
    binsw = np.arange(xmin, xmax + binwidth, binwidth)
    binsh  = np.arange(ymin, ymax + binheight, binheight)
    axHistx.hist(x, bins=binsw)
    axHisty.hist(y, bins=binsh, orientation='horizontal')

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )

    plt.show()
    
# plot RAPS curve
def plot_raps(im):
    import matplotlib.pyplot as plt
    from numpy import log
    #im = image_normalize(im)
    val, freq = image_raps(im)
    #val = image_atodB(val)
    #val = log(val)
    plt.semilogy(freq, val)
    plt.xlabel('Nyquist frequency')
    plt.ylabel('Power spectrum')
    plt.grid(True)
    plt.show()

# plot FRC curve
def plot_frc(im1, im2):
    import matplotlib.pyplot as plt
    frc, freq = image_frc(im1, im2)
    plt.plot(freq, frc)
    plt.xlabel('Nyquist frequency')
    plt.ylabel('FRC')
    plt.axis([0, 0.5, 0, 1.0])
    plt.grid(True)
    plt.show()

# plot profil of any filter
def plot_filter_profil(H):
    import matplotlib.pyplot as plt
    p, f = filter_profil(H)
    plt.plot(f, p)
    plt.xlabel('Nyquist frequency')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.show()
    
# smooth curve
def curve_smooth(a, order):
    from numpy import zeros

    for o in xrange(order):
        b     = zeros(a.shape, a.dtype)
        n     = a.size - 1
        b[-1] = a[-1]
        for i in xrange(n):
            b[i] = (a[i] + a[i+1]) / 2.0

        a = b.copy()

    return a
    
    

# ==== Filtering ============================
# ===========================================

def filter_3d_Metz(vol, N, sig):
    from kernel import kernel_3Dconv_cuda

    smax    = max(vol.shape)
    H       = filter_build_3d_Metz(smax, N, sig)
    Hpad    = filter_pad_3d_cuda(H)
    z, y, x = vol.shape
    vol     = volume_pack_cube(vol)
    kernel_3Dconv_cuda(vol, Hpad)
    vol     = volume_unpack_cube(vol, z, y, x)

    return vol

def filter_3d_Gaussian(vol, sig):
    from kernel import kernel_3Dconv_cuda

    smax    = max(vol.shape)
    H       = filter_build_3d_Gaussian(smax, sig)
    Hpad    = filter_pad_3d_cuda(H)
    z, y, x = vol.shape
    vol     = volume_pack_cube(vol)
    kernel_3Dconv_cuda(vol, Hpad)
    vol     = volume_unpack_cube(vol, z, y, x)

    return vol

def filter_3d_Butterworth_lp(vol, order, fc):
    from kernel import kernel_3Dconv_cuda

    smax    = max(vol.shape)
    H       = filter_build_3d_Butterworth_lp(smax, order, fc)
    Hpad    = filter_pad_3d_cuda(H)
    z, y, x = vol.shape
    vol     = volume_pack_cube(vol)
    kernel_3Dconv_cuda(vol, Hpad)
    vol     = volume_unpack_cube(vol, z, y, x)

    return vol

def filter_3d_tanh_lp(vol, a, fc):
    from kernel import kernel_3Dconv_cuda

    smax    = max(vol.shape)
    H       = filter_build_3d_tanh_lp(smax, a, fc)
    Hpad    = filter_pad_3d_cuda(H)
    z, y, x = vol.shape
    vol     = volume_pack_cube(vol)
    kernel_3Dconv_cuda(vol, Hpad)
    vol     = volume_unpack_cube(vol, z, y, x)

    return vol

def filter_2d_Metz(im, N, sig):
    smax = max(im.shape)
    H    = filter_build_2d_Metz(smax, N, sig)

    return image_ifft(image_fft(im) * H)

def filter_2d_tanh_lp(im, a, fc):
    smax = max(im.shape)
    H    = filter_build_2d_tanh_lp(smax, a, fc)

    return image_ifft(image_fft(im) * H)

def filter_build_3d_Metz(size, N, sig):
    from numpy import zeros
    from math  import exp
    
    c  = size // 2
    H  = zeros((size, size, size), 'float32')
    N += 1
    for k in xrange(size):
        for i in xrange(size):
            for j in xrange(size):
                fi   = i - c
                fj   = j - c
                fk   = k - c
                f    = (fi*fi + fj*fj + fk*fk)**(0.5)
                f   /= size
                gval = exp(-(f*f) / (2*sig*sig))
                H[k, i, j] = (1 - (1 - gval*gval)**N) / gval

    return H

def filter_build_2d_Metz(size, N, sig):
    from numpy import zeros
    from math  import exp
    
    c  = size // 2
    H  = zeros((size, size), 'float32')
    N += 1
    for i in xrange(size):
        for j in xrange(size):
            fi   = i - c
            fj   = j - c
            f    = (fi*fi + fj*fj)**(0.5)
            f   /= size
            gval = exp(-(f*f) / (2*sig*sig))
            H[i, j] = (1 - (1 - gval*gval)**N) / gval
                
    return H

def filter_build_1d_Metz(size, N, sig):
    from numpy import zeros
    from math  import exp
    
    c  = size // 2
    H  = zeros((size), 'float32')
    N += 1
    for i in xrange(size):
        f    = abs((i - c) / float(size))
        gval = exp(-(f*f) / (2*sig*sig))
        H[i] = (1 - (1 - gval*gval)**N) / gval
                
    return H
    
def filter_build_3d_Gaussian(size, sig):
    from numpy import zeros
    from math  import exp

    c = size // 2
    H = zeros((size, size, size), 'float32')

    for k in xrange(size):
        for i in xrange(size):
            for j in xrange(size):
                fi   = i - c
                fj   = j - c
                fk   = k - c
                f    = (fi*fi + fj*fj + fk*fk)**(0.5)
                f   /= size
                H[k, i, j] = exp(-(f*f) / (2*sig*sig))

    return H

def filter_build_2d_Gaussian(size, sig):
    from numpy import zeros
    from math  import exp

    c = size // 2
    H = zeros((size, size), 'float32')

    for i in xrange(size):
        for j in xrange(size):
            fi   = i - c
            fj   = j - c
            f    = (fi*fi + fj*fj)**(0.5)
            f   /= size
            H[i, j] = exp(-(f*f) / (2*sig*sig))

    return H

def filter_build_1d_Gaussian(size, sig):
    from numpy import zeros
    from math  import exp

    c = size // 2
    H = zeros((size), 'float32')

    for i in xrange(size):
        f    = abs((i - c) / float(size))
        H[i] = exp(-(f*f) / (2*sig*sig))

    return H

def filter_build_3d_Butterworth_lp(size, order, fc):
    from numpy import zeros, array
    
    order *= 2
    c      = size // 2
    H      = zeros((size, size, size), 'float32')
    for k in xrange(size):
        for i in xrange(size):
            for j in xrange(size):
                f          = ((i-c)*(i-c) + (j-c)*(j-c) + (k-c)*(k-c))**(0.5) # radius
                f         /= size                                            # fequency
                H[k, i, j] = 1 / (1 + (f / fc)**order)**0.5                   # filter

    return H

def filter_build_2d_Butterworth_lp(size, order, fc):
    from numpy import zeros, array
    
    order *= 2
    c      = size // 2
    H      = zeros((size, size), 'float32')
    for i in xrange(size):
        for j in xrange(size):
            f       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f      /= size                                            # fequency
            H[i, j] = 1 / (1 + (f / fc)**order)**0.5                   # filter

    return H

def filter_build_1d_Butterworth_lp(size, order, fc):
    from numpy import zeros, array
    
    order *= 2
    c      = size // 2
    H      = zeros((size), 'float32')
    for i in xrange(size):
        f    = abs((i-c) / float(size))
        H[i] = 1 / (1 + (f / fc)**order)**0.5

    return H

def filter_build_3d_tanh_lp(size, a, fc):
    from numpy import zeros, array
    from math  import tanh, pi

    c = size // 2
    H = zeros((size, size, size), 'float32')
    for k in xrange(size):
        for i in xrange(size):
            for j in xrange(size):
                f          = ((i-c)*(i-c) + (j-c)*(j-c) + (k-c)*(k-c))**(0.5) # radius
                f         /= size                                            # fequency
                v          = (pi * (f - fc)) / (2 * a * fc)
                H[k, i, j] = 0.5 - (0.5 * tanh(v))                           # filter

    return H

def filter_build_2d_tanh_lp(size, a, fc):
    from numpy import zeros, array
    from math  import tanh, pi

    c = size // 2
    H = zeros((size, size), 'float32')
    for i in xrange(size):
        for j in xrange(size):
            f       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f      /= size                               # fequency
            v       = (pi * (f - fc)) / (2 * a * fc)
            H[i, j] = 0.5 - (0.5 * tanh(v))              # filter

    return H

def filter_build_1d_tanh_lp(size, a, fc):
    from numpy import zeros, array
    from math  import tanh, pi

    c = size // 2
    H = zeros((size), 'float32')
    for i in xrange(size):
        f    = abs((i-c) / float(size))
        v    = (pi * (f - fc)) / (2 * a * fc)
        H[i] = 0.5 - (0.5 * tanh(v))              # filter

    return H

def filter_build_3d_tanh_hp(size, a, fc):
    H = filter_build_3d_tanh_lp(size, a, fc)
    H = 1 - H
    
    return H

def filter_build_2d_tanh_hp(size, a, fc):
    H = filter_build_2d_tanh_lp(size, a, fc)
    H = 1 - H
    
    return H

def filter_build_1d_tanh_hp(size, a, fc):
    filter_build_1d_tanh_lp(size, a, fc)
    H = 1 - H
    
    return H

def filter_pad_3d_cuda(H):
    from numpy import zeros
    size, size, size = H.shape
    c                = size // 2
    nc               = (size // 2) + 1
    Hpad             = zeros((size, size, size), 'float32')

    for k in xrange(size):
        for i in xrange(size):
            for j in xrange(size):
                padi = i - c
                padj = j - c
                padk = k - c
                if padi < 0: padi = size + padi
                if padj < 0: padj = size + padj
                if padk < 0: padk = size + padk

                Hpad[padk, padi, padj] = H[k, i, j]

    return Hpad[:, :, :nc]

# Return the profil of any filter
def filter_profil(H):
    from numpy import arange
    dim = len(H.shape)
    if dim == 3:
        nz, ny, nx = H.shape
        cz = nz // 2
        im = H[cz, :, :]
        cx = nx // 2
        cy = ny // 2
        p  = im[cy, cx:]
    elif dim == 2:
        ny, nx = H.shape
        cx = nx // 2
        cy = ny // 2
        p  = H[cy, cx:]

    f = arange(len(p), dtype='float32') / float(nx)

    return p, f
