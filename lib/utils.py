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

# export image
def image_write(im, name):
    from PIL import Image
    slice = im.copy()
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
        print 'Image must be square!'
        return -1
    if w % 2 == 1: imf = fft.fft2(im)
    else:          imf = fft.fft2(im, s=(w+1, w+1))
    imf  = fft.fftshift(imf)

    return imf

# compute ifft of image
def image_ifft(imf, wo = -1):
    from numpy import fft
    l, w = imf.shape
    if wo == -1: wo = w
    if l!= w:
        print 'Image must be square!'
        return -1
    if wo % 2 == 1: im = fft.ifft2(imf)
    else:           im = fft.ifft2(imf, s=(wo, wo))

    return abs(im)

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
def image_noise(w, h, sigma):
    from numpy import zeros
    from random import gauss

    mu = 0.5
    im = zeros((w, h), 'float32')
    for i in xrange(h):
        for j in xrange(w):
            im[j, i] = gauss(mu, sigma)

    im -= im.min()
    im /= im.max()
    
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
    
    freq  = range(0, wo // 2 + 1)
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
    im1    = image_normalize(im1)
    im2    = image_normalize(im2)
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
    fsc[0] = imf1[c, c] * imf2c[c, c]
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

    freq  = range(0, wo // 2 + 1)
    freq  = array(freq, 'float32')
    freq /= float(wo)

    return fsc, freq

# Compute SNR
def image_snr(im):
    wo, ho = im.shape
    noise  = image_noise(wo, ho, 1.0)
    noise  = image_normalize(noise)
    im     = image_normalize(im)
    Pim    = image_pows(im)
    Pnoise = image_pows(noise)
    print Pnoise.mean()
    w, h   = Pim.shape
    c      = (w - 1) // 2
    rmax   = c
    snr    = 0.0
    ct     = 0
    # calculate inside a mask circle
    for i in xrange(c - rmax, c + rmax + 1):
        for j in xrange(c - rmax, c + rmax + 1):
            r = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5)
            if r >= rmax: continue
            snr += (Pim[i, j] / Pnoise[i, j])
            ct  += 1

    return snr/ct
            

            
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


# ==== Volume ===============================
# ===========================================

# open a raw volume (datatype = 'uint8', 'uint16', etc.)
def volume_open(name, nz, ny, nx, datatype):
    import numpy
    
    data = open(name, 'rb').read()
    vol  = numpy.fromstring(data, datatype)
    vol  = vol.astype('float32')
    '''
    if   datatype == 'uint8':  val /= 255.0
    elif datatype == 'uint16': val /= 65535.0
    elif datatype == 'uint32'; val /= 4294967295.0
    '''
    #vol *= 1.0 / vol.max()
    vol  = vol.reshape(nz, ny, nx)

    return vol

# get a slice from a volume
def volume_slice(vol, pos=0, axe='z'):
    if   axe == 'z': return vol[pos]
    elif axe == 'x': return vol[:, :, pos]
    elif axe == 'y': return vol[:, pos, :]

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
def volume_mosaic(vol):
    from numpy import zeros
    
    z, y, x = vol.shape
    wi  = int(z**0.5 + 0.5)
    hi  = int(z / wi + 0.5)
    ct  = 0
    mos = zeros((hi * y, wi * x), 'float32')
    zi  = 0
    for i in xrange(hi):
        for j in xrange(wi):
            im  = volume_slice(vol, zi, 'z')
            mos[i*y:(i+1)*y, j*x:(j+1)*x] = im
            zi += 1
            if zi >= z: break
        if zi >= z: break

    return mos

# compute volume fft
def volume_fft(vol):
    from numpy import fft
    z, y, x = vol.shape
    if z != y or z != x or x != y:
        print 'Volume must be square!'
        return -1
    if x % 2 == 1: volf = fft.fftn(vol)
    else:          volf = fft.fftn(vol, s=(z+1, y+1, x+1))
    volf = fft.fftshift(volf)

    return volf

# compute volume ifft
def volume_ifft(volf, xo = -1):
    from numpy import fft
    z, y, x = volf.shape
    if xo == -1: xo = x
    if z != y or z != x or x != y:    
        print 'Volume must be square!'
        return -1
    if xo % 2 == 1: vol = fft.ifftn(volf)
    else:           vol = fft.ifftn(volf, s=(zo, yo, xo))

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
def volume_mask_box(wb, hb, db, w, h, d):
    from numpy import zeros
    vol  = zeros((d, h, w), 'float32')
    cv   = (w-1)  // 2
    cb   = (wb-1) // 2
    staw = cv - cb
    stow = staw + wb
    cv   = (h-1)  // 2
    cb   = (hb-1) // 2
    stah = cv - cb
    stoh = stah + hb
    cv   = (d-1)  // 2
    cb   = (db-1) // 2
    stad = cv - cb
    stod = stad + db
    # draw mask box
    for k in xrange(stad, stod):
        for j in xrange(staw, stow):
            for i in xrange(stah, stoh):
                vol[k, j, i] = 1.0

    return vol

# pack a non-isovolume to a cube
def volume_pack_cube(vol):
    from numpy import zeros
    oz, oy, ox = vol.shape
    c = max(oz, oy, ox)
    padx = (c-ox) // 2
    pady = (c-oy) // 2
    padz = (c-oz) // 2
    newvol = zeros((c, c, c), 'float32')
    for z in xrange(oz):
        for y in xrange(oy):
            for x in xrange(ox):
                newvol[z+padz, y+pady, x+padx] = vol[z, y, x]

    return newvol


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

def plot_raps(im):
    import matplotlib.pyplot as plt
    im = image_normalize(im)
    val, freq = image_raps(im)
    val = image_atodB(val)
    plt.plot(freq, val)
    plt.xlabel('Nyquist frequency')
    plt.ylabel('Power spectrum (dB)')
    plt.show()

# ==== List-Mode ============================
# ===========================================

# Open list-mode pre-compute data set (int format), values are entry-exit point of SRM matrix
def listmode_open_xyz_int(basename):
    from numpy import fromfile
    
    f  = open(basename + '.x1', 'rb')
    x1 = fromfile(file=f, dtype='uint16')
    x1 = x1.astype('uint16')
    f.close()

    f  = open(basename + '.y1', 'rb')
    y1 = fromfile(file=f, dtype='uint16')
    y1 = y1.astype('uint16')
    f.close()
    
    f  = open(basename + '.z1', 'rb')
    z1 = fromfile(file=f, dtype='uint16')
    z1 = z1.astype('uint16')
    f.close()
    
    f  = open(basename + '.x2', 'rb')
    x2 = fromfile(file=f, dtype='uint16')
    x2 = x2.astype('uint16')
    f.close()
    
    f  = open(basename + '.y2', 'rb')
    y2 = fromfile(file=f, dtype='uint16')
    y2 = y2.astype('uint16')
    f.close()
    
    f  = open(basename + '.z2', 'rb')
    z2 = fromfile(file=f, dtype='uint16')
    z2 = z2.astype('uint16')
    f.close()

    return x1, y1, z1, x2, y2, z2

# Open list-mode pre-compute data set (float format), values are entry-exit point of SRM matrix
def listmode_open_xyz_float(basename):
    from numpy import fromfile
    
    f  = open(basename + '.x1', 'rb')
    x1 = fromfile(file=f, dtype='float32')
    f.close()

    f  = open(basename + '.y1', 'rb')
    y1 = fromfile(file=f, dtype='float32')
    f.close()
    
    f  = open(basename + '.z1', 'rb')
    z1 = fromfile(file=f, dtype='float32')
    f.close()
    
    f  = open(basename + '.x2', 'rb')
    x2 = fromfile(file=f, dtype='float32')
    f.close()
    
    f  = open(basename + '.y2', 'rb')
    y2 = fromfile(file=f, dtype='float32')
    f.close()
    
    f  = open(basename + '.z2', 'rb')
    z2 = fromfile(file=f, dtype='float32')
    f.close()

    return x1, y1, z1, x2, y2, z2


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
    from numpy import fromfile

    '''
    data = open(filename, 'r').readlines()
    Ns   = len(data)
    SM   = zeros((Ns), 'float32')
    for n in xrange(Ns):
        SM[n] = float(data[n])
    del data
    '''
    f  = open(filename, 'rb')
    SM = fromfile(file=f, dtype='int32')
    SM = SM.astype('float32')

    return SM

# PET 2D ring scan create LOR event from a simple simulate phantom (three activities) ONLY FOR ALLEGRO SCANNER
def listmode_simu_circle_phantom(nbparticules, ROIsize = 141, radius_detector = 432, respix = 4.0, rnd = 10):
    from numpy        import zeros, array
    from numpy.random import poisson
    from numpy.random import seed as seed2
    from random       import seed, random, randrange
    from math         import pi, sqrt, cos, sin, asin, acos
    seed(rnd)
    seed2(rnd)

    # Cst Allegro
    Ldetec    = 97.0 / float(respix)  # detector width (mm)
    Lcryst    = 4.3  / float(respix) # crystal width (mm)
    radius    = radius_detector / float(respix)
    border    = int((2*radius - ROIsize) / 2.0)
    cxo = cyo = ROIsize // 2
    image     = zeros((ROIsize, ROIsize), 'float32')
    source    = []

    # Generate an activity map with three differents circles
    cx0, cy0, r0 = cxo,    cyo,    16
    cx1, cy1, r1 = cx0+4,  cy0+4,   7
    cx2, cy2, r2 = cx0-6,  cy0-6,   2
    r02          = r0*r0
    r12          = r1*r1
    r22          = r2*r2
    for y in xrange(ROIsize):
        for x in xrange(ROIsize):
            if ((cx0-x)*(cx0-x) + (cy0-y)*(cy0-y)) <= r02:
                # inside the first circle
                if ((cx1-x)*(cx1-x) + (cy1-y)*(cy1-y)) <= r12:
                    # inside the second circle (do nothing)
                    continue
                if ((cx2-x)*(cx2-x) + (cy2-y)*(cy2-y)) <= r22:
                    # inside the third circle
                    source.extend([x, y, 5])
                    #image[y, x] = 5
                else:
                    # inside the first circle
                    source.extend([x, y, 1])
                    #image[y, x] = 1
                    
    nbpix  = len(source) // 3
    pp1    = poisson(lam=1.0, size=(nbparticules)).astype('int32')
    ps1    = [randrange(-1, 2) for i in xrange(nbparticules)]
    pp2    = poisson(lam=1.0, size=(nbparticules)).astype('int32')
    ps2    = [randrange(-1, 2) for i in xrange(nbparticules)]
    alpha  = [random()*2*pi for i in xrange(nbparticules)]
    ind    = [randrange(nbpix) for i in xrange(nbparticules)]
    
    IDD1 = zeros((nbparticules), 'int32')
    IDC1 = zeros((nbparticules), 'int32')
    IDD2 = zeros((nbparticules), 'int32')
    IDC2 = zeros((nbparticules), 'int32')
    p    = 0
    while p < nbparticules:
        x     = int(source[3*ind[p]]   + (ps1[p] * pp1[p]))
        y     = int(source[3*ind[p]+1] + (ps2[p] * pp2[p]))
        val   = source[3*ind[p]+2]
        image[y, x] += (source[3*ind[p]+2] * 10.0)
        x    += border
        y    += border
        a     = alpha[p]
        # compute line intersection with a circle (detectors)
	dx    = cos(a)
	dy    = sin(a)
	b     = 2 * (dx*(x-radius) + dy*(y-radius))
	c     = 2*radius*radius + x*x + y*y - 2*(radius*x + radius*y) - radius*radius
	d     = b*b - 4*c
	k0    = (-b + sqrt(d)) / 2.0
	k1    = (-b - sqrt(d)) / 2.0
	x1    = x + k0*dx
	y1    = y + k0*dy
	x2    = x + k1*dx
	y2    = y + k1*dy
	# compute angle from center of each point
	dx = x1 - radius #- border
	dy = y1 - radius #- border
	if abs(dx) > abs(dy):
            phi1 = asin(dy / float(radius))
            if phi1 < 0: # asin return -pi/2 < . < pi/2
                if dx < 0: phi1 = pi - phi1
                else:      phi1 = 2*pi + phi1
            else:
                if dx < 0: phi1 = pi - phi1 # mirror according y axe
	else:
            phi1 = acos(dx / float(radius))
            if dy < 0:     phi1 = 2*pi - phi1 # mirror according x axe
 	dx = x2 - radius #- border
	dy = y2 - radius #- border
	if abs(dx) > abs(dy):
            phi2 = asin(dy / float(radius))
            if phi2 < 0: # asin return -pi/2 < . < pi/2
                if dx < 0: phi2 = pi - phi2
                else:      phi2 = 2*pi + phi2
            else:
                if dx < 0: phi2 = pi - phi2 # mirror according y axe
	else:
            phi2 = acos(dx / float(radius))
            if dy < 0:     phi2 = 2*pi - phi2 # mirror according x axe

        # convert arc distance to ID detector and crystal
        phi1  = (2 * pi - phi1 + pi / 2.0) % (2 * pi)
        phi2  = (2 * pi - phi2 + pi / 2.0) % (2 * pi)
        pos1  = phi1 * radius
        pos2  = phi2 * radius
        idd1  = int(pos1 / Ldetec)
        pos1 -= (idd1 * Ldetec)
        idc1  = int(pos1 / Lcryst)
        idd2  = int(pos2 / Ldetec)
        pos2 -= (idd2 * Ldetec)
        idc2  = int(pos2 / Lcryst)

        for n in xrange(val):
            if p >= nbparticules: break
            IDD1[p] = idd1
            IDC1[p] = idc1
            IDD2[p] = idd2
            IDC2[p] = idc2
            p += 1
        if p >= nbparticules: break
        
    return image, IDC1, IDD1, IDC2, IDD2
