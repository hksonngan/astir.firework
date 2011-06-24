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
    from numpy    import zeros
    from firework import image_normalize
    
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
        print 'Volume must be cube!'
        return -1
    volf = fft.fftn(vol)
    volf = fft.fftshift(volf)

    return volf

# compute volume ifft
def volume_ifft(volf):
    from numpy import fft
    
    z, y, x = volf.shape
    if z != y or z != x or x != y:    
        print 'Volume must be cube!'
        return -1
    vol = fft.ifftn(volf)
    vol = abs(vol)
    
    return vol.astype('float32')

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

# create ball mask
def volume_mask_ball(nz, ny, nx, rad):
    from numpy import zeros

    vol = zeros((nz, ny, nx), 'float32')
    cx  = nx // 2
    cy  = ny // 2
    cz  = nz // 2
    for z in xrange(nz):
        for y in xrange(ny):
            for x in xrange(nx):
                rxyz = ((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz))**(0.5)
                if rxyz > rad: continue

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

# Compute volume RA (radial average)
def volume_ra(vol):
    from numpy import zeros

    nz, ny, nx = vol.shape
    c          = (nx - 1) // 2
    rmax       = c
    val        = zeros((rmax + 1), 'float32')
    ct         = zeros((rmax + 1), 'float32')
    val[0]     = vol[c, c, c] # central value
    ct[0]      = 1.0
    for k in xrange(c - rmax, c + rmax + 1):
        dk = k - c
        if abs(dk) > rmax: continue
        for i in xrange(c - rmax, c + rmax + 1):
            di = i - c
            if abs(di) > rmax: continue
            for j in xrange(c - rmax, c + rmax + 1):
                dj = j - c
                r = (di*di + dj*dj + dk*dk)**(0.5)
                ir = int(r)
                if ir > rmax: continue
                cir   = ir + 1
                frac  = r  - ir
                cfrac = 1  - frac

                val[ir]  += (cfrac * vol[k, j, i])
                ct[ir]   += cfrac
                if cir <= rmax:
                    val[cir] += (frac  * vol[k, j, i])
                    ct[cir]  += frac

    val /= ct
    
    return val

# Compute volume RAPS (radial averaging power spectrum)
def volume_raps(vol):
    from numpy import array

    nzo, nyo, nxo = vol.shape
    vol           = volume_pows(vol)
    nz, ny, nx    = vol.shape
    c      = (nx - 1) // 2
    val    = volume_ra(vol)
    
    freq  = range(0, c + 1) # TODO should be wo // 2 + 1 coefficient need to fix!!
    freq  = array(freq, 'float32')
    freq /= float(nxo)
    
    return val, freq

# compute power spectrum of volume
def volume_pows(vol):
    volf   = volume_fft(vol)
    volf   = volf * volf.conj()
    volf   = volf.real
    nz, ny, nx = vol.shape
    volf  /= (float(nz * ny * nx))
    
    return volf

# Compute FSC curve (Fourier Shell Correlation)
def volume_fsc(vol1, vol2):
    from numpy import zeros, array

    nzo, nyo, nxo = vol1.shape
    volf1   = volume_fft(vol1)
    volf2   = volume_fft(vol2)
    volf2c  = volf2.conj()
    nz, ny, nx = volf1.shape
    c      = (nx - 1) // 2
    rmax   = c
    fsc    = zeros((rmax + 1), 'float32')
    nf1    = zeros((rmax + 1), 'float32')
    nf2    = zeros((rmax + 1), 'float32')
    
    # center
    fsc[0] = abs(volf1[c, c, c] * volf2c[c, c, c])
    nf1[0] = abs(volf1[c, c, c])**2
    nf2[0] = abs(volf2[c, c, c])**2
    
    # over all shells
    for k in xrange(c-rmax, c+rmax+1):
        dk = k-c
        if abs(dk) > rmax: continue
        for i in xrange(c-rmax, c+rmax+1):
            di = i-c
            if abs(di) > rmax: continue
            for j in xrange(c-rmax, c+rmax+1):
                dj = j-c
                r = (di*di + dj*dj + dk*dk)**(0.5)
                ir    = int(r)
                if ir > rmax: continue
                cir      = ir + 1
                frac     = r - ir
                cfrac    = 1 - frac
                ifsc     = volf1[k, j, i] * volf2c[k, j, i]
                inf1     = abs(volf1[k, j, i])**2
                inf2     = abs(volf2[k, j, i])**2
                fsc[ir] += (cfrac * ifsc)
                nf1[ir] += (cfrac * inf1)
                nf2[ir] += (cfrac * inf2)
                if cir <= rmax:
                    fsc[cir] += (frac * ifsc)
                    nf1[cir] += (frac * inf1)
                    nf2[cir] += (frac * inf2)
    
    fsc = fsc / (nf1 * nf2)**0.5

    freq  = range(rmax + 1)  # TODO should be 0, wo // 2 + 1) need to fix!!
    freq  = array(freq, 'float32')
    freq /= float(nxo)

    return fsc, freq

# Compute ZNCC between 2 volumes
def volume_zncc(vol1, vol2):
    from numpy import sqrt, sum
    
    vol1 -= vol1.mean()
    vol2 -= vol2.mean()
    s1    = sqrt(sum(vol1*vol1))
    s2    = sqrt(sum(vol2*vol2))
    
    return sum(vol1 * vol2) / (s1 * s2)

# Compute SNR based on ZNCC
def volume_snr_from_zncc(signal, noise):
    from math import sqrt
    
    ccc = volume_zncc(signal, noise)
    snr = sqrt(ccc / (1 - ccc))

    return snr

# Compute SNR based on Lodge's method
def volume_snr_from_Lodge(vol1, vol2, s1, s2, mask):
    from firework import image_pick_undermask
    
    S   = s2 - s1 + 1
    snr = 0
    for i in xrange(s1, s2):
        im1 = volume_slice(vol1, i)
        im2 = volume_slice(vol2, i)

        im1 = image_pick_undermask(im1, mask)
        im2 = image_pick_undermask(im2, mask)

        dj  = im1 - im2
        mj  = (im1 + im2) / 2.0
        ai  = mj.mean()
        ni  = len(dj)

        dsdi  = ni * (dj*dj).sum() - (dj.sum())**2
        dsdi /= float(ni*ni - ni)
        dsdi  = dsdi**(0.5)

        snr += (ai / float(dsdi))

    return snr * (2**(0.5) / float(S))

# flip up to down a volume
def volume_flip_ud(vol):
    from numpy import flipud, zeros
    
    nz, ny, nx = vol.shape
    newv       = zeros(vol.shape, vol.dtype)
    for n in xrange(nz):
        newv[n, :, :] = flipud(vol[n, :, :])

    return newv

