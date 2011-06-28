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

# ==== Filtering ============================
# ===========================================

'''
TODO
# Low pass filter (Butterworth)
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
'''


'''
# 3D Metz filter
def filter_3d_Metz(vol, N, sig):
    from firekernel import kernel_3D_conv_wrap_cuda
    from numpy      import where

    smax    = max(vol.shape)
    H       = filter_build_3d_Metz(smax, N, sig)
    Hpad    = filter_pad_3d_cuda(H)
    z, y, x = vol.shape
    vol     = volume_pack_cube(vol)
    kernel_3D_conv_wrap_cuda(vol, Hpad)
    id      = where(vol < 0)
    vol[id] = 0
    vol     = volume_unpack_cube(vol, z, y, x)

    return vol

# 3D Gaussian filter
def filter_3d_Gaussian(vol, sig):
    from kernel import kernel_3D_conv_wrap_cuda

    smax    = max(vol.shape)
    H       = filter_build_3d_Gaussian(smax, sig)
    Hpad    = filter_pad_3d_cuda(H)
    z, y, x = vol.shape
    vol     = volume_pack_cube(vol)
    kernel_3D_conv_wrap_cuda(vol, Hpad)
    vol     = volume_unpack_cube(vol, z, y, x)

    return vol

# 3D Butterworth low pass filter
def filter_3d_Butterworth_lp(vol, order, fc):
    from kernel import kernel_3D_conv_wrap_cuda

    smax    = max(vol.shape)
    H       = filter_build_3d_Butterworth_lp(smax, order, fc)
    Hpad    = filter_pad_3d_cuda(H)
    z, y, x = vol.shape
    vol     = volume_pack_cube(vol)
    kernel_3D_conv_wrap_cuda(vol, Hpad)
    vol     = volume_unpack_cube(vol, z, y, x)

    return vol

# 3D tangent hyperbolic low pass filter
def filter_3d_tanh_lp(vol, a, fc):
    from kernel import kernel_3D_conv_wrap_cuda

    smax    = max(vol.shape)
    H       = filter_build_3d_tanh_lp(smax, a, fc)
    Hpad    = filter_pad_3d_cuda(H)
    z, y, x = vol.shape
    vol     = volume_pack_cube(vol)
    kernel_3D_conv_wrap_cuda(vol, Hpad)
    vol     = volume_unpack_cube(vol, z, y, x)

    return vol
'''

# 2d low pass filter (Butterworth)
def filter_2d_butterworth_lp(im, fc, order):
    smax = max(im.shape)
    H    = filter_build_2d_butterworth_lp(smax, order, fc)

    return image_ifft(image_fft(im) * H)

# 2d high pass filter (Butterworth)
def filter_2d_butterworth_hp(im, fc, order):
    smax = max(im.shape)
    H    = filter_build_2d_butterworth_hp(smax, order, fc)

    return image_ifft(image_fft(im) * H)

# 2d band pass filter (Butterworth)
def filter_2d_butterworth_bp(im, fl, fh, order):
    smax = max(im.shape)
    H    = filter_build_2d_butterworth_bp(size, order, fl, fh)

    return image_ifft(image_fft(im) * H)

# 2d Metz filter
def filter_2d_metz(im, N, sig):
    smax = max(im.shape)
    H    = filter_build_2d_Metz(smax, N, sig)

    return image_ifft(image_fft(im) * H)

# 2d hyperbolic tangent low pass filter
def filter_2d_tanh_lp(im, a, fc):
    smax = max(im.shape)
    H    = filter_build_2d_tanh_lp(smax, a, fc)

    return image_ifft(image_fft(im) * H)

# 2d Gaussian filter
def filter_2d_gaussian(im, sig):
    smax = max(im.shape)
    H    = filter_build_2d_Gaussian(smax, sig)

    return image_ifft(image_fft(im) * H)

# 2d Lanczos2 filter 
def filter_2d_lanczos2(im):
    smax = max(im.shape)
    H    = filter_build_2d_lanczos(smax, a=2)

    return image_ifft(image_fft(im) * H)

# 2d Lanczos3 filter 
def filter_2d_lanczos3(im):
    smax = max(im.shape)
    H    = filter_build_2d_lanczos(smax, a=3)

    return image_ifft(image_fft(im) * H)

# ==== Build filter =========================
# ===========================================

def filter_build_3d_metz(size, N, sig):
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

def filter_build_2d_metz(size, N, sig):
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

def filter_build_1d_metz(size, N, sig):
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
    
def filter_build_3d_gaussian(size, sig):
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

def filter_build_2d_gaussian(size, sig):
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

def filter_build_1d_gaussian(size, sig):
    from numpy import zeros
    from math  import exp

    c = size // 2
    H = zeros((size), 'float32')

    for i in xrange(size):
        f    = abs((i - c) / float(size))
        H[i] = exp(-(f*f) / (2*sig*sig))

    return H

def filter_build_3d_butterworth_lp(size, order, fc):
    from numpy import zeros
    
    order *= 2
    c      = size // 2
    H      = zeros((size, size, size), 'float32')
    for k in xrange(size):
        for i in xrange(size):
            for j in xrange(size):
                f          = ((i-c)*(i-c) + (j-c)*(j-c) + (k-c)*(k-c))**(0.5) # radius
                f         /= size                                             # frequency
                H[k, i, j] = 1 / (1 + (f / fc)**order)**0.5                   # filter

    return H

def filter_build_2d_butterworth_lp(size, order, fc):
    from numpy import zeros
    
    order *= 2
    c      = size // 2
    H      = zeros((size, size), 'float32')
    for i in xrange(size):
        for j in xrange(size):
            f       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f      /= size                               # frequency
            H[i, j] = 1 / (1 + (f / fc)**order)**0.5     # filter

    return H

def filter_build_2d_butterworth_hp(size, order, fc):
    from numpy import zeros

    order *= 2
    c      = size // 2
    H      = zeros((size, size), 'float32')
    for i in xrange(size):
        for j in xrange(size):
            r       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f       = r / (w-1)                          # frequency
            # build like a low pass filter with fc = 0.5 - fc and mirror f (0.5 - f)
            H[i, j] = 1 / (1 + ((0.5-f) / (0.5-fc))**order)**0.5

    return H

def filter_build_2d_butterworth_bp(size, order, fl, fh):
    from numpy import zeros

    order *= 2
    c      = size // 2
    H      = zeros((size, size), 'float32')
    for i in xrange(size):
        for j in xrange(size):
            r       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f       = r / (w-1)                          # frequency
            # low pass filter
            a1      = 1 / (1 + (f / fh)**order)**0.5 
            # high pass filter
            a2      = 1 / (1 + ((0.5-f) / (0.5-fl))**order)**0.5
            # band pass filter
            H[i, j] = a1 * a2

    return H

def filter_build_1d_butterworth_lp(size, order, fc):
    from numpy import zeros
    
    order *= 2
    c      = size // 2
    H      = zeros((size), 'float32')
    for i in xrange(size):
        f    = abs((i-c) / float(size))
        H[i] = 1 / (1 + (f / fc)**order)**0.5

    return H

def filter_build_3d_tanh_lp(size, a, fc):
    from numpy import zeros
    from math  import tanh, pi

    c = size // 2
    H = zeros((size, size, size), 'float32')
    for k in xrange(size):
        for i in xrange(size):
            for j in xrange(size):
                f          = ((i-c)*(i-c) + (j-c)*(j-c) + (k-c)*(k-c))**(0.5) # radius
                f         /= size                                             # frequency
                v          = (pi * (f - fc)) / (2 * a * fc)
                H[k, i, j] = 0.5 - (0.5 * tanh(v))                            # filter

    return H

def filter_build_2d_tanh_lp(size, a, fc):
    from numpy import zeros, array
    from math  import tanh, pi

    c = size // 2
    H = zeros((size, size), 'float32')
    for i in xrange(size):
        for j in xrange(size):
            f       = ((i-c)*(i-c) + (j-c)*(j-c))**(0.5) # radius
            f      /= size                               # frequency
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

def filter_build_2d_lanczos(size, a=2):
    from numpy import zeros, sinc

    a  = float(a)
    c  = size // 2
    H  = zeros((size, size), 'float32')
    p  = a / 0.5
    for i in xrange(size):
        for j in xrange(size):
            fi   = i - c
            fj   = j - c
            f    = (fi*fi + fj*fj)**(0.5)
            f   /= size
            f   *= p
            H[i, j] = sinc(f)*sinc(f / a)
                
    return H

def filter_build_1d_lanczos(size, a=2):
    from numpy import zeros, sinc

    a = float(a)
    c = size // 2
    H = zeros((size), 'float32')
    p = a / 0.5
    for i in xrange(size):
        f = p * abs((i-c) / float(size))
        H[i] = sinc(f)*sinc(f / a)

    return H

'''
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
'''

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
