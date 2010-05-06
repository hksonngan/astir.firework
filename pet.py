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

# simulation of gamma photon to PET with four detectors
def pet_2D_square_test_sim_LOR(posx, posy, nx):
    from math         import pi, cos, sin
    from numpy.random import poisson
    from random       import random, expovariate

    # detector
      # # #
    # o o o #
    # o x o #
    # o o o #
      # # #
    
    # Poisson distribution
    v = poisson(lam=2.0, size=(1))[0]
    r1 = random()
    r2 = random()
    #v = expovariate(1.0)
    if   r1 >= 0.66: x0 = posx + v
    elif r1 >= 0.33: x0 = posx - v
    else:            x0 = posx
    if   r2 >= 0.66: y0 = posy + v
    elif r2 >= 0.33: y0 = posy - v
    else:            y0 = posy
    
    # uniform angle
    alpha = random() * pi

    # photon back-to-back simulation
    g1_x, g2_x   = x0, x0
    g1_y, g2_y   = y0, y0
    id1, id2     = -1, -1
    if alpha >= (pi / 4.0) and alpha <= (3 * pi / 4.0):
        incx = cos(alpha)
        incy = 1
        while 1:
            g1_x += incx
            g1_y -= incy
            if g1_x <= 0:
                id1 = 2 * nx + 2 * nx - int(g1_y) - 1
                break
            if g1_x >= nx:
                id1 = nx + int(g1_y) + 1
                break
            if g1_y <= 0:
                id1 = int(g1_x) + 1
                break
        while 1:
            g2_x -= incx
            g2_y += incy
            if g2_x >= nx:
                id2 = nx + int(g2_y) + 1
                break
            if g2_x <= 0:
                id2 = 2 * nx + 2 * nx - int(g2_y) + 1
                break
            if g2_y >= nx:
                id2 = 2 * nx + nx - int(g2_x) + 1
                break
    else:
        if alpha >= (3 * pi / 4.0): incx = -1
        else:                       incx =  1
        incy = sin(alpha)
        while 1:
            g1_x += incx
            g1_y -= incy
            if g1_x <= 0:
                id1 = 2 * nx + 2 * nx - int(g1_y) + 1
                break
            if g1_x >= nx:
                id1 = nx + int(g1_y) + 1
                break
            if g1_y <= 0:
                id1 = int(g1_x) + 1
                break
        while 1:
            g2_x -= incx
            g2_y += incy
            if g2_x >= nx:
                id2 = nx + int(g2_y) + 1
                break
            if g2_x <= 0:
                id2 = 2 * nx + 2 * nx - int(g2_y) + 1
                break
            if g2_y >= nx:
                id2 = 2 * nx + nx - int(g2_x) + 1
                break
            
    return id1, id2, x0, y0
    

# create a list-mode from a simple simulate data
def pet2D_square_test_LOR(nx, posx, posy, nbp, rnd = 10):
    from numpy        import zeros, array
    from random       import seed
    seed(10)

    crystals = zeros((nx*nx, nx*nx), 'f')
    image    = zeros((nx, nx), 'f')
    
    for p in xrange(nbp):
        id1, id2, x0, y0 = pet_2D_square_test_sim_LOR(posx, posy, nx)
        crystals[id2, id1] += 1.0
        image[y0, x0]      += 1.0

    # build LOR
    LOR_val = []
    LOR_id1 = []
    LOR_id2 = []
    for id2 in xrange(nx*nx):
        for id1 in xrange(nx*nx):
            val = int(crystals[id2, id1])
            if val != 0:
                LOR_val.append(val)
                LOR_id1.append(id1)
                LOR_id2.append(id2)

    LOR_val = array(LOR_val, 'i')
    LOR_id1 = array(LOR_id1, 'i')
    LOR_id2 = array(LOR_id2, 'i')

    return LOR_val, LOR_id1, LOR_id2, image

# build the sensibility matrix based in the system response matrix for all possible LOR
def pet2D_square_build_SM(nx):
    from numpy  import zeros, ones
    from kernel import kernel_build_2D_SRM_BLA
    from utils  import image_1D_projection

    nlor    = 6 * nx * nx  # pet 4 heads
    SRM     = zeros((nlor, nx * nx), 'i')
    line    = zeros((4 * nlor), 'i')
    LOR_val = ones((nlor), 'i')

    # first head
    i = 0
    for x1 in xrange(nx):
        for y2 in xrange(nx):
            line[i:i+4] = [x1, 0, nx-1, y2]
            i += 4
        for x2 in xrange(nx):
            line[i:i+4] = [x1, 0, x2, nx-1]
            i += 4
        for y2 in xrange(nx):
            line[i:i+4] = [x1, 0, 0, y2]
            i += 4
    # second head
    for y1 in xrange(nx):
        for x2 in xrange(nx):
            line[i:i+4] = [nx-1, y1, x2, nx-1]
            i += 4
        for y2 in xrange(nx):
            line[i:i+4] = [nx-1, y1, 0, y2]
            i += 4
    # third head
    for x1 in xrange(nx):
        for y2 in xrange(nx):
            line[i:i+4] = [x1, nx-1, 0, y2]
            i += 4

    kernel_build_2D_SRM_BLA(SRM, LOR_val, line, nx)
    norm = image_1D_projection(SRM, 'x')
    SRM  = SRM.astype('f')
    for i in xrange(nlor): SRM[i] /= float(norm[i])
    
    return image_1D_projection(SRM, 'y')

# build the System Response Matrix for a list of LOR detected
def pet2D_square_build_SRM_LOR(LOR_val, LOR_id1, LOR_id2, nx):
    from numpy import zeros
    from kernel import kernel_build_2D_SRM_BLA
    
    nlor = len(LOR_val)
    SRM  = zeros((nlor, nx*nx), 'int32')
    N    = len(LOR_val)
    line = zeros((4 * N), 'i') # format [x1, y1, x2, y2, ct]

    # transform LOR index in x, y image space according the detector
    for n in xrange(N):
        id1  = LOR_id1[n]
        id2  = LOR_id2[n]
        val  = LOR_val[n]
        face = id1 // nx
        res  = id1 % nx
        if   face == 0:
            y1 = 0
            x1 = res
        elif face == 1:
            x1 = nx - 1
            y1 = res
        elif face == 2:
            y1 = nx - 1
            x1 = nx - res - 1
        elif face == 3:
            x1 = 0
            y1 = nx - res - 1

        face = id2 // nx
        res  = id2 % nx
        if   face == 0:
            y2 = 0
            x2 = res
        elif face == 1:
            x2 = nx - 1
            y2 = res
        elif face == 2:
            y2 = nx - 1
            x2 = nx - res - 1
        elif face == 3:
            x2 = 0
            y2 = nx - res - 1

        line[4*n:4*(n+1)] = [x1, y1, x2, y2]

    kernel_build_2D_SRM_BLA(SRM, LOR_val, line, nx)
            
    return SRM

    
