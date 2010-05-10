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

# create a list-mode from a simple simulate data with only one point in space
def pet2D_square_test_1point_LOR(nx, posx, posy, nbp, rnd = 10):
    from kernel       import kernel_pet2D_square_gen_sim_ID
    from numpy        import zeros, array
    from numpy.random import poisson
    from random       import seed, random, randrange
    from math         import pi
    seed(10)

    crystals = zeros((nx*nx, nx*nx), 'f')
    image    = zeros((nx, nx), 'f')
    pp1      = poisson(lam=2.0, size=(nbp)).astype('f')
    ps1      = [randrange(-1, 2) for i in xrange(nbp)]
    pp2      = poisson(lam=2.0, size=(nbp)).astype('f')
    ps2      = [randrange(-1, 2) for i in xrange(nbp)]
    alpha    = [random()*pi for i in xrange(nbp)]
    res      = zeros((2), 'i')
    for p in xrange(nbp):
        x = posx + (ps1[p] * pp1[p])
        y = posy + (ps2[p] * pp2[p])
        kernel_pet2D_square_gen_sim_ID(res, x, y, alpha[p], nx)
        id1, id2 = res
        crystals[id2, id1] += 1.0
        image[y, x]  += 1.0

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

# create a list-mode from a simple simulate data with circle phantom
# image size is fixed to 65x65 with three differents activities
def pet2D_square_test_circle_LOR(nbp, rnd = 10):
    from kernel       import kernel_pet2D_square_gen_sim_ID
    from numpy        import zeros, array
    from numpy.random import poisson
    from numpy.random import seed as seed2
    from random       import seed, random, randrange
    from math         import pi
    seed(rnd)
    seed2(rnd)

    nx        = 65
    crystals  = zeros((nx*nx, nx*nx), 'f')
    image     = zeros((nx, nx), 'f')
    source    = []
    
    # three differents circle
    cx0, cy0, r0 = 32, 32, 16
    cx1, cy1, r1 = 36, 36, 7
    cx2, cy2, r2 = 26, 26, 2
    r02          = r0*r0
    r12          = r1*r1
    r22          = r2*r2
    for y in xrange(nx):
        for x in xrange(nx):
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
    pp1    = poisson(lam=1.0, size=(nbp)).astype('f')
    ps1    = [randrange(-1, 2) for i in xrange(nbp)]
    pp2    = poisson(lam=1.0, size=(nbp)).astype('f')
    ps2    = [randrange(-1, 2) for i in xrange(nbp)]
    alpha  = [random()*pi for i in xrange(nbp)]
    ind    = [randrange(nbpix) for i in xrange(nbp)]
    res    = zeros((2), 'i')
    for p in xrange(nbp):
        x   = source[3*ind[p]]   + (ps1[p] * pp1[p])
        y   = source[3*ind[p]+1] + (ps2[p] * pp2[p])
        val = source[3*ind[p]+2]
        kernel_pet2D_square_gen_sim_ID(res, x, y, alpha[p], nx)
        id1, id2 = res
        crystals[id2, id1] += val
        image[y, x]  += source[3*ind[p]+2]

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
    SRM     = zeros((nlor, nx * nx), 'float32')
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
    SRM  = zeros((nlor, nx*nx), 'float32')
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

###############################################################################
# PET 2D ring scanner
###############################################################################

# build the sensibility Matrix to 2D ring PET scan according all possibles LORs
def pet2D_ring_build_SM(nbcrystals):
    from numpy  import zeros
    from math   import cos, sin, pi, sqrt
    from kernel import kernel_pet2D_ring_build_SM
    from utils  import image_1D_projection

    radius = int(nbcrystals / 2.0 / pi + 0.5)      # radius PET
    dia    = 2 * radius + 1                        # dia PET must be odd
    cxo    = cyo = radius                          # center PET
    Nlor   = (nbcrystals-1) * (nbcrystals-1) / 2   # nb all possible LOR
    radius = float(radius)

    # build SRM for only the square image inside the ring of the PET
    SM = zeros((dia * dia), 'float32')
    for i in xrange(nbcrystals):
        nlor  = nbcrystals-(i+1)
        index = 0
        SRM   = zeros((nlor, dia * dia), 'float32')
        for j in xrange(i+1, nbcrystals):
            alpha1 = i / radius
            alpha2 = j / radius
            x1     = int(cxo + radius * cos(alpha1) + 0.5)
            x2     = int(cxo + radius * cos(alpha2) + 0.5)
            y1     = int(cyo - radius * sin(alpha1) + 0.5)
            y2     = int(cyo - radius * sin(alpha2) + 0.5)
            kernel_pet2D_ring_build_SM(SRM, x1, y1, x2, y2, dia, index)
            index += 1
        # sum by step in order to decrease the memory for this stage
        norm = image_1D_projection(SRM, 'x')
        SRM  = SRM.astype('f')
        for i in xrange(nlor): SRM[i] /= float(norm[i])
        res = image_1D_projection(SRM, 'y')
        SM += res
 
    return SM

# PET 2D ring scan create LOR event from a simple simulate phantom (three activities)
def pet2D_ring_simu_circle_phantom(nbcrystals, nbparticules, rnd = 10):
    from numpy        import zeros, array
    from numpy.random import poisson
    from numpy.random import seed as seed2
    from random       import seed, random, randrange
    from math         import pi, sqrt, cos, sin
    from kernel       import kernel_pet2D_ring_gen_sim_ID, kernel_draw_2D_line_BLA
    seed(rnd)
    seed2(rnd)

    radius = int(nbcrystals / 2.0 / pi + 0.5)      # radius PET
    dia    = 2 * radius + 1                        # dia PET must be odd
    cxo    = cyo = radius                          # center PET
    crystals = zeros((nbcrystals, nbcrystals), 'float32')
    image    = zeros((dia, dia), 'float32')
    source   = []

    # three differents circle
    cx0, cy0, r0 = cxo,    cyo,    16
    cx1, cy1, r1 = cx0+4,  cy0+4,   7
    cx2, cy2, r2 = cx0-6,  cy0-6,   2
    r02          = r0*r0
    r12          = r1*r1
    r22          = r2*r2
    for y in xrange(dia):
        for x in xrange(dia):
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
    alpha  = [random()*pi for i in xrange(nbparticules)]
    ind    = [randrange(nbpix) for i in xrange(nbparticules)]
    res    = zeros((2), 'int32')
    lines  = zeros((4), 'int32')
    for p in xrange(nbparticules):
        x   = int(source[3*ind[p]]   + (ps1[p] * pp1[p]))
        y   = int(source[3*ind[p]+1] + (ps2[p] * pp2[p]))
        val = source[3*ind[p]+2]
        kernel_pet2D_ring_gen_sim_ID(res, x, y, alpha[p], radius, lines)
        id1, id2 = res
        crystals[id2, id1] += val
        print lines
        kernel_draw_2D_line_BLA(image, int(lines[0]), int(lines[1]), int(lines[2]), int(lines[3]), val)
        #image[y, x]  += source[3*ind[p]+2]

    # build LOR
    LOR_val = []
    LOR_id1 = []
    LOR_id2 = []
    for id2 in xrange(nbcrystals):
        for id1 in xrange(nbcrystals):
            val = int(crystals[id2, id1])
            if val != 0:
                LOR_val.append(val)
                LOR_id1.append(id1)
                LOR_id2.append(id2)

    LOR_val = array(LOR_val, 'int32')
    LOR_id1 = array(LOR_id1, 'int32')
    LOR_id2 = array(LOR_id2, 'int32')

    return LOR_val, LOR_id1, LOR_id2, image
