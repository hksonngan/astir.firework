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


###############################################################################
# PET 2D ring scanner
###############################################################################

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
    for i in xrange(nbcrystals-1):
        nlor  = nbcrystals-(i+1)
        index = 0
        SRM   = zeros((nlor, dia * dia), 'float32')
        for j in xrange(i+1, nbcrystals):
            alpha1 = i / radius
            alpha2 = j / radius
            x1     = int(cxo + radius * cos(alpha1) + 0.5)
            x2     = int(cxo + radius * cos(alpha2) + 0.5)
            y1     = int(cyo + radius * sin(alpha1) + 0.5)
            y2     = int(cyo + radius * sin(alpha2) + 0.5)
            kernel_pet2D_ring_build_SM(SRM, x1, y1, x2, y2, dia, index)
            index += 1
            
        # sum by step in order to decrease the memory for this stage
        norm = image_1D_projection(SRM, 'x')
        for n in xrange(nlor): SRM[n] /= float(norm[n])
        res = image_1D_projection(SRM, 'y')
        SM += res
 
    return SM

