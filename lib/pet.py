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

