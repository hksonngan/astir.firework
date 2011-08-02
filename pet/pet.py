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

# Convert a volume from CT to attenuation mumap
def pet_ct_to_mumap(CT):
    from numpy import zeros

    nz, ny, nx = CT.shape
    mmap = zeros((nz, ny, nx), 'float32')

    mu_pet_water = 0.096
    mu_pet_bone  = 0.172
    mu_ct_water  = 0.184
    mu_ct_bone   = 0.428

    a = mu_ct_water * (mu_pet_bone - mu_pet_water)
    b = 1000 * (mu_ct_bone - mu_ct_water)

    for z in xrange(nz):
        for y in xrange(ny):
            for x in xrange(nx):
                ict = CT[z, y, x]
                if ict <= 0: mu = mu_pet_water * (ict + 1000) / 1000.0
                else:        mu = mu_pet_water + ict * (a / b)
                mmap[z, y, x] = mu

    return mmap

