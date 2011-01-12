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

import optparse, os, sys

progname = os.path.basename(sys.argv[0])
usage    = progname + ' vol_in.vol vol_out.vol'
topic    = 'Masking and filtering volume'
p        = optparse.OptionParser(usage, description=topic)
#p.add_option('--Nite',    type='int',    default=1,       help='Number of iterations (default 1)')

(options, args) = p.parse_args()
if len(args) < 2:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()
    
src = args[0]
trg = args[1]

from firework import *
from numpy    import *

vol   = volume_open(src)
mask  = volume_mask_cylinder(47, 127, 127, 47, 60)
#mask  = volume_mask_box(47, 127, 127, 121, 121, 47)
#volf  = filter_3d_Metz(vol, 2, 0.16) # Ny=0.3
volf   = filter_3d_Metz(vol, 3, 0.2) # Ny=0.4
volf *= mask

volume_write(volf, trg)
