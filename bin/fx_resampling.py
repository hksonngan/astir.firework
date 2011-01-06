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
usage    = progname + ' vol_src.vol vol_trg.vol nz ny nx --method'
topic    = 'Resampling a volume'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--method',  action='store', type='choice', choices=['L3', 'L2'], default='L3',
             help='"L3" Lancsoz3, "L2" Lancsoz2 (default is L3)')

(options, args) = p.parse_args()
if len(args) < 5:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()
    
src = args[0]
trg = args[1]
nz  = int(args[2])
ny  = int(args[3])
nx  = int(args[4])

from firework import *
from numpy    import *

svol = volume_open(src)
tvol = zeros((nz, ny, nx), 'float32')
if   options.method == 'L3': kernel_resampling_3d_Lanczos3(svol, tvol)
elif options.method == 'L2': kernel_resampling_3d_Lanczos2(svol, tvol)
volume_write(tvol, trg)
