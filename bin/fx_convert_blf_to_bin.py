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
usage    = progname + ' filename.blf output.bin'
topic    = 'Convert GE blf list-mode data to coincidence binary data (ID crystals and modules)'
p        = optparse.OptionParser(usage, description=topic)

(options, args) = p.parse_args()
if len(args) < 2:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()

src    = args[0]
trg    = args[1]

from firework import *
from numpy    import *
from time     import time

t = time()
kernel_discovery_blftobin(src, trg)
print 'running time', time_format(time()-t)


