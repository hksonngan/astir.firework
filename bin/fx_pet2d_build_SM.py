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
usage    = progname + ' SM.txt --nb_crystals=289'
topic    = 'Build the sensibility matrix (SM) based on 2D PET ring scan'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--nb_crystals', type='int',     default=289,  help='Number of crystals')

(options, args) = p.parse_args()
if len(args) < 1:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()

from firework import *

sm_name  = args[0]

# build SM
SM = pet2D_ring_build_SM(options.nb_crystals)
f = open(sm_name, 'w')
for n in xrange(len(SM)):
    f.write('%f\n' % SM[n])
f.close()
