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
# FIREwork Copyright (C) 2008 - 2011 Julien Bert 

from time     import time
from struct   import pack
import sys

if len(sys.argv) != 3:
    print 'Convert .dat Gate coincidence file to bin data'
    print 'fx_convert_Gate_ascii_to_bin.py source.dat target.bin'
    sys.exit()

src   = sys.argv[1]
trg   = sys.argv[2]

f    = open(src, 'r')
o    = open(trg, 'wb')
line = f.readline()
ct   = 0
t    = time()
while line:
    data = line.split()
    o.write(pack('i', int(data[15])))
    o.write(pack('i', int(data[12])))
    o.write(pack('i', int(data[38])))
    o.write(pack('i', int(data[35])))
    
    ct  += 1
    line = f.readline()

print ct, ' number of LORs, build in ', time()-t, ' s'    

f.close()
o.close()

