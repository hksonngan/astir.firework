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

# Module compilation
all : _dev_c.so

_dev_c.so : dev.o dev_wrap.o
	g++ -o _dev_c.so -shared dev.o dev_wrap.o

# Python wrapping
dev_wrap.o : dev_wrap.c
	g++ -o dev_wrap.o -c -fPIC dev_wrap.c -I$(PYTHONDIR)/include/python2.7 -I$(PYTHONDIR)/lib/python2.7
dev_wrap.c : dev.i
	swig -python dev.i

# C code
dev.o : dev.c
	g++ -o dev.o -c dev.c -fPIC 

clean :
	rm *.pyc || true
	rm *.o || true
	rm *.so || true
	rm dev_wrap.c || true
	rm dev_c.py || true

