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

all : _kernel.so

_kernel.so : kernel.o kernel_wrap.o
	g++ -o _kernel.so -shared kernel.o kernel_wrap.o -fopenmp -lGL

kernel.o : kernel.cpp
	g++ -c -fPIC kernel.cpp -fopenmp -lGL

kernel_wrap.o : kernel_wrap.c
	g++ -c -fPIC kernel_wrap.c -I/usr/include/python2.6 -I/usr/lib/python2.6

kernel_wrap.c : kernel.i
	swig -python kernel.i

clean :
	rm *.o *.so *_wrap.c *.pyc kernel.py