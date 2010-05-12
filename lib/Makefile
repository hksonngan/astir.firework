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

all : _kernel.so

_kernel.so : kernel.o kernel_wrap.o kernel_cuda.so
	g++ -o _kernel.so -shared kernel.o kernel_wrap.o kernel_cuda.so -fopenmp -lGL

kernel.o : kernel.cpp
	g++ -c -fPIC kernel.cpp -fopenmp -lGL
# not need -fopenmp and -lGL here ??

kernel_wrap.o : kernel_wrap.c
	g++ -c -fPIC kernel_wrap.c -I/usr/include/python2.6 -I/usr/lib/python2.6

kernel_wrap.c : kernel.i
	swig -python kernel.i

kernel_cuda.so : kernel_cuda.o
	nvcc -o kernel_cuda.so -shared kernel_cuda.o -lcublas

kernel_cuda.o : kernel_cuda.cu
	nvcc -c kernel_cuda.cu --compiler-options '-fPIC'

clean :
	rm *.o *.so *_wrap.c *.pyc kernel.py