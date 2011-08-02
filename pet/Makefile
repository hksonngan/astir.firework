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


# :: C Code ::
# Module compilation
all : _pet_c.so

_pet_c.so : pet.o pet_wrap.o
	g++ -o _pet_c.so -shared pet.o pet_wrap.o

# Python wrapping
pet_wrap.o : pet_wrap.c
	g++ -o pet_wrap.o -c -fPIC pet_wrap.c -I$(PYTHONDIR)/include/python2.7 -I$(PYTHONDIR)/lib/python2.7
pet_wrap.c : pet.i
	swig -python pet.i

# C code
pet.o : pet.c
	g++ -o pet.o -c pet.c -fPIC

# :: CUDA Code ::
# Module compilation
cuda : _pet_cuda.so

_pet_cuda.so : libpetcuda.so pet_cuda_wrap.o
	g++ -o _pet_cuda.so -shared pet_cuda_wrap.o -lpetcuda

# Python wrapping
pet_cuda_wrap.o : pet_cuda_wrap.c
	g++ -o pet_cuda_wrap.o -c -fPIC pet_cuda_wrap.c -I$(PYTHONDIR)/include/python2.7 -I$(PYTHONDIR)/lib/python2.7
pet_cuda_wrap.c : pet_cuda.i
	swig -python pet_cuda.i

# CUDA code
libpetcuda.so : pet_cuda.cu
	nvcc -o pet_cuda.o -c pet_cuda.cu --compiler-options '-fPIC' -use_fast_math -arch compute_20
	nvcc -o libpetcuda.so -shared pet_cuda.o


clean :
	rm *.pyc || true
	rm *.o || true
	rm *.so || true
	rm pet_wrap.c || true
	rm pet_cuda_wrap.c || true
	rm pet_c.py || true
	rm pet_cuda.py || true
