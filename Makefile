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
	rm *.o *.so *_wrap.c *.pyc