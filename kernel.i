%module kernel

%{
#define SWIG_FILE_WITH_INIT
#include "kernel.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

void omp_vec_square(float* INPLACE_ARRAY1, int DIM1);
void kernel_draw_voxels(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1);
