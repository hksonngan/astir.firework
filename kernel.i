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
void kernel_draw_voxels(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float gamma, float thres);
void kernel_draw_voxels_edge(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float thres);
void kernel_draw_2D_line_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);