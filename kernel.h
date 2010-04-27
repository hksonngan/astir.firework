#include <GL/gl.h>
#include <omp.h>

void omp_vec_square(float* data, int n);
void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float gamma, float thres);
void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float thres);
