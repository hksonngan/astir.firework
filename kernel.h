#include <GL/gl.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

void omp_vec_square(float* data, int n);

void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float gamma, float thres);
void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float thres);

void kernel_draw_2D_line_DDA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_BLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_WLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_WALA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_DDAA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);

void kernel_draw_3D_line_DDA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val);
