// This file is part of FIREwork
// 
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// FIREwork Copyright (C) 2008 - 2010 Julien Bert 

//#include <GL/gl.h>
//#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

void omp_vec_square(float* data, int n);

// Volume rendering
//void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float gamma, float thres);
//void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float thres);

// 2D drawing line
void kernel_draw_2D_line_DDA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_DDA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_lines_DDAA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_lines_DDAA2(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_BLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_BLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_WLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_WALA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_DDAA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_SIDDON(float* mat, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int res, int b, int matsize);

// 3D drawing line
void kernel_draw_3D_line_DDA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val);

// PET Scan Allegro
void kernel_pet2D_SRM_entryexit(float* px, int npx, float* py, int npy, float* qx, int nqx, float* qy, int nqy, int b, int res, int srmsize, int* enable, int nenable);
void kernel_pet2D_SRM_DDA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_DDAA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_DDAA2(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_BLA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_SIDDON(float* SRM, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int res, int b, int matsize);

void kernel_allegro_idtopos(int* id_crystal1, int nidc1, int* id_detector1, int nidd1,
							float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
							int* id_crystal2, int nidc2, int* id_detector2, int nidd2,
							float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
							float respix, int sizespacexy, int sizespacez);

// PET 2D Simulated four heads
//void kernel_pet2D_square_gen_sim_ID(int* RES, int nres, float posx, float posy, float alpha, int nx);
//void kernel_build_2D_SRM_BLA(float* SRM, int sy, int sx, int* LOR_val, int nval, int* lines, int nvec, int wx);

// PET 2D reconstruction
void kernel_pet2D_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval);
void kernel_pet2D_LM_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim);
void kernel_pet2D_LM_EMML_COO_iter(float* SRMvals, int nvals, int* SRMrows, int nrows, int* SRMcols, int ncols, float* S, int nbs, float* im, int npix, int nevents);
void kernel_pet2D_EMML_iter_MPI(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval, int N_start, int N_stop);
void kernel_pet2D_EMML_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval, float* S, int ns, int maxit);

// PET 2D  Simulated ring scan
//void kernel_pet2D_ring_build_SM(float* SRM, int sy, int sx, int x1, int y1, int x2, int y2, int nx, int numlor);
//void kernel_pet2D_ring_gen_sim_ID(int* RES, int nres, int posx, int posy, float alpha, int radius);
//void kernel_pet2D_ring_LOR_SRM_BLA(float* SRM, int sy, int sx, int* LOR_val, int nval, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals);
//int kernel_pet2D_ring_LM_SRM_BLA(float* SRM, int sy, int sx, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals);
//int kernel_pet2D_ring_LM_SRM_DDA(float* SRM, int sy, int sx, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals);
//void kernel_pet2D_ring_LM_SRM_WALA(float* SRM, int sy, int sx, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals);

// Utils
void kernel_matrix_mat2coo(float* mat, int ni, int nj, float* vals, int nvals, int* rows, int nrows, int* cols, int ncols, int roffset, int coffset);
void kernel_matrix_coo_sumcol(float* vals, int nvals, int* cols, int ncols, float* im, int npix);
void kernel_matrix_coo_saxy(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres);
void kernel_matrix_coo_satxy(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres);

void kernel_matrix_mat2csr(float* mat, int ni, int nj, float* vals, int nvals, int* ptr, int nptr, int* cols, int ncols);
void kernel_matrix_csr_saxy(float* vals, int nvals, int* cols, int ncols, int* ptrs, int nptrs, float* y, int ny, float* res, int nres);
void kernel_matrix_csr_satxy(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres);

void kernel_matrix_saxy(float* mat, int ni, int nj, float* y, int ny, float* res, int nres);
int kernel_matrix_nonzeros(float* mat, int ni, int nj);
void kernel_matrix_sumcol(float* mat, int ni, int nj, float* im, int npix);
