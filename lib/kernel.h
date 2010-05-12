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

void kernel_draw_2D_alllines_BLA(float* mat, int wy, int wx, int* vec, int nvec);
void kernel_pet2D_square_gen_sim_ID(int* RES, int nres, float posx, float posy, float alpha, int nx);
void kernel_build_2D_SRM_BLA(float* SRM, int sy, int sx, int* LOR_val, int nval, int* lines, int nvec, int wx);

// PET 2D reconstruction
void kernel_pet2D_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval);
void kernel_pet2D_EMML_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval);

// PET 2D  ring scan
void kernel_pet2D_ring_build_SM(float* SRM, int sy, int sx, int x1, int y1, int x2, int y2, int nx, int numlor);
void kernel_pet2D_ring_gen_sim_ID(int* RES, int nres, int posx, int posy, float alpha, int radius);
void kernel_pet2D_ring_LOR_SRM_BLA(float* SRM, int sy, int sx, int* LOR_val, int nval, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals);

void kernel_draw_3D_line_DDA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val);
