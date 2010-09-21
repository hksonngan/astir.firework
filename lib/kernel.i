/* -*- C -*-  (not really, but good for syntax highlighting) */
// This file is part of FIREwire
// 
// FIREwire is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwire is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwire.  If not, see <http://www.gnu.org/licenses/>.
//
// FIREwire Copyright (C) 2008 - 2010 Julien Bert 

%module kernel

%{
#define SWIG_FILE_WITH_INIT
#include "kernel.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

//void omp_vec_square(float* INPLACE_ARRAY1, int DIM1);

// Volume rendering
void kernel_draw_voxels(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float gamma, float thres);
void kernel_draw_voxels_edge(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float thres);
void kernel_draw_pixels(float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2);
void kernel_color_image(float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1);

// 2D Raytracer
void kernel_draw_2D_line_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);
void kernel_draw_2D_lines_DDAA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);
void kernel_draw_2D_lines_DDAA2(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);
void kernel_draw_2D_line_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);
void kernel_draw_2D_line_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);
void kernel_draw_2D_line_WALA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_DDAA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_SIDDON(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, int res, int b, int matsize);

// PET Scan Allegro
void kernel_pet2D_SRM_entryexit(float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int b, int srmsize, int* INPLACE_ARRAY1, int DIM1);
void kernel_pet2D_SRM_clean_entryexit_int(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1);
void kernel_pet2D_SRM_clean_entryexit_float(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_pet2D_SRM_clean_LOR_center(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int border, int size_im);
void kernel_pet2D_SRM_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int width_image);
void kernel_pet2D_SRM_ELL_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int width_image);
void kernel_pet2D_SRM_DDA_cuda(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int width_image); // cuda version
void kernel_pet2D_SRM_DDA_omp(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int width_image); // omp version
void kernel_pet2D_SRM_DDAA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int width_image);
void kernel_pet2D_SRM_DDAA2(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int width_image);
void kernel_pet2D_SRM_ELL_DDAA2(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int width_image);
void kernel_pet2D_SRM_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int width_image);
void kernel_pet2D_SRM_SIDDON(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, int matsize);
void kernel_pet2D_SRM_ELL_SIDDON(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, int matsize);
void kernel_pet2D_SRM_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int wim);
void kernel_allegro_idtopos(int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float respix, int sizespacexy, int sizespacez, int rnd);
void kernel_allegro_build_all_LOR(unsigned short int* INPLACE_ARRAY1, int DIM1, unsigned short int* INPLACE_ARRAY1, int DIM1, unsigned short int* INPLACE_ARRAY1, int DIM1, unsigned short int* INPLACE_ARRAY1, int DIM1);
void kernel_pet3D_SRM_raycasting(float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int border, int ROIxy, int ROIz);
void kernel_pet3D_SRM_clean_LOR_int(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1);
void kernel_pet3D_SRM_clean_LOR_float(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_pet3D_SRM_ELL_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, int wim);
void kernel_pet3D_IM_SRM_DDA(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wim);
void kernel_pet3D_IM_SRM_ELL_DDA_iter(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wim, int ndata);
void kernel_pet3D_IM_SRM_ELL_DDA_iter_vec(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wim, int ndata);
void kernel_pet3D_IM_SRM_SIDDON(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wim, int dim);
void kernel_pet3D_IM_SRM_SIDDON_iter(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wim);
void kernel_pet3D_IM_SRM_COO_SIDDON(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wim, int isub);
void kernel_pet3D_IM_SRM_COO_SIDDON_iter_vec(float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int N, int isub);
void kernel_pet3D_IM_SRM_COO_ON_SIDDON_iter(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wim, int dim);
void kernel_pet3D_IM_ATT_SRM_COO_ON_SIDDON_iter(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, int wim, int dim);
void kernel_pet3D_IM_SRM_COO_SIDDON_iter_mat(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int N, int isub);
void kernel_pet3D_IM_SRM_ELL_SIDDON_iter(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wim, int ndata);



// PET 2D Simulated four squares
//void kernel_pet2D_square_gen_sim_ID(int* INPLACE_ARRAY1, int DIM1, float posx, float posy, float alpha, int nx);
//void kernel_build_2D_SRM_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int wx);

// PET 2D Reconstruction
void kernel_pet2D_EMML_iter(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);
void kernel_pet2D_LM_EMML_iter(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_pet2D_LM_EMML_COO_iter_mat(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int nevents);
void kernel_pet2D_LM_EMML_COO_iter_vec(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int nevents);
void kernel_pet2D_LM_EMML_ELL_iter(float* IN_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1);
void kernel_pet2D_EMML_cuda(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, int maxit);
void kernel_pet2D_EMML_iter_MPI(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int N_start, int N_stop);
void kernel_pet2D_LM_EMML_DDA_ELL_cuda(int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wsrm, int wim, int maxite);
void kernel_pet2D_IM_SRM_DDA_ELL_cuda(int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wsrm, int wim);
void kernel_pet2D_IM_SRM_DDA_ELL_iter_cuda(int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wsrm, int wim);
void kernel_pet3D_IM_SRM_DDA_ELL_cuda(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wsrm, int wim, int ID);
void kernel_pet3D_IM_SRM_DDA_ELL_iter_cuda(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int wsrm, int wim, int ID);


// PET 2D Simulated ring scan
//void kernel_pet2D_ring_build_SM(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, int nx, int numlor);
//void kernel_pet2D_ring_gen_sim_ID(int* INPLACE_ARRAY1, int DIM1, int posx, int posy, float alpha, int radius);
//void kernel_pet2D_ring_LOR_SRM_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int nbcrystals);
//int kernel_pet2D_ring_LM_SRM_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int nbcrystals);
//int kernel_pet2D_ring_LM_SRM_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int nbcrystals);
//void kernel_pet2D_ring_LM_SRM_WALA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int nbcrystals);


// 3D Raytracer
void kernel_draw_3D_line_DDA(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, int x1, int y1, int z1, int x2, int y2, int z2, float val);

// Utils
void kernel_matrix_mat2coo(float* IN_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int roffset, int coffset);

void kernel_matrix_coo_sumcol(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_coo_spmv(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_coo_spmtv(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);

void kernel_matrix_mat2csr(float* IN_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_csr_sumcol(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_csr_spmv(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_csr_spmtv(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);

void kernel_matrix_mat2ell(float* IN_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2);
void kernel_matrix_ell_sumcol(float* IN_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_ell_spmv(float* IN_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_ell_spmv_cuda(float* IN_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_ell_spmtv(float* IN_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);

void kernel_matrix_spmv(float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_spmtv(float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);
int kernel_matrix_nonzeros(float* IN_ARRAY2, int DIM1, int DIM2);
void kernel_matrix_nonzeros_rows(float* IN_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_sumcol(float* IN_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_lp_H(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, float fc, int order);
void kernel_flatvolume_gaussian_filter_3x3x3(float* INPLACE_ARRAY1, int DIM1, int nk, int nj, int ni);

int kernel_vector_nonzeros(float* IN_ARRAY1, int DIM1);

void kernel_listmode_open_subset_xyz_int(unsigned short int* INPLACE_ARRAY1, int DIM1, unsigned short int* INPLACE_ARRAY1, int DIM1, unsigned short int* INPLACE_ARRAY1, int DIM1, unsigned short int* INPLACE_ARRAY1, int DIM1, unsigned short int* INPLACE_ARRAY1, int DIM1, unsigned short int* INPLACE_ARRAY1, int DIM1, int n_start, int n_stop, char* basename);

void kernel_listmode_open_subset_xyz_float(float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int n_start, int n_stop, char* basename);

void kernel_listmode_open_subset_ID_int(int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,	int n_start, int n_stop, char* name);

void toto(char* name);



void kernel_pet3D_IM_DEV_cuda(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int wim, int ID);

void kernel_pet3D_IM_SRM_DDA_ON_iter_cuda(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1, float* IN_ARRAY3, int DIM1, int DIM2, int DIM3, float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, int wim, int ID);

void kernel_pet3D_IM_ATT_SRM_DDA_ON_iter_cuda(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
											  unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
											  unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
											  float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
											  float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
											  float* IN_ARRAY3, int DIM1, int DIM2, int DIM3, int wim, int ID);

void kernel_mip_volume_rendering(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3, float* INPLACE_ARRAY2, int DIM1, int DIM2, float alpha, float beta, float scale);

void kernel_filter_2d_median(float* IN_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY2, int DIM1, int DIM2, int w);
void kernel_filter_3d_median(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3, float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, int w);
void kernel_filter_2d_adaptive_median(float* IN_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY2, int DIM1, int DIM2, int w, int wmax);
void kernel_filter_3d_adaptive_median(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3, float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, int w, int wmax);

void kernel_3Dconv_cuda(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, float* IN_ARRAY3, int DIM1, int DIM2, int DIM3);

int kernel_pack_id(int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
					int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
					unsigned short int* INPLACE_ARRAY1, int DIM1, int id, int flag);

void kernel_SRM_to_HD(int isub);

void kernel_allegro_save_all_LOR(int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
								 int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
								 int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

