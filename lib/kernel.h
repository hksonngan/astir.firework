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

//void omp_vec_square(float* data, int n);

// Volume rendering
void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float* valthr, int nthr, float gamma, float thres);
void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float* valthr, int nthr, float thres);
void kernel_draw_pixels(float* mapr, int him, int wim, float* mapg, int himg, int wimg, float* mapb, int himb, int wimb);
void kernel_color_image(float* im, int him, int wim,
						float* mapr, int him1, int wim1, float* mapg, int him2, int wim2, float* mapb, int him3, int wim3,
						float* lutr, int him4, float* lutg, int him5, float* lutb, int him6);

// 2D drawing line
void kernel_draw_2D_line_DDA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_DDA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_lines_DDA_fixed(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_lines_DDAA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_lines_DDAA2(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_BLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_BLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_WLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_WLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_WALA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_WALA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_DDAA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_SIDDON(float* mat, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int res, int b, int matsize);

// 3D drawing line
void kernel_draw_3D_line_DDA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val);
void kernel_draw_3D_line_BLA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val);

// PET 2D Scan Allegro
void kernel_pet2D_SRM_entryexit(float* px, int npx, float* py, int npy, float* qx, int nqx, float* qy, int nqy, int b, int srmsize, int* enable, int nenable);
void kernel_pet2D_SRM_clean_entryexit_int(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
									  int* xi1, int nxi1, int* yi1, int nyi1, int* xi2, int nxi2, int* yi2, int nyi2);
void kernel_pet2D_SRM_clean_entryexit_float(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
											float* xf1, int nxf1, float* yf1, int nyf1, float* xf2, int nxf2, float* yf2, int nyf2);
void kernel_pet2D_SRM_clean_LOR_center(float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
									   float* xc1, int nxc1, float* yc1, int nyc1, float* xc2, int nxc2, float* yc2, int ncy2, int border, int size_im);
void kernel_pet2D_SRM_DDA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_ELL_DDA(float* vals, int niv, int njv, int* cols, int nic, int njc, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_DDA_cuda(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image); // CUDA wrapper
void kernel_pet2D_SRM_DDA_omp(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image); // OMP version
void kernel_pet2D_SRM_DDAA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_DDAA2(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_ELL_DDAA2(float* SRMvals, int niv, int njv, int* SRMcols, int nic, int njc, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_BLA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_pet2D_SRM_SIDDON(float* SRM, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int matsize);
void kernel_pet2D_SRM_ELL_SIDDON(float* SRMvals, int niv, int njv, int* SRMcols, int nic, int njc, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int matsize);
void kernel_pet2D_SRM_WLA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int wim);

// PET 3D Scan Allegro
void kernel_pet3D_SRM_ELL_DDA(float* vals, int niv, int njv, int* cols, int nic, int njc,
							  unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1, unsigned short int* Z1, int nz1,
							  unsigned short int* X2, int nx2, unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2, int wim);
void kernel_pet3D_IM_SRM_ELL_DDA_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1, unsigned short int* Z1, int nz1,
										  unsigned short int* X2, int nx2, unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
										  float* im, int nim, float* F, int nf, int wim, int ndata);
void kernel_pet3D_IM_SRM_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
									 float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2, float* im, int nim, float* F, int nf, int wim);
void kernel_pet3D_IM_SRM_COO_SIDDON(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
									float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2, float* im, int nim, int wim, int isub);
void kernel_pet3D_IM_SRM_COO_SIDDON_iter_vec(float* im, int nim, float* F, int nf, int N, int isub);

void kernel_pet3D_IM_SRM_COO_SIDDON_iter_mat(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* im, int nim, float* F, int nf, int N, int isub);
void kernel_pet3D_IM_SRM_ELL_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
										 float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
										 float* im, int nim, float* F, int nf, int wim, int ndata);

void kernel_pet3D_IM_SRM_DDA_ELL_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
									  unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
									  float* im, int nim, int wsrm, int wim, int ID);
void kernel_pet3D_IM_SRM_DDA_ELL_iter_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
										   unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										   float* im, int nim, float* F, int nf, int wsrm, int wim, int ID);

// used
void kernel_allegro_idtopos(int* id_crystal1, int nidc1, int* id_detector1, int nidd1,
							float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
							int* id_crystal2, int nidc2, int* id_detector2, int nidd2,
							float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
							float respix, int sizespacexy, int sizespacez, int rnd);
// used
void kernel_allegro_build_all_LOR(unsigned short int* idc1, int n1, unsigned short int* idd1, int n2,
								  unsigned short int* idc2, int n3, unsigned short int* idd2, int n4);

// used
void kernel_discovery_blftobin(char* blffilename, char* binfilename);

// used
void kernel_discovery_idtopos(int* id_crystal1, int nidc1, int* id_detector1, int nidd1,
							  float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
							  int* id_crystal2, int nidc2, int* id_detector2, int nidd2,
							  float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
							  float respix, int sizespacexy, int sizespacez, int rnd);

// used
void kernel_pet3D_SRM_raycasting(float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
								float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
								 int* enable, int nenable, int border, int ROIxy, int ROIz);
// used
void kernel_pet3D_SRM_clean_LOR_int(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
									float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
									int* xi1, int nxi1, int* yi1, int nyi1, int* zi1, int nzi1,
									int* xi2, int nxi2, int* yi2, int nyi2, int* zi2, int nzi2);
// used
void kernel_pet3D_SRM_clean_LOR_float(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
									  float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
									  float* xf1, int nxi1, float* yf1, int nyi1, float* zf1, int nzi1,
									  float* xf2, int nxi2, float* yf2, int nyi2, float* zf2, int nzi2);
// used
void kernel_pet3D_IM_SRM_SIDDON(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
								float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
								float* im, int nim, int wim, int dim);
// used
void kernel_pet3D_IM_SRM_DDA_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
								  unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
								  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
								  int* im, int nim1, int nim2, int nim3, int wim, int ID);
// used
void kernel_pet3D_IM_SRM_DDA_ON_iter_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
										  unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
										  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										  float* im, int nim1, int nim2, int nim3, float* F, int nf1, int nf2, int nf3,
										  int wim, int ID);
// used
void kernel_pet3D_IM_ATT_SRM_DDA_ON_iter_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
											  unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
											  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
											  float* im, int nim1, int nim2, int nim3,
											  float* F, int nf1, int nf2, int nf3,
											  float* mumap, int nmu1, int nmu2, int nmu3, int wim, int ID);
// used
void kernel_pet3D_IM_ATT_SRM_COO_ON_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
												float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
												float* im, int nim, float* F, int nf, float* mumap, int nmu,
												int wim, int dim);
// used
void kernel_pet3D_IM_SRM_COO_ON_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
											float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
											float* im, int nim, float* F, int nf, int wim, int dim);
// used
void kernel_pet3D_IM_SRM_DDA( unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
							  unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
							  unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
							  float* im, int nim1, int nim2, int nim3, int wim);
// to test
void kernel_pet3D_IM_SRM_DDA_fixed(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
								   unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
								   unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
								   float* im, int nim1, int nim2, int nim3, int wim);

// used
void kernel_pet3D_IM_SRM_ELL_DDA_ON_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
										 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
										 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
										 float* im, int nim1, int nim2, int nim3,
										 float* F, int nf1, int nf2, int nf3, int wim, int ndata);

// the best one
void kernel_pet3D_IM_SRM_ELL_DDA_fixed_ON_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
										 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
										 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
										 float* im, int nim1, int nim2, int nim3,
										 float* F, int nf1, int nf2, int nf3, int wim, int ndata);

// to compare
void kernel_pet3D_IM_SRM_COO_DDA_ON_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
										 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
										 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
										 float* im, int nim1, int nim2, int nim3,
										 float* F, int nf1, int nf2, int nf3, int wim);

// to compare
void kernel_pet3D_IM_SRM_RAW_DDA_ON_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
										 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
										 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
										 float* im, int nim1, int nim2, int nim3,
										 float* F, int nf1, int nf2, int nf3, int wim);

// option
void kernel_pet3D_IM_SRM_BLA(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
							 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
							 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
							 float* im, int nim1, int nim2, int nim3, int wim);

// used (new code)
void kernel_pet3D_OPLEM_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
							 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
							 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
							 float* im, int nim1, int nim2, int nim3,
							 float* NM, int NM1, int NM2, int NM3, int Nsub, int ID);
// used (new code)
void kernel_pet3D_OPLEM_att_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
								 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
								 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
								 float* im, int nim1, int nim2, int nim3,
								 float* NM, int NM1, int NM2, int NM3,
								 float* at, int nat1, int nat2, int nat3,
								 int Nsub, int ID);



// PET 2D Simulated four heads
//void kernel_pet2D_square_gen_sim_ID(int* RES, int nres, float posx, float posy, float alpha, int nx);
//void kernel_build_2D_SRM_BLA(float* SRM, int sy, int sx, int* LOR_val, int nval, int* lines, int nvec, int wx);

// PET 2D reconstruction
void kernel_pet2D_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval);
void kernel_pet2D_LM_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim);
void kernel_pet2D_LM_EMML_COO_iter_mat(float* SRMvals, int nvals, int* SRMrows, int nrows, int* SRMcols, int ncols, float* S, int nbs, float* im, int npix, int nevents);
void kernel_pet2D_LM_EMML_COO_iter_vec(float* SRMvals, int nvals, int* SRMrows, int nrows, int* SRMcols, int ncols, float* S, int nbs, float* im, int npix, int nevents);
void kernel_pet2D_LM_EMML_ELL_iter(float* SRMvals, int nivals, int njvals, int* SRMcols, int nicols, int njcols, float* S, int ns, float* im, int npix);
void kernel_pet2D_EMML_iter_MPI(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval, int N_start, int N_stop);

// CUDA wrapper
void kernel_pet2D_EMML_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval, float* S, int ns, int maxit);
void kernel_pet2D_LM_EMML_DDA_ELL_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, float* S, int ns, int wsrm, int wim, int maxite);
void kernel_pet2D_IM_SRM_DDA_ELL_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, int wsrm, int wim);
void kernel_pet2D_IM_SRM_DDA_ELL_iter_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* S, int ns, float* im, int nim, int wsrm, int wim);

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
void kernel_matrix_coo_spmv(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres);
void kernel_matrix_coo_spmtv(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres);

void kernel_matrix_mat2csr(float* mat, int ni, int nj, float* vals, int nvals, int* ptr, int nptr, int* cols, int ncols);
void kernel_matrix_csr_sumcol(float* vals, int nvals, int* cols, int ncols, float* im, int npix);
void kernel_matrix_csr_spmv(float* vals, int nvals, int* cols, int ncols, int* ptrs, int nptrs, float* y, int ny, float* res, int nres);
void kernel_matrix_csr_spmtv(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres);

void kernel_matrix_mat2ell(float* mat, int ni, int nj, float* vals, int nivals, int njvals, int* cols, int nicols, int njcols);
void kernel_matrix_ell_sumcol(float* vals, int niv, int njv, int* cols, int nic, int njc, float* im, int npix);
void kernel_matrix_ell_spmv(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres);
void kernel_matrix_ell_spmv_cuda(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres);
void kernel_matrix_ell_spmtv(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres);

void kernel_matrix_spmv(float* mat, int ni, int nj, float* y, int ny, float* res, int nres);
void kernel_matrix_spmtv(float* mat, int ni, int nj, float* y, int ny, float* res, int nres);
int kernel_matrix_nonzeros(float* mat, int ni, int nj);
void kernel_matrix_nonzeros_rows(float* mat, int ni, int nj, int* rows, int nrows);
void kernel_matrix_sumcol(float* mat, int ni, int nj, float* im, int npix);
void kernel_matrix_lp_H(float* mat, int nk, int nj, int ni, float fc, int order);

void kernel_flatvolume_gaussian_filter_3x3x3(float* mat, int nmat, int nk, int nj, int ni);

int kernel_vector_nonzeros(float* mat, int ni);

void kernel_listmode_open_subset_xyz_int(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
										 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
										 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										 int n_start, int n_stop, char* basename);

void kernel_listmode_open_subset_xyz_float(float* x1, int nx1, float* y1, int ny1, float* z1, int nz1, 
										   float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
										   int n_start, int n_stop, char* basename);

void kernel_listmode_open_subset_ID_int(int* idc1, int n1, int* idd1, int n2, int* idc2, int n3, int* idd2, int n4,
										int n_start, int n_stop, char* name);

void toto(char* name);

void kernel_mip_volume_rendering(float* vol, int nz, int ny, int nx, float* mip, int wim, int him, float alpha, float beta, float scale);

// filter
void kernel_filter_2d_median(float* im, int ny, int nx, float* res, int nyr, int nxr, int w);
void kernel_filter_3d_median(float* im, int nz, int ny, int nx, float* res, int nzr, int nyr, int nxr, int w);
void kernel_filter_2d_adaptive_median(float* im, int ny, int nx, float* res, int nyr, int nxr, int w, int wmax);
void kernel_filter_3d_adaptive_median(float* im, int nz, int ny, int nx,
									  float* res, int nzr, int nyr, int nxr, int w, int wmax);
// Utils function
void inkernel_quicksort(float* vec, int m, int n);
void inkernel_bubblesort(float* vec, int n);
int inkernel_mono(int i, int j);
float inkernel_randf();
float inkernel_randgf(float mean, float std);
void inkernel_randg2f(float mean, float std, float* z0, float* z1);

// dev
void kernel_3Dconv_cuda(float* vol, int nz, int ny, int nx, float* H, int a, int b, int c);

int kernel_pack_id(int* d1, int nd1, int* c1, int nc1,
					int* d2, int nd2, int* c2, int nc2,
					unsigned short int* pack, int np, int id, int flag);

void kernel_SRM_to_HD(int isub);

void kernel_allegro_save_all_LOR(int* id1, int n1, int* id2, int n2,
								 int* x1, int nx1, int* y1, int ny1, int* z1, int nz1,
								 int* x2, int nx2, int* y2, int ny2, int* z2, int nz2);

void kernel_pet3D_IM_SRM_HD(int* idc1, int nc1, int* idd1, int nd1, int* idc2, int nc2, int* idd2, int nd2,
							float* im, int nz, int ny, int nx, char* nametoc, char* namesrm);
