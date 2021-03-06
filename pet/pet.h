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
// FIREwork Copyright (C) 2008 - 2011 Julien Bert 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Open a phase-space
void pet_c_phasespace_open(char* filename,
						   int* type, int ntype,
						   float* E, int nE,
						   float* px, int npx, float* py, int npy, float* pz, int npz,
						   float* dx, int ndx, float* dy, int ndy, float* dz, int ndz);

// Allegro scanner
void pet_c_allegro_idtopos(int* id_crystal1, int nidc1, int* id_detector1, int nidd1,
						   float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
						   int* id_crystal2, int nidc2, int* id_detector2, int nidd2,
						   float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
						   float respix, int sizespacexy, int sizespacez, int rnd);

void pet_c_allegro_build_all_lor(unsigned short int* idc1, int n1, unsigned short int* idd1, int n2,
								 unsigned short int* idc2, int n3, unsigned short int* idd2, int n4);

void pet_c_allegro_save_rnd_lor(char* savename, int nlor);

// Discovery scanner
void pet_c_discovery_blftobin(char* blffilename, char* binfilename);

void pet_c_discovery_idtopos(int* id_crystal1, int nidc1, int* id_detector1, int nidd1,
							 float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
							 int* id_crystal2, int nidc2, int* id_detector2, int nidd2,
							 float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
							 float respix, int sizespacexy, int sizespacez, int rnd);

void pet_c_discovery_save_rnd_lor(char* savename, int nlor);

// Utils
void pet_c_raycasting(float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
					  float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
					  int* enable, int nenable, int border, int ROIxy, int ROIz);

void pet_c_clean_lor_int(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
						 float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
						 int* xi1, int nxi1, int* yi1, int nyi1, int* zi1, int nzi1,
						 int* xi2, int nxi2, int* yi2, int nyi2, int* zi2, int nzi2);

void pet_c_clean_lor_float(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
						   float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
						   float* xf1, int nxi1, float* yf1, int nyi1, float* zf1, int nzi1,
						   float* xf2, int nxi2, float* yf2, int nyi2, float* zf2, int nzi2);

void pet_c_listmode_open_subset_xyz_int(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
										unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
										unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										int n_start, int n_stop, char* basename);

void pet_c_listmode_open_subset_xyz_int_sort(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1, 
											 unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
											 int* ID, int nid, int n_start, int n_stop, char* basename);

void pet_c_listmode_open_subset_xyz_float(float* x1, int nx1, float* y1, int ny1, float* z1, int nz1, 
										  float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
										  int n_start, int n_stop, char* basename);

void pet_c_listmode_open_subset_id_int(int* idc1, int n1, int* idd1, int n2, int* idc2, int n3, int* idd2, int n4,
									   int n_start, int n_stop, char* name);

// LM-OSEM reconstruction
void pet_c_lmosem_siddon(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
						 float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
						 float* im, int nim, float* F, int nf, int wim, int dim, int border);

void pet_c_lmosem_siddon_att(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
							 float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
							 float* im, int nim, float* F, int nf, float* mumap, int nmu,
							 int wim, int dim, int border);

void pet_c_lmosem_dda(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
					  unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
					  unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
					  float* im, int nim1, int nim2, int nim3,
					  float* F, int nf1, int nf2, int nf3, int wim, int ndata);

void pet_c_lmosem_dda_att(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
						  unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
						  unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
						  float* im, int nim1, int nim2, int nim3,
						  float* F, int nf1, int nf2, int nf3,
						  float* A, int na1, int na2, int na3,
						  int wim, int ndata);

// OPLEM reconstruction
void pet_c_oplem_dda(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
					 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
					 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
					 float* im, int nim1, int nim2, int nim3,
					 float* NM, int nm1, int nm2, int nm3,
					 int nsub);

void pet_c_oplem_dda_att(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
						 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
						 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
						 float* im, int nim1, int nim2, int nim3,
						 float* NM, int nm1, int nm2, int nm3,
						 float* AM, int am1, int am2, int am3,
						 int nsub);

void pet_c_oplem_sid(float* X1, int nx1, float* Y1, int ny1,
					 float* Z1, int nz1, float* X2, int nx2,
					 float* Y2, int ny2, float* Z2, int nz2,
					 float* im, int nim1, int nim2, int nim3,
					 float* NM, int nm1, int nm2, int nm3,
					 int nsub, int border);

void pet_c_oplem_sid_att(float* X1, int nx1, float* Y1, int ny1,
						 float* Z1, int nz1, float* X2, int nx2,
						 float* Y2, int ny2, float* Z2, int nz2,
						 float* im, int nim1, int nim2, int nim3,
						 float* NM, int nm1, int nm2, int nm3,
						 float* AM, int am1, int am2, int am3,
						 int nsub, int border);
