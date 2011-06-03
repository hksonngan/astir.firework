/* -*- C -*-  (not really, but good for syntax highlighting) */
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

%module firekernel

%{
#define SWIG_FILE_WITH_INIT
#include "pet_c.h"	
%}

%include "numpy.i"

%init %{
import_array();
%}

// Open a phase-space
void kernel_phasespace_open(char* filename,
							int* INPLACE_ARRAY1, int DIM1,
							float* INPLACE_ARRAY1, int DIM1,
							float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);


// PET 3D Allegro
void kernel_allegro_idtopos(int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
							float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
							int* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							float respix, int sizespacexy, int sizespacez, int rnd);

void kernel_allegro_build_all_LOR(unsigned short int* INPLACE_ARRAY1, int DIM1,
								  unsigned short int* INPLACE_ARRAY1, int DIM1,
								  unsigned short int* INPLACE_ARRAY1, int DIM1,
								  unsigned short int* INPLACE_ARRAY1, int DIM1);

void kernel_allegro_save_rnd_LOR(char* savename, int nlor);

// PET 3D Discorvery
void kernel_discovery_blftobin(char* blffilename, char* binfilename);
void kernel_discovery_idtopos(int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
							  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							  float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
							  int* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							  float respix, int sizespacexy, int sizespacez, int rnd);

void kernel_discovery_save_rnd_LOR(char* savename, int nlor);

// PET 3D
void kernel_pet3D_SRM_raycasting(float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
								 float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
								 float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
								 int* INPLACE_ARRAY1, int DIM1, int border, int ROIxy, int ROIz);

void kernel_pet3D_SRM_clean_LOR_int(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1,	int DIM1,
									float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
									float* IN_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
									int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
									int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1);

void kernel_pet3D_SRM_clean_LOR_float(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
									  float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
									  float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
									  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
									  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
									  float* INPLACE_ARRAY1, int DIM1);

void kernel_listmode_open_subset_xyz_int(unsigned short int* INPLACE_ARRAY1, int DIM1,
										 unsigned short int* INPLACE_ARRAY1, int DIM1,
										 unsigned short int* INPLACE_ARRAY1, int DIM1,
										 unsigned short int* INPLACE_ARRAY1, int DIM1,
										 unsigned short int* INPLACE_ARRAY1, int DIM1,
										 unsigned short int* INPLACE_ARRAY1, int DIM1,
										 int n_start, int n_stop, char* basename);

void kernel_listmode_open_subset_xyz_int_sort(unsigned short int* INPLACE_ARRAY1, int DIM1,
											  unsigned short int* INPLACE_ARRAY1, int DIM1,
											  unsigned short int* INPLACE_ARRAY1, int DIM1,
											  unsigned short int* INPLACE_ARRAY1, int DIM1,
											  unsigned short int* INPLACE_ARRAY1, int DIM1,
											  unsigned short int* INPLACE_ARRAY1, int DIM1,
											  int* IN_ARRAY1, int DIM1, int n_start, int n_stop, char* basename);


void kernel_listmode_open_subset_xyz_float(float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
										   float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
										   float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
										   int n_start, int n_stop, char* basename);

void kernel_listmode_open_subset_ID_int(int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
										int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
										int n_start, int n_stop, char* name);

// PET 3D LM-OSEM
void kernel_pet3D_LMOSEM_dda(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
							 unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
							 unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
							 float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
							 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
							 int wim, int ndata);

void kernel_pet3D_LMOSEM_dda_att(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
								 unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
								 unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
								 float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
								 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
								 float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
								 int wim, int ndata);

void kernel_pet3D_LMOSEM_siddon(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
								float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
								float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
								int wim, int dim, int border);

void kernel_pet3D_LMOSEM_siddon_att(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
									float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
									float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
									int wim, int dim, int border);

// PET 3D OPLEM
void kernel_pet3D_OPLEM(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
						float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
						int Nsub);

void kernel_pet3D_OPLEM_att(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
							unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
							unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
							float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
							float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
							float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
							int Nsub);

void kernel_pet3D_OPLEM_sid(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
							float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
							float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
							float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
							float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
							int nsub, int border);

void kernel_pet3D_OPLEM_sid_att(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
								float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
								float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
								float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
								float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
								float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
								int nsub, int border);
