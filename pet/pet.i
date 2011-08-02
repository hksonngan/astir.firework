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

%module pet_c

%{
#define SWIG_FILE_WITH_INIT
#include "pet.h"	
%}

%include "numpy.i"

%init %{
import_array();
%}

// Open a phase-space
void pet_c_phasespace_open(char* filename,
						   int* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1);


// Allegro scanner
void pet_c_allegro_idtopos(int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
						   int* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
						   float respix, int sizespacexy, int sizespacez, int rnd);

void pet_c_allegro_build_all_lor(unsigned short int* INPLACE_ARRAY1, int DIM1,
								 unsigned short int* INPLACE_ARRAY1, int DIM1,
								 unsigned short int* INPLACE_ARRAY1, int DIM1,
								 unsigned short int* INPLACE_ARRAY1, int DIM1);

void pet_c_allegro_save_rnd_lor(char* savename, int nlor);

// Discorvery scanner
void pet_c_discovery_blftobin(char* blffilename, char* binfilename);

void pet_c_discovery_idtopos(int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
							 float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							 float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
							 int* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							 float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
							 float respix, int sizespacexy, int sizespacez, int rnd);

void pet_c_discovery_save_rnd_lor(char* savename, int nlor);

// Utils
void pet_c_raycasting(float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
					  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
					  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
					  int* INPLACE_ARRAY1, int DIM1, int border, int ROIxy, int ROIz);

void pet_c_clean_lor_int(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1,	int DIM1,
						 float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						 float* IN_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
						 int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
						 int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1);

void pet_c_clean_lor_float(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						   float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						   float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
						   float* INPLACE_ARRAY1, int DIM1);

void pet_c_listmode_open_subset_xyz_int(unsigned short int* INPLACE_ARRAY1, int DIM1,
										unsigned short int* INPLACE_ARRAY1, int DIM1,
										unsigned short int* INPLACE_ARRAY1, int DIM1,
										unsigned short int* INPLACE_ARRAY1, int DIM1,
										unsigned short int* INPLACE_ARRAY1, int DIM1,
										unsigned short int* INPLACE_ARRAY1, int DIM1,
										int n_start, int n_stop, char* basename);

void pet_c_listmode_open_subset_xyz_int_sort(unsigned short int* INPLACE_ARRAY1, int DIM1,
											 unsigned short int* INPLACE_ARRAY1, int DIM1,
											 unsigned short int* INPLACE_ARRAY1, int DIM1,
											 unsigned short int* INPLACE_ARRAY1, int DIM1,
											 unsigned short int* INPLACE_ARRAY1, int DIM1,
											 unsigned short int* INPLACE_ARRAY1, int DIM1,
											 int* IN_ARRAY1, int DIM1, int n_start, int n_stop, char* basename);

void pet_c_listmode_open_subset_xyz_float(float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
										  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
										  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
										  int n_start, int n_stop, char* basename);

void pet_c_listmode_open_subset_id_int(int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
									   int* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
									   int n_start, int n_stop, char* name);

// LM-OSEM reconstruction
void pet_c_lmosem_dda(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
					  unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
					  unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
					  float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
					  float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
					  int wim, int ndata);

void pet_c_lmosem_dda_att(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						  unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						  unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						  float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
						  float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
						  float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
						  int wim, int ndata);

void pet_c_lmosem_siddon(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						 float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						 float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
						 int wim, int dim, int border);

void pet_c_lmosem_siddon_att(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
							 float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
							 float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
							 int wim, int dim, int border);

// OPLEM reconstruction
void pet_c_oplem_dda(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
					 unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
					 unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
					 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
					 float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
					 int Nsub);

void pet_c_oplem_dda_att(unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						 unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						 unsigned short int* IN_ARRAY1, int DIM1, unsigned short int* IN_ARRAY1, int DIM1,
						 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
						 float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
						 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
						 int Nsub);

void pet_c_oplem_sid(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
					 float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
					 float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
					 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
					 float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
					 int nsub, int border);

void pet_c_oplem_sid_att(float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						 float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						 float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
						 float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
						 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
						 int nsub, int border);
