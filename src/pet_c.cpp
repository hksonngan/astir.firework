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
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <kernel_c.h>

/********************************************************************************
 * Phase-Space
 ********************************************************************************/
/*
void kernel_phasespace_open(char* filename,
							char* type, int ntype,
							float* E, int nE,
							float* px, int npx, float* py, int npy, float* pz, int npz,
							float* dx, int ndx, float* dy, int ndy, float* dz, int ndz,
							float* w, int nw) {	

	FILE * pfile = fopen(filename, "rb");

	for (int i = 0; i < nbele; ++i) {
		
	}

	dim_phantom.z = 46;
	dim_phantom.y = 63;
	dim_phantom.x = 128;
	float size_voxel = 4.0f;  // used latter
	int nb = dim_phantom.z * dim_phantom.y * dim_phantom.x;
	unsigned int mem_phantom = nb * sizeof(unsigned short int);
	unsigned short int* phantom = (unsigned short int*)malloc(mem_phantom);
	fread(phantom, sizeof(unsigned short int), nb, pfile);
	fclose(pfile);
}
*/

/********************************************************************************
 * PET Scan Allegro      
 ********************************************************************************/

#define pi  3.141592653589
#define twopi 6.283185307179
// Convert ID event from allegro scanner to global position in 3D space
void kernel_allegro_idtopos(int* id_crystal1, int nidc1, int* id_detector1, int nidd1,
							float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
							int* id_crystal2, int nidc2, int* id_detector2, int nidd2,
							float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
							float respix, int sizespacexy, int sizespacez, int rnd) {
	// NOTE: ref system will be changed, from ref GATE system to image system
	// GATE             SPACE
	//    X             Z
	//    |__ Z         /_ X
	//   /             |
	//  Y              Y
	
	// cst system
	int nic = 22;      // number of crystals along i
	int njc = 29;      // number of crystals along j
	int nd  = 28;      // number of detectors
	float dcz = 6.3;   // delta position of crystal along z (mm)
	float dcx = 4.3;   // delta position of crystal along x (mm)
	float rcz = 88.2;  // org translation of coordinate along z (mm)
	float rcx = 45.15; // org translation of coordinate along x (mm)
	float tsc = 432;   // translation scanner detector along y (mm)
	//float ROI = 565;   // ROI scanner is 565 mm (GANTRY of 56.5 cm)
	//float cxyimage = ROI / respix / 2.0; // center of ROI image along x and y (square image)
	//float czimage  = rcz / respix;       // center of ROI image along z (volume)
	float cxyimage = (float)sizespacexy / 2.0f;
	float czimage = (float)sizespacez / 2.0f;
	float xi, yi, zi, a, newx, newy, newz;
	float cosa, sina;
	float dex, dez;
	int n, ID;
	if (rnd) {printf("Random pos\n");}
	else {printf("No random pos\n");}
	// to add fluctuation (due to DDA line drawing)
	if (rnd) {srand(rnd);}
	for (n=0; n<nidc1; ++n) {
		// ID1
		////////////////////////////////
		// global position in GATE space
		ID = id_crystal1[n];
		zi = float(ID / nic) * dcz - rcz;
		xi = float(ID % nic) * dcx - rcx;
		yi = tsc;
		// random position to the crystal aera
		if (rnd) {
			//inkernel_randg2f(0.0, 0.25, &dex, &dez);
			dex = 4.0f * inkernel_randf() - 2.0f;
			dez = 4.0f * inkernel_randf() - 2.0f;
			xi += dex;
			zi += dez;
		}
		// rotation accoring ID detector
		a = (float)id_detector1[n] * (-twopi / (float)nd) - pi / 2.0f;
		cosa = cos(a);
		sina = sin(a);
		newx = xi*cosa - yi*sina;
		newy = xi*sina + yi*cosa;
		// change to image org
		newx += cxyimage;           // change origin (left upper corner)
		newy  = (-newy) + cxyimage; // inverse y axis
		newz  = zi + czimage;
		newx /= respix;             // scale factor to match with ROI (image)
		newy /= respix;
		newz /= respix;
		x1[n] = newx;
		y1[n] = newy;
		z1[n] = newz;
		// ID2
		////////////////////////////////
		// global position in GATE space
		ID = id_crystal2[n];
		zi = float(ID / nic) * dcz - rcz;
		xi = float(ID % nic) * dcx - rcx;
		yi = tsc;
		// random position to the crystal aera
		if (rnd) {
			//inkernel_randg2f(0.0, 0.25, &dex, &dez);
			dex = 4.0f * inkernel_randf() - 2.0f;
			dez = 4.0f * inkernel_randf() - 2.0f;
			xi += dex;
			zi += dez;
		}
		// rotation accoring ID detector
		a = (float)id_detector2[n] * (-twopi / (float)nd) - pi / 2.0f;
		cosa = cos(a);
		sina = sin(a);
		newx = xi*cosa - yi*sina;
		newy = xi*sina + yi*cosa;
		// change to image org
		newx += cxyimage;           // change origin (left upper corner)
		newy  = (-newy) + cxyimage; // inverse y axis
		newz  = zi + czimage;
		newx /= respix;             // scale factor to match with ROI (image)
		newy /= respix;
		newz /= respix;
		x2[n] = newx;
		y2[n] = newy;
		z2[n] = newz;
	}
}
#undef pi
#undef twopi

// build the list of all LOR in order to compute S matrix
void kernel_allegro_build_all_LOR(unsigned short int* idc1, int n1, unsigned short int* idd1, int n2,
								  unsigned short int* idc2, int n3, unsigned short int* idd2, int n4) {
	int idmax = 22*29;
	int ndete = 28;
	int N = idmax*ndete;
	int Ntot = (N*N - N) / 2;
	int i1, i2;
	unsigned int dc1, dd1, dc2, dd2;
	int ct=0;

	for (i1=0; i1<(N-1); ++i1) {
		dd1 = i1 / idmax;
		dc1 = i1 % idmax;
		for (i2=i1+1; i2<N; ++i2) {
			dd2 = i2 / idmax;
			dc2 = i2 % idmax;
			idc1[ct] = dc1;
			idd1[ct] = dd1;
			idc2[ct] = dc2;
			idd2[ct] = dd2;
			++ct;
		}
	}
}

// build a random list of LOR in order to compute normalize matrix of Allegro scanner
void kernel_allegro_save_rnd_LOR(char* savename, int nlor) {
	
	FILE * pfile_lors;
	int cmax = 22*29;
	int dmax = 28;
	int c1, d1, c2, d2;
	int n;
	unsigned int sizeint = sizeof(int);
	
	pfile_lors = fopen(savename, "wb");
	for (n=0; n<nlor; ++n) {
		c1 = (int) (inkernel_randf() * cmax);
		d1 = (int) (inkernel_randf() * dmax);
		c2 = (int) (inkernel_randf() * cmax);
		d2 = (int) (inkernel_randf() * dmax);
		fwrite(&c1, sizeint, 1, pfile_lors);
		fwrite(&d1, sizeint, 1, pfile_lors);
		fwrite(&c2, sizeint, 1, pfile_lors);
		fwrite(&d2, sizeint, 1, pfile_lors);
	}
	fclose(pfile_lors);
	
}


/********************************************************************************
 * PET Scan GE Discovery      
 ********************************************************************************/

// Convert blf file from GE scanner to ID (crystals and modules)
// blf format word of 32 bits
// 0         1         2         3
// 01234567890123456789012345678901
// x|        ||   |||        ||   | coincidence (if 0)
//  xxxxxxxxxx|   |||        ||   | ID 1
//            xxxxx||        ||   | Ring 1
//                 x|        ||   | prompt or delay (1 or 0)
//                  xxxxxxxxxx|   | ID 2
//                            xxxxx ring 2
void kernel_discovery_blftobin(char* blffilename, char* binfilename) {
	// vars
	unsigned int word;
	int M1, R1, C1, ID1;
	int M2, R2, C2, ID2;
	int nid;
	int n, c;
	int p;
	// init file
	FILE * pfile_blf;
	FILE * pfile_bin;
	pfile_blf = fopen(blffilename, "rb");
	pfile_bin = fopen(binfilename, "wb");
	// Read number of words
	fseek(pfile_blf, 0, SEEK_END);
	nid = ftell(pfile_blf);
	rewind(pfile_blf);
	nid /= 4;
	printf("nid %i\n", nid);
	c = 0;
	// Reading loop
	for (n=0; n<nid; ++n) {
		// read a word
		fread(&word, sizeof(word), 1, pfile_blf);
		// check if coincidence
		if ((word & 0x80000000) >> 31 == 1) {continue;}
		// extract ID1
		ID1 = (word & 0x7fe00000) >> 21;
		// extract R1
		R1 = (word & 0x001f0000) >> 16;
		if (R1 >= 24) {continue;}
		// extract p
		p = (word & 0x00008000) >> 15;
		// extract ID2
		ID2 = (word & 0x00007ff0) >> 5;
		// extract R2
		R2 = (word & 0x0000001f);
		if (R2 >= 24) {continue;}
		// convert
		M1 = ID1 / 16;
		M2 = ID2 / 16;
		if (M1 >= 35 || M2 >= 35) {continue;}
		C1 = ID1 - M1 * 16;
		C2 = ID2 - M2 * 16;
		if (C1 >= 16 || C2 >= 16) {continue;}
		C1 += (16 * R1);
		C2 += (16 * R2);
		// save in binary format
		fwrite(&C1, sizeof(int), 1, pfile_bin);
		fwrite(&M1, sizeof(int), 1, pfile_bin);
		fwrite(&C2, sizeof(int), 1, pfile_bin);
		fwrite(&M2, sizeof(int), 1, pfile_bin);
		c++;

	}
	printf("tot %i\n", c);
	fclose(pfile_blf);
	fclose(pfile_bin);
}

#define pi  3.141592653589
#define twopi 6.283185307179
// Convert ID event from GE DSTE scanner to global position in 3D space 
void kernel_discovery_idtopos(int* id_crystal1, int nidc1, int* id_detector1, int nidd1,
							  float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
							  int* id_crystal2, int nidc2, int* id_detector2, int nidd2,
							  float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
							  float respix, int sizespacexy, int sizespacez, int rnd) {
	// SPACE
	//   Z
	//  /_ X
	// |
	// Y 
	
	// cst system
	int nic = 16;      // number of crystals along i
	int njc = 24;      // number of crystals along j
	int nd  = 35;      // number of detectors
	float dcz = 6.52;  // delta position of crystal along z (mm)
	float dcx = 4.9;   // delta position of crystal along x (mm)
	float rcz = 78.24; // org translation of coordinate along z (mm)
	float rcx = 39.2;  // org translation of coordinate along x (mm)
	float tsc = 440.5; // translation scanner detector along y (mm)
	float cxyimage = (float)sizespacexy / 2.0f;
	float czimage = (float)sizespacez / 2.0f;
	float xi, yi, zi, a, newx, newy, newz;
	float cosa, sina;
	float dex, dez;
	int n, ID;
	if (rnd) {printf("Random pos\n");}
	else {printf("No random pos\n");}
	// to add fluctuation (due to DDA line drawing)
	if (rnd) {srand(rnd);}
	for (n=0; n<nidc1; ++n) {
		// ID1
		////////////////////////////////
		// global position in GATE space
		ID = id_crystal1[n];
		zi = float(ID / nic) * dcz - rcz;
		xi = float(ID % nic) * dcx - rcx;
		yi = tsc;
		// random position to the crystal aera
		if (rnd) {
			//inkernel_randg2f(0.0, 0.25, &dex, &dez);
			dex = 4.0f * inkernel_randf() - 2.0f;
			dez = 4.0f * inkernel_randf() - 2.0f;
			xi += dex;
			zi += dez;
		}
		// rotation accoring ID detector
		//a = (float)id_detector1[n] * (-twopi / (float)nd) - pi / 2.0f;
		a = (float)id_detector1[n] * (-twopi / (float)nd);
		cosa = cos(a);
		sina = sin(a);
		newx = xi*cosa - yi*sina;
		newy = xi*sina + yi*cosa;
		// change to image org
		newx += cxyimage;           // change origin (left upper corner)
		newy  = (-newy) + cxyimage; // inverse y axis
		//newy += cxyimage;
		newz  = zi + czimage;
		newx /= respix;             // scale factor to match with ROI (image)
		newy /= respix;
		newz /= respix;
		x1[n] = newx;
		y1[n] = newy;
		z1[n] = newz;
		// ID2
		////////////////////////////////
		// global position in GATE space
		ID = id_crystal2[n];
		zi = float(ID / nic) * dcz - rcz;
		xi = float(ID % nic) * dcx - rcx;
		yi = tsc;
		// random position to the crystal aera
		if (rnd) {
			//inkernel_randg2f(0.0, 0.25, &dex, &dez);
			dex = 4.0f * inkernel_randf() - 2.0f;
			dez = 4.0f * inkernel_randf() - 2.0f;
			xi += dex;
			zi += dez;
		}
		// rotation accoring ID detector
		//a = (float)id_detector2[n] * (-twopi / (float)nd) - pi / 2.0f;
		a = (float)id_detector2[n] * (-twopi / (float)nd);
		cosa = cos(a);
		sina = sin(a);
		newx = xi*cosa - yi*sina;
		newy = xi*sina + yi*cosa;
		// change to image org
		newx += cxyimage;           // change origin (left upper corner)
		newy  = (-newy) + cxyimage; // inverse y axis
		//newy += cxyimage;
		newz  = zi + czimage;
		newx /= respix;             // scale factor to match with ROI (image)
		newy /= respix;
		newz /= respix;
		x2[n] = newx;
		y2[n] = newy;
		z2[n] = newz;
	}
}
#undef pi
#undef twopi

// build a random list of LOR in order to compute normalize matrix of Discovery scanner
void kernel_discovery_save_rnd_LOR(char* savename, int nlor) {
	
	FILE * pfile_lors;
	int cmax = 16*24;
	int dmax = 35;
	int c1, d1, c2, d2;
	int n;
	unsigned int sizeint = sizeof(int);
	
	pfile_lors = fopen(savename, "wb");
	for (n=0; n<nlor; ++n) {
		c1 = (int) (inkernel_randf() * cmax);
		d1 = (int) (inkernel_randf() * dmax);
		c2 = (int) (inkernel_randf() * cmax);
		d2 = (int) (inkernel_randf() * dmax);
		fwrite(&c1, sizeint, 1, pfile_lors);
		fwrite(&d1, sizeint, 1, pfile_lors);
		fwrite(&c2, sizeint, 1, pfile_lors);
		fwrite(&d2, sizeint, 1, pfile_lors);
	}
	fclose(pfile_lors);
	
}

/********************************************************************************
 * PET Scan       
 ********************************************************************************/

// SRM Raycasting, Compute ray intersection with the 3D SRM
void kernel_pet3D_SRM_raycasting(float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
								 float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
								 int* enable, int nenable, int border, int ROIxy, int ROIz) {
	// Smith's algorithm ray-box AABB intersection
	int i, chk1, chk2;
	float xd, yd, zd, xmin, ymin, zmin, xmax, ymax, zmax;
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	float xi1, yi1, zi1, xp1, yp1, zp1, xp2, yp2, zp2;
	// define box and ray direction
	xmin = float(border);
	xmax = float(border + ROIxy);
	ymin = float(border);
	ymax = float(border + ROIxy);
	zmin = 0.0f;
	zmax = float(ROIz);
	for (i=0; i<nx1; ++i) {
		xi1 = x1[i];
		yi1 = y1[i];
		zi1 = z1[i];
		xd = x2[i] - xi1;
		yd = y2[i] - yi1;
		zd = z2[i] - zi1;
		tmin = -1e9f;
		tmax = 1e9f;
		// on x
		if (xd != 0.0f) {
			tmin = (xmin - xi1) / xd;
			tmax = (xmax - xi1) / xd;
			if (tmin > tmax) {
				buf = tmin;
				tmin = tmax;
				tmax = buf;
			}
		}
		// on y
		if (yd != 0.0f) {
			tymin = (ymin - yi1) / yd;
			tymax = (ymax - yi1) / yd;
			if (tymin > tymax) {
				buf = tymin;
				tymin = tymax;
				tymax = buf;
			}
			if (tymin > tmin) {tmin = tymin;}
			if (tymax < tmax) {tmax = tymax;}
		}
		// on z
		if (zd != 0.0f) {
			tzmin = (zmin - zi1) / zd;
			tzmax = (zmax - zi1) / zd;
			if (tzmin > tzmax) {
				buf = tzmin;
				tzmin = tzmax;
				tzmax = buf;
			}
			if (tzmin > tmin) {tmin = tzmin;}
			if (tzmax < tmax) {tmax = tzmax;}
		}
		// compute points
		xp1 = xi1 + xd * tmin;
		yp1 = yi1 + yd * tmin;
		zp1 = zi1 + zd * tmin;
		xp2 = xi1 + xd * tmax;
		yp2 = yi1 + yd * tmax;
		zp2 = zi1 + zd * tmax;
		//printf("p1 %f %f %f - p2 %f %f %f\n", xp1, yp1, zp1, xp2, yp2, zp2);
		// check point p1
		chk1 = 0;
		chk2 = 0;
		if (xp1 >= xmin && xp1 <= xmax) {
			if (yp1 >= ymin && yp1 <= ymax) {
				if (zp1 >= zmin && zp1 <= zmax) {
					xp1 -= border;
					yp1 -= border;
					//zp1 -= border;
					if (int(xp1+0.5) == ROIxy) {xp1 = ROIxy-1.0f;}
					if (int(yp1+0.5) == ROIxy) {yp1 = ROIxy-1.0f;}
					if (int(zp1+0.5) == ROIz) {zp1 = ROIz-1.0f;}
					x1[i] = xp1;
					y1[i] = yp1;
					z1[i] = zp1;
					chk1 = 1;
				} else {continue;}
			} else {continue;}
		} else {continue;}
		// check point p2
		if (xp2 >= xmin && xp2 <= xmax) {
			if (yp2 >= ymin && yp2 <= ymax) {
				if (zp2 >= zmin && zp2 <= zmax) {
					xp2 -= border;
					yp2 -= border;
					//zp2 -= border;
					if (int(xp2+0.5) == ROIxy) {xp2 = ROIxy-1.0f;}
					if (int(yp2+0.5) == ROIxy) {yp2 = ROIxy-1.0f;}
					if (int(zp2+0.5) == ROIz) {zp2 = ROIz-1.0f;}
					x2[i] = xp2;
					y2[i] = yp2;
					z2[i] = zp2;
					chk2 = 1;
				} else {continue;}
			} else {continue;}
		} else {continue;}
		if (chk1 && chk2) {
			if (int(xp1) == int(xp2) && int(yp1) == int(yp2) && int(zp1) == int(zp2)) {continue;}
			enable[i] = 1;
		}
	}
}

// Cleanning LORs outside of ROI based on SRM raycasting intersection calculation (return int)
void kernel_pet3D_SRM_clean_LOR_int(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
									float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
									int* xi1, int nxi1, int* yi1, int nyi1, int* zi1, int nzi1,
									int* xi2, int nxi2, int* yi2, int nyi2, int* zi2, int nzi2) {
	int i, c;
	c = 0;
	for (i=0; i<nx1; ++i) {
		if (enable[i]) {
			xi1[c] = (int)(x1[i]);
			yi1[c] = (int)(y1[i]);
			zi1[c] = (int)(z1[i]);
			xi2[c] = (int)(x2[i]);
			yi2[c] = (int)(y2[i]);
			zi2[c] = (int)(z2[i]);
			++c;
		}
	}
}

// Cleanning LORs outside of ROI based on SRM raycasting intersection calculation (return float)
void kernel_pet3D_SRM_clean_LOR_float(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
									  float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
									  float* xf1, int nxi1, float* yf1, int nyi1, float* zf1, int nzi1,
									  float* xf2, int nxi2, float* yf2, int nyi2, float* zf2, int nzi2) {
	int i, c;
	c = 0;
	for (i=0; i<nx1; ++i) {
		if (enable[i]) {
			xf1[c] = x1[i];
			yf1[c] = y1[i];
			zf1[c] = z1[i];
			xf2[c] = x2[i];
			yf2[c] = y2[i];
			zf2[c] = z2[i];
			++c;
		}
	}
}

// Read a subset of list-mode data set (int data).
void kernel_listmode_open_subset_xyz_int(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1, 
										 unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										 int n_start, int n_stop, char* basename) {

	// init file
	FILE * pfile_x1;
	FILE * pfile_y1;
	FILE * pfile_z1;
	FILE * pfile_x2;
	FILE * pfile_y2;
	FILE * pfile_z2;
	char namex1 [100];
	char namey1 [100];
	char namez1 [100];
	char namex2 [100];
	char namey2 [100];
	char namez2 [100];
	sprintf(namex1, "%s.x1", basename);
	sprintf(namey1, "%s.y1", basename);
	sprintf(namez1, "%s.z1", basename);
	sprintf(namex2, "%s.x2", basename);
	sprintf(namey2, "%s.y2", basename);
	sprintf(namez2, "%s.z2", basename);
	pfile_x1 = fopen(namex1, "rb");
	pfile_y1 = fopen(namey1, "rb");
	pfile_z1 = fopen(namez1, "rb");
	pfile_x2 = fopen(namex2, "rb");
	pfile_y2 = fopen(namey2, "rb");
	pfile_z2 = fopen(namez2, "rb");
	// position file
	long int pos = n_start * sizeof(unsigned short int);
	fseek(pfile_x1, pos, SEEK_SET);
	fseek(pfile_y1, pos, SEEK_SET);
	fseek(pfile_z1, pos, SEEK_SET);
	fseek(pfile_x2, pos, SEEK_SET);
	fseek(pfile_y2, pos, SEEK_SET);
	fseek(pfile_z2, pos, SEEK_SET);

	// read data
	int i;
	unsigned short int xi1, yi1, zi1, xi2, yi2, zi2;
	int N = n_stop - n_start;
	for (i=0; i<N; ++i) {
		fread(&xi1, 1, sizeof(unsigned short int), pfile_x1);
		fread(&yi1, 1, sizeof(unsigned short int), pfile_y1);
		fread(&zi1, 1, sizeof(unsigned short int), pfile_z1);
		fread(&xi2, 1, sizeof(unsigned short int), pfile_x2);
		fread(&yi2, 1, sizeof(unsigned short int), pfile_y2);
		fread(&zi2, 1, sizeof(unsigned short int), pfile_z2);
		x1[i] = xi1;
		y1[i] = yi1;
		z1[i] = zi1;
		x2[i] = xi2;
		y2[i] = yi2;
		z2[i] = zi2;
	}
	// close files
	fclose(pfile_x1);
	fclose(pfile_y1);
	fclose(pfile_z1);
	fclose(pfile_x2);
	fclose(pfile_y2);
	fclose(pfile_z2);

}

// Read a subset of list-mode data set (int data) and sort according ID vectors (usefull to shuflle LORs).
void kernel_listmode_open_subset_xyz_int_sort(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1, 
											  unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
											  int* ID, int nid, int n_start, int n_stop, char* basename) {

	// init file
	FILE * pfile_x1;
	FILE * pfile_y1;
	FILE * pfile_z1;
	FILE * pfile_x2;
	FILE * pfile_y2;
	FILE * pfile_z2;
	char namex1 [100];
	char namey1 [100];
	char namez1 [100];
	char namex2 [100];
	char namey2 [100];
	char namez2 [100];
	sprintf(namex1, "%s.x1", basename);
	sprintf(namey1, "%s.y1", basename);
	sprintf(namez1, "%s.z1", basename);
	sprintf(namex2, "%s.x2", basename);
	sprintf(namey2, "%s.y2", basename);
	sprintf(namez2, "%s.z2", basename);
	pfile_x1 = fopen(namex1, "rb");
	pfile_y1 = fopen(namey1, "rb");
	pfile_z1 = fopen(namez1, "rb");
	pfile_x2 = fopen(namex2, "rb");
	pfile_y2 = fopen(namey2, "rb");
	pfile_z2 = fopen(namez2, "rb");
	// position file
	long int pos = n_start * sizeof(unsigned short int);
	fseek(pfile_x1, pos, SEEK_SET);
	fseek(pfile_y1, pos, SEEK_SET);
	fseek(pfile_z1, pos, SEEK_SET);
	fseek(pfile_x2, pos, SEEK_SET);
	fseek(pfile_y2, pos, SEEK_SET);
	fseek(pfile_z2, pos, SEEK_SET);

	// read data
	int i;
	unsigned short int xi1, yi1, zi1, xi2, yi2, zi2;
	int N = n_stop - n_start;
	unsigned short int* xt1 = (unsigned short int*)malloc(N * sizeof(unsigned short int));
	unsigned short int* yt1 = (unsigned short int*)malloc(N * sizeof(unsigned short int));
	unsigned short int* zt1 = (unsigned short int*)malloc(N * sizeof(unsigned short int));
	unsigned short int* xt2 = (unsigned short int*)malloc(N * sizeof(unsigned short int));
	unsigned short int* yt2 = (unsigned short int*)malloc(N * sizeof(unsigned short int));
	unsigned short int* zt2 = (unsigned short int*)malloc(N * sizeof(unsigned short int));
	for (i=0; i<N; ++i) {
		fread(&xi1, 1, sizeof(unsigned short int), pfile_x1);
		fread(&yi1, 1, sizeof(unsigned short int), pfile_y1);
		fread(&zi1, 1, sizeof(unsigned short int), pfile_z1);
		fread(&xi2, 1, sizeof(unsigned short int), pfile_x2);
		fread(&yi2, 1, sizeof(unsigned short int), pfile_y2);
		fread(&zi2, 1, sizeof(unsigned short int), pfile_z2);
		xt1[i] = xi1;
		yt1[i] = yi1;
		zt1[i] = zi1;
		xt2[i] = xi2;
		yt2[i] = yi2;
		zt2[i] = zi2;
	}
	// sort LORs according ID vectors
	i=0;
	while(i<N) {
		x1[i] = xt1[ID[i]];
		y1[i] = yt1[ID[i]];
		z1[i] = zt1[ID[i]];
		x2[i] = xt2[ID[i]];
		y2[i] = yt2[ID[i]];
		z2[i] = zt2[ID[i]];
		++i;
	}
	// close files
	fclose(pfile_x1);
	fclose(pfile_y1);
	fclose(pfile_z1);
	fclose(pfile_x2);
	fclose(pfile_y2);
	fclose(pfile_z2);
	// free mem
	free(xt1);
	free(yt1);
	free(zt1);
	free(xt2);
	free(yt2);
	free(zt2);
}


// Read a subset of list-mode data set (float data).
void kernel_listmode_open_subset_xyz_float(float* x1, int nx1, float* y1, int ny1, float* z1, int nz1, 
										   float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
										   int n_start, int n_stop, char* basename) {

	// init file
	FILE * pfile_x1;
	FILE * pfile_y1;
	FILE * pfile_z1;
	FILE * pfile_x2;
	FILE * pfile_y2;
	FILE * pfile_z2;
	char namex1 [100];
	char namey1 [100];
	char namez1 [100];
	char namex2 [100];
	char namey2 [100];
	char namez2 [100];
	sprintf(namex1, "%s.x1", basename);
	sprintf(namey1, "%s.y1", basename);
	sprintf(namez1, "%s.z1", basename);
	sprintf(namex2, "%s.x2", basename);
	sprintf(namey2, "%s.y2", basename);
	sprintf(namez2, "%s.z2", basename);
	pfile_x1 = fopen(namex1, "rb");
	pfile_y1 = fopen(namey1, "rb");
	pfile_z1 = fopen(namez1, "rb");
	pfile_x2 = fopen(namex2, "rb");
	pfile_y2 = fopen(namey2, "rb");
	pfile_z2 = fopen(namez2, "rb");
	// position file
	long int pos = n_start * sizeof(float);
	fseek(pfile_x1, pos, SEEK_SET);
	fseek(pfile_y1, pos, SEEK_SET);
	fseek(pfile_z1, pos, SEEK_SET);
	fseek(pfile_x2, pos, SEEK_SET);
	fseek(pfile_y2, pos, SEEK_SET);
	fseek(pfile_z2, pos, SEEK_SET);

	// read data
	int i;
	float xf1, yf1, zf1, xf2, yf2, zf2;
	int N = n_stop - n_start;
	for (i=0; i<N; ++i) {
		fread(&xf1, 1, sizeof(float), pfile_x1);
		fread(&yf1, 1, sizeof(float), pfile_y1);
		fread(&zf1, 1, sizeof(float), pfile_z1);
		fread(&xf2, 1, sizeof(float), pfile_x2);
		fread(&yf2, 1, sizeof(float), pfile_y2);
		fread(&zf2, 1, sizeof(float), pfile_z2);
		x1[i] = xf1;
		y1[i] = yf1;
		z1[i] = zf1;
		x2[i] = xf2;
		y2[i] = yf2;
		z2[i] = zf2;
	}
	// close files
	fclose(pfile_x1);
	fclose(pfile_y1);
	fclose(pfile_z1);
	fclose(pfile_x2);
	fclose(pfile_y2);
	fclose(pfile_z2);

}

// Read a subset of list-mode data set (Id of crystals and detectors).
void kernel_listmode_open_subset_ID_int(int* idc1, int n1, int* idd1, int n2, int* idc2, int n3, int* idd2, int n4,
										int n_start, int n_stop, char* name) {

	// init file
	FILE * pfile;
	pfile = fopen(name, "rb");
	// position file
	long int pos = 4 * n_start * sizeof(int);
	fseek(pfile, pos, SEEK_SET);
	// read data
	int i;
	int c1, c2, d1, d2;
	int N = n_stop - n_start;
	for (i=0; i<N; ++i) {
		fread(&c1, 1, sizeof(int), pfile);
		fread(&d1, 1, sizeof(int), pfile);
		fread(&c2, 1, sizeof(int), pfile);
		fread(&d2, 1, sizeof(int), pfile);
		idc1[i] = c1;
		idd1[i] = d1;
		idc2[i] = c2;
		idd2[i] = d2;
	}
	// close files
	fclose(pfile);

}

/********************************************************************************
 * 3D LM-OSEM       
 ********************************************************************************/

// Update image online, SRM is build with DDA's Line Algorithm in fixed point, store in ELL format and update with LM-OSEM
#define CONST int(pow(2, 23))
#define float2fixed(X) ((int) X * CONST)
#define intfixed(X) (X >> 23)
void kernel_pet3D_LMOSEM_dda(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
							 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
							 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
							 float* im, int nim1, int nim2, int nim3,
							 float* F, int nf1, int nf2, int nf3, int wim, int ndata) {
	
	int length, lengthy, lengthz, i, j, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	int fxinc, fyinc, fzinc, fx, fy, fz;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	step = wim*wim;

	// alloc mem
	//float* vals = (float*)malloc(ndata * sizeof(float));
	int* cols = (int*)malloc(ndata * sizeof(int));
	int LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi;

	for (i=0; i< nx1; ++i) {
		Qi = 0.0f;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		z1 = Z1[i];
		z2 = Z2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		diffz = z2-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = 1.0f / (float)length;
		fxinc = float2fixed(diffx * flength);
		fyinc = float2fixed(diffy * flength);
		fzinc = float2fixed(diffz * flength);
		fx = float2fixed(x1);
		fy = float2fixed(y1);
		fz = float2fixed(z1);
		for (n=0; n<length; ++n) {
			//vals[n] = 1.0f;
			vcol = intfixed(fz) * step + intfixed(fy) * wim + intfixed(fx);
			cols[n] = vcol;
			Qi += im[vcol];
			fx = fx + fxinc;
			fy = fy + fyinc;
			fz = fz + fzinc;
		}
		// eof
		//vals[n] = -1;
		cols[n] = -1;
		// compute F
		if (Qi==0.0f) {continue;}
		Qi = 1.0f / Qi;
		vcol = cols[0];
		j = 0;
		while (vcol != -1) {
			F[vcol] += Qi;
			++j;
			vcol = cols[j];
			}
	}
	//free(vals);
	free(cols);
}
#undef CONST
#undef float2fixed
#undef intfixed

// Update image online, SRM is build with DDA's Line Algorithm in fixed point,
// store in ELL format and update with LM-OSEM and corrected the attenuation
#define CONST int(pow(2, 23))
#define float2fixed(X) ((int) X * CONST)
#define intfixed(X) (X >> 23)
void kernel_pet3D_LMOSEM_dda_att(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
								 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
								 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
								 float* im, int nim1, int nim2, int nim3,
								 float* F, int nf1, int nf2, int nf3,
								 float* A, int na1, int na2, int na3,
								 int wim, int ndata) {
	
	int length, lengthy, lengthz, i, j, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	int fxinc, fyinc, fzinc, fx, fy, fz;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	step = wim*wim;

	// alloc mem
	//float* vals = (float*)malloc(ndata * sizeof(float));
	int* cols = (int*)malloc(ndata * sizeof(int));
	int LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi, Ai;

	for (i=0; i< nx1; ++i) {
		Qi = 0.0f;
		Ai = 0.0f;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		z1 = Z1[i];
		z2 = Z2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		diffz = z2-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = 1.0f / (float)length;
		fxinc = float2fixed(diffx * flength);
		fyinc = float2fixed(diffy * flength);
		fzinc = float2fixed(diffz * flength);
		fx = float2fixed(x1);
		fy = float2fixed(y1);
		fz = float2fixed(z1);
		for (n=0; n<length; ++n) {
			//vals[n] = 1.0f;
			vcol = intfixed(fz) * step + intfixed(fy) * wim + intfixed(fx);
			cols[n] = vcol;
			Qi += im[vcol];
			Ai -= A[vcol];
			fx = fx + fxinc;
			fy = fy + fyinc;
			fz = fz + fzinc;
		}
		// eof
		//vals[n] = -1;
		cols[n] = -1;
		// compute F
		if (Qi==0.0f) {continue;}
		Qi = Qi * exp(Ai / 2.0f);
		Qi = 1.0f / Qi;
		vcol = cols[0];
		j = 0;
		while (vcol != -1) {
			F[vcol] += Qi;
			++j;
			vcol = cols[j];
			}
	}
	//free(vals);
	free(cols);
}
#undef CONST
#undef float2fixed
#undef intfixed

// Update image online, SRM is build with Siddon's Line Algorithm in COO format, and update with LM-OSEM
void kernel_pet3D_LMOSEM_siddon(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
								float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
								float* im, int nim, float* F, int nf, int wim, int dim, int border) {
	int n, ct;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;
	double Qi;
	float* vals = NULL;
	int* cols = NULL;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		float* vals = NULL;
		int* cols = NULL;
		Qi = 0.0f;
		ct = 0;
		// draw the line
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		px -= border;
		py -= border;
		qx -= border;
		qy -= border;
		initl = inkernel_randf();
		//initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		initl = initl * 0.4 + 0.3; // rnd number between 0.3 to 0.7
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
		if (ei < 0.0f || ei >= wim || ej < 0.0f || ej >= wim || ek < 0.0f || ek >= dim) {continue;}
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qz-tz>0) {
			w=ek+1;
			stepk=1;
		}
		if (qz-tz<0) {
			w=ek;
			stepk=-1;
		}
		if (qz-tz==0) {
			w=ej;
			stepk=0;
		}
		
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		if (qz==pz) {divz=1.0;}
		else {divz = float(qz-pz);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		azstart = (w-pz) / divz;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		if (azstart > astart) {astart = azstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)+(qz-pz)*(qz-pz));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		stepz = fabs(pq / divz);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		if (stepz < valmax) {valmax = stepz;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz == startl) {
			k += stepk;
			runz += stepz;
		}
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<dim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				++ct;
				vals = (float*)realloc(vals, ct*sizeof(float));
				cols = (int*)realloc(cols, ct*sizeof(int));
				vals[ct-1] = val;
				cols[ct-1] = k * wim2 + j * wim + i;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		if (pz-tz>0) {stepk=1;}
		if (pz-tz<0) {stepk=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz==startl) {
			k += stepk;
			runz += stepz;
		}
		++ct;
		vals = (float*)realloc(vals, ct*sizeof(float));
		cols = (int*)realloc(cols, ct*sizeof(int));
		vals[ct-1] = 0.707f;
		cols[ct-1] = ek * wim2 + ej * wim + ei;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<dim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				++ct;
				vals = (float*)realloc(vals, ct*sizeof(float));
				cols = (int*)realloc(cols, ct*sizeof(int));
				vals[ct-1] = val;
				cols[ct-1] = k * wim2 + j * wim + i;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// first compute Qi
		for (i=0; i<ct; ++i) {Qi += (vals[i] * im[cols[i]]);}
		if (Qi == 0.0f) {continue;}
		// accumulate to F
		for(i=0; i<ct; ++i) {
			if (im[cols[i]] != 0.0f) {
				F[cols[i]] += (vals[i] / Qi);
			}

		}
		free(vals);
		free(cols);
		
	} // LORs loop
	
}

// Update image online, SRM is build with Siddon's Line Algorithm in COO format, and update with LM-OSEM
// Use attenuation correction
void kernel_pet3D_LMOSEM_siddon_att(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
									float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
									float* im, int nim, float* F, int nf, float* mumap, int nmu,
									int wim, int dim, int border) {

	int n, ct;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;
	float Qi, Ai;
	float* vals = NULL;
	int* cols = NULL;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		float* vals = NULL;
		int* cols = NULL;
		Qi = 0.0f;
		Ai = 0.0f;
		ct = 0;
		// draw the line
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		px -= border;
		py -= border;
		qx -= border;
		qy -= border;
		initl = inkernel_randf();
		//initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		initl = initl * 0.4 + 0.3; // rnd number between 0.3 to 0.7
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
		if (ei < 0.0f || ei >= wim || ej < 0.0f || ej >= wim || ek < 0.0f || ek >= dim) {continue;}
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qz-tz>0) {
			w=ek+1;
			stepk=1;
		}
		if (qz-tz<0) {
			w=ek;
			stepk=-1;
		}
		if (qz-tz==0) {
			w=ej;
			stepk=0;
		}
		
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		if (qz==pz) {divz=1.0;}
		else {divz = float(qz-pz);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		azstart = (w-pz) / divz;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		if (azstart > astart) {astart = azstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)+(qz-pz)*(qz-pz));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		stepz = fabs(pq / divz);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		if (stepz < valmax) {valmax = stepz;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz == startl) {
			k += stepk;
			runz += stepz;
		}
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<dim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				++ct;
				vals = (float*)realloc(vals, ct*sizeof(float));
				cols = (int*)realloc(cols, ct*sizeof(int));
				vals[ct-1] = val;
				cols[ct-1] = k * wim2 + j * wim + i;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		if (pz-tz>0) {stepk=1;}
		if (pz-tz<0) {stepk=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz==startl) {
			k += stepk;
			runz += stepz;
		}
		++ct;
		vals = (float*)realloc(vals, ct*sizeof(float));
		cols = (int*)realloc(cols, ct*sizeof(int));
		vals[ct-1] = 0.707f;
		cols[ct-1] = ek * wim2 + ej * wim + ei;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<dim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				++ct;
				vals = (float*)realloc(vals, ct*sizeof(float));
				cols = (int*)realloc(cols, ct*sizeof(int));
				vals[ct-1] = val;
				cols[ct-1] = k * wim2 + j * wim + i;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// first compute Qi
		for (i=0; i<ct; ++i) {Qi += (vals[i] * im[cols[i]]);}
		if (Qi == 0.0f) {continue;}
		// second compute Ai
		for (i=0; i<ct; ++i) {Ai -= (vals[i] * mumap[cols[i]]);}
		Qi = Qi * exp(Ai / 2.0f);
		// accumulate to F
		for(i=0; i<ct; ++i) {
			if (im[cols[i]] != 0.0f) {
				F[cols[i]] += (vals[i] / Qi);
			}
		}
		free(vals);
		free(cols);
		
	} // LORs loop
}

/********************************************************************************
 * 3D OPLEM       
 ********************************************************************************/

// OPLEM: DDA's Line Algorithm in fixed point and memory handling with ELLPACK format
#define CONST int(pow(2, 23))
#define float2fixed(X) ((int) X * CONST)
#define intfixed(X) (X >> 23)
void kernel_pet3D_OPLEM(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
						unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
						unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
						float* im, int nim1, int nim2, int nim3,
						float* NM, int nm1, int nm2, int nm3,
						int nsub) {
	
	// vars DDA
	int length, lengthy, lengthz, ilor, n;
	float flength, Qi;
	float lx, ly, lz;
	int fxinc, fyinc, fzinc, fx, fy, fz;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step = nim2 * nim3;
	int ndata, ind;
	if (nim1 > nim2) {ndata = nim1;}
	if (nim3 > ndata) {ndata = nim3;}
	ndata += ndata / 3;
	int* buf = (int*)malloc(ndata * sizeof(int));
	step = nim2*nim3;

	// vars subset
	int lor_start, lor_stop, nlor, isub;
	int nvox = step * nim1;
	unsigned int mem_size_F = nvox * sizeof(float);
	float* F = (float*)malloc(mem_size_F);

	// prepare normalize matrix
	for (n=0; n<nvox; ++n) {NM[n] = 1.0f / NM[n];}

	// sub loop
	for (isub=0; isub<nsub; ++isub) {
		// boundary lor
		lor_start = int(float(nx1) / nsub * isub + 0.5f);
		lor_stop = int(float(nx1) / nsub * (isub+1) + 0.5f);
		nlor = lor_stop - lor_start;
		// init F
		memset(F, 0, mem_size_F);
		// DDA-ELL ray-projector
		for (ilor=lor_start; ilor<lor_stop; ++ilor) {
			Qi = 0.0f;
			x1 = X1[ilor];
			x2 = X2[ilor];
			y1 = Y1[ilor];
			y2 = Y2[ilor];
			z1 = Z1[ilor];
			z2 = Z2[ilor];
			diffx = x2-x1;
			diffy = y2-y1;
			diffz = z2-z1;
			lx = abs(diffx);
			ly = abs(diffy);
			lz = abs(diffz);
			length = ly;
			if (lx > length) {length = lx;}
			if (lz > length) {length = lz;}
			flength = 1.0f / (float)length;
			fxinc = float2fixed(diffx * flength);
			fyinc = float2fixed(diffy * flength);
			fzinc = float2fixed(diffz * flength);
			fx = float2fixed(x1);
			fy = float2fixed(y1);
			fz = float2fixed(z1);
			for (n=0; n<length; ++n) {
				ind = intfixed(fz) * step + intfixed(fy) * nim3 + intfixed(fx);
				buf[n] = ind;
				Qi += im[ind];
				fx = fx + fxinc;
				fy = fy + fyinc;
				fz = fz + fzinc;
			}
			// compute F
			if (Qi==0.0f) {continue;}
			Qi = 1.0f / Qi;
			for (n=0; n<length; ++n) {F[buf[n]] += Qi;}
		} // ilor
		// update im
		for (n=0; n<nvox; ++n) {im[n] = im[n] * F[n] * NM[n];}
	} // isub
	// release mem
	free(buf);
	free(F);
}
#undef CONST
#undef float2fixed
#undef intfixed

// OPLEM: DDA's Line Algorithm in fixed point, memory handling with ELLPACK format, and attenuation correction
#define CONST int(pow(2, 23))
#define float2fixed(X) ((int) X * CONST)
#define intfixed(X) (X >> 23)
void kernel_pet3D_OPLEM_att(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
							unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
							unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
							float* im, int nim1, int nim2, int nim3,
							float* NM, int nm1, int nm2, int nm3,
							float* AM, int am1, int am2, int am3,
							int nsub) {
	
	// vars DDA
	int length, lengthy, lengthz, ilor, n;
	float flength, Qi, Ai;
	float lx, ly, lz;
	int fxinc, fyinc, fzinc, fx, fy, fz;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step = nim2 * nim3;
	int ndata, ind;
	if (nim1 > nim2) {ndata = nim1;}
	if (nim3 > ndata) {ndata = nim3;}
	ndata += ndata / 3;
	int* buf = (int*)malloc(ndata * sizeof(int));
	step = nim2*nim3;

	// vars subset
	int lor_start, lor_stop, nlor, isub;
	int nvox = step * nim1;
	unsigned int mem_size_F = nvox * sizeof(float);
	float* F = (float*)malloc(mem_size_F);

	// prepare normalize matrix
	for (n=0; n<nvox; ++n) {NM[n] = 1.0f / NM[n];}

	// sub loop
	for (isub=0; isub<nsub; ++isub) {
		// boundary lor
		lor_start = int(float(nx1) / nsub * isub + 0.5f);
		lor_stop = int(float(nx1) / nsub * (isub+1) + 0.5f);
		nlor = lor_stop - lor_start;
		// init F
		memset(F, 0, mem_size_F);
		// DDA-ELL ray-projector
		for (ilor=lor_start; ilor<lor_stop; ++ilor) {
			Qi = 0.0f;
			Ai = 0.0f;
			x1 = X1[ilor];
			x2 = X2[ilor];
			y1 = Y1[ilor];
			y2 = Y2[ilor];
			z1 = Z1[ilor];
			z2 = Z2[ilor];
			diffx = x2-x1;
			diffy = y2-y1;
			diffz = z2-z1;
			lx = abs(diffx);
			ly = abs(diffy);
			lz = abs(diffz);
			length = ly;
			if (lx > length) {length = lx;}
			if (lz > length) {length = lz;}
			flength = 1.0f / (float)length;
			fxinc = float2fixed(diffx * flength);
			fyinc = float2fixed(diffy * flength);
			fzinc = float2fixed(diffz * flength);
			fx = float2fixed(x1);
			fy = float2fixed(y1);
			fz = float2fixed(z1);
			for (n=0; n<length; ++n) {
				ind = intfixed(fz) * step + intfixed(fy) * nim3 + intfixed(fx);
				buf[n] = ind;
				Qi += im[ind];
				Ai -= AM[ind];
				fx = fx + fxinc;
				fy = fy + fyinc;
				fz = fz + fzinc;
			}
			// compute F
			if (Qi==0.0f) {continue;}
			Qi = Qi * exp(Ai / 2.0f);
			Qi = 1.0f / Qi;
			for (n=0; n<length; ++n) {F[buf[n]] += Qi;}
		} // ilor
		// update im
		for (n=0; n<nvox; ++n) {im[n] = im[n] * F[n] * NM[n];}
	} // isub
	// release mem
	free(buf);
	free(F);
}
#undef CONST
#undef float2fixed
#undef intfixed
