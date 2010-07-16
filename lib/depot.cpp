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
#include <math.h>
#include "kernel_cuda.h"

void omp_vec_square(float* data, int n) {
	int i;
	#pragma omp parallel for shared(data) private(i)
	for(i=0; i<n; ++i) {data[i] = data[i] * data[i];}
}

void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float gamma, float thres){
	int ind, n, x, y, z;
	float l;
	for (n=0; n<nval; ++n) {
		ind = 3 * n;
		x = posxyz[ind];
		y = posxyz[ind+1];
		z = posxyz[ind+2];
		l = val[n];
		if (l <= thres) {continue;}
		l *= gamma;
		glColor4f(1.0, 1.0, 1.0, l);
		// face 0
		glBegin(GL_QUADS);
		glNormal3f(-1, 0, 0);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		// face 1
		glBegin(GL_QUADS);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		// face 2 
		glBegin(GL_QUADS);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		// face 3
		glBegin(GL_QUADS);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		// face 4
		glBegin(GL_QUADS);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		// face 5
		glBegin(GL_QUADS);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
		
	}
	glColor4f(1.0, 1.0, 1.0, 1.0);
}

void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float thres){
	int ind, n, x, y, z;
	float l;
	for (n=0; n<nval; ++n) {
		ind = 3 * n;
		x = posxyz[ind];
		y = posxyz[ind+1];
		z = posxyz[ind+2];
		l = val[n];
		if (l <= thres) {continue;}
		// face 0
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(-1, 0, 0);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		// face 1
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		// face 2
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		// face 3
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		// face 4
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		// face 5
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
	}
	glColor4f(1.0, 1.0, 1.0, 1.0);
}

// fill the system response matrix according the LOR
void kernel_build_2D_SRM_BLA(float* SRM, int sy, int sx, int* LOR_val, int nval, int* lines, int nvec, int wx) {
	int l, x1, y1, x2, y2, val, ind, offset;
	int x, y, dx, dy, xinc, yinc, balance;

	for (l=0; l<nval; ++l) {
		ind = 4 * l;
		offset = sx * l;
		x1 = lines[ind];
		y1 = lines[ind+1];
		x2 = lines[ind+2];
		y2 = lines[ind+3];

		if (x2 >= x1) {
			dx = x2 - x1;
			xinc = 1;
		} else {
			dx = x1 - x2;
			xinc = -1;
		}
		if (y2 >= y1) {
			dy = y2 - y1;
			yinc = 1;
		} else {
			dy = y1 - y2;
			yinc = -1;
		}
		x = x1;
		y = y1;
		if (dx >= dy) {
			dy <<= 1;
			balance = dy - dx;
			dx <<= 1;
			while (x != x2) {
				SRM[offset + y * wx + x] = LOR_val[l];
				if (balance >= 0) {
					y = y + yinc;
					balance = balance - dx;
				}
				balance = balance + dy;
				x = x + xinc;
			}
			SRM[offset + y * wx + x] = LOR_val[l];
		} else {
			dx <<= 1;
			balance = dx - dy;
			dy <<= 1;
			while (y != y2) {
				SRM[offset + y * wx + x] = LOR_val[l];
				if (balance >= 0) {
					x = x + xinc;
					balance = balance - dy;
				}
				balance = balance + dx;
				y = y + yinc;
			}
			SRM[offset + y * wx + x] = LOR_val[l];
		}
	}
}

#define pi 3.141592653589
// simulate a gamma photon in PET four detectors
//    # # #
//  # o o o #
//  # o x o #
//  # o o o #
//    # # #
void kernel_pet2D_square_gen_sim_ID(int* RES, int nres, float posx, float posy, float alpha, int nx) {
	double g1_x, g2_x, g1_y, g2_y, incx, incy;
	int id1, id2;
	g1_x = (double)posx;
	g2_x = (double)posx;
	g1_y = (double)posy;
	g2_y = (double)posy;
	alpha = (double)alpha;
	if ((alpha >= (pi / 4.0)) && (alpha <= (3 * pi / 4.0))) {
		incx = cos(alpha);
		incy = 1;
		while (1) {
			g1_x += incx;
			g1_y -= incy;
			if (g1_x <= 0.0) {
				id1 = 4 * nx - (int)g1_y - 1;
				break;
			}
			if (g1_x >= nx) {
				id1 = nx + (int)g1_y + 1;
				break;
			}
			if (g1_y <= 0.0) {
				id1 = (int)g1_x + 1;
				break;
			}					
		}
		while (1) {
			g2_x -= incx;
			g2_y += incy;
			if (g2_x >= nx) {
				id2 = nx + (int)g2_y + 1;
				break;
			}
			if (g2_x <= 0.0) {
				id2 = 4 * nx - (int)g2_y + 1;
				break;
			}
			if (g2_y >= nx) {
				id2 = 3 * nx - (int)g2_x + 1;
				break;
			}
		} 
	} else {
		if (alpha >= (3 * pi / 4.0)) {incx = -1;}
		else {incx = 1;}
		incy = sin(alpha);
		while (1) {
			g1_x += incx;
			g1_y -= incy;
			if (g1_x <= 0) {
				id1 = 4 * nx - (int)g1_y + 1;
				break;
			}
			if (g1_x >= nx) {
				id1 = nx + (int)g1_y + 1;
				break;
			}
			if (g1_y <= 0) {
				id1 = (int)g1_x + 1;
				break;
			}
		}
		while (1) {
			g2_x -= incx;
			g2_y += incy;
			if (g2_x >= nx) {
				id2 = nx + (int)g2_y + 1;
				break;
			}
			if (g2_x <= 0.0) {
				id2 = 4 * nx - (int)g2_y + 1;
				break;
			}
			if (g2_y >= nx) {
				id2 = 3 * nx - (int)g2_x + 1;
				break;
			}
		}
	}
	RES[0] = id1;
	RES[1] = id2;
}
#undef pi

/**************************************************************
 * 2D PET SCAN      Simulated ring scanner
 **************************************************************/

// use to fill the SRM in order to compute the sensibility matrix
void kernel_pet2D_ring_build_SM(float* SRM, int sy, int sx, int x1, int y1, int x2, int y2, int nx, int numlor) {
	int offset;
	int x, y, dx, dy, xinc, yinc, balance;

	offset = sx * numlor;
	if (x2 >= x1) {
		dx = x2 - x1;
		xinc = 1;
	} else {
		dx = x1 - x2;
		xinc = -1;
	}
	if (y2 >= y1) {
		dy = y2 - y1;
		yinc = 1;
	} else {
		dy = y1 - y2;
		yinc = -1;
	}
	x = x1;
	y = y1;
	if (dx >= dy) {
		dy <<= 1;
		balance = dy - dx;
		dx <<= 1;
		while (x != x2) {
			SRM[offset + y * nx + x] = 1.0;
			if (balance >= 0) {
				y = y + yinc;
				balance = balance - dx;
			}
			balance = balance + dy;
			x = x + xinc;
		}
		SRM[offset + y * nx + x] = 1.0;
	} else {
		dx <<= 1;
		balance = dx - dy;
		dy <<= 1;
		while (y != y2) {
			SRM[offset + y * nx + x] = 1.0;
			if (balance >= 0) {
				x = x + xinc;
				balance = balance - dy;
			}
			balance = balance + dx;
			y = y + yinc;
		}
		SRM[offset + y * nx + x] = 1.0;
	}
}

#define pi 3.141592653589
// simulate a gamma photon in 2D PET ring detectors
void kernel_pet2D_ring_gen_sim_ID(int* RES, int nres, int posx, int posy, float alpha, int radius) {
	double dx, dy, b, c, d, k0, k1;
	double x1, y1, x2, y2;
	int id1, id2;
	dx = cos(alpha);
	dy = sin(alpha);
	b  = 2 * (dx*(posx-radius) + dy*(posy-radius));  // radius = cxo = cyo
	c  = 2*radius*radius + posx*posx + posy*posy - 2*(radius*posx + radius*posy) - radius*radius;
	d  = b*b - 4*c;
	k0 = (-b + sqrt(d)) / 2.0;
	k1 = (-b - sqrt(d)) / 2.0;
	x1 = posx + k0*dx;
	y1 = posy + k0*dy;
	x2 = posx + k1*dx;
	y2 = posy + k1*dy;
	// convert xy to id crystal
	dx = x1 - radius;
	dy = y1 - radius;
	if (abs((int)dx) > abs((int)dy)) {
		alpha = asin(dy / (double)radius);
		if (alpha < 0) {  // asin return -pi/2 < . < pi/2
			if (dx < 0) {alpha = pi - alpha;}
			else {alpha = 2*pi + alpha;}
		}
		else {
			if (dx < 0) {alpha = pi - alpha;} // mirror according y axe
		}
	} else {
		alpha = acos(dx / (double)radius);
		if (dy < 0) {alpha = 2*pi - alpha;} // mirror according x axe
	}
	id1 = int(radius * alpha + 0.5); // id crystal is the arc
	dx = x2 - radius;
	dy = y2 - radius;
	if (abs((int)dx) > abs((int)dy)) {
		alpha = asin(dy / (double)radius);
		if (alpha < 0) {  // asin return -pi/2 < . < pi/2
			if (dx < 0) {alpha = pi - alpha;}
			else {alpha = 2*pi + alpha;}
		}
		else {
			if (dx < 0) {alpha = pi - alpha;} // mirror according y axe
		}
	} else {
		alpha = acos(dx / (double)radius);
		if (dy < 0) {alpha = 2*pi - alpha;} // mirror according x axe
	}
	id2 = int(radius * alpha + 0.5); // id crystal is the arc
	RES[0] = id1;
	RES[1] = id2;
}
#undef pi

#define pi 3.141592653589
// fill the system response matrix according the LOR (binary mode)
void kernel_pet2D_ring_LOR_SRM_BLA(float* SRM, int sy, int sx, int* LOR_val, int nval, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals) {
	int l, x1, y1, x2, y2, val, ind, offset;
	int x, y, dx, dy, xinc, yinc, balance;
	double alpha, coef;
	double radius = (double)int(nbcrystals / 2.0 / pi + 0.5);
	int wx = 2*radius+1;

	for (l=0; l<nval; ++l) {
		ind = 4 * l;
		offset = sx * l;
		// convert id crystal to x, y
		alpha = (double)ID1[l] / radius;
		x1 = int(radius + radius * cos(alpha) + 0.5);
		y1 = int(radius + radius * sin(alpha) + 0.5);
		alpha = (double)ID2[l] / radius;
		x2 = int(radius + radius * cos(alpha) + 0.5);
		y2 = int(radius + radius * sin(alpha) + 0.5);
		// integral line must be equal to one
		coef = 1.0 / sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
		// drawing line
		if (x2 >= x1) {
			dx = x2 - x1;
			xinc = 1;
		} else {
			dx = x1 - x2;
			xinc = -1;
		}
		if (y2 >= y1) {
			dy = y2 - y1;
			yinc = 1;
		} else {
			dy = y1 - y2;
			yinc = -1;
		}
		x = x1;
		y = y1;
		if (dx >= dy) {
			dy <<= 1;
			balance = dy - dx;
			dx <<= 1;
			while (x != x2) {
				SRM[offset + y * wx + x] += (LOR_val[l] * coef);
				if (balance >= 0) {
					y = y + yinc;
					balance = balance - dx;
				}
				balance = balance + dy;
				x = x + xinc;
			}
			SRM[offset + y * wx + x] += (LOR_val[l] * coef);
		} else {
			dx <<= 1;
			balance = dx - dy;
			dy <<= 1;
			while (y != y2) {
				SRM[offset + y * wx + x] += (LOR_val[l] * coef);
				if (balance >= 0) {
					x = x + xinc;
					balance = balance - dy;
				}
				balance = balance + dx;
				y = y + yinc;
			}
			SRM[offset + y * wx + x] += (LOR_val[l] * coef);
		}
	}
}
#undef pi

#define pi 3.141592653589
// Raytracing with BLA method: fill the system response matrix according the events (list-mode)
int kernel_pet2D_ring_LM_SRM_BLA(float* SRM, int sy, int sx, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals) {
	int l, x1, y1, x2, y2, ind, offset;
	int x, y, dx, dy, xinc, yinc, balance;
	double alpha, coef;
	double radius = (double)int(nbcrystals / 2.0 / pi + 0.5);
	int wx = 2*radius+1;
	int ct = 0;

	for (l=0; l<nid1; ++l) {
		ind = 4 * l;
		offset = sx * l;
		// convert id crystal to x, y
		alpha = (double)ID1[l] / radius;
		x1 = int(radius + radius * cos(alpha) + 0.5);
		y1 = int(radius + radius * sin(alpha) + 0.5);
		alpha = (double)ID2[l] / radius;
		x2 = int(radius + radius * cos(alpha) + 0.5);
		y2 = int(radius + radius * sin(alpha) + 0.5);
		// integral line must be equal to one
		coef = 1.0 / sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
		// drawing line
		if (x2 >= x1) {
			dx = x2 - x1;
			xinc = 1;
		} else {
			dx = x1 - x2;
			xinc = -1;
		}
		if (y2 >= y1) {
			dy = y2 - y1;
			yinc = 1;
		} else {
			dy = y1 - y2;
			yinc = -1;
		}
		x = x1;
		y = y1;
		if (dx >= dy) {
			dy <<= 1;
			balance = dy - dx;
			dx <<= 1;
			while (x != x2) {
				SRM[offset + y * wx + x] += coef;
				++ct;
				if (balance >= 0) {
					y = y + yinc;
					balance = balance - dx;
				}
				balance = balance + dy;
				x = x + xinc;
			}
			SRM[offset + y * wx + x] += coef;
			++ct;
		} else {
			dx <<= 1;
			balance = dx - dy;
			dy <<= 1;
			while (y != y2) {
				SRM[offset + y * wx + x] += coef;
				++ct;
				if (balance >= 0) {
					x = x + xinc;
					balance = balance - dy;
				}
				balance = balance + dx;
				y = y + yinc;
			}
			SRM[offset + y * wx + x] += coef;
			++ct;
		}
	}
	return ct;
}
#undef pi

#define pi 3.141592653589
// Raytracing with DDA method: fill the system response matrix according the events (list-mode)
int kernel_pet2D_ring_LM_SRM_DDA(float* SRM, int sy, int sx, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals) {
	int l, x1, y1, x2, y2, ind, offset;
	double x, y, xinc, yinc;
	double alpha, coef;
	double radius = (double)int(nbcrystals / 2.0 / pi + 0.5);
	int wx = 2*radius+1;
	int ct = 0;
	int length, i;

	for (l=0; l<nid1; ++l) {
		ind = 4 * l;
		offset = sx * l;
		// convert id crystal to x, y
		alpha = (double)ID1[l] / radius;
		x1 = int(radius + radius * cos(alpha) + 0.5);
		y1 = int(radius + radius * sin(alpha) + 0.5);
		alpha = (double)ID2[l] / radius;
		x2 = int(radius + radius * cos(alpha) + 0.5);
		y2 = int(radius + radius * sin(alpha) + 0.5);
		// integral line must be equal to one
		coef = 1.0 / sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
		// draw the line
		length = abs(x2- x1);
		ct += length;
		if (abs(y2 - y1) > length) {length = abs(y2 - y1);}
		xinc = (double)(x2 - x1) / (double) length;
		yinc = (double)(y2 - y1) / (double) length;
		x    = x1 + 0.5;
		y    = y1 + 0.5;
		for (i=0; i<=length; ++i) {
			SRM[offset + (int)y * wx + (int)x] += coef;
			x = x + xinc;
			y = y + yinc;
		}
	}
	return ct;
}
#undef pi

// Raytracing with WALA method: fill the system response matrix according the events (list-mode)
#define ipart_(X) ((int) X)
#define round_(X) ((int)(((double)(X)) + 0.5))
#define fpart_(X) ((double)(X) - (double)ipart_(X))
#define rfpart_(X) (1.0 - fpart_(X))
#define swap_(a, b) do{ __typeof__(a) tmp; tmp = a; a = b; b = tmp; }while(0)
#define pi 3.141592653589
void kernel_pet2D_ring_LM_SRM_WALA(float* SRM, int sy, int sx, int* ID1, int nid1, int* ID2, int nid2, int nbcrystals) {
	int l, x1, y1, x2, y2, ind, offset;
	double alpha, coef;
	double radius = (double)int(nbcrystals / 2.0 / pi + 0.5);
	int wx = 2*radius+1;

	for (l=0; l<nid1; ++l) {
		ind = 4 * l;
		offset = sx * l;
		// convert id crystal to x, y
		alpha = (double)ID1[l] / radius;
		x1 = int(radius + radius * cos(alpha) + 0.5);
		y1 = int(radius + radius * sin(alpha) + 0.5);
		alpha = (double)ID2[l] / radius;
		x2 = int(radius + radius * cos(alpha) + 0.5);
		y2 = int(radius + radius * sin(alpha) + 0.5);
		// integral line must be equal to one
		coef = 1.0 / sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
		// draw the line
		double dx = (double)x2 - (double)x1;
		double dy = (double)y2 - (double)y1;
		if (fabs(dx) > fabs(dy)) {
			if (x2 < x1) {
				swap_(x1, x2);
				swap_(y1, y2);
			}
		
			double gradient = dy / dx;
			double xend = round_(x1);
			double yend = y1 + gradient * (xend - x1);
			double xgap = rfpart_(x1 + 0.5);
			int xpxl1 = xend;
			int ypxl1 = ipart_(yend);
			SRM[offset + ypxl1 * wx + xpxl1] += (rfpart_(yend) * xgap * coef);
			SRM[offset + (ypxl1 + 1) * wx + xpxl1] += (fpart_(yend) * xgap * coef);
			double intery = yend + gradient;
		
			xend = round_(x2);
			yend = y2 + gradient*(xend - x2);
			xgap = fpart_(x2+0.5);
			int xpxl2 = xend;
			int ypxl2 = ipart_(yend);
			SRM[offset + ypxl2 * wx + xpxl2] += (rfpart_(yend) * xgap * coef);
			SRM[offset + (ypxl2 + 1) * wx + xpxl2] += (fpart_(yend) * xgap * coef);
			int x;
			for (x=xpxl1+1; x <= (xpxl2-1); x++) {
				SRM[offset + ipart_(intery) * wx + x] += (rfpart_(intery) * coef);
				SRM[offset + (ipart_(intery) + 1) * wx + x] += (fpart_(intery) * coef);
				intery += gradient;
			}
		} else {
			if (y2 < y1) {
				swap_(x1, x2);
				swap_(y1, y2);
			}
			double gradient = dx / dy;
			double yend = round_(y1);
			double xend = x1 + gradient*(yend - y1);
			double ygap = rfpart_(y1 + 0.5);
			int ypxl1 = yend;
			int xpxl1 = ipart_(xend);
			SRM[offset + ypxl1 * wx + xpxl1] += (rfpart_(xend) * ygap * coef);
			SRM[offset + (ypxl1 + 1) * wx + xpxl1] += (fpart_(xend) * ygap * coef);
			double interx = xend + gradient;

			yend = round_(y2);
			xend = x2 + gradient*(yend - y2);
			ygap = fpart_(y2+0.5);
			int ypxl2 = yend;
			int xpxl2 = ipart_(xend);
			SRM[offset + ypxl2 * wx + xpxl2] += (rfpart_(xend) * ygap * coef);
			SRM[offset + (ypxl2 + 1) * wx + xpxl2] += (fpart_(xend) * ygap * coef);

			int y;
			for(y=ypxl1+1; y <= (ypxl2-1); y++) {
				SRM[offset + y * wx + ipart_(interx)] += (rfpart_(interx) * coef);
				SRM[offset + y * wx + ipart_(interx) + 1] += (fpart_(interx) * coef);
				interx += gradient;
			}
		}

		
	}

}
#undef pi
#undef swap_
#undef ipart_
#undef fpart_
#undef round_
#undef rfpart_
