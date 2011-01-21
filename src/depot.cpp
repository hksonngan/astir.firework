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

// SRM Raytracing (transversal algorithm), Compute entry and exit point on SRM of the ray
void kernel_pet2D_SRM_entryexit(float* px, int npx, float* py, int npy, float* qx, int nqx, float* qy, int nqy, int b, int srmsize, int* enable, int nenable) {
	float divx, divy, fsrmsize;
	float axn, ax0, ayn, ay0;
	float amin, amax, buf1, buf2;
	float x1, y1, x2, y2;
	float pxi, pyi, qxi, qyi;
	int i;
		
	b = (float)b;
	fsrmsize = (float)srmsize;

	for (i=0; i<npx; ++i) {
		pxi = px[i];
		pyi = py[i];
		qxi = qx[i];
		qyi = qy[i];
		
		if (pxi == qxi) {divx = 1.0;}
		else {divx = pxi - qxi;}
		if (pyi == qyi) {divy = 1.0;}
		else {divy = pyi - qyi;}
		axn = (fsrmsize + b - qxi) / divx;
		ax0 = (b - qxi) / divx;
		ayn = (fsrmsize + b - qyi) / divy;
		ay0 = (b - qyi) / divy;

		buf1 = ax0;
		if (axn < ax0) {buf1 = axn;}
		buf2 = ay0;
		if (ayn < ay0) {buf2 = ayn;}
		amin = buf2;
		if (buf1 > buf2) {amin = buf1;}
		buf1 = ax0;
		if (axn > ax0) {buf1 = axn;}
		buf2 = ay0;
		if (ayn > ay0) {buf2 = ayn;}
		amax = buf2;
		if (buf1 < buf2) {amax = buf1;}

		x1 = (qxi + amax * (pxi - qxi) - b);
		y1 = (qyi + amax * (pyi - qyi) - b);
		x2 = (qxi + amin * (pxi - qxi) - b);
		y2 = (qyi + amin * (pyi - qyi) - b);

		// format
		if (x1 == fsrmsize) {x1 = fsrmsize-1.0f;}
		if (y1 == fsrmsize) {y1 = fsrmsize-1.0f;}
		if (x2 == fsrmsize) {x2 = fsrmsize-1.0f;}
		if (y2 == fsrmsize) {y2 = fsrmsize-1.0f;}
		// check if ray through the image
		enable[i] = 1;
		if (x1 < 0 || x1 > fsrmsize-1 || y1 < 0 || y1 > fsrmsize-1) {enable[i] = 0;}
		if (x2 < 0 || x2 > fsrmsize-1 || y2 < 0 || y2 > fsrmsize-1) {enable[i] = 0;}
		// check if the ray is > 0
		if (int(x1) == int(x2) && int(y1) == int(y2)) {enable[i] = 0;}
		px[i] = x1;
		py[i] = y1;
		qx[i] = x2;
		qy[i] = y2;
	}
}

// Cleanning LORs outside of ROi based on SRM entry-exit point calculation (return int)
void kernel_pet2D_SRM_clean_entryexit_int(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
									  int* xi1, int nxi1, int* yi1, int nyi1, int* xi2, int nxi2, int* yi2, int nyi2) {
	int i, c;
	c = 0;
	for (i=0; i<nx1; ++i) {
		if (enable[i]) {
			xi1[c] = (int)x1[i];
			yi1[c] = (int)y1[i];
			xi2[c] = (int)x2[i];
			yi2[c] = (int)y2[i];
			++c;
		}
	}
}
// Cleanning LORs outside of ROi based on SRM entry-exit point calculation (return float)
void kernel_pet2D_SRM_clean_entryexit_float(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
									  float* xf1, int nxf1, float* yf1, int nyf1, float* xf2, int nxf2, float* yf2, int nyf2) {
	int i, c;
	c = 0;
	for (i=0; i<nx1; ++i) {
		if (enable[i]) {
			xf1[c] = x1[i];
			yf1[c] = y1[i];
			xf2[c] = x2[i];
			yf2[c] = y2[i];
			++c;
		}
	}
}

// Cleanning LORs outside of ROI based on center LOR position (used by SIDDON to start drawing)
void kernel_pet2D_SRM_clean_LOR_center(float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
									   float* xc1, int nxc1, float* yc1, int nyc1, float* xc2, int nxc2, float* yc2, int ncy2, int border, int size_im) {
	int i, c;
	float tx, ty;
	float lx1, ly1, lx2, ly2;
	float lxc1, lyc1, lxc2, lyc2;
	c = 0;
	for (i=0; i<nx1; ++i) {
		lx1 = x1[i];
		ly1 = y1[i];
		lx2 = x2[i];
		ly2 = y2[i];
		tx = (lx2 - lx1) * 0.5 + lx1;
		ty = (ly2 - ly1) * 0.5 + ly1;
		if (tx<border || ty<border) {continue;}
		if (tx>=(border+size_im) || ty>=(border+size_im)) {continue;}
		xc1[c] = lx1;
		yc1[c] = ly1;
		xc2[c] = lx2;
		yc2[c] = ly2;
		++c;
	}
}

// Raytrace SRM matrix with DDA algorithm
void kernel_pet2D_SRM_DDA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength, val;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy;
	int LOR_ind;
	
	for (i=0; i< nx1; ++i) {
		LOR_ind = i * wx;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		val  = 1 / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		for (n=0; n<=length; ++n) {
			SRM[LOR_ind + (int)y * width_image + (int)x] = val;
			x = x + xinc;
			y = y + yinc;
		}
	}
}

// Raytrace SRM matrix with DDA algorithm in ELL sparse matrix format
void kernel_pet2D_SRM_ELL_DDA(float* vals, int niv, int njv, int* cols, int nic, int njc, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength, val;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy;
	int LOR_ind;
	val = 1.0f;
	for (i=0; i< nx1; ++i) {
		LOR_ind = i * njv;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		//val  = 1 / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		for (n=0; n<=length; ++n) {
			vals[LOR_ind + n] = val;
			cols[LOR_ind + n] = (int)y * width_image + (int)x;
			x = x + xinc;
			y = y + yinc;
		}
		cols[LOR_ind + n] = -1; // eof
	}
}

// Raytrace SRM matrix with DDA algorithm with GPU
void kernel_pet2D_SRM_DDA_cuda(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
#ifdef CUDA
	kernel_pet2D_SRM_DDA_wrap_cuda(SRM, wy, wx, X1, nx1, Y1, ny1, X2, nx2, Y2, ny2, width_image);
#else
	printf("CUDA not supported!");
#endif
}

// OMP version DOES NOT WORK
void kernel_pet2D_SRM_DDA_omp(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength, val;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy;
	int LOR_ind;
	int myid, ncpu;
	int Nstart, Nstop;
#pragma omp parallel num_threads(4)
{
	ncpu = 4; //omp_get_num_threads();
	myid = omp_get_thread_num();
	Nstart = int(float(nx1) / float(ncpu) * float(myid) + 0.5);
	Nstop = int(float(nx1) / float(ncpu) * float(myid + 1) + 0.5);
	printf("myid %i / %i - %i %i\n", myid, ncpu, Nstart, Nstop);
    //#pragma omp parallel for shared(SRM, X1, Y1, X2, Y2) private(i)
	//#pragma omp parallel for private(i)
	for (i=Nstart; i < Nstop; ++i) {
		LOR_ind = i * wx;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		val  = 1 / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
        
		for (n=0; n<=length; ++n) {
			SRM[LOR_ind + (int)y * width_image + (int)x] = val;
			x = x + xinc;
			y = y + yinc;
		}
	}
}
}

// Draw lines in SRM with DDA anti-aliased version 1 pix
void kernel_pet2D_SRM_DDAA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint;
	int LOR_ind;

	for (i=0; i< nx1; ++i) {
		LOR_ind = i * wx;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			SRM[LOR_ind + yint * width_image + xint] = (1 - fabs(x - (xint + 0.5)));
			x = x + xinc;
			y = y + yinc;
		}
	}
}

// Draw lines in SRM with DDA anti-aliased version 2 pix 
void kernel_pet2D_SRM_DDAA2(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint, ind;
	float val, vd, vu;
	int LOR_ind;

	for (i=0; i< nx1; ++i) {
		LOR_ind = i * wx;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;

		// first pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		SRM[LOR_ind + yint * width_image + xint] = val;
		x = x + xinc;
		y = y + yinc;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			ind = LOR_ind + yint * width_image + xint;
			val = 1 - fabs(x - (xint + 0.5));
			vu = (x - xint) * 0.5;
			// vd = 0.5 - vu;
			SRM[ind+1] = vu;
			SRM[ind] = val;
			x = x + xinc;
			y = y + yinc;
		}
		// last pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		SRM[LOR_ind + yint * width_image + xint] = val;
	}
}

// Draw lines in SRM with DDA anti-aliased version 2 pix, SRM is in ELL sparse matrix format 
void kernel_pet2D_SRM_ELL_DDAA2(float* SRMvals, int niv, int njv, int* SRMcols, int nic, int njc, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint, ind, ind2;
	float val, vd, vu;
	int LOR_ind;

	for (i=0; i< nx1; ++i) {
		LOR_ind = i * njv;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;

		// first pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		SRMvals[LOR_ind] = val;
		SRMcols[LOR_ind] = yint * width_image + xint;
		//SRM[LOR_ind + yint * width_image + xint] = val;
		x = x + xinc;
		y = y + yinc;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			ind = yint * width_image + xint;
			val = 1 - fabs(x - (xint + 0.5));
			vu = (x - xint) * 0.5;
			// vd = 0.5 - vu;
			ind2 = LOR_ind + 2*n;
			SRMvals[ind2] = vu;
			SRMcols[ind2] = ind + 1;
			SRMvals[ind2 + 1] = val;
			SRMcols[ind2 + 1] = ind;
			//SRM[ind+1] = vu;
			//SRM[ind] = val;
			x = x + xinc;
			y = y + yinc;
		}
		// last pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		ind2 = LOR_ind + 2*n;
		SRMvals[ind2] = val;
		SRMcols[ind2] = yint * width_image + xint;
		//SRM[LOR_ind + yint * width_image + xint] = val;
	}
}

// Draw lines in SRM by Bresenham's Line Algorithm (modified version 1D)
void kernel_pet2D_SRM_BLA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int x, y, n;
	int x1, y1, x2, y2;
	int dx, dy;
	int xinc, yinc;
	int balance;
	float val;
	int LOR_ind;

	for (n=0; n<nx1; ++n) {
		LOR_ind = n * wx;
		x1 = X1[n];
		y1 = Y1[n];
		x2 = X2[n];
		y2 = Y2[n];

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
			val = 1 / (float)dx;
			dy <<= 1;
			balance = dy - dx;
			dx <<= 1;
			while (x != x2) {
				SRM[LOR_ind + y * width_image + x] = val;
				if (balance >= 0) {
					y = y + yinc;
					balance = balance - dx;
				}
				balance = balance + dy;
				x = x + xinc;
			}
			SRM[LOR_ind + y * width_image + x] = val;
		} else {
			val = 1 / (float)dy;
			dx <<= 1;
			balance = dx - dy;
			dy <<= 1;
			while (y != y2) {
				SRM[LOR_ind + y * width_image + x] = val;
				if (balance >= 0) {
					x = x + xinc;
					balance = balance - dy;
				}
				balance = balance + dx;
				y = y + yinc;
			}
			SRM[LOR_ind + y * width_image + x] = val;
		}
	}
}

// Draw lines in SRM by Siddon's Line Algorithm (modified version 1D)
void kernel_pet2D_SRM_SIDDON(float* SRM, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int matsize) {
	int n, LOR_ind;
	float tx, ty, px, qx, py, qy;
	int ei, ej, u, v, i, j;
	int stepi, stepj;
	float divx, divy, runx, runy, oldv, newv, val, valmax;
	float axstart, aystart, astart, pq, stepx, stepy, startl, initl;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * wx;
		px = X2[n];
		py = Y2[n];
		qx = X1[n];
		qy = Y1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		ei = int(tx);
		ej = int(ty);
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
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			SRM[LOR_ind + j * matsize + i] = val;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		SRM[LOR_ind + ej * matsize + ei] = val;
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			SRM[LOR_ind + j * matsize + i] = val;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}
	}
}

// Draw lines in SRM by Siddon's Line Algorithm (modified version 1D), SRM is in ELL sparse matrix format
void kernel_pet2D_SRM_ELL_SIDDON(float* SRMvals, int niv, int njv, int* SRMcols, int nic, int njc, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int matsize) {
	int n, LOR_ind;
	float tx, ty, px, qx, py, qy;
	int ei, ej, u, v, i, j, ct;
	int stepi, stepj;
	float divx, divy, runx, runy, oldv, newv, val, valmax;
	float axstart, aystart, astart, pq, stepx, stepy, startl, initl;
	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * njv;
		ct = 0;
		px = X2[n];
		py = Y2[n];
		qx = X1[n];
		qy = Y1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		ei = int(tx);
		ej = int(ty);
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
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			SRMvals[LOR_ind + ct] = val;
			SRMcols[LOR_ind + ct] = j * matsize + i;
			ct++;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}

		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		SRMvals[LOR_ind + ct] = val;
		SRMcols[LOR_ind + ct] = ej * matsize + ei;
		ct++;
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			SRMvals[LOR_ind + ct] = val;
			SRMcols[LOR_ind + ct] = j * matsize + i;
			ct++;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}
	}
}

// Draw a list of lines in SRM by Wu's Line Algorithm (modified version 1D)
void kernel_pet2D_SRM_WLA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int wim) {
	int dx, dy, stepx, stepy, n, LOR_ind;
	int length, extras, incr2, incr1, c, d, i;
	int x1, y1, x2, y2;
	float val;
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * wx;
		x1 = X1[n];
		y1 = Y1[n];
		x2 = X2[n];
		y2 = Y2[n];
	    dy = y2 - y1;
		dx = x2 - x1;
	
		if (dy < 0) { dy = -dy;  stepy = -1; } else { stepy = 1; }
		if (dx < 0) { dx = -dx;  stepx = -1; } else { stepx = 1; }
		if (dx > dy) {val = 1 / float(dx);}
		else {val = 1 / float(dy);}

		SRM[LOR_ind + y1 * wim + x1] = val;
		SRM[LOR_ind + y2 * wim + x2] = val;
		if (dx > dy) {
			length = (dx - 1) >> 2;
			extras = (dx - 1) & 3;
			incr2 = (dy << 2) - (dx << 1);
			if (incr2 < 0) {
				c = dy << 1;
				incr1 = c << 1;
				d =  incr1 - dx;
				for (i = 0; i < length; i++) {
					x1 = x1 + stepx;
					x2 = x2 - stepx;
					if (d < 0) {                            // Pattern:
						SRM[LOR_ind + y1 * wim + x1] = val; //
						x1 = x1 + stepx;                    // x o o
						SRM[LOR_ind + y1 * wim + x1] = val;
						SRM[LOR_ind + y2 * wim + x2] = val;
						x2 = x2 - stepx;
						SRM[LOR_ind + y2 * wim + x2] = val;
						d += incr1;
					} else {
						if (d < c) {                                 // Pattern:
							SRM[LOR_ind + y1 * wim + x1] = val;      //       o
							x1 = x1 + stepx;                         //   x o
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
							
						} else {
							y1 = y1 + stepy;                      // Pattern
							SRM[LOR_ind + y1 * wim + x1] = val;   //    o o
							x1 = x1 + stepx;                      //  x
							SRM[LOR_ind + y1 * wim + x1] = val;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d < 0) {
						x1 = x1 + stepx;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else 
	                if (d < c) {
						x1 = x1 + stepx;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
	                }
				}
			} else {
			    c = (dy - dx) << 1;
				incr1 = c << 1;
				d =  incr1 + dx;
				for (i = 0; i < length; i++) {
					x1 = x1 + stepx;
					x2 = x2 - stepx;
					if (d > 0) {
						y1 = y1 + stepy;                     // Pattern
						SRM[LOR_ind + y1 * wim + x1] = val;  //      o
						x1 = x1 + stepx;                     //    o
						y1 = y1 + stepy;                     //   x
						SRM[LOR_ind + y1 * wim + x1] = val;
						y2 = y2 - stepy;
						SRM[LOR_ind + y2 * wim + x2] = val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						SRM[LOR_ind + y2 * wim + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							SRM[LOR_ind + y1 * wim + x1] = val;  // Pattern
							x1 = x1 + stepx;                     //      o
							y1 = y1 + stepy;                     //  x o
							SRM[LOR_ind + y1 * wim + x1] = val;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						} else {
							y1 = y1 + stepy;                    // Pattern
							SRM[LOR_ind + y1 * wim + x1] = val; //    o  o
							x1 = x1 + stepx;                    //  x
							SRM[LOR_ind + y1 * wim + x1] = val;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d > 0) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else 
	                if (d < c) {
						x1 = x1 + stepx;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							if (d > c) {
								x2 = x2 - stepx;
								y2 = y2 - stepy;
								SRM[LOR_ind + y2 * wim + x2] = val;
							} else {
								x2 = x2 - stepx;
								SRM[LOR_ind + y2 * wim + x2] = val;
							}
						}
					}
				}
			}
	    } else {
		    length = (dy - 1) >> 2;
			extras = (dy - 1) & 3;
			incr2 = (dx << 2) - (dy << 1);
			if (incr2 < 0) {
				c = dx << 1;
				incr1 = c << 1;
				d =  incr1 - dy;
				for (i = 0; i < length; i++) {
					y1 = y1 + stepy;
					y2 = y2 - stepy;
					if (d < 0) {
						SRM[LOR_ind + y1 * wim + x1] = val;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						SRM[LOR_ind + y2 * wim + x2] = val;
						y2 = y2 - stepy;
						SRM[LOR_ind + y2 * wim + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							SRM[LOR_ind + y1 * wim + x1] = val;
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						} else {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d < 0) {
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else 
	                if (d < c) {
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
	                } else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
	                }
				}
	        } else {
				c = (dx - dy) << 1;
				incr1 = c << 1;
				d =  incr1 + dy;
				for (i = 0; i < length; i++) {
					y1 = y1 + stepy;
					y2 = y2 - stepy;
					if (d > 0) {
						x1 = x1 + stepx;
						SRM[LOR_ind + y1 * wim + x1] = val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						x2 = x2 - stepx;
						SRM[LOR_ind + y2 * wim + x2] = val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						SRM[LOR_ind + y2 * wim + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							SRM[LOR_ind + y1 * wim + x1] = val;
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						} else {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d > 0) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else
	                if (d < c) {
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
	                    if (extras > 2) {
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							if (d > c)  {
								x2 = x2 - stepx;
								y2 = y2 - stepy;
								SRM[LOR_ind + y2 * wim + x2] = val;
							} else {
								y2 = y2 - stepy;
								SRM[LOR_ind + y2 * wim + x2] = val;
							}
						}
					}
				}
			}
		}
	}
}


// Raytrace SRM matrix with DDA algorithm in ELL sparse matrix format
void kernel_pet3D_SRM_ELL_DDA(float* vals, int niv, int njv, int* cols, int nic, int njc,
							  unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1, unsigned short int* Z1, int nz1,
							  unsigned short int* X2, int nx2, unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2, int wim) {
	int length, lengthy, lengthz, i, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int LOR_ind;
	int step;
	val = 1.0f;
	step = wim*wim;
	
	for (i=0; i< nx1; ++i) {
		LOR_ind = i * njv;
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
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		zinc = diffz / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		z = z1 + 0.5;
		for (n=0; n<=length; ++n) {
			vals[LOR_ind + n] = val;
			cols[LOR_ind + n] = (int)z * step + (int)y * wim + (int)x;
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		cols[LOR_ind + n] = -1; // eof
	}
}

// Compute the first image with DDA algorithm
void kernel_pet3D_IM_SRM_DDA(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
							 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
							 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
							 float* im, int nim1, int nim2, int nim3, int wim) {
	
	int length, lengthy, lengthz, i, n;
	float flength;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	step = wim*wim;
	
	for (i=0; i< nx1; ++i) {
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
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			im[(int)z * step + (int)y * wim + (int)x] += 1.0f;
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}
}

// Compute the first image with DDA algorithm and fixed point
#define CONST int(pow(2, 23))
#define float2fixed(X) ((int) X * CONST)
#define intfixed(X) (X >> 23)
void kernel_pet3D_IM_SRM_DDA_fixed(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
								   unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
								   unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
								   float* im, int nim1, int nim2, int nim3, int wim) {
	
	int length, lengthy, lengthz, i, n;
	float flength, val;
	float lx, ly, lz;
	//float xinc, yinc, zinc;
	int fxinc, fyinc, fzinc, fx, fy, fz;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	step = wim*wim;
	
	for (i=0; i< nx1; ++i) {
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
		for (n=0; n<length; ++n) {  // change <= to < to avoid a bug
			im[intfixed(fz) * step + intfixed(fy) * wim + intfixed(fx)] += 1.0f;
			fx = fx + fxinc;
			fy = fy + fyinc;
			fz = fz + fzinc;
		}
	}
}
#undef CONST
#undef float2fixed
#undef intfixed

// Compute the first image with BLA algorithm
void kernel_pet3D_IM_SRM_BLA(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
							 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
							 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
							 float* im, int nim1, int nim2, int nim3, int wim) {

	float val = 1.0f;
	int x, y, z;
	int x1, y1, z1, x2, y2, z2;
	int dx, dy, dz;
	int xinc, yinc, zinc;
	int balance1, balance2;
	int step = wim*wim;
	int i;

	for (i=0; i< nx1; ++i) {
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		z1 = Z1[i];
		z2 = Z2[i];

		dx = x2 - x1;
		dy = y2 - y1;
		dz = z2 - z1;
		if (dx < 0) {
			xinc = -1;
			dx = -dx;
		} else {xinc = 1;}
		if (dy < 0) {
			yinc = -1;
			dy = -dy;
		} else {yinc = 1;}
		if (dz < 0) {
			zinc = -1;
			dz = -dz;
		} else {zinc = 1;}

		x = x1;
		y = y1;
		z = z1;
		if (dx >= dy && dx >= dz) {
			dy <<= 1;
			dz <<= 1;
			balance1 = dy - dx;
			balance2 = dz - dx;
			dx <<= 1;
			while (x != x2) {
				im[z * step + y * wim + x] += val;
				if (balance1 >= 0) {
					y = y + yinc;
					balance1 = balance1 - dx;
				}
				if (balance2 >= 0) {
					z = z + zinc;
					balance2 = balance2 - dx;
				}
				balance1 = balance1 + dy;
				balance2 = balance2 + dz;
				x = x + xinc;
			}
			im[z * step + y * wim + x] += val;
		} else {
			if (dy >= dx && dy >= dz) {
				dx <<= 1;
				dz <<= 1;
				balance1 = dx - dy;
				balance2 = dz - dy;
				dy <<= 1;
				while (y != y2) {
					im[z * step + y * wim + x] += val;
					if (balance1 >= 0) {
						x = x + xinc;
						balance1 = balance1 - dy;
					}
					if (balance2 >= 0) {
						z = z + zinc;
						balance2 = balance2 - dy;
					}
					balance1 = balance1 + dx;
					balance2 = balance2 + dz;
					y = y + yinc;
				}
				im[z * step + y * wim + x] += val;
			} else {
				dx <<= 1;
				dy <<= 1;
				balance1 = dx - dz;
				balance2 = dy - dz;
				dz <<= 1;
				while (z != z2) {
					im[z * step + y * wim + x] += val;
					if (balance1 >= 0) {
						x = x + xinc;
						balance1 = balance1 - dz;
					}
					if (balance2 >= 0) {
						y = y + yinc;
						balance2 = balance2 - dz;
					}
					balance1 = balance1 + dx;
					balance2 = balance2 + dy;
					z = z + zinc;
				}
				im[z * step + y * wim + x] += val;
			}
		}
	}
}

// Update image online, SRM is build with DDA's Line Algorithm, store in ELL format and update with LM-OSEM
void kernel_pet3D_IM_SRM_ELL_DDA_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1, unsigned short int* Z1, int nz1,
									  unsigned short int* X2, int nx2, unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
									  float* im, int nim, float* F, int nf, int wim, int ndata) {
	int length, lengthy, lengthz, i, j, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	val = 1.0f;
	step = wim*wim;

	// alloc mem
	float* vals = (float*)malloc(nx1 * ndata * sizeof(float));
	int* cols = (int*)malloc(nx1 * ndata * sizeof(int));
	float* Q = (float*)calloc(nx1, sizeof(float));
	int LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi;

	for (i=0; i< nx1; ++i) {
		LOR_ind = i * ndata;
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
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			vals[LOR_ind + n] = val;
			cols[LOR_ind + n] = (int)z * step + (int)y * wim + (int)x;
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// eof
		vals[LOR_ind + n] = -1;
		cols[LOR_ind + n] = -1;
	}

	// Sparse matrix operation Q = SRM * im
	for (i=0; i<nx1; ++i) {
		LOR_ind = i * ndata;
		vcol = cols[LOR_ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (vals[LOR_ind+j] * im[vcol]);
			++j;
			vcol = cols[LOR_ind+j];
		}
		Q[i] = sum;
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nx1; ++i) {
		LOR_ind = i * ndata;
		vcol = cols[LOR_ind];
		j = 0;
		Qi = Q[i];
		if (Qi==0.0f) {continue;}
		while (vcol != -1) {
			F[vcol] += (vals[LOR_ind+j] / Qi);
			++j;
			vcol = cols[LOR_ind+j];
		}
	}

	free(vals);
	free(cols);
	free(Q);

	
}


// Update image online, SRM is build with DDA's Line Algorithm, store in ELL format and update with LM-OSEM
void kernel_pet3D_IM_SRM_ELL_DDA_ON_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
										 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
										 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
										 float* im, int nim1, int nim2, int nim3,
										 float* F, int nf1, int nf2, int nf3, int wim, int ndata) {
	
	int length, lengthy, lengthz, i, j, n;
	float flength;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	step = wim*wim;

	// alloc mem
	float* vals = (float*)malloc(ndata * sizeof(float));
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
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			vals[n] = 1.0f;
			vcol = (int)z * step + (int)y * wim + (int)x;
			cols[n] = vcol;
			Qi += im[vcol];
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// eof
		vals[n] = -1;
		cols[n] = -1;
		// compute F
		if (Qi==0.0f) {continue;}
		Qi = 1.0f / Qi;
		vcol = cols[0];
		j = 0;
		while (vcol != -1) {
			F[vcol] += (vals[j] * Qi);
			++j;
			vcol = cols[j];
		}
	}
	free(vals);
	free(cols);
}

// Update image online, SRM is build with DDA's Line Algorithm, store in COO format and update with LM-OSEM
void kernel_pet3D_IM_SRM_COO_DDA_ON_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
										 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
										 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
										 float* im, int nim1, int nim2, int nim3,
										 float* F, int nf1, int nf2, int nf3, int wim) {
	
	int length, lengthy, lengthz, i, j, n, ct;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	val = 1.0f;
	step = wim*wim;

	// alloc mem
	int LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi;

	for (i=0; i< nx1; ++i) {
		float* vals = NULL;
		int* cols = NULL;
		Qi = 0.0f;
		ct = 0;
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
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			++ct;
			vals = (float*)realloc(vals, ct*sizeof(float));
			cols = (int*)realloc(cols, ct*sizeof(int));
			vals[ct-1] = val;
			vcol = (int)z * step + (int)y * wim + (int)x;
			cols[ct-1] = vcol;
			Qi += (val * im[vcol]);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// compute F
		if (Qi==0.0f) {continue;}
		for(j=0; j<ct; ++j) {
			if (im[cols[j]] != 0.0f) {
				F[cols[j]] += (vals[j] / Qi);
			}
		}
		free(vals);
		free(cols);
	}
}

// Update image online, SRM is build with DDA's Line Algorithm, store in raw format and update with LM-OSEM
void kernel_pet3D_IM_SRM_RAW_DDA_ON_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1,
										 unsigned short int* Z1, int nz1, unsigned short int* X2, int nx2,
										 unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
										 float* im, int nim1, int nim2, int nim3,
										 float* F, int nf1, int nf2, int nf3, int wim) {
	
	int length, lengthy, lengthz, i, j, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	val = 1.0f;
	step = wim*wim;
	int ntot = nim1*nim2*nim3;
	int LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi;

	for (i=0; i< nx1; ++i) {
		float* subim = (float*)malloc(ntot * sizeof(float));
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
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			vcol = (int)z * step + (int)y * wim + (int)x;
			subim[vcol] = val;
			Qi += (val * im[vcol]);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// compute F
		if (Qi==0.0f) {continue;}
		for(j=0; j<ntot; ++j) {
			if (im[j] != 0.0f) {
				F[j] += (subim[j] / Qi);
			}
		}
		free(subim);
	}
}


// Compute first image ionline by Siddon's Line Algorithm
void kernel_pet3D_IM_SRM_SIDDON(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
								float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2, float* im, int nim, int wim, int dim) {
	int n;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		px -= 55.0f;
		py -= 55.0f;
		qx -= 55.0f;
		qy -= 55.0f;
		initl = inkernel_randf();
		//initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		//initl = initl * 0.4 + 0.1; // rnd number between 0.1 to 0.5
		initl = initl * 0.4 + 0.3; // rnd number between 0.3 to 0.7
		//initl = 0.5f;
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
			if (oldi != i || oldj != j || oldk != k) {im[k * wim2 + j * wim + i] += val;}
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
		im[ek * wim2 + ej * wim + ei] += 0.707f; //val;
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
			if (oldi != i || oldj != j || oldk != k) {im[k * wim2 + j * wim + i] += val;}
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
	}
}

// Update image online, SRM is build with Siddon's Line Algorithm, and update with LM-OSEM
void kernel_pet3D_IM_SRM_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
									 float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
									 float* im, int nim, float* F, int nf, int wim) {
	int n;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;
	double Qi;
	float* SRM = (float*)malloc(nim * sizeof(float));

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		//printf("%i\n", n);
		// init SRM and Qi
		//for (i=0; i<nim; ++i) {SRM[i] = 0.0f;}
		memset(SRM, 0, nim*sizeof(float));
		Qi = 0.0f;
		// draw the line
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		initl = (float)rand() / (float)RAND_MAX;
		//initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		initl = initl * 0.4 + 0.1;
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
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
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {SRM[k * wim2 + j * wim + i] += val;}
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
		SRM[ek * wim2 + ej * wim + ei] += 0.707f; //val;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {SRM[k * wim2 + j * wim + i] += val;}
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
		for (i=0; i<nim; ++i) {Qi += (SRM[i] * im[i]);}
		if (Qi == 0.0f) {continue;}
		// accumulate to F
		for (i=0; i<nim; ++i) {
			if (im[i] != 0.0f) {
				F[i] += (SRM[i] / Qi);
			}
		}
		
	} // LORs loop
	free(SRM);
	
}


// Compute first image online by Siddon's Line Algorithm, and store SRM matrix to the harddrive with COO format
void kernel_pet3D_IM_SRM_COO_SIDDON(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
									float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2, float* im, int nim, int wim, int isub) {
	int n;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;
	int col, ct;

	// init file
	FILE * pfile_vals;
	FILE * pfile_rows;
	FILE * pfile_cols;
	char namevals [20];
	char namecols [20];
	char namerows [20];
	sprintf(namevals, "SRMvals_%i.coo", isub);
	sprintf(namecols, "SRMcols_%i.coo", isub);
	sprintf(namerows, "SRMrows_%i.coo", isub);
	pfile_vals = fopen(namevals, "wb");
	pfile_cols = fopen(namecols, "wb");
	pfile_rows = fopen(namerows, "wb");

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		//printf("%i %f %f %f - %f %f %f\n", n, px, py, pz, qx, qy, qz);
		ct = 0;
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
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
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				col = k * wim2 + j * wim + i;
				im[col] += val;
				fwrite(&val, sizeof(float), 1, pfile_vals);
				fwrite(&col, sizeof(int), 1, pfile_cols);
				fwrite(&n, sizeof(int), 1, pfile_rows);
				++ct;
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
		col = ek * wim2 + ej * wim + ei;
		val = 0.707f;
		im[col] += val;
		fwrite(&val, sizeof(float), 1, pfile_vals);
		fwrite(&col, sizeof(int), 1, pfile_cols);
		fwrite(&n, sizeof(int), 1, pfile_rows);
		++ct;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				col = k * wim2 + j * wim + i;
				im[col] += val;
				fwrite(&val, sizeof(float), 1, pfile_vals);
				fwrite(&col, sizeof(int), 1, pfile_cols);
				fwrite(&n, sizeof(int), 1, pfile_rows);
				++ct;
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
	}
	// close files
	fclose(pfile_vals);
	fclose(pfile_cols);
	fclose(pfile_rows);
}

// Update image online, SRM is read from the hard-drive and update with LM-OSEM
void kernel_pet3D_IM_SRM_COO_SIDDON_iter_vec(float* im, int nim, float* F, int nf, int N, int isub) {
	// open files
	FILE * pfile_vals;
	FILE * pfile_rows;
	FILE * pfile_cols;
	char namevals [20];
	char namecols [20];
	char namerows [20];
	sprintf(namevals, "SRMvals_%i.coo", isub);
	sprintf(namecols, "SRMcols_%i.coo", isub);
	sprintf(namerows, "SRMrows_%i.coo", isub);
	pfile_vals = fopen(namevals, "rb");
	pfile_cols = fopen(namecols, "rb");
	pfile_rows = fopen(namerows, "rb");

	// init
	//float* SRM = (float*)malloc(nim * sizeof(float));
	int* Ni = (int*)calloc(N, sizeof(int));
	float Qi, ival;
	int i, n, icol;
	// compute number of elements per rows
	int nbele;
	fseek(pfile_rows, 0, SEEK_END);
	nbele = ftell(pfile_rows);
	rewind(pfile_rows);
	nbele /= sizeof(float);
	int irows;
	for (i=0; i<nbele; ++i) {
		fread(&irows, 1, sizeof(int), pfile_rows);
		Ni[irows] += 1;
	}
	// create a static memory
	int max = 0;
	for (i=0; i<N; ++i) {
		if (Ni[i]>max) {max=Ni[i];}
	}
	float* vals = (float*)malloc(max * sizeof(float));
	int* cols = (int*)malloc(max * sizeof(int));
	
	// read SRM
	for (n=0; n<N; ++n) {
		
		// init SRM and Qi
		Qi = 0.0f;
		for (i=0; i<Ni[n]; ++i) {
			fread(&icol, 1, sizeof(int), pfile_cols);
			fread(&ival, 1, sizeof(float), pfile_vals);
			vals[i] = ival;
			cols[i] = icol;
			Qi += (ival * im[icol]);
		}
		if (Qi == 0.0f) {continue;}
		// accumulate to F
		for (i=0; i<Ni[n]; ++i) {
			F[cols[i]] += (vals[i] / Qi);
		}

	}
	// close files
	fclose(pfile_vals);
	fclose(pfile_cols);
	fclose(pfile_rows);
	free(Ni);
	free(vals);
	free(cols);
	
}

// Update image online, SRM is read from the hard-drive and update with LM-OSEM
void kernel_pet3D_IM_SRM_COO_SIDDON_iter_mat(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* im, int nim, float* F, int nf, int N, int isub) {
	int i, j, ind;
	float buf;
	float* Q = (float*)malloc(N * sizeof(float));

	// init Q and F
	for (i=0; i<N; ++i) {Q[i] = 0.0f;}
	
	// Sparse matrix multiplication Q = SRM * im
	for (i=0; i<nvals; ++i) {
		Q[rows[i]] += (vals[i] * im[cols[i]]);
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nvals; ++i) {
		if (Q[rows[i]] == 0.0f) {continue;}
		F[cols[i]] += (vals[i] / Q[rows[i]]);
	}
	/*
	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		if (buf != 0) {
			im[j] = buf / S[j] * F[j];
		}
	}
	*/
	free(Q);
}

// Update image online, SRM is build with Siddon's Line Algorithm, store in ELL format and update with LM-OSEM
void kernel_pet3D_IM_SRM_ELL_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
										 float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
										 float* im, int nim, float* F, int nf, int wim, int ndata) {
	int n;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;

	// alloc mem
	float* vals = (float*)malloc(nx1 * ndata * sizeof(float));
	int* cols = (int*)malloc(nx1 * ndata * sizeof(int));
	float* Q = (float*)calloc(nx1, sizeof(float));
	int ct, LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * ndata;
		ct = 0;
		// draw the line
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
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
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				vals[LOR_ind + ct] = val;
				cols[LOR_ind + ct] = k * wim2 + j * wim + i;
				++ct;
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
		vals[LOR_ind + ct] = 0.707f;
		cols[LOR_ind + ct] = ek * wim2 + ej * wim + ei;
		++ct;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				vals[LOR_ind + ct] = val;
				cols[LOR_ind + ct] = k * wim2 + j * wim + i;
				++ct;
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
		// eof		
		vals[LOR_ind + ct] = -1;
		cols[LOR_ind + ct] = -1;
		//printf("ct %i\n", ct);
	} // LORs loop

	// Sparse matrix operation Q = SRM * im
	for (i=0; i<nx1; ++i) {
		LOR_ind = i * ndata;
		vcol = cols[LOR_ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (vals[LOR_ind+j] * im[vcol]);
			++j;
			vcol = cols[LOR_ind+j];
		}
		Q[i] = sum;
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nx1; ++i) {
		LOR_ind = i * ndata;
		vcol = cols[LOR_ind];
		j = 0;
		Qi = Q[i];
		if (Qi==0.0f) {continue;}
		while (vcol != -1) {
			F[vcol] += (vals[LOR_ind+j] / Qi);
			++j;
			vcol = cols[LOR_ind+j];
		}
	}
	free(vals);
	free(cols);
	free(Q);
}

// Draw lines in 2D space with DDA anti-aliased version 1 pix
void kernel_draw_2D_lines_DDAA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint;

	for (i=0; i< nx1; ++i) {
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			mat[yint*wx + xint] += (1 - fabs(x - (xint + 0.5)));
			x = x + xinc;
			y = y + yinc;
		}
	}
}

// Draw lines in 2D space with DDA anti-aliased version 2 pix 
void kernel_draw_2D_lines_DDAA2(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint, ind;
	float val, vd, vu;

	for (i=0; i< nx1; ++i) {
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;

		// first pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		mat[yint * wx + xint] += val;
		x = x + xinc;
		y = y + yinc;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			ind = yint*wx + xint;
			val = 1 - fabs(x - (xint + 0.5));
			vu = (x - xint) * 0.5;
			// vd = 0.5 - vu;
			mat[ind+1] += vu;
			mat[ind] += val;
			x = x + xinc;
			y = y + yinc;
		}
		// last pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		mat[yint * wx + xint] += val;
	}
}

// THIS MUST BE CHANGE!!!!!!!!!
// Draw a line in 2D space by DDA with my antialiaser
#define ipart_(X) ((int) X)
#define round_(X) ((int)(((double)(X)) + 0.5))
#define fpart_(X) ((double)(X) - (double)ipart_(X))
void kernel_draw_2D_line_DDAA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int length, i;
	double x, y;
	double xinc, yinc;
	
	length = abs(x2 - x1);
	if (abs(y2 - y1) > length) {length = abs(y2 - y1);}
	xinc = (double)(x2 - x1) / (double) length;
	yinc = (double)(y2 - y1) / (double) length;
	//x    = x1 + 0.5;
	//y    = y1 + 0.5;
	x = x1;
	y = y1;
	float hval = val / 2.0;
	float vx, vy;
	int ix, iy;
	for (i=0; i<=length; ++i) {
		vx = fpart_(x);
		vy = fpart_(y);
		ix = round_(x);
		iy = round_(y);
		mat[iy * wx + ix] += hval;
		
		if (vx > 0.5) {mat[iy * wx + ix + 1] += hval;}
		else if (vx < 0.5) {mat[iy * wx + ix - 1] += hval;}
		if (vy > 0.5) {mat[(iy + 1) * wx + ix] += hval;}
		else if (vy > 0.5) {mat[(iy - 1) * wx + ix] += hval;}
		
		x = x + xinc;
		y = y + yinc;
	}
}
#undef ipart_
#undef fpart_
#undef round_

// EM-ML algorithm, only one iteration (bin mode)
void kernel_pet2D_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval) {
	int i, j, ind;
	float qi, buf, f;
	float* Q = (float*)malloc(nlor * sizeof(float));

	// compute expected value
	for (i=0; i<nlor; ++i) {
		qi = 0.0;
		ind = i * npix;
		for (j=0; j<npix; ++j) {qi += (SRM[ind+j] * im[j]);}
		Q[i] = qi;
	}

	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		
		if (buf != 0) {
			f = 0.0;
			for (i=0; i<nlor; ++i) {
				f += (LOR_val[i] * SRM[i * npix + j] / Q[i]);
			}
			im[j] = buf / S[j] * f;
		}
	}
	free(Q);
}

// EM-ML algorithm, only one iteration (list-mode), Naive implementation as define by the method
void kernel_pet2D_LM_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim) {
	int i, j, ind;
	float qi, buf, f;
	float* Q = (float*)malloc(nlor * sizeof(float));

	// compute expected value
	for (i=0; i<nlor; ++i) {
		qi = 0.0;
		ind = i * npix;
		for (j=0; j<npix; ++j) {qi += (SRM[ind+j] * im[j]);}
		if (qi == 0.0) {qi = 1.0f;}
		Q[i] = qi;
	}

	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		if (buf != 0) {
			f = 0.0;
			for (i=0; i<nlor; ++i) {
				f += (SRM[i * npix + j] / Q[i]);
			}
			im[j] = buf / S[j] * f;
		}
	}
	free(Q);
}

// EM-ML algorithm with sparse matrix (COO), only one iteration (list-mode), matrix operation
void kernel_pet2D_LM_EMML_COO_iter_mat(float* SRMvals, int nvals, int* SRMrows, int nrows, int* SRMcols, int ncols, float* S, int nbs, float* im, int npix, int nevents) {
	int i, j, ind;
	float buf;
	float* Q = (float*)malloc(nevents * sizeof(float));
	float* F = (float*)malloc(npix * sizeof(float));

	// init Q and F
	for (i=0; i<nevents; ++i) {Q[i] = 0.0f;}
	for (i=0; i<npix; ++i) {F[i] = 0.0f;}
	
	// Sparse matrix multiplication Q = SRM * im
	for (i=0; i<nvals; ++i) {
		Q[SRMrows[i]] += (SRMvals[i] * im[SRMcols[i]]);
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nvals; ++i) {
		F[SRMcols[i]] += (SRMvals[i] / Q[SRMrows[i]]);
	}
	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		if (buf != 0) {
			im[j] = buf / S[j] * F[j];
		}
	}
	free(F);
	free(Q);
}

// EM-ML algorithm with sparse matrix (COO), only one iteration (list-mode), naive method scalar operation
void kernel_pet2D_LM_EMML_COO_iter_vec(float* SRMvals, int nvals, int* SRMrows, int nrows, int* SRMcols, int ncols, float* S, int nbs, float* im, int npix, int nevents) {
	int i, j, ind;
	float buf, f;
	float* Q = (float*)malloc(nevents * sizeof(float));

	// init Q and F
	for (i=0; i<nevents; ++i) {Q[i] = 0.0f;}
	
	// Sparse matrix multiplication Q = SRM * im
	for (i=0; i<nvals; ++i) {
		Q[SRMrows[i]] += (SRMvals[i] * im[SRMcols[i]]);
	}
	// update pixel
	for (j=0; j<npix; ++j) {
		printf("%i\n", j);
		buf = im[j];
		if (buf != 0) {
			f = 0.0;
			for (i=0; i<ncols; ++i) {
				if (SRMcols[i] == j) {
					f += (SRMvals[i] / Q[SRMrows[i]]);
				}
			}
			im[j] = buf / S[j] * f;
		}
	}
	free(Q);
}


// EM-ML algorithm with sparse matrix (ELL)
void kernel_pet2D_LM_EMML_ELL_iter(float* SRMvals, int nivals, int njvals, int* SRMcols, int nicols, int njcols, float* S, int ns, float* im, int npix) {
	int i, j, ind, vcol;
	float buf, sum;
	float* Q = (float*)calloc(nivals, sizeof(float));
	float* F = (float*)calloc(npix, sizeof(float));

	// Sparse matrix operation Q = SRM * im
	for (i=0; i<nivals; ++i) {
		ind = i * njvals;
		vcol = SRMcols[ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (SRMvals[ind+j] * im[vcol]);
			++j;
			vcol = SRMcols[ind+j];
		}
		Q[i] = sum;
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nivals; ++i) {
		ind = i * njvals;
		vcol = SRMcols[ind];
		j = 0;
		while (vcol != -1) {
			F[vcol] += (SRMvals[ind+j] / Q[i]);
			++j;
			vcol = SRMcols[ind+j];
		}
	}
	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		if (buf != 0) {
			im[j] = buf / S[j] * F[j];
		}
	}
	free(Q);
	free(F);
}

// EM-ML algorithm, only one iteration MPI version
void kernel_pet2D_EMML_iter_MPI(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval, int N_start, int N_stop) {
	int i, j, ind;
	float qi, buf, f;
	float* Q = (float*)malloc(nlor * sizeof(float));
	
	// compute expected value
	for (i=0; i<nlor; ++i) {
		qi = 0.0;
		ind = i * npix;
		for (j=0; j<npix; ++j) {qi += (SRM[ind+j] * im[j]);}
		Q[i] = qi;
	}

	// update pixel
	for (j=N_start; j<N_stop; ++j) {
		buf = im[j];
		
		if (buf != 0) {
			f = 0.0;
			for (i=0; i<nlor; ++i) {
				f += (LOR_val[i] * SRM[i * npix + j] / Q[i]);
			}
			im[j] = buf / S[j] * f;
		}
	}
	free(Q);
}

// EM-ML algorithm, all iterations GPU version
void kernel_pet2D_EMML_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval, float* S, int ns, int maxit) {
	kernel_pet2D_EMML_wrap_cuda(SRM, nlor, npix, im, npixim, LOR_val, nval, S, ns, maxit);
}


// List mode 2D rexonstruction with DDA and ELL format, all iterations are perform on GPU
void kernel_pet2D_LM_EMML_DDA_ELL_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, float* S, int ns, int wsrm, int wim, int maxite) {
	kernel_pet2D_LM_EMML_DDA_ELL_wrap_cuda(x1, nx1, y1, ny1, x2, nx2, y2, ny2, im, nim, S, ns, wsrm, wim, maxite);
}

// Compute first image in 2D-LM-OSEM reconstruction (from IM, x, y build SRM in ELL format then compute IM+=IM)
void kernel_pet2D_IM_SRM_DDA_ELL_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, int wsrm, int wim) {
	kernel_pet2D_IM_SRM_DDA_ELL_wrap_cuda(x1, nx1, y1, ny1, x2, nx2, y2, ny2, im, nim, wsrm, wim);
}

// Update image for the 2D-LM-OSEM reconstruction (from x, y, IM and S, build SRM in ELL format then update IM)
void kernel_pet2D_IM_SRM_DDA_ELL_iter_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* S, int ns, float* im, int nim, int wsrm, int wim) {
	kernel_pet2D_IM_SRM_DDA_ELL_iter_wrap_cuda(x1, nx1, y1, ny1, x2, nx2, y2, ny2, S, ns, im, nim, wsrm, wim);
}

// Compute first image in 3D-LM-OSEM reconstruction (from IM, x, y build SRM in ELL format then compute IM+=IM)
void kernel_pet3D_IM_SRM_DDA_ELL_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
									  unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
									  float* im, int nim, int wsrm, int wim, int ID) {
	kernel_pet3D_IM_SRM_DDA_ELL_wrap_cuda(x1, nx1, y1, ny1, z1, nz1, x2, nx2, y2, ny2, z2, nz2, im, nim, wsrm, wim, ID);
}

// Update image for the 3D-LM-OSEM reconstruction (from x, y, IM and S, build SRM in ELL format then return F)
void kernel_pet3D_IM_SRM_DDA_ELL_iter_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
										   unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										   float* im, int nim, float* F, int nf, int wsrm, int wim, int ID) {
	kernel_pet3D_IM_SRM_DDA_ELL_iter_wrap_cuda(x1, nx1, y1, ny1, z1, nz1, x2, nx2, y2, ny2, z2, nz2, im, nim, F, nf, wsrm, wim, ID);
}

// Convert dense matrix to sparse one with COO format
void kernel_matrix_mat2coo(float* mat, int ni, int nj, float* vals, int nvals, int* rows, int nrows, int* cols, int ncols, int roffset, int coffset) {
	// roffset and coffset are rows and colums shiftment, if mat is a tile of a big matrix indexes must adjust
	int i, j, ind;
	int ct = 0;
	float buf;
	for (i=0; i<ni; ++i) {
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			buf = mat[ind + j];
			if (buf != 0.0f) {
				rows[ct] = i + roffset;
				cols[ct] = j + coffset;
				vals[ct] = buf;
				++ct;
			}
		}
		
	}
}

// Compute col sum of COO matrix
void kernel_matrix_coo_sumcol(float* vals, int nvals, int* cols, int ncols, float* im, int npix) {
	int n;
	for (n=0; n<nvals; ++n) {
		im[cols[n]] += vals[n];
	}
}

// Compute spmv matrix/vector multiplication with sparse COO matrix
void kernel_matrix_coo_spmv(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres) {
	int n;
	for (n=0; n<nvals; ++n) {
		res[rows[n]] += (vals[n] * y[cols[n]]);
	}
}

// Compute spmtv (t for transpose) matrix/vector multiplication with sparse COO matrix 
void kernel_matrix_coo_spmtv(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres) {
	int n;
	for (n=0; n<nvals; ++n) {
		res[cols[n]] += (vals[n] * y[rows[n]]);
	}
}

// Convert dense matrix to sparse one with CSR format
void kernel_matrix_mat2csr(float* mat, int ni, int nj, float* vals, int nvals, int* ptr, int nptr, int* cols, int ncols) {
	int i, j, ind;
	int ct = 0;
	float buf;
	for (i=0; i<ni; ++i) {
		ptr[i] = -1;
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			buf = mat[ind + j];
			if (buf != 0) {
				if (ptr[i] == -1) {ptr[i] = ct;}
				cols[ct] = j;
				vals[ct] = buf;
				++ct;
			}
		}
		if (ptr[i] == -1) {ptr[i] = ct;}
	}
	ptr[ni] = nvals;
}

// Compute col sum of CSR matrix
void kernel_matrix_csr_sumcol(float* vals, int nvals, int* cols, int ncols, float* im, int npix) {
	int n;
	for (n=0; n<nvals; ++n) {
		im[cols[n]] += vals[n];
	}
}


// Compute spmv matrix/vector multiplication with sparse CSR matrix
void kernel_matrix_csr_spmv(float* vals, int nvals, int* cols, int ncols, int* ptrs, int nptrs, float* y, int ny, float* res, int nres) {
	int iptr, k;
	for (iptr=0; iptr<(nptrs-1); ++iptr) {
		for (k=ptrs[iptr]; k<ptrs[iptr+1]; ++k) {
			res[iptr] += (vals[k] * y[cols[k]]);
		}
	}
}

// Compute spmtv (t for transpose) matrix/vector multiplication with sparse CSR matrix 
void kernel_matrix_csr_spmtv(float* vals, int nvals, int* cols, int ncols, int* ptrs, int nptrs, float* y, int ny, float* res, int nres) {
	int iptr, k;
	for (iptr=0; iptr<(nptrs-1); ++iptr) {
		for (k=ptrs[iptr]; k<ptrs[iptr+1]; ++k) {
			res[cols[k]] += (vals[k] * y[iptr]);
		}
	}
}

// Convert dense matrix to sparse one with ELL format
void kernel_matrix_mat2ell(float* mat, int ni, int nj, float* vals, int nivals, int njvals, int* cols, int nicols, int njcols) {
	int i, j, ind1, ind2, icol;
	float buf;
	for (i=0; i<ni; ++i) {
		ind1 = i*nj;
		ind2 = i*njvals;
		icol = 0;
		for (j=0; j<nj; ++j) {
			buf = mat[ind1+j];
			if (buf != 0.0f) {
				vals[ind2+icol] = buf;
				cols[ind2+icol] = j;
				++icol;
			}
		}
		cols[ind2+icol] = -1; // eof
	}
}

// Compute col sum of ELL matrix
void kernel_matrix_ell_sumcol(float* vals, int niv, int njv, int* cols, int nic, int njc, float* im, int npix) {
	int i, j, ind, vcol;
	for (i=0; i<niv; ++i) {
		ind = i * njv;
		vcol = cols[ind];
		j = 0;
		while (vcol != -1) {
			im[vcol] += vals[ind+j];
			++j;
			vcol = cols[ind+j];
		}
	}
}

// Compute spmv matrix/vector multiplication with sparse ELL matrix
void kernel_matrix_ell_spmv(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres) {
	int i, j, ind, vcol;
	float sum;
	for (i=0; i<niv; ++i) {
		ind = i * njv;
		vcol = cols[ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (vals[ind+j] * y[vcol]);
			++j;
			vcol = cols[ind+j];
		}
		res[i] = sum;
	}
}

// Compute spmv matrix/vector multiplication with sparse ELL matrix using GPU
void kernel_matrix_ell_spmv_cuda(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres) {
	kernel_matrix_ell_spmv_wrap_cuda(vals, niv, njv, cols, nic, njc, y, ny, res, nres);
}

// Compute spmtv matrix/vector multiplication with sparse ELL matrix
void kernel_matrix_ell_spmtv(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres) {
	int i, j, ind, vcol;
	for (i=0; i<niv; ++i) {
		ind = i * njv;
		vcol = cols[ind];
		j = 0;
		while (vcol != -1) {
			res[vcol] += (vals[ind+j] * y[i]);
			++j;
			vcol = cols[ind+j];
		}
	}
}

// Compute spmv matrix/vector multiplication
void kernel_matrix_spmv(float* mat, int ni, int nj, float* y, int ny, float* res, int nres) {
	int i, j, ind;
	float sum;
	for (i=0; i<ni; ++i) {
		sum = 0.0f;
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			sum += (mat[ind+j] * y[j]);
		}
		res[i] = sum;
	}
}

// Compute spmtv matrix/vector multiplication
void kernel_matrix_spmtv(float* mat, int ni, int nj, float* y, int ny, float* res, int nres) {
	int i, j, ind;
	float sum;
	for (j=0; j<nj; ++j) {
		sum = 0.0f;
		for (i=0; i<ni; ++i) {
			sum += (mat[i*nj + j] * y[i]);
		}
		res[i] = sum;
	}
}

void kernel_3Dconv_cuda(float* vol, int nz, int ny, int nx, float* H, int a, int b, int c) {
	//timeval start, end;
	//double t1, t2, diff;
	//gettimeofday(&start, NULL);
	//t1 = start.tv_sec + start.tv_usec / 1000000.0;
	kernel_3Dconv_wrap_cuda(vol, nz, ny, nx, H, a, b, c);
	//gettimeofday(&end, NULL);
	//t2 = end.tv_sec + end.tv_usec / 1000000.0;
	//diff = t2 - t1;
	//printf("C time %f s\n", diff);
}

int kernel_pack_id(int* d1, int nd1, int* c1, int nc1,
				   int* d2, int nd2, int* c2, int nc2,
				   unsigned short int* pack, int np, int id, int flag) {
	int i;
	// return only the number of elements
	if (flag) {
		int ct = 0;
		for (i=0; i<nd1; ++i) {
			if (d1[i]==id) {++ct;}
		}
		return ct;
	// pack values
	} else {
		int ind = 0;
		for (i=0; i<nd1; ++i) {
			if (d1[i]==id) {
				pack[ind] = (unsigned short int)d1[i];
				pack[ind+1] = (unsigned short int)c1[i];
				pack[ind+2] = (unsigned short int)d2[i];
				pack[ind+3] = (unsigned short int)c2[i];
				ind += 4;
			}
		}
		return ind;

	}

}

// build the list of all LOR in order to compute S matrix
void kernel_allegro_save_all_LOR(int* id1, int n1, int* id2, int n2,
								 int* x1, int nx1, int* y1, int ny1, int* z1, int nz1,
								 int* x2, int nx2, int* y2, int ny2, int* z2, int nz2) {
	FILE * pfile_lors;
	char namelors [20];

	int i, N, isub, ct;
	char xc1, yc1, zc1, xc2, yc2, zc2;
	N = int(n1 / 10000000);
	ct = 0;
	for (isub=0; isub<N; ++isub) {
		printf("Save iter %i\n", isub);
		sprintf(namelors, "lors_%i.bin", isub);
		pfile_lors = fopen(namelors, "wb");
		for (i=0; i<10000000; ++i) {
			fwrite(&id1[ct], sizeof(int), 1, pfile_lors);
			fwrite(&id2[ct], sizeof(int), 1, pfile_lors);
			xc1 = (char)x1[ct];
			yc1 = (char)y1[ct];
			zc1 = (char)z1[ct];
			xc2 = (char)x2[ct];
			yc2 = (char)y2[ct];
			zc2 = (char)z2[ct];
			fwrite(&xc1, sizeof(char), 1, pfile_lors);
			fwrite(&yc1, sizeof(char), 1, pfile_lors);
			fwrite(&zc1, sizeof(char), 1, pfile_lors);
			fwrite(&xc2, sizeof(char), 1, pfile_lors);
			fwrite(&yc2, sizeof(char), 1, pfile_lors);
			fwrite(&zc2, sizeof(char), 1, pfile_lors);
			++ct;
		}
		fclose(pfile_lors);
	}
	sprintf(namelors, "lors_%i.bin", N);
	pfile_lors = fopen(namelors, "wb");
	while (ct<n1) {
		fwrite(&id1[ct], sizeof(int), 1, pfile_lors);
		fwrite(&id2[ct], sizeof(int), 1, pfile_lors);
		xc1 = (char)x1[ct];
		yc1 = (char)y1[ct];
		zc1 = (char)z1[ct];
		xc2 = (char)x2[ct];
		yc2 = (char)y2[ct];
		zc2 = (char)z2[ct];
		fwrite(&xc1, sizeof(char), 1, pfile_lors);
		fwrite(&yc1, sizeof(char), 1, pfile_lors);
		fwrite(&zc1, sizeof(char), 1, pfile_lors);
		fwrite(&xc2, sizeof(char), 1, pfile_lors);
		fwrite(&yc2, sizeof(char), 1, pfile_lors);
		fwrite(&zc2, sizeof(char), 1, pfile_lors);
		++ct;
	}
	fclose(pfile_lors);
	
}


void kernel_SRM_to_HD(int isub) {
	// vars
	int length, lengthy, lengthz, i, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	long long int ID, ptr, size;
	int ind;
	int id1, id2;
	unsigned char buf;
	
	// TOBE change
	int wim = 141;
	step = wim*wim;
	// init file
	FILE * pfile_toc;
	FILE * pfile_srm;
	FILE * pfile_ID;
	char nametoc [20];
	char namesrm [20];
	char nameID [20];
	//sprintf(nametoc, "toc.bin");
	sprintf(nametoc, "toctmp.bin");
	sprintf(namesrm, "srm.bin");
	sprintf(nameID, "lors_%i.bin", isub);
	//pfile_toc = fopen(nametoc, "rb+");
	pfile_toc = fopen(nametoc, "ab+");
	pfile_srm = fopen(namesrm, "ab+");
	pfile_ID = fopen(nameID, "rb");

	int nid;
	fseek(pfile_ID, 0, SEEK_END);
	nid = ftell(pfile_ID);
	rewind(pfile_ID);
	nid /= 14;

	fseek(pfile_srm, 0, SEEK_END);
	
	for (i=0; i<nid; ++i) {
		fread(&id1, 1, sizeof(int), pfile_ID);
		fread(&id2, 1, sizeof(int), pfile_ID);
		fread(&buf, 1, sizeof(char), pfile_ID);
		x1 = (int)buf;
		fread(&buf, 1, sizeof(char), pfile_ID);
		y1 = (int)buf;
		fread(&buf, 1, sizeof(char), pfile_ID);
		z1 = (int)buf;
		fread(&buf, 1, sizeof(char), pfile_ID);
		x2 = (int)buf;
		fread(&buf, 1, sizeof(char), pfile_ID);
		y2 = (int)buf;
		fread(&buf, 1, sizeof(char), pfile_ID);
		z2 = (int)buf;

		//printf("%i - id %i %i - p1 %i %i %i - p2 %i %i %i\n", i, id1, id2, x1, y1, z1, x2, y2, z2);

		diffx = x2-x1;
		diffy = y2-y1;
		diffz = z2-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = 1 / (float)length;
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;

		// save info to toc file
		ID = inkernel_mono(id1, id2);
		ptr = ftell(pfile_srm);
		fwrite(&ID, sizeof(long long int), 1, pfile_toc);
		//fseek(pfile_toc, (3*ID + 1)*sizeof(int), SEEK_SET);
		fwrite(&ptr, sizeof(long long int), 1, pfile_toc);
		//fseek(pfile_toc, (3*ID + 2)*sizeof(int), SEEK_SET);
		size = length + 1;
		fwrite(&size, sizeof(long long int), 1, pfile_toc);

		//printf("ID %i ptr %i size %i\n", ID, ptr, size);
		x = x1 + 0.5;
		y = y1 + 0.5;
		z = z1 + 0.5;
		//debug += size;
		for (n=0; n<=length; ++n) {
			ind = (int)z * step + (int)y * wim + (int)x;
			fwrite(&ind, sizeof(int), 1, pfile_srm);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		
	}

	fclose(pfile_toc);
	fclose(pfile_srm);
	fclose(pfile_ID);
	//printf("tot %i\n", debug);

}


// Draw first image from pre-calculate SRM save on the hard drive
#define SWAP(a, b) {int tmp=(a); (a)=(b); (b)=tmp;}
void kernel_pet3D_IM_SRM_HD_(int* idc1, int nc1, int* idd1, int nd1, int* idc2, int nc2, int* idd2, int nd2,
							float* im, int nz, int ny, int nx, char* nametoc, char* namesrm) {
	int id1, id2, id, n, ind;
	long long int ptr, size;
	long long int i;
	int idmax = 22*29;

	// init file
	FILE * pfile_toc;
	FILE * pfile_srm;
	//char nametoc [20];
	//char namesrm [20];
	//sprintf(nametoc, "toc.bin");
	//sprintf(namesrm, "srm.bin");
	pfile_toc = fopen(nametoc, "rb");
	pfile_srm = fopen(namesrm, "rb");

	for (n=0; n<nc1; ++n) {
		id1 = idc1[n] + (idmax * idd1[n]);
		id2 = idc2[n] + (idmax * idd2[n]);
		if (id2 > id1) {SWAP(id1, id2);} // only the lower triangular matrix was stored
		id = inkernel_mono(id1, id2);
		//printf("id %i\n", id);
		
		fseek(pfile_toc, id*16, SEEK_SET); // ptr, and size in 64bits
		fread(&ptr, 1, sizeof(long long int), pfile_toc);
		fread(&size, 1, sizeof(long long int), pfile_toc);

		//printf("ptr %lli size %lli\n", ptr, size);
		fseek(pfile_srm, ptr, SEEK_SET);
		for (i=0; i<size; ++i) {
			fread(&ind, 1, sizeof(int), pfile_srm);
			im[ind] += 1.0f;
		}

	}
	fclose(pfile_toc);
	fclose(pfile_srm);

}
#undef SWAP

// Draw first image from pre-calculate SRM save on the hard drive
#define SWAP(a, b) {int tmp=(a); (a)=(b); (b)=tmp;}
void kernel_pet3D_IM_SRM_HD(int* idc1, int nc1, int* idd1, int nd1, int* idc2, int nc2, int* idd2, int nd2,
							float* im, int nz, int ny, int nx, char* nametoc, char* namesrm) {
	int id1, id2, id, idi, n, ind;
	long long int ptr, size;
	long long int i;
	int idmax = 22*29;

	// init file
	FILE * pfile_toc;
	FILE * pfile_srm;
	//char nametoc [20];
	//char namesrm [20];
	//sprintf(nametoc, "toc.bin");
	//sprintf(namesrm, "srm.bin");
	pfile_toc = fopen(nametoc, "rb");
	pfile_srm = fopen(namesrm, "rb");
	n=0;
	for (id=0; id<159552316; ++id) {
		id1 = idc1[n] + (idmax * idd1[n]);
		id2 = idc2[n] + (idmax * idd2[n]);
		if (id2 > id1) {SWAP(id1, id2);} // only the lower triangular matrix was stored
		idi = inkernel_mono(id1, id2);

		fseek(pfile_toc, id*16, SEEK_SET); // ptr, and size in 64bits
		fread(&ptr, 1, sizeof(long long int), pfile_toc);
		fread(&size, 1, sizeof(long long int), pfile_toc);
		fseek(pfile_srm, ptr, SEEK_SET);
		for (i=0; i<size; ++i) {
			fread(&ind, 1, sizeof(int), pfile_srm);
			im[ind] += 1.0f;
		}


		if (fmod(id, 1000000.0) == 0) {printf("%i\n", int((float)id/1000000.0f));}
	}
	//
	//printf("ptr %lli size %lli\n", ptr, size);

	/*

	for (n=0; n<nc1; ++n) {
		id1 = idc1[n] + (idmax * idd1[n]);
		id2 = idc2[n] + (idmax * idd2[n]);
		if (id2 > id1) {SWAP(id1, id2);} // only the lower triangular matrix was stored
		id = inkernel_mono(id1, id2);
		//printf("id %i\n", id);
		
		fseek(pfile_toc, id*16, SEEK_SET); // ptr, and size in 64bits
		fread(&ptr, 1, sizeof(long long int), pfile_toc);
		fread(&size, 1, sizeof(long long int), pfile_toc);

		//printf("ptr %lli size %lli\n", ptr, size);
		fseek(pfile_srm, ptr, SEEK_SET);
		for (i=0; i<size; ++i) {
			fread(&ind, 1, sizeof(int), pfile_srm);
			im[ind] += 1.0f;
		}

	}
	*/
	fclose(pfile_toc);
	fclose(pfile_srm);

}
#undef SWAP

