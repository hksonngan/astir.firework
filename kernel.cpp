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


#include <GL/gl.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

// Draw a line in 2D space by Digital Differential Analyzer method (modified version to 1D)
void kernel_draw_2D_line_DDA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int length, i;
	double x, y;
	double xinc, yinc;
	
	length = abs(x2 - x1);
	if (abs(y2 - y1) > length) {length = abs(y2 - y1);}
	xinc = (double)(x2 - x1) / (double) length;
	yinc = (double)(y2 - y1) / (double) length;
	x    = x1 + 0.5;
	y    = y1 + 0.5;
	for (i=0; i<=length; ++i) {
		mat[(int)y * wx + (int)x] += val;
		x = x + xinc;
		y = y + yinc;
	}
}

// Draw a line in 3D space by DDA method
void kernel_draw_3D_line_DDA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val) {
	int length, lengthy, lengthz, i, step;
	double x, y, z, xinc, yinc, zinc;
	step = wx * wy;
	length = abs(x2 - x1);
	lengthy = abs(y2 - y1);
	lengthz = abs(z2 - z1);
	if (lengthy > length) {length = lengthy;}
	if (lengthz > length) {length = lengthz;}

	xinc = (double)(x2 - x1) / (double) length;
	yinc = (double)(y2 - y1) / (double) length;
	zinc = (double)(z2 - z1) / (double) length;
	x    = x1 + 0.5;
	y    = y1 + 0.5;
	z    = z1 + 0.5;
	for (i=0; i<=length; ++i) {
		mat[(int)z*step + (int)y*wx + (int)x] += val;
		x = x + xinc;
		y = y + yinc;
		z = z + zinc;
	}
}

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

// Draw a line in 2D space by Bresenham's Line Algorithm (modified version 1D)
void kernel_draw_2D_line_BLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int x, y;
	int dx, dy;
	int xinc, yinc;
	int balance;
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
		printf("dx %i dy %i\n", dx, dy);
		dy <<= 1;
		balance = dy - dx;
		dx <<= 1;
		printf("dx %i dy %i balance %i\n", dx, dy, balance);
		while (x != x2) {
			mat[y * wx + x] += val;
			if (balance >= 0) {
				y = y + yinc;
				balance = balance - dx;
			}
			balance = balance + dy;
			x = x + xinc;
		}
		mat[y * wx + x] += val;
	} else {
		dx <<= 1;
		balance = dx - dy;
		dy <<= 1;
		while (y != y2) {
			mat[y * wx + x] += val;
			if (balance >= 0) {
				x = x + xinc;
				balance = balance - dy;
			}
			balance = balance + dx;
			y = y + yinc;
		}
		mat[y * wx + x] += val;
	}
}

// Draw a line in 2D space by Wu's Antialiasing Line Algorithm (modified version 1D)
#define ipart_(X) ((int) X)
#define round_(X) ((int)(((double)(X)) + 0.5))
#define fpart_(X) ((double)(X) - (double)ipart_(X))
#define rfpart_(X) (1.0 - fpart_(X))
#define swap_(a, b) do{ __typeof__(a) tmp; tmp = a; a = b; b = tmp; }while(0)
void kernel_draw_2D_line_WALA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
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
		mat[ypxl1 * wx + xpxl1] += (rfpart_(yend) * xgap * val);
		mat[(ypxl1 + 1) * wx + xpxl1] += (fpart_(yend) * xgap * val);
		double intery = yend + gradient;
		
		xend = round_(x2);
		yend = y2 + gradient*(xend - x2);
		xgap = fpart_(x2+0.5);
		int xpxl2 = xend;
		int ypxl2 = ipart_(yend);
		mat[ypxl2 * wx + xpxl2] += (rfpart_(yend) * xgap * val);
		mat[(ypxl2 + 1) * wx + xpxl2] += (fpart_(yend) * xgap * val);
		int x;
		for (x=xpxl1+1; x <= (xpxl2-1); x++) {
			mat[ipart_(intery) * wx + x] += (rfpart_(intery) * val);
			mat[(ipart_(intery) + 1) * wx + x] += (fpart_(intery) * val);
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
		mat[ypxl1 * wx + xpxl1] += (rfpart_(xend) * ygap * val);
		mat[(ypxl1 + 1) * wx + xpxl1] += (fpart_(xend) * ygap * val);
		double interx = xend + gradient;

		yend = round_(y2);
		xend = x2 + gradient*(yend - y2);
		ygap = fpart_(y2+0.5);
		int ypxl2 = yend;
		int xpxl2 = ipart_(xend);
		mat[ypxl2 * wx + xpxl2] += (rfpart_(xend) * ygap * val);
		mat[(ypxl2 + 1) * wx + xpxl2] += (fpart_(xend) * ygap * val);

		int y;
		for(y=ypxl1+1; y <= (ypxl2-1); y++) {
			mat[y * wx + ipart_(interx)] += (rfpart_(interx) * val);
			mat[y * wx + ipart_(interx) + 1] += (fpart_(interx) * val);
			interx += gradient;
		}
	}
}
#undef swap_
#undef ipart_
#undef fpart_
#undef round_
#undef rfpart_

// Draw a line in 2D space by Wu's Line Algorithm (modified version 1D)
void kernel_draw_2D_line_WLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int dy = y2 - y1;
	int dx = x2 - x1;
	int stepx, stepy;

	if (dy < 0) { dy = -dy;  stepy = -1; } else { stepy = 1; }
	if (dx < 0) { dx = -dx;  stepx = -1; } else { stepx = 1; }

	mat[y1 * wx + x1] += val;
	mat[y2 * wx + x2] += val;
	if (dx > dy) {
		int length = (dx - 1) >> 2;
		int extras = (dx - 1) & 3;
		int incr2 = (dy << 2) - (dx << 1);
		if (incr2 < 0) {
			int c = dy << 1;
			int incr1 = c << 1;
			int d =  incr1 - dx;
			for (int i = 0; i < length; i++) {
				x1 = x1 + stepx;
				x2 = x2 - stepx;
				if (d < 0) {                    // Pattern:
					mat[y1 * wx + x1] += val;   //
					x1 = x1 + stepx;            // x o o
					mat[y1 * wx + x1] += val;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {                                 // Pattern:
						mat[y1 * wx + x1] += val;                //       o
						x1 = x1 + stepx;                         //   x o
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						
					} else {
						y1 = y1 + stepy;                      // Pattern
						mat[y1 * wx + x1] += val;             //    o o
						x1 = x1 + stepx;                      //  x
						mat[y1 * wx + x1] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d < 0) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
                }
			}
		} else {
		    int c = (dy - dx) << 1;
			int incr1 = c << 1;
			int d =  incr1 + dx;
			for (int i = 0; i < length; i++) {
				x1 = x1 + stepx;
				x2 = x2 - stepx;
				if (d > 0) {
					y1 = y1 + stepy;           // Pattern
					mat[y1 * wx + x1] += val;  //      o
					x1 = x1 + stepx;           //    o
					y1 = y1 + stepy;           //   x
					mat[y1 * wx + x1] += val;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;  // Pattern
						x1 = x1 + stepx;           //      o
						y1 = y1 + stepy;           //  x o
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						y1 = y1 + stepy;          // Pattern
						mat[y1 * wx + x1] += val; //    o  o
						x1 = x1 + stepx;          //  x
						mat[y1 * wx + x1] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d > 0) {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						if (d > c) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
						}
					}
				}
			}
		}
    } else {
	    int length = (dy - 1) >> 2;
		int extras = (dy - 1) & 3;
		int incr2 = (dx << 2) - (dy << 1);
		if (incr2 < 0) {
			int c = dx << 1;
			int incr1 = c << 1;
			int d =  incr1 - dy;
			for (int i = 0; i < length; i++) {
				y1 = y1 + stepy;
				y2 = y2 - stepy;
				if (d < 0) {
					mat[y1 * wx + x1] += val;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					mat[y2 * wx + x2] += val;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d < 0) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
                } else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
                }
			}
        } else {
			int c = (dx - dy) << 1;
			int incr1 = c << 1;
			int d =  incr1 + dy;
			for (int i = 0; i < length; i++) {
				y1 = y1 + stepy;
				y2 = y2 - stepy;
				if (d > 0) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					x2 = x2 - stepx;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val; 
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d > 0) {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else
                if (d < c) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
                    if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						if (d > c)  {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
					}
				}
			}
		}
	}
}
