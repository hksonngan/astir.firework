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
#include <GL/gl.h>
#include <assert.h>

/**************************************************************
 * Utils (functions know ony by the kernel)
 **************************************************************/

#define SWAP(a, b) {float tmp=(a); (a)=(b); (b)=tmp;}
// Quick sort O(n(log n))
void inkernel_quicksort(float* vec, int l, int r) {
	int key, i, j, k;

	if (l < r) {
		int i, j;
		float pivot;
		pivot = vec[l];
		i = l;
		j = r+1;

		while (1) {
			do ++i; while(vec[i] <= pivot && i <= r);
			do --j; while(vec[j] > pivot);
			if (i >= j) break;
			SWAP(vec[i], vec[j]);
		}
		SWAP(vec[l], vec[j]);
		inkernel_quicksort(vec, l, j-1);
		inkernel_quicksort(vec, j+1, r);

	}
}
// Bubble sort O(n2)
void inkernel_bubblesort(float* vec, int n) {
	bool move = true;
	int i;

	while (move) {
		move = false;
		for (i=0; i<(n-1); ++i) {
			if (vec[i] > vec[i+1]) {
				SWAP(vec[i], vec[i+1]);
				move = true;
			}
		}
	}
}
#undef SWAP

// Mono function
int inkernel_mono(int i, int j) {
	int ma, mi;
if (i>j) {ma = i;}
	else {ma = j;}
	if (i<j) {mi = i;}
	else {mi = j;}

	return mi + ma * (ma - 1) / 2;
}

// Float uniform random generator
float inkernel_randf() {
	return (float)rand() / (float)(RAND_MAX+1.0f);
}

// Float Gauss random generator
#define pi 3.141592653589793238462643383279
float inkernel_randgf(float mean, float std) {
	float u1 = (float)rand() / (float)(RAND_MAX+1.0f);
	float u2 = (float)rand() / (float)(RAND_MAX+1.0f);
	float r1 = sqrt(-2.0f * log(u1));
	float r2 = 2.0f * pi * u2;
	float z0 = r1 * cos(r2);
	//float z1 = r1 * sin(r2);  // 2D Case
	z0 *= std;
	z0 += mean;

	return z0;
}
#undef pi

// Float Gauss random generator 2D Case
#define pi 3.141592653589793238462643383279
void inkernel_randg2f(float mean, float std, float* z0, float* z1) {
	float u1 = (float)rand() / (float)(RAND_MAX+1.0f);
	float u2 = (float)rand() / (float)(RAND_MAX+1.0f);
	float r1 = sqrt(-2.0f * log(u1));
	float r2 = 2.0f * pi * u2;
	*z0 = r1 * cos(r2);
	*z1 = r1 * sin(r2);
	*z0 *= std;
	*z0 += mean;
	*z1 *= std;
	*z1 += mean;
}
#undef pi

/********************************************************************************
 * GENERAL      volume rendering
 ********************************************************************************/

// helper function to rendering volume
void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float* valthr, int nthr, float gamma, float thres){
	int ind, n, x, y, z;
	float r, g, b, l;
	for (n=0; n<nthr; ++n) {
		l = valthr[n];
		if (l <= thres) {continue;}
		ind = 3 * n;
		x = posxyz[ind];
		y = posxyz[ind+1];
		z = posxyz[ind+2];
		r = val[ind];
		g = val[ind+1];
		b = val[ind+2];
		l *= gamma;
		glColor4f(r, g, b, l);
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
// helper function to rendering volume (with edge)
void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float* valthr, int nthr,  float thres){
	int ind, n, x, y, z;
	float r, g, b, l;
	for (n=0; n<nthr; ++n) {
		ind = 3 * n;
		x = posxyz[ind];
		y = posxyz[ind+1];
		z = posxyz[ind+2];
		r = val[ind];
		g = val[ind+1];
		b = val[ind+2];
		l = valthr[n];
		if (l <= thres) {continue;}
		// face 0
		glColor4f(r, g, b, l);
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

// helper to rendering image with OpenGL
void kernel_draw_pixels(float* mapr, int him, int wim, float* mapg, int himg, int wimg, float* mapb, int himb, int wimb) {
	int i, j;
	int npix = him * wim;
	float val;
	int ct = 0;
	glPointSize(1.0f);
	for (i=0; i<him; ++i) {
		for (j=0; j<wim; ++j) {
			//if (val == 0.0f) {continue;}
			glBegin(GL_POINTS);
			glColor3f(mapr[ct], mapg[ct], mapb[ct]);
			glVertex2i(j, i);
			glEnd();
			++ct;
		}
	}
	glColor3f(1.0f, 1.0f, 1.0f);
}

// helper function to colormap image (used in OpenGL MIP rendering)
void kernel_color_image(float* im, int him, int wim,
						float* mapr, int him1, int wim1, float* mapg, int him2, int wim2, float* mapb, int him3, int wim3,
						float* lutr, int him4, float* lutg, int him5, float* lutb, int him6) {
	float val;
	int i, j, ind, pos;
	for (i=0; i<him; ++i) {
		for (j=0; j<wim; ++j) {
			pos = i*wim + j;
			val = im[pos];
			val *= 255.0;
			ind = (int)val;
			mapr[pos] = lutr[ind];
			mapg[pos] = lutg[ind];
			mapb[pos] = lutb[ind];
		}
	}
}


#define pi  3.141592653589
void kernel_mip_volume_rendering(float* vol, int nz, int ny, int nx, float* mip, int him, int wim, float alpha, float beta, float scale) {
	// first some var
	float ts = sqrt(nz*nz + nx*nx) + 1;
	float sizeworld = 2 * wim;
	float center_world = sizeworld / 2.0;
	float center_imx = wim / 2.0;
	float center_imy = him / 2.0;
	float padx = (sizeworld-nx) / 2.0;
	float pady = (sizeworld-ny) / 2.0;
	float padz = (sizeworld-nz) / 2.0;
	int step = nx*ny;
	//printf("ts %f size %f center %f imx %f imy %f\n", ts, sizeworld, center_world, center_imx, center_imy);
	int x, y;
	float xw, yw, zw, x1, y1, z1, x2, y2, z2;
	float xd, yd, zd, xmin, ymin, zmin, xmax, ymax, zmax;
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	float xp1, yp1, zp1, xp2, yp2, zp2;
	int length, lengthy, lengthz, i;
	float xinc, yinc, zinc, maxval, val;

	float ca, sa, cb, sb;
	ca = cos(alpha);
	sa = sin(alpha);
	cb = cos(beta);
	sb = sin(beta);

	for (y=0; y<him; ++y) {
		for (x=0; x<wim; ++x) {
			// init image
			mip[y*wim + x] = 0.0f;
			// origin centre in the world
			xw = x - center_imx;
			yw = y - center_imy;
			zw = -ts;

			// magnefication
			xw = xw * scale;
			yw = yw * scale;
			
			// Rotation 2 axes
			x1 = xw*ca + zw*sa;
			y1 = xw*sb*sa + yw*cb - zw*sb*ca;
			z1 = -xw*sa*cb + yw*sb + zw*cb*ca;
			zw = ts;
			x2 = xw*ca + zw*sa;
			y2 = xw*sb*sa + yw*cb - zw*sb*ca;
			z2 = -xw*sa*cb + yw*sb + zw*cb*ca;
			
			/* One axe
			x1 = xw*cos(alpha) + zw*sin(alpha);
			y1 = yw;
			z1 = -xw*sin(alpha) + zw*cos(alpha);
			zw = ts;
			x2 = xw*cos(alpha) + zw*sin(alpha);
			y2 = yw;
			z2 = -xw*sin(alpha) + zw*cos(alpha);
			*/

			//printf("%f %f %f\n", x1, y1, z1);
			//printf("%f %f %f\n", x2, y2, z2);
			// change origin to raycasting
			x1 += center_world;
			y1 += center_world;
			z1 += center_world;
			x2 += center_world;
			y2 += center_world;
			z2 += center_world;
			// define box and ray direction
			xmin = padx;
			xmax = padx+float(nx);
			ymin = pady;
			ymax = pady+float(ny);
			zmin = padz;
			zmax = padz+float(nz);
			// Rayscasting Smits's algorithm ray-box AABB intersection
			xd = x2 - x1;
			yd = y2 - y1;
			zd = z2 - z1;
			tmin = -1e9f;
			tmax = 1e9f;
			// to fix the singularity  ../xd ../yd ../zd
			if(xd == 0.0f) {xd=0.0000000001f;}
			if(yd == 0.0f) {yd=0.0000000001f;}
			if(zd == 0.0f) {zd=0.0000000001f;}

			// on x
			tmin = (xmin - x1) / xd;
			tmax = (xmax - x1) / xd;
			if (tmin > tmax) {
				buf = tmin;
				tmin = tmax;
				tmax = buf;
			}

			// on y
			tymin = (ymin - y1) / yd;
			tymax = (ymax - y1) / yd;
			if (tymin > tymax) {
				buf = tymin;
				tymin = tymax;
				tymax = buf;
			}
			if (tymin > tmin) {tmin = tymin;}
			if (tymax < tmax) {tmax = tymax;}

			// on z
			tzmin = (zmin - z1) / zd;
			tzmax = (zmax - z1) / zd;
			if (tzmin > tzmax) {
				buf = tzmin;
				tzmin = tzmax;
				tzmax = buf;
			}
			if (tzmin > tmin) {tmin = tzmin;}
			if (tzmax < tmax) {tmax = tzmax;}

			// compute points
			xp1 = x1 + xd * tmin;
			yp1 = y1 + yd * tmin;
			zp1 = z1 + zd * tmin;
			xp2 = x1 + xd * tmax;
			yp2 = y1 + yd * tmax;
			zp2 = z1 + zd * tmax;
			//printf("p1 %f %f %f - p2 %f %f %f\n", xp1, yp1, zp1, xp2, yp2, zp2);
			// check point p1
			if (xp1 >= xmin && xp1 <= xmax) {
				if (yp1 >= ymin && yp1 <= ymax) {
					if (zp1 >= zmin && zp1 <= zmax) {
						xp1 -= padx;
						yp1 -= pady;
						zp1 -= padz;
						if (int(xp1+0.5) == nx) {xp1 = nx-1.0f;}
						if (int(yp1+0.5) == ny) {yp1 = ny-1.0f;}
						if (int(zp1+0.5) == nz) {zp1 = nz-1.0f;}
					} else {continue;}
				} else {continue;}
			} else {continue;}
			// check point p2
			if (xp2 >= xmin && xp2 <= xmax) {
				if (yp2 >= ymin && yp2 <= ymax) {
					if (zp2 >= zmin && zp2 <= zmax) {
						xp2 -= padx;
						yp2 -= pady;
						zp2 -= padz;
						if (int(xp2+0.5) == nx) {xp2 = nx-1.0f;}
						if (int(yp2+0.5) == ny) {yp2 = ny-1.0f;}
						if (int(zp2+0.5) == nz) {zp2 = nz-1.0f;}
					} else {continue;}
				} else {continue;}
			} else {continue;}

			//printf("e %f %f %f    s %f %f %f\n", xp1, yp1, zp1, xp2, yp2, zp2);

			// walk the ray to choose the max intensity with the DDA algorithm
			step = nx * ny;
			length = abs(xp2 - xp1);
			lengthy = abs(yp2 - yp1);
			lengthz = abs(zp2 - zp1);
			if (lengthy > length) {length = lengthy;}
			if (lengthz > length) {length = lengthz;}
			
			xinc = (xp2 - xp1) / (float)length;
			yinc = (yp2 - yp1) / (float)length;
			zinc = (zp2 - zp1) / (float)length;
			xp1 += 0.5;
			yp1 += 0.5;
			zp1 += 0.5;
			maxval = 0.0f;
			for (i=0; i<=length; ++i) {
				val = vol[(int)zp1*step + (int)yp1*nx + (int)xp1];
				if (val > maxval) {maxval = val;}
				xp1 += xinc;
				yp1 += yinc;
				zp1 += zinc;
			}
			
			// Assign new value
			mip[y*wim + x] = maxval;
			
		} // loop j
	} // loop i


}
#undef pi

#define pi  3.141592653589
void kernel_volume_rendering(float* vol, int nz, int ny, int nx, float* mip, int him, int wim, float alpha, float beta, float scale, float th) {
	// first some var
	float ts = sqrt(nz*nz + nx*nx) + 1;
	float sizeworld = 2 * wim;
	float center_world = sizeworld / 2.0;
	float center_imx = wim / 2.0;
	float center_imy = him / 2.0;
	float padx = (sizeworld-nx) / 2.0;
	float pady = (sizeworld-ny) / 2.0;
	float padz = (sizeworld-nz) / 2.0;
	int step = nx*ny;
	//printf("ts %f size %f center %f imx %f imy %f\n", ts, sizeworld, center_world, center_imx, center_imy);
	int x, y;
	float xw, yw, zw, x1, y1, z1, x2, y2, z2;
	float xd, yd, zd, xmin, ymin, zmin, xmax, ymax, zmax;
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	float xp1, yp1, zp1, xp2, yp2, zp2;
	int length, lengthy, lengthz, i;
	float xinc, yinc, zinc, val, newval;
	float light;

	float ca, sa, cb, sb;
	ca = cos(alpha);
	sa = sin(alpha);
	cb = cos(beta);
	sb = sin(beta);

	for (y=0; y<him; ++y) {
		for (x=0; x<wim; ++x) {
			// init image
			mip[y*wim + x] = 0.0f;
			// origin centre in the world
			xw = x - center_imx;
			yw = y - center_imy;
			zw = -ts;

			// magnefication
			xw = xw * scale;
			yw = yw * scale;
			
			// Rotation 2 axes
			x1 = xw*ca + zw*sa;
			y1 = xw*sb*sa + yw*cb - zw*sb*ca;
			z1 = -xw*sa*cb + yw*sb + zw*cb*ca;
			zw = ts;
			x2 = xw*ca + zw*sa;
			y2 = xw*sb*sa + yw*cb - zw*sb*ca;
			z2 = -xw*sa*cb + yw*sb + zw*cb*ca;
			
			/* One axe
			x1 = xw*cos(alpha) + zw*sin(alpha);
			y1 = yw;
			z1 = -xw*sin(alpha) + zw*cos(alpha);
			zw = ts;
			x2 = xw*cos(alpha) + zw*sin(alpha);
			y2 = yw;
			z2 = -xw*sin(alpha) + zw*cos(alpha);
			*/

			//printf("%f %f %f\n", x1, y1, z1);
			//printf("%f %f %f\n", x2, y2, z2);
			// change origin to raycasting
			x1 += center_world;
			y1 += center_world;
			z1 += center_world;
			x2 += center_world;
			y2 += center_world;
			z2 += center_world;
			// define box and ray direction
			xmin = padx;
			xmax = padx+float(nx);
			ymin = pady;
			ymax = pady+float(ny);
			zmin = padz;
			zmax = padz+float(nz);
			// Rayscasting Smits's algorithm ray-box AABB intersection
			xd = x2 - x1;
			yd = y2 - y1;
			zd = z2 - z1;
			tmin = -1e9f;
			tmax = 1e9f;
			// to fix the singularity  ../xd ../yd ../zd
			if(xd == 0.0f) {xd=0.0000000001f;}
			if(yd == 0.0f) {yd=0.0000000001f;}
			if(zd == 0.0f) {zd=0.0000000001f;}
			
			// on x
			tmin = (xmin - x1) / xd;
			tmax = (xmax - x1) / xd;
			if (tmin > tmax) {
				buf = tmin;
				tmin = tmax;
				tmax = buf;
			}

			// on y
			tymin = (ymin - y1) / yd;
			tymax = (ymax - y1) / yd;
			if (tymin > tymax) {
				buf = tymin;
				tymin = tymax;
				tymax = buf;
			}
			if (tymin > tmin) {tmin = tymin;}
			if (tymax < tmax) {tmax = tymax;}

			// on z
			tzmin = (zmin - z1) / zd;
			tzmax = (zmax - z1) / zd;
			if (tzmin > tzmax) {
				buf = tzmin;
				tzmin = tzmax;
				tzmax = buf;
			}
			if (tzmin > tmin) {tmin = tzmin;}
			if (tzmax < tmax) {tmax = tzmax;}
			if (tmin > tmax) {continue;}
			
			// compute points
			xp1 = x1 + xd * tmin;
			yp1 = y1 + yd * tmin;
			zp1 = z1 + zd * tmin;
			xp2 = x1 + xd * tmax;
			yp2 = y1 + yd * tmax;
			zp2 = z1 + zd * tmax;

			// light (ray length)
			light = sqrt((xp1-x1)*(xp1-x1) + (yp1-y1)*(yp1-y1) + (zp1-z1)*(zp1-z1));

			// check point p1
			if (xp1 >= xmin && xp1 <= xmax) {
				if (yp1 >= ymin && yp1 <= ymax) {
					if (zp1 >= zmin && zp1 <= zmax) {
						xp1 -= padx;
						yp1 -= pady;
						zp1 -= padz;
						if (int(xp1+0.5) == nx) {xp1 = nx-1.0f;}
						if (int(yp1+0.5) == ny) {yp1 = ny-1.0f;}
						if (int(zp1+0.5) == nz) {zp1 = nz-1.0f;}
					} else {continue;}
				} else {continue;}
			} else {continue;}
			// check point p2
			if (xp2 >= xmin && xp2 <= xmax) {
				if (yp2 >= ymin && yp2 <= ymax) {
					if (zp2 >= zmin && zp2 <= zmax) {
						xp2 -= padx;
						yp2 -= pady;
						zp2 -= padz;
						if (int(xp2+0.5) == nx) {xp2 = nx-1.0f;}
						if (int(yp2+0.5) == ny) {yp2 = ny-1.0f;}
						if (int(zp2+0.5) == nz) {zp2 = nz-1.0f;}
					} else {continue;}
				} else {continue;}
			} else {continue;}

			//printf("e %f %f %f    s %f %f %f\n", xp1, yp1, zp1, xp2, yp2, zp2);

			// walk the ray and stop of > to the th
			step = nx * ny;
			length = abs(xp2 - xp1);
			lengthy = abs(yp2 - yp1);
			lengthz = abs(zp2 - zp1);
			if (lengthy > length) {length = lengthy;}
			if (lengthz > length) {length = lengthz;}
			
			xinc = (xp2 - xp1) / (float)length;
			yinc = (yp2 - yp1) / (float)length;
			zinc = (zp2 - zp1) / (float)length;
			xp1 += 0.5;
			yp1 += 0.5;
			zp1 += 0.5;
			newval = 0.0f;
			for (i=0; i<=length; ++i) {
				val = vol[(int)zp1*step + (int)yp1*nx + (int)xp1];
				if (val > th) {
					newval = val;
					break;
				}
				xp1 += xinc;
				yp1 += yinc;
				zp1 += zinc;
			}

			// light (ray distance)
			light += i;
			light /= (2*ts);
			light  = 1 - light;
			
			// Assign new value
			light = 1;
			mip[y*wim + x] = newval*light;
			
			
		} // loop j
	} // loop i

	//printf("pix: %i", c);
}
#undef pi

/********************************************************************************
 * GENERAL      line drawing
 ********************************************************************************/

// Draw a line in 2D space by Digital Differential Analyzer method (modified version to 1D)
void kernel_draw_2D_line_DDA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int length, i;
	double x, y;
	double xinc, yinc;
	
	length = abs(x2 - x1);
	if (abs(y2 - y1) > length) {length = abs(y2 - y1);}
	xinc = (double)(x2 - x1) / (double) length;
	yinc = (double)(y2 - y1) / (double) length;
	x    = x1;
	y    = y1;
	for (i=0; i<=length; ++i) {
		mat[(int)y * wx + (int)x] += val;
		x = x + xinc;
		y = y + yinc;
	}
}

// Draw lines in 2D space with DDA
void kernel_draw_2D_lines_DDA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy;
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
		flength = 1.0f / (float)length;
		xinc = diffx * flength;
		yinc = diffy * flength;
		x = x1;
		y = y1;
		for (n=0; n<=length; ++n) {
			mat[(int)y * wx + (int)x] += 1.0f;
			x = x + xinc;
			y = y + yinc;
		}
	}
}

// Draw lines in 2D space with fixed point DDA
#define CONST int(pow(2, 23))
#define float2fixed(X) ((int) X * CONST)
#define intfixed(X) (X >> 23)
void kernel_draw_2D_lines_DDA_fixed(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int length, i, n;
	float flength;
	float lx, ly;
	int fxinc, fyinc, fx, fy;
	int x1, y1, x2, y2, diffx, diffy;

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
		fxinc = float2fixed(diffx / flength);
		fyinc = float2fixed(diffy / flength);
		fx = float2fixed(x1);
		fy = float2fixed(y1);
		for (n=0; n<=length; ++n) {
			mat[intfixed(fy) * wx + intfixed(fx)] = 1.0f;
			fx = fx + fxinc;
			fy = fy + fyinc;
		}
	}
}
#undef CONST
#undef float2fixed
#undef intfixed

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
		dy <<= 1;
		balance = dy - dx;
		dx <<= 1;
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

// Draw lines in 2D space by Bresenham's Line Algorithm (modified version 1D)
void kernel_draw_2D_lines_BLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int x, y, n;
	int x1, y1, x2, y2;
	int dx, dy;
	int xinc, yinc;
	int balance;
	float val = 1.0f;

	for (n=0; n<nx1; ++n) {
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
			//val = 1 / (float)dx;
			dy <<= 1;
			balance = dy - dx;
			dx <<= 1;
			while (x != x2) {
				mat[y * wx + x] = val;
				if (balance >= 0) {
					y = y + yinc;
					balance = balance - dx;
				}
				balance = balance + dy;
				x = x + xinc;
			}
			mat[y * wx + x] = val;
		} else {
			//val = 1 / (float)dy;
			dx <<= 1;
			balance = dx - dy;
			dy <<= 1;
			while (y != y2) {
				mat[y * wx + x] = val;
				if (balance >= 0) {
					x = x + xinc;
					balance = balance - dy;
				}
				balance = balance + dx;
				y = y + yinc;
			}
			mat[y * wx + x] = val;
		}
	}
}

// Draw lines in 2D space by Siddon's Line Algorithm (modified version 1D)
void kernel_draw_2D_lines_SIDDON(float* mat, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int res, int b, int matsize) {
	int n;
	float tx, ty, px, qx, py, qy;
	int ei, ej, u, v, i, j, oldi, oldj;
	int stepi, stepj;
	float divx, divy, runx, runy, oldv, newv, val, valmax;
	float axstart, aystart, astart, pq, stepx, stepy, startl, initl;
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		px = X2[n];
		py = Y2[n];
		qx = X1[n];
		qy = Y1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.4 + 0.1;
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		ei = int((tx-b) / (float)res);
		ej = int((ty-b) / (float)res);
		
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

		axstart = ((u*res)+b-px) / divx;
		aystart = ((v*res)+b-py) / divy;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py));
		stepx = fabs((res*pq / divx));
		stepy = fabs((res*pq / divy));
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		valmax = valmax + valmax*0.01f;
		//valmax = sqrt(stepx * stepx + stepy * stepy);

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
		oldi = -1;
		oldj = -1;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j) {mat[j * wx + i] += val;}
			oldv = newv;
			oldi = i;
			oldj = j;
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
		oldv = startl;
		mat[ej * wx + ei] += 0.707f;
		oldi = -1;
		oldj = -1;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j) {mat[j * wx + i] += val;}
			oldv = newv;
			oldi = i;
			oldj = j;
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

// Draw a list of line in 2D space by Wu's Antialiasing Line Algorithm (modified version 1D)
#define ipart_(X) ((int) X)
#define round_(X) ((int)(((double)(X)) + 0.5))
#define fpart_(X) ((double)(X) - (double)ipart_(X))
#define rfpart_(X) (1.0 - fpart_(X))
#define swap_(a, b) do{ __typeof__(a) tmp; tmp = a; a = b; b = tmp; }while(0)
void kernel_draw_2D_lines_WALA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	double dx, dy, gradient, xend, yend, xgap, ygap, intery, interx;
	int xpxl1, ypxl1, xpxl2, ypxl2, x, y, n;
	float x1, y1, x2, y2;
	float val = 1.0f;
	
	for (n=0; n<nx1; ++n) {
	x1 = X1[n];
	y1 = Y1[n];
	x2 = X2[n];
	y2 = Y2[n];
	dx = (double)x2 - (double)x1;
	dy = (double)y2 - (double)y1;

	if (fabs(dx) > fabs(dy)) {
		if (x2 < x1) {
			swap_(x1, x2);
			swap_(y1, y2);
		}
		
	    gradient = dy / dx;
		xend = round_(x1);
		yend = y1 + gradient * (xend - x1);
		xgap = rfpart_(x1 + 0.5);
		xpxl1 = xend;
		ypxl1 = ipart_(yend);
		mat[ypxl1 * wx + xpxl1] += (rfpart_(yend) * xgap * val);
		mat[(ypxl1 + 1) * wx + xpxl1] += (fpart_(yend) * xgap * val);
		intery = yend + gradient;
		
		xend = round_(x2);
		yend = y2 + gradient*(xend - x2);
		xgap = fpart_(x2+0.5);
		xpxl2 = xend;
		ypxl2 = ipart_(yend);
		mat[ypxl2 * wx + xpxl2] += (rfpart_(yend) * xgap * val);
		mat[(ypxl2 + 1) * wx + xpxl2] += (fpart_(yend) * xgap * val);
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
		gradient = dx / dy;
		yend = round_(y1);
		xend = x1 + gradient*(yend - y1);
		ygap = rfpart_(y1 + 0.5);
		ypxl1 = yend;
		xpxl1 = ipart_(xend);
		mat[ypxl1 * wx + xpxl1] += (rfpart_(xend) * ygap * val);
		mat[(ypxl1 + 1) * wx + xpxl1] += (fpart_(xend) * ygap * val);
		interx = xend + gradient;

		yend = round_(y2);
		xend = x2 + gradient*(yend - y2);
		ygap = fpart_(y2+0.5);
		ypxl2 = yend;
		xpxl2 = ipart_(xend);
		mat[ypxl2 * wx + xpxl2] += (rfpart_(xend) * ygap * val);
		mat[(ypxl2 + 1) * wx + xpxl2] += (fpart_(xend) * ygap * val);

		for(y=ypxl1+1; y <= (ypxl2-1); y++) {
			mat[y * wx + ipart_(interx)] += (rfpart_(interx) * val);
			mat[y * wx + ipart_(interx) + 1] += (fpart_(interx) * val);
			interx += gradient;
		}
	}
	} // for n
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

// Draw a list of lines in 2D space by Wu's Line Algorithm (modified version 1D)
void kernel_draw_2D_lines_WLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int dx, dy, stepx, stepy, n;
	int length, extras, incr2, incr1, c, d, i;
	int x1, y1, x2, y2;
	float val=1.0f;
	for (n=0; n<nx1; ++n) {
		x1 = X1[n];
		y1 = Y1[n];
		x2 = X2[n];
		y2 = Y2[n];
	    dy = y2 - y1;
		dx = x2 - x1;
	
		if (dy < 0) { dy = -dy;  stepy = -1; } else { stepy = 1; }
		if (dx < 0) { dx = -dx;  stepx = -1; } else { stepx = 1; }
		//if (dx > dy) {val = 1 / float(dx);}
		//else {val = 1 / float(dy);}
	
		mat[y1 * wx + x1] = val;
		mat[y2 * wx + x2] = val;
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
					if (d < 0) {                    // Pattern:
						mat[y1 * wx + x1] = val;   //
						x1 = x1 + stepx;            // x o o
						mat[y1 * wx + x1] = val;
						mat[y2 * wx + x2] = val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] = val;
						d += incr1;
					} else {
						if (d < c) {                                 // Pattern:
							mat[y1 * wx + x1] = val;                //       o
							x1 = x1 + stepx;                         //   x o
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
							mat[y2 * wx + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
							
						} else {
							y1 = y1 + stepy;                      // Pattern
							mat[y1 * wx + x1] = val;             //    o o
							x1 = x1 + stepx;                      //  x
							mat[y1 * wx + x1] = val;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
							x2 = x2 - stepx;
							mat[y2 * wx + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d < 0) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] = val;
						}
					} else 
	                if (d < c) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
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
						y1 = y1 + stepy;           // Pattern
						mat[y1 * wx + x1] = val;  //      o
						x1 = x1 + stepx;           //    o
						y1 = y1 + stepy;           //   x
						mat[y1 * wx + x1] = val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] = val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							mat[y1 * wx + x1] = val;  // Pattern
							x1 = x1 + stepx;           //      o
							y1 = y1 + stepy;           //  x o
							mat[y1 * wx + x1] = val;
							mat[y2 * wx + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						} else {
							y1 = y1 + stepy;          // Pattern
							mat[y1 * wx + x1] = val; //    o  o
							x1 = x1 + stepx;          //  x
							mat[y1 * wx + x1] = val;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
							x2 = x2 - stepx;
							mat[y2 * wx + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d > 0) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						}
					} else 
	                if (d < c) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							if (d > c) {
								x2 = x2 - stepx;
								y2 = y2 - stepy;
								mat[y2 * wx + x2] = val;
							} else {
								x2 = x2 - stepx;
								mat[y2 * wx + x2] = val;
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
						mat[y1 * wx + x1] = val;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						mat[y2 * wx + x2] = val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							mat[y1 * wx + x1] = val;
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
							mat[y2 * wx + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						} else {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] = val;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
							x2 = x2 - stepx;
							mat[y2 * wx + x2] = val;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d < 0) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						}
					} else 
	                if (d < c) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						}
	                } else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
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
						mat[y1 * wx + x1] = val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] = val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							mat[y1 * wx + x1] = val;
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
							mat[y2 * wx + x2] = val; 
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						} else {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] = val;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
							x2 = x2 - stepx;
							mat[y2 * wx + x2] = val;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d > 0) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						}
					} else
	                if (d < c) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
	                    if (extras > 2) {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							mat[y1 * wx + x1] = val;
						}
						if (extras > 2) {
							if (d > c)  {
								x2 = x2 - stepx;
								y2 = y2 - stepy;
								mat[y2 * wx + x2] = val;
							} else {
								y2 = y2 - stepy;
								mat[y2 * wx + x2] = val;
							}
						}
					}
				}
			}
		}
	}
}

// Draw a line in 3D space by Bresenham's Line Algorithm (modified version 1D)
void kernel_draw_3D_line_BLA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val) {
	int x, y, z;
	int dx, dy, dz;
	int xinc, yinc, zinc;
	int balance1, balance2;
	int step = wx*wy;

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
			mat[z * step + y * wx + x] += val;
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
		mat[z * step + y * wx + x] += val;
	} else {
		if (dy >= dx && dy >= dz) {
			dx <<= 1;
			dz <<= 1;
			balance1 = dx - dy;
			balance2 = dz - dy;
			dy <<= 1;
			while (y != y2) {
				mat[z * step + y * wx + x] += val;
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
			mat[z * step + y * wx + x] += val;
		} else {
			dx <<= 1;
			dy <<= 1;
			balance1 = dx - dz;
			balance2 = dy - dz;
			dz <<= 1;
			while (z != z2) {
				mat[z * step + y * wx + x] += val;
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
			mat[z * step + y * wx + x] += val;
		}
	}
}

/**************************************************************
 * Vector/Matrix operations
 **************************************************************/

// Count non-zeros elements inside the matrix
int kernel_matrix_nonzeros(float* mat, int ni, int nj) {
	int i, j, ind;
	int c=0;
	for (i=0; i<ni; ++i) {
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			if (mat[ind + j] != 0) {++c;}
		}
	}
	return c;
}

// Count non-zeros elements per rows inside a matrix
void kernel_matrix_nonzeros_rows(float* mat, int ni, int nj, int* rows, int nrows) {
	int i, j, ind;
	int c = 0;
	for(i=0; i<ni; ++i) {
		ind = i*nj;
		c = 0;
		for (j=0; j<nj; ++j) {
			if (mat[ind + j] != 0) {++c;}
		}
		rows[i] = c;
	}
}

// Compute matrix col sum
void kernel_matrix_sumcol(float* mat, int ni, int nj, float* im, int npix) {
	int i, j, ind;
	for (i=0; i<ni; ++i) {
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			im[j] += mat[ind + j];
		}
	}
}

// Count non-zeros elements inside the matrix
int kernel_vector_nonzeros(float* mat, int ni) {
	int i;
	int c=0;
	for (i=0; i<ni; ++i) {
		if (mat[i] != 0) {++c;}
	}
	return c;
}

/**************************************************************
 * Filteration                                               
 **************************************************************/

// Helper to build H matrix for a low pass filter
void kernel_matrix_lp_H(float* mat, int nk, int nj, int ni, float fc, int order) {
	int i, j, k, step;
	float c, r, size, fi, fj, fk, forder;
	
	forder = (float)order * 2.0f;
	step = nj*ni;
	c = ((float)ni - 1.0f) / 2.0f;
	size = (float)nj - 1.0f;
	for (k=0; k<nk; ++k) {
		for (j=0; j<nj; ++j) {
			for (i=0; i<ni; ++i) {
				fi = (float)i;
				fj = (float)j;
				fk = (float)k;
				r = sqrt((fi-c)*(fi-c) + (fj-c)*(fj-c) + (fk-c)*(fk-c));
				r = r / size;
				r = pow((r / fc), forder);
				r = sqrt(1 + r);
				mat[k*step + i*nj +j] = 1 / r;
			}
		}
	}

}

// Quick Gaussian filter on volume
void kernel_flatvolume_gaussian_filter_3x3x3(float* mat, int nmat, int nk, int nj, int ni) {
	float kernel[] = {4.0f/14.0f, 6.0f/14.0f, 4.0f/14.0f};
	float sum;
	int i, j, k, indi, indk;
	float* res = (float*)calloc(nmat, sizeof(float));
	int step = ni*nj;
	// first on x
	for (k=1; k<(nk-1); ++k) {
		indk = k*step;
		for (i=1; i<(ni-1); ++i) {
			indi = indk + i*nj;
			for (j=1; j<(nj-1); ++j) {
				sum = 0.0f;
				sum += (mat[indi+j-1] * kernel[0]);
				sum += (mat[indi+j] * kernel[1]);
				sum += (mat[indi+j+1] * kernel[2]);
				res[indi+j] = sum;
			}
		}
	}
	// then on y
	for (k=1; k<(nk-1); ++k) {
		indk = k*step;
		for (j=1; i<(nj-1); ++j) {
			for (i=1; i<(ni-1); ++i) {
				sum = 0.0f;
				sum += (mat[indk+(i-1)*nj+j] * kernel[0]);
				sum += (mat[indk+i*nj+j] * kernel[1]);
				sum += (mat[indk+(i+1)*nj+j] * kernel[2]);
				res[indk+i*nj+j] = sum;
			}
		}
	}
	// at the end on z
	for (i=1; i<(ni-1); ++i) {
		indi = i*nj;
		for (j=1; j<(nj-1); ++j) {
			indk = indi+j;
			for (k=1; k<(nk-1); ++k) {
				sum = 0.0f;
				sum += (mat[(k-1)*step+indk] * kernel[0]);
				sum += (mat[k*step+indk] * kernel[1]);
				sum += (mat[(k+1)*step+indk] * kernel[2]);
				res[k*step+indk] = sum;
			}
		}
	}
	// swap result
	memcpy(mat, res, nmat*sizeof(float));
	free(res);
}


// 2d median filter
void kernel_filter_2d_median(float* im, int ny, int nx, float* res, int nyr, int nxr, int w) {
	int nwin = w*w;
	float* win = (float*)malloc(nwin * sizeof(float));
	int edgex = w / 2;
	int edgey = w / 2;
	int mpos = nwin / 2;
	int x, y, wx, wy, ind, indy, indw;
	for (y=edgey; y<(ny-edgey); ++y) {
		ind = y*ny;
		for (x=edgex; x<(nx-edgex); ++x) {
			for (wy=0; wy<w; ++wy) {
				indw = wy*w;
				indy = ny*(y + wy - edgey);
				for (wx=0; wx<w; ++wx) {
					win[indw + wx] = im[indy + x + wx - edgex];
				}
			}
			// sort win
			inkernel_quicksort(win, 0, nwin-1);
			// select mpos
			res[ind + x] = win[mpos];
		}
	}
}

// 3d median filter
void kernel_filter_3d_median(float* im, int nz, int ny, int nx, float* res, int nzr, int nyr, int nxr, int w) {
	int nwin = w*w*w;
	float* win = (float*)malloc(nwin * sizeof(float));
	int edgex = w / 2;
	int edgey = w / 2;
	int edgez = w / 2;
	int mpos = nwin / 2;
	int step = ny*nx;
	int x, y, z, wx, wy, wz, ind, indy, indz, indw;
	int nwa;
	for (z=edgez; z<(nz-edgez); ++z) {
		indz = z * step;
		for (y=edgey; y<(ny-edgey); ++y) {
			ind = indz + y*ny;
			for (x=edgex; x<(nx-edgex); ++x) {
				nwa = 0;
				for (wz=0; wz<w; ++wz) {
					indw = step * (z + wz - edgez);
					for (wy=0; wy<w; ++wy) {
						indy = indw + ny*(y + wy - edgey);
						for (wx=0; wx<w; ++wx) {
							win[nwa] = im[indy + x + wx - edgex];
							++nwa;
						}
					}
				}
				// sort win
				inkernel_quicksort(win, 0, nwin-1);
				// select mpos
				res[ind + x] = win[mpos];
			}
		}
	}
}

// 2d adaptive median filter
void kernel_filter_2d_adaptive_median(float* im, int ny, int nx, float* res, int nyr, int nxr, int w, int wmax) {
	int nwin = wmax*wmax;
	float* win = (float*)malloc(nwin * sizeof(float));
	int size_mem_im = ny * nx * sizeof(float);
	float smin, smead, smax;
	int edgex, edgey;
	int wa, nwa;
	int x, y, wx, wy, ind, indy;

	for (wa=w; wa<=wmax; wa+=2) {
		edgex = wa / 2;
		edgey = wa / 2;
		for (y=edgey; y<(ny-edgey); ++y) {
			ind = y * ny;
			for (x=edgex; x<(nx-edgex); ++x) {
				// read windows
				nwa = 0;
				for (wy=0; wy<wa; ++wy) {
					indy = ny * (y + wy - edgey);
					for (wx=0; wx<wa; ++wx) {
						win[nwa] = im[indy + x + wx - edgex];
						++nwa;
					} // wx
				} // wy
				// sort win
				inkernel_quicksort(win, 0, nwa-1);
				// get values
				smin = win[0];
				smead = win[nwa/2];
				smax = win[nwa-1];
				// median filter
				if ((smin < smead) && (smead < smax)) {
					// step 5.
					if ((smin < im[ind + x]) && (im[ind + x] < smax)) {
						res[ind + x] = im[ind + x];
					} else {
						res[ind + x] = smead;
					}
				} else {
					res[ind + x] = smead;
				}

			} // x
		} // y
		if (wa != wmax) {memcpy(im, res, size_mem_im);} 
	} // wa

}

// 3d adaptive median filter
void kernel_filter_3d_adaptive_median(float* im, int nz, int ny, int nx,
									  float* res, int nzr, int nyr, int nxr, int w, int wmax) {
	int nwin = wmax*wmax*wmax;
	float* win = (float*)malloc(nwin * sizeof(float));
	int size_mem_im = nz * ny * nx * sizeof(float);
	int step = ny * nx;
	float smin, smead, smax;
	int edgex, edgey, edgez;
	int wa, nwa;
	int x, y, z, wx, wy, wz, ind, indimz, indy, indz;

	for (wa=w; wa<=wmax; wa+=2) {
		edgex = wa / 2;
		edgey = wa / 2;
		edgez = wa / 2;
		for (z=edgez; z<(nz-edgez); ++z) {
			indimz = step * z;
			for (y=edgey; y<(ny-edgey); ++y) {
				ind = indimz + y * ny;
				for (x=edgex; x<(nx-edgex); ++x) {
					// read windows
					nwa = 0;
					for (wz=0; wz<wa; ++wz) {
						indz = step * (z + wz - edgez);
						for (wy=0; wy<wa; ++wy) {
							indy = indz + ny * (y + wy - edgey);
							for (wx=0; wx<wa; ++wx) {
								win[nwa] = im[indy + x + wx - edgex];
								++nwa;
							} // wx
						} // wy
					} // wz
					// sort win
					inkernel_quicksort(win, 0, nwa-1);
					// get values
					smin = win[0];
					smead = win[nwa/2];
					smax = win[nwa-1];
					// median filter
					res[ind + x] = smead;
					if ((smin < smead) && (smead < smax)) {
						// step 5.
						if ((smin < im[ind + x]) && (im[ind + x] < smax)) {
							res[ind + x] = im[ind + x];
						} else {
							res[ind + x] = smead;
						}
					} else {
						res[ind + x] = smead;
					}

				} // x
			} // y
		} // z
		if (wa != wmax) {memcpy(im, res, size_mem_im);} 
	} // wa
}

/**************************************************************
 * Interpolation
 **************************************************************/

// 3D Resampling by Lanczos3 (uses backwarp mapping)
#define pi 3.141592653589793238462643383279
#define SINC(x) ((x)==(0)?1:sin(pi*(x))/(pi*(x)))
void kernel_resampling_3d_Lanczos3(float* org, int noz, int noy, int nox, float* trg, int nz, int ny, int nx) {
	// scale factor
	float scalez = noz / float(nz);
	float scaley = noy / float(ny);
	float scalex = nox / float(nx);
	int stepo = nox*noy;
	int stept = nx*ny;
	// backward mapping, thus scan from the target
	int x, y, z;
	int xi, yi, zi;
	float xt, yt, zt;
	int u, v, w;
	int wz, wy, wx;
	float p, q, r;
	float dx, dy, dz;
	for (z=0; z<nz; ++z) {
		printf("slice z = %i / %i\n", z+1, nz); 
		zt = z * scalez;
		zi = (int)zt;
		
		for (y=0; y<ny; ++y) {
			yt = y * scaley;
			yi = (int)yt;
			
			for (x=0; x<nx; ++x) {
				xt = x * scalex;
				xi = (int)xt;

				// window loop
				r = 0;
				for (wz = -2; wz < 4; ++wz) {
					w = zi + wz;
					if (w >= noz) {continue;}
					if (w < 0) {continue;}
					dz = zt - w;
					if (abs(dz) > 3.0f) {dz = 3.0f;}
					q = 0;
					for (wy = -2; wy < 4; ++wy) {
						v = yi + wy;
						if (v >= noy) {continue;}
						if (v < 0) {continue;}
						dy = yt - v;
						if (abs(dy) > 3.0f) {dy = 3.0f;}
						p = 0;
						for (wx = -2; wx < 4; ++wx) {
							u = xi + wx;
							if (u >= nox) {continue;}
							if (u < 0) {continue;}
							dx = xt - u;
							if (abs(dx) > 3.0f) {dx = 3.0f;}
							p = p + org[w*stepo + v*nox + u] * SINC(dx) * SINC(dx * 0.333333f);
						} // wx
						q = q + p * SINC(dy) * SINC(dy * 0.333333f);
					} // wy
					r = r + q * SINC(dz) * SINC(dz * 0.333333f);
				} // wz

				// assign the new value
				trg[z*stept + y*nx + x] = r;
				
			} // x
		} // y
	} // z

}
#undef pi
#undef SINC

// 3D Resampling by Lanczos2 (uses backwarp mapping)
#define pi 3.141592653589793238462643383279
#define SINC(x) ((x)==(0)?1:sin(pi*(x))/(pi*(x)))
void kernel_resampling_3d_Lanczos2(float* org, int noz, int noy, int nox, float* trg, int nz, int ny, int nx) {
	// scale factor
	float scalez = noz / float(nz);
	float scaley = noy / float(ny);
	float scalex = nox / float(nx);
	int stepo = nox*noy;
	int stept = nx*ny;
	// backward mapping, thus scan from the target
	int x, y, z;
	int xi, yi, zi;
	float xt, yt, zt;
	int u, v, w;
	int wz, wy, wx;
	float p, q, r;
	float dx, dy, dz;
	for (z=0; z<nz; ++z) {
		printf("slice z = %i / %i\n", z+1, nz); 
		zt = z * scalez;
		zi = (int)zt;
		
		for (y=0; y<ny; ++y) {
			yt = y * scaley;
			yi = (int)yt;
			
			for (x=0; x<nx; ++x) {
				xt = x * scalex;
				xi = (int)xt;

				// window loop
				r = 0;
				for (wz = -1; wz < 3; ++wz) {
					w = zi + wz;
					if (w >= noz) {continue;}
					if (w < 0) {continue;}
					dz = zt - w;
					if (abs(dz) > 2.0f) {dz = 2.0f;}
					q = 0;
					for (wy = -1; wy < 3; ++wy) {
						v = yi + wy;
						if (v >= noy) {continue;}
						if (v < 0) {continue;}
						dy = yt - v;
						if (abs(dy) > 2.0f) {dy = 2.0f;}
						p = 0;
						for (wx = -1; wx < 3; ++wx) {
							u = xi + wx;
							if (u >= nox) {continue;}
							if (u < 0) {continue;}
							dx = xt - u;
							if (abs(dx) > 2.0f) {dx = 2.0f;}
							p = p + org[w*stepo + v*nox + u] * SINC(dx) * SINC(dx * 0.5f);
						} // wx
						q = q + p * SINC(dy) * SINC(dy * 0.5f);
					} // wy
					r = r + q * SINC(dz) * SINC(dz * 0.5f);
				} // wz

				// assign the new value
				trg[z*stept + y*nx + x] = r;
				
			} // x
		} // y
	} // z

}
#undef pi
#undef SINC

// 2D Resampling by Lanczos2 (uses backwarp mapping)
#define pi 3.141592653589793238462643383279
#define SINC(x) ((x)==(0)?1:sin(pi*(x))/(pi*(x)))
void kernel_resampling_2d_Lanczos2(float* org, int noy, int nox, float* trg, int ny, int nx) {
	// scale factor
	float scaley = noy / float(ny);
	float scalex = nox / float(nx);
	// backward mapping, thus scan from the target
	int x, y;
	int xi, yi;
	float xt, yt;
	int u, v;
	int wy, wx;
	float p, q;
	float dx, dy;

	for (y=0; y<ny; ++y) {
		yt = y * scaley;
		yi = (int)yt;
			
		for (x=0; x<nx; ++x) {
			xt = x * scalex;
			xi = (int)xt;

			// window loop
			q = 0;
			for (wy = -1; wy < 3; ++wy) {
				v = yi + wy;
				if (v >= noy) {continue;}
				if (v < 0) {continue;}
				dy = yt - v;
				if (abs(dy) > 2.0f) {dy = 2.0f;}
				p = 0;
				for (wx = -1; wx < 3; ++wx) {
					u = xi + wx;
					if (u >= nox) {continue;}
					if (u < 0) {continue;}
					dx = xt - u;
					if (abs(dx) > 2.0f) {dx = 2.0f;}
					p = p + org[v*nox+u] * SINC(dx) * SINC(dx * 0.5f);
				} // wx
				q = q + p * SINC(dy) * SINC(dy * 0.5f);
			} // wy

			// assign the new value
			trg[y*nx + x] += q;
				
		} // x
	} // y
}
#undef pi
#undef SINC

