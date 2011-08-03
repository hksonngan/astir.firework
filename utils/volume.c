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

#include "volume.h"

// 3D Resampling by Lanczos2 (uses backwarp mapping)
#define pi 3.141592653589793238462643383279
#define SINC(x) ((x)==(0)?1:sin(pi*(x))/(pi*(x)))
void volume_c_resampling_lanczos2(float* org, int noz, int noy, int nox, float* trg, int nz, int ny, int nx) {
	// scale factor
	float scalez = noz / (float)nz;
	float scaley = noy / (float)ny;
	float scalex = nox / (float)nx;
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

// 3D Resampling by Lanczos3 (uses backwarp mapping)
#define pi 3.141592653589793238462643383279
#define SINC(x) ((x)==(0)?1:sin(pi*(x))/(pi*(x)))
void volume_c_resampling_lanczos3(float* org, int noz, int noy, int nox, float* trg, int nz, int ny, int nx) {
	// scale factor
	float scalez = noz / (float)nz;
	float scaley = noy / (float)ny;
	float scalex = nox / (float)nx;
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

