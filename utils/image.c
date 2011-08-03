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

#include "image.h"

// 2D Resampling by Lanczos2 (uses backwarp mapping)
#define pi 3.141592653589793238462643383279
#define SINC(x) ((x)==(0)?1:sin(pi*(x))/(pi*(x)))
void image_c_resampling_lanczos2(float* org, int noy, int nox, float* trg, int ny, int nx) {
	// scale factor
	float scaley = noy / (float)ny;
	float scalex = nox / (float)nx;
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

