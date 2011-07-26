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

#include "filter.ch"

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
#undef SWAP

/**************************************************************
 * Filter 
 **************************************************************/

// 2d median filter
void filter_c_2d_median(float* im, int ny, int nx, float* res, int nyr, int nxr, int w) {
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
void filter_c_3d_median(float* im, int nz, int ny, int nx, float* res, int nzr, int nyr, int nxr, int w) {
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
void filter_c_2d_adaptive_median(float* im, int ny, int nx, float* res, int nyr, int nxr, int w, int wmax) {
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
void filter_c_3d_adaptive_median(float* im, int nz, int ny, int nx,
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
