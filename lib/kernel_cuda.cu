#include "kernel_cuda.h"
#include <stdio.h>
#include <cublas.h>
#include <cufft.h>
#include <sys/time.h>
#include <math_constants.h>

// textures
texture<float, 1, cudaReadModeElementType> tex1;
texture<float, 1, cudaReadModeElementType> tex_im;
texture<float, 1, cudaReadModeElementType> tex_mumap;
texture<unsigned short, 1, cudaReadModeElementType> tex_x1;
texture<unsigned short, 1, cudaReadModeElementType> tex_y1;
texture<unsigned short, 1, cudaReadModeElementType> tex_z1;
texture<unsigned short, 1, cudaReadModeElementType> tex_x2;
texture<unsigned short, 1, cudaReadModeElementType> tex_y2;
texture<unsigned short, 1, cudaReadModeElementType> tex_z2;

/*
// DEV draw one pixel per thread, if the thread is alon the line. Too slow...
__global__ void dev_draw(float* d_im, unsigned short int* d_x1, unsigned short int* d_y1,
						 unsigned short int* d_z1, unsigned short int* d_x2, unsigned short int* d_y2,
						 unsigned short int* d_z2, int wim, int nx1, int nim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x1, y1, z1, x2, y2, z2, x, y, z, n, step, color;
	//unsigned short int toto;
	float dx, dy, dz, mag, u, xt, yt, zt, d;
	step = wim*wim;
	if (idx < nim) {
		//color = d_im[idx];
		color = 0;
		for (n=0; n<nx1; ++n) {
			
			z = idx / step;
			x = (idx - (z * step));
			y = x / wim;
			x = (x - (y * wim));
			//x1 = d_x1[n];
			//y1 = d_y1[n];
			//z1 = d_z1[n];
			//x1 = tex1Dfetch(tex_x1, n);
			//y1 = tex1Dfetch(tex_y1, n);
			//z1 = tex1Dfetch(tex_z1, n);
			//x2 = d_x2[n];
			//y2 = d_y2[n];
			//z2 = d_z2[n];
			x2 = 20;
			x1 = 10;
			y1 = 10;
			y2 = 10;
			z1 = 20;
			z2 = 10;
			dx = x2-x1;
			dy = y2-y1;
			dz = z2-z1;
			mag = __powf(dx*dx + dy*dy + dz*dz, 0.5);
			u = ((x-x1)*dx + (y-y1)*dy + (z-z1)*dz) / (mag*mag);
			xt = x1 + u*dx;
			yt = y1 + u*dy;
			zt = z1 + u*dz;
			d = __powf((x-xt)*(x-xt) + (y-yt)*(y-yt) + (z-zt)*(z-zt), 0.5);
			//d_im[idx] = d;
			if (d < .707f) {color++;}
			//d = d + 0.5f;
			//d = int(1 / d);
			//color = d;
			//d = 0.5f;
			//color += (x1 + y1 + z1);
			//if (x < 0.707f) {color++;}
			//color += d;
			//__syncthreads();
			}
		d_im[idx] = color;
		
	}
}
*/

__device__ inline void atomicFloatAdd(float* address, float val) {
	int i_val = __float_as_int(val);
	int tmp0 = 0;
	int tmp1;

	while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
	{
		tmp0 = tmp1;
		i_val = __float_as_int(val + __int_as_float(tmp1));
	}

}

// kernel to update image in pet2D EMML algorithm
__global__ void pet2D_im_update(float* im, float* S, float* F, int npix) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < npix && F[idx] != 0.0f) {
		im[idx] = im[idx] / S[idx] * F[idx];
		//im[idx] = F[idx];
	}
}
// kernel to update Q value in pet2D EMML algorithm
__global__ void pet2D_Q_update(int* d_lorval, float* d_Q, int nval) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nval) {
		d_Q[idx] = (float)d_lorval[idx] / d_Q[idx];
	}
}
// kernel to raytrace line in SRM with DDA algorithm
__global__ void pet2D_SRM_DDA(float* d_SRM, int* d_X1, int* d_Y1, int* d_X2, int* d_Y2, int wx, int nx1, int width_image) {
	int length, n, x1, y1, x2, y2, diffx, diffy, LOR_ind;
	float flength, val, x, y, lx, ly, xinc, yinc;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nx1) {
		LOR_ind = idx * wx;
		x1 = d_X1[idx];
		y1 = d_Y1[idx];
		x2 = d_X2[idx];
		y2 = d_Y2[idx];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		val = 1.0f / flength;
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		for (n=0; n<=length; ++n) {
			d_SRM[LOR_ind + (int)y * width_image + (int)x] = val;
			x = x + xinc;
			y = y + yinc;
		}
	}
}
// kernel to raytrace line in SRM with DDA algorithm and ELL sparse matrix format 
__global__ void pet2D_SRM_DDA_ELL(float* d_SRM_vals, int* d_SRM_cols, int* d_x1, int* d_y1, int* d_x2, int* d_y2, int wsrm, int wim, int nx1) {
	int length, n, x1, y1, x2, y2, diffx, diffy, LOR_ind;
	float flength, val, x, y, lx, ly, xinc, yinc;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	val = 1.0f;
	if (idx < nx1) {
		LOR_ind = idx * wsrm;
		x1 = d_x1[idx];
		x2 = d_x2[idx];
		y1 = d_y1[idx];
		y2 = d_y2[idx];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		//val  = 1.0f / flength;
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		for (n=0; n<=length; ++n) {
			d_SRM_vals[LOR_ind + n] = val;
			d_SRM_cols[LOR_ind + n] = (int)y * wim + (int)x;
			x = x + xinc;
			y = y + yinc;
		}
		d_SRM_cols[LOR_ind + n] = -1; // eof
	}
}
/*
// DEV
__global__ void pet3D_IM_SRM_DDA_DEV(unsigned short int* d_x1, unsigned short int* d_y1, unsigned short int* d_z1,
								  unsigned short int* d_x2, unsigned short int* d_y2, unsigned short int* d_z2,
								  int wsrm, int wim, int nx1) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	if (idx < wim) {
		for (i=0;i<nx1;++i) {
			


		}



	}
}
*/


// kernel to raytrace 3D line in SRM with DDA algorithm and ELL sparse matrix format 
__global__ void pet3D_SRM_DDA_ELL(float* d_SRM_vals, int* d_SRM_cols, int wsrm, int wim, int nx1) {
	int length, n, diffx, diffy, diffz, LOR_ind, step;
	float flength, val, x, y, z, lx, ly, lz, xinc, yinc, zinc;
	unsigned short int x1, y1, z1, x2, y2, z2;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	val = 1.0f;
	step = wim*wim;
	
	if (idx < nx1) {
		LOR_ind = idx * wsrm;
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		x2 = tex1Dfetch(tex_x2, idx);
		y2 = tex1Dfetch(tex_y2, idx);
		z2 = tex1Dfetch(tex_z2, idx);

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
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		z = z1 + 0.5f;
		for (n=0; n<=length; ++n) {
			d_SRM_vals[LOR_ind + n] = val;
			d_SRM_cols[LOR_ind + n] = (int)z * step + (int)y * wim + (int)x;
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		d_SRM_cols[LOR_ind + n] = -1; // eof
	}

}
// kernel to raytrace 3D line in SRM with DDA algorithm on-line
__global__ void pet3D_SRM_DDA_ON(int* d_im, int wim, int nx1, int nim) {

	int length, n, diffx, diffy, diffz, step;
	float flength, x, y, z, lx, ly, lz, xinc, yinc, zinc;
	unsigned short int x1, y1, z1, x2, y2, z2;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	step = wim*wim;
	
	if (idx < nx1) {
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		x2 = tex1Dfetch(tex_x2, idx);
		y2 = tex1Dfetch(tex_y2, idx);
		z2 = tex1Dfetch(tex_z2, idx);
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
			atomicAdd(&d_im[int(z) * step + (int)y * wim + (int)x], 1);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}
}

// kernel to raytrace 3D line in SRM with DDA algorithm on-line
__global__ void pet3D_SRM_DDA_fixed_ON(int* d_im, int wim, int nx1, int nim) {

	int length, n, diffx, diffy, diffz, step;
	float flength, lx, ly, lz;
	int x, y, z, xinc, yinc, zinc;
	int xt, yt, zt;
	unsigned short int x1, y1, z1, x2, y2, z2;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	step = wim*wim;
	
	if (idx < nx1) {
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		x2 = tex1Dfetch(tex_x2, idx);
		y2 = tex1Dfetch(tex_y2, idx);
		z2 = tex1Dfetch(tex_z2, idx);
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
		xinc = (int)(diffx * flength * 8388608);
		yinc = (int)(diffy * flength * 8388608);
		zinc = (int)(diffz * flength * 8388608);
		x = (int)(x1 * 8388608);
		y = (int)(y1 * 8388608);
		z = (int)(z1 * 8388608);
		for (n=0; n<=length; ++n) {
			xt = x;
			yt = y;
			zt = z;
			atomicAdd(&d_im[(zt >> 23) * step + (yt >> 23) * wim + (xt >> 23)], 1);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}
}


// kernel to raytrace 3D line in SRM with DDA algorithm and compute F on-line
__global__ void pet3D_SRM_DDA_F_ON(unsigned int* d_F, int wim, int nx1, int nim, float scale) {

	int length, n, diffx, diffy, diffz, step;
	float flength, x, y, z, lx, ly, lz, xinc, yinc, zinc, Qi;
	unsigned short int x1, y1, z1;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	step = wim*wim;
	
	if (idx < nx1) {
		Qi = 0.0f;
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		diffx = tex1Dfetch(tex_x2, idx)-x1;
		diffy = tex1Dfetch(tex_y2, idx)-y1;
		diffz = tex1Dfetch(tex_z2, idx)-z1;
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
			Qi = Qi + tex1Dfetch(tex_im, (int)z * step + (int)y * wim + (int)x);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}

		// compute F
		if (Qi==0.0f) {return;}
		Qi = 1 / Qi;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			//atomicFloatAdd(&d_F[(int)z * step + (int)y * wim + (int)x], Qi);
			atomicAdd(&d_F[(int)z * step + (int)y * wim + (int)x], (unsigned int)(Qi*scale));
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}
}

// Same as pet3D_SRM_DDA_F_ON with attenuation correction
__global__ void pet3D_SRM_DDA_F_ATT_ON(unsigned int* d_F, int wim, int nx1, int nim, float scale) {

	int length, n, diffx, diffy, diffz, step, ind;
	float flength, x, y, z, lx, ly, lz, xinc, yinc, zinc, Qi, Ai;
	unsigned short int x1, y1, z1;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	step = wim*wim;
	
	if (idx < nx1) {
		Qi = 0.0f;
		Ai = 0.0f;
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		diffx = tex1Dfetch(tex_x2, idx)-x1;
		diffy = tex1Dfetch(tex_y2, idx)-y1;
		diffz = tex1Dfetch(tex_z2, idx)-z1;
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
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		z = z1 + 0.5f;
		for (n=0; n<=length; ++n) {
			ind = (int)z * step + (int)y * wim + (int)x;
			Qi = Qi + tex1Dfetch(tex_im, ind);
			Ai = Ai - tex1Dfetch(tex_mumap, ind);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// compute F
		if (Qi==0.0f) {return;}
		if (Ai < -5.0f) {Ai = -5.0f;}
		Qi = Qi * __expf(Ai);
		Qi = 1 / Qi;
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		z = z1 + 0.5f;
		for (n=0; n<=length; ++n) {
			//atomicFloatAdd(&d_F[(int)z * step + (int)y * wim + (int)x], Qi);
			atomicAdd(&d_F[(int)z * step + (int)y * wim + (int)x], (unsigned int)(Qi*scale));
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}

}
// kernel to raytrace 3D line in SRM with DDA algorithm and ELL sparse matrix format 
__global__ void pet3D_SRM_DDA_ELL_Q(float* d_SRM_vals, int* d_SRM_cols, float* d_im, float* d_Q,
									unsigned short int* d_x1, unsigned short int* d_y1, unsigned short int* d_z1,
									unsigned short int* d_x2, unsigned short int* d_y2, unsigned short int* d_z2,
									int wsrm, int wim, int nx1) {
	int length, n, x1, y1, z1, x2, y2, z2, diffx, diffy, diffz, LOR_ind, step, vcol;
	float flength, val, x, y, z, lx, ly, lz, xinc, yinc, zinc, Qi;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	val = 1.0f;
	step = wim*wim;
	Qi = 0.0f;
	
	if (idx < nx1) {
		LOR_ind = idx * wsrm;
		x1 = d_x1[idx];
		x2 = d_x2[idx];
		y1 = d_y1[idx];
		y2 = d_y2[idx];
		z1 = d_z1[idx];
		z2 = d_z2[idx];
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
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		z = z1 + 0.5f;
		for (n=0; n<=length; ++n) {
			d_SRM_vals[LOR_ind + n] = val;
			vcol = (int)z * step + (int)y * wim + (int)x;
			d_SRM_cols[LOR_ind + n] = vcol;
			Qi = Qi + d_im[vcol];
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		d_SRM_cols[LOR_ind + n] = -1; // eof
		d_Q[idx] = Qi;
	}

}
// kernel to raytrace line in SRM with DDA anti-aliased version 2 pix, SRM is in ELL sparse matrix format 
__global__ void pet2D_SRM_DDAA_ELL(float* d_SRM_vals, int* d_SRM_cols, int* d_x1, int* d_y1, int* d_x2, int* d_y2, int wsrm, int wim, int nx1) {
	int length, n, x1, y1, x2, y2, diffx, diffy, LOR_ind, ind, ind2, xint, yint;
	float flength, val, vu, x, y, lx, ly, xinc, yinc;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nx1) {
		LOR_ind = idx * wsrm;
		x1 = d_x1[idx];
		x2 = d_x2[idx];
		y1 = d_y1[idx];
		y2 = d_y2[idx];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		// first pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5f));
		d_SRM_vals[LOR_ind] = val;
		d_SRM_cols[LOR_ind] = yint * wim + xint;
		x = x + xinc;
		y = y + yinc;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			ind = yint * wim + xint;
			val = 1 - fabs(x - (xint + 0.5f));
			vu = (x - xint) * 0.5f;
			// vd = 0.5 - vu;
			ind2 = LOR_ind + 2*n;
			d_SRM_vals[ind2] = vu;
			d_SRM_cols[ind2] = ind + 1;
			d_SRM_vals[ind2 + 1] = val;
			d_SRM_cols[ind2 + 1] = ind;
			x = x + xinc;
			y = y + yinc;
		}
		// last pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5f));
		ind2 = LOR_ind + 2*n;
		d_SRM_vals[ind2] = val;
		d_SRM_cols[ind2] = yint * wim + xint;
	}
}
// init SRM to zeros with the format ELL
__global__ void pet2D_SRM_ELL_init(float* d_SRM_vals, int* d_SRM_cols, int wsrm, int nx) {
	int j, ind;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nx) {
		// due to limitation of gridsize < 65536
		for (j=0; j<wsrm; ++j) {
			ind = idx * wsrm + j;
			d_SRM_vals[ind] = 0.0f;
			d_SRM_cols[ind] = -1;
		}
	}
}
// init Q and F to zeros
__global__ void pet2D_QF_init(float* d_Q, float* d_F, int nq, int nf) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nq) {d_Q[idx] = 0.0f;}
	if (idx < nf) {d_F[idx] = 0.0f;}
}
// init Q to zeros
__global__ void pet2D_Q_init(float* d_Q, int nq) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nq) {d_Q[idx] = 0.0f;}
}
// Compute Q vector by SRM * IM (ELL sparse matrix format)
__global__ void pet2D_ell_spmv(float* d_SRM_vals, int* d_SRM_cols, float* d_Q, float* d_im,  int niv, int njv) {
	int j, ind, vcol;
	float sum;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < niv) {
		ind = idx * njv;
		vcol = d_SRM_cols[ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (d_SRM_vals[ind+j] * d_im[vcol]);
			++j;
			vcol = d_SRM_cols[ind+j];
		}
		d_Q[idx] = sum;
	}
}
// Compute F vector by SRM^T / Q (ELL sparse matrix format)
__global__ void pet2D_ell_F(float* d_SRM_vals, int* d_SRM_cols, float* d_F, float* d_Q, int niv, int njv) {
	int i, ind, vcol;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float Qi;
	if (idx < njv) {
		for (i=0; i < niv; ++i) {
			Qi = d_Q[i];
			if (Qi==0.0f) {continue;}
			ind = i * njv + idx;
			vcol = d_SRM_cols[ind];
			if (vcol != -1) {d_F[vcol] += (d_SRM_vals[ind] / Qi);}
			__syncthreads();
		}
	}
}

// Compute col sum of ELL matrix (to get im from SRM)
__global__ void matrix_ell_sumcol(float* d_vals, int niv, int njv, int* d_cols, float* d_im) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, vcol, ind;
	if (idx < njv) {
		for (i=0; i<niv; ++i) {
			ind = i * njv + idx;
			vcol = d_cols[ind];
			if (vcol != -1) {d_im[vcol] += d_vals[ind];}
			__syncthreads();
		}
	}
}


__global__ void matrix_ell_spmv(float* d_vals, int* d_cols, float* d_res, int niv, int njv) {
	int j, ind, vcol;
	float sum;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < niv) {
		ind = idx * njv;
		vcol = d_cols[ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (d_vals[ind+j] * tex1D(tex1, vcol));
			++j;
			vcol = d_cols[ind+j];
		}
		d_res[idx] = sum;
	}
}

// Perform a multiplication between a complex and a real vectors
__global__ void vector_complex_x_real(cufftComplex* dcpx, float* dr, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float r, j, h;
	if (idx < n) {
		r = dcpx[idx].x;
		j = dcpx[idx].y;
		h = dr[idx];
		r = r * h;
		j = j * h;
		dcpx[idx].x = r;
		dcpx[idx].y = j;
	}
}

// Perform a mulitplication between a real vectors and an alpha value
__global__ void vector_real_x_cst(float* dr, float alpha, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float val;
	if (idx < n) {
		val = dr[idx];
		val = val * alpha;
		dr[idx] = val;
	}
}

void kernel_pet2D_EMML_wrap_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval, float* S, int ns, int maxit) {
	// select a GPU
	cudaSetDevice(0);
	// init cublas
	cublasStatus status;
	status = cublasInit();
	// allocate device memory for SRM, im, Q and F
	int size_SRM = nlor * npix;
	float* d_SRM;
	float* d_im;
	float* d_Q;
	float* d_F;
	float* d_S;
	int* d_lorval;
	status = cublasAlloc(size_SRM, sizeof(float), (void**)&d_SRM);
	status = cublasAlloc(npixim, sizeof(float), (void**)&d_im);
	status = cublasAlloc(nlor, sizeof(float), (void**)&d_Q);
	status = cublasAlloc(nlor, sizeof(float), (void**)&d_F);
	status = cublasAlloc(ns, sizeof(float), (void**)&d_S);
	status = cublasAlloc(nval, sizeof(int), (void**)&d_lorval);
	// load SRM, SM, LOR_val and im to the device
	status = cublasSetVector(size_SRM, sizeof(float), SRM, 1, d_SRM, 1);
	status = cublasSetVector(npixim, sizeof(float), im, 1, d_im, 1);
	status = cublasSetVector(ns, sizeof(float), S, 1, d_S, 1);
	status = cublasSetVector(nval, sizeof(int), LOR_val, 1, d_lorval, 1);
	if (status != 0) {exit(0);}
	int ite, block_size1, grid_size1, block_size2, grid_size2;
	block_size1 = 256;
	grid_size1 = (nlor + block_size1 - 1) / block_size1;
	block_size2 = 64;
	grid_size2 = (npix + block_size2 - 1) / block_size2;
	dim3 threads1(block_size1);
	dim3 grid1(grid_size1);
	dim3 threads2(block_size2);
	dim3 grid2(grid_size2);
	for (ite=0; ite<maxit; ++ite) {
		// compute Q
		cublasSgemv('t', npix, nlor, 1.0, d_SRM, npix, d_im, 1, 0.0, d_Q, 1);
		pet2D_Q_update<<< grid1, threads1 >>>(d_lorval, d_Q, nval);
		// compute f = sum{SRMi / qi} for each i LOR
		cublasSgemv('n', npix, nlor, 1.0, d_SRM, npix, d_Q, 1, 0.0, d_F, 1);
		// update image
		pet2D_im_update<<< grid2, threads2 >>>(d_im, d_S, d_F, npix);
	}
	// get results
	status = cublasGetError();
	status = cublasGetVector(npix, sizeof(float), d_im, 1, im, 1);
	// free memory
	status = cublasFree(d_SRM);
	status = cublasFree(d_im);
	status = cublasFree(d_Q);
	status = cublasFree(d_F);
	status = cublasFree(d_S);
	// prepare to quit
	status = cublasShutdown();
}

void kernel_pet2D_SRM_DDA_wrap_cuda(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	// select a GPU
	cudaSetDevice(0);
	// some vars
	int size_SRM = wy * wx;
	unsigned int mem_size_SRM = sizeof(float) * size_SRM;
	unsigned int mem_size_point = sizeof(int) * nx1;
	// alloacte device memory for SRM, x1, y1, x2, and y2
	float* d_SRM;
	int* d_X1;
	int* d_Y1;
	int* d_X2;
	int* d_Y2;
	cudaMalloc((void**) &d_SRM, mem_size_SRM);
	cudaMemset(d_SRM, 0.0f, mem_size_SRM);
	cudaMalloc((void**) &d_X1, mem_size_point);
	cudaMalloc((void**) &d_Y1, mem_size_point);
	cudaMalloc((void**) &d_X2, mem_size_point);
	cudaMalloc((void**) &d_Y2, mem_size_point);
	// copy host memory to device
	cudaMemcpy(d_X1, X1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y1, Y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_X2, X2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y2, Y2, mem_size_point, cudaMemcpyHostToDevice);
	// setup execution parameters
	int block_size, grid_size;
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size;
	dim3 threads(block_size);
	dim3 grid(grid_size);

	//timeval start, end;
	//double t1, t2, diff;
	//gettimeofday(&start, NULL);
	//t1 = start.tv_sec + start.tv_usec / 1000000.0;
	// DDA kernel
	pet2D_SRM_DDA<<< grid, threads >>>(d_SRM, d_X1, d_Y1, d_X2, d_Y2, wx, nx1, width_image);
	cudaThreadSynchronize();
	// get back results to the host
	cudaMemcpy(SRM, d_SRM, mem_size_SRM, cudaMemcpyDeviceToHost);
	//gettimeofday(&end, NULL);
	//t2 = end.tv_sec + end.tv_usec / 1000000.0;
	//diff = t2 - t1;
	//printf("time %f s\n", diff);
	// clean up memory
	cudaFree(d_SRM);
	cudaFree(d_X1);
	cudaFree(d_Y1);
	cudaFree(d_X2);
	cudaFree(d_Y2);
}

void kernel_matrix_ell_spmv_wrap_cuda(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres) {
	// select a GPU
	cudaSetDevice(0);
	// some vars
	int size_data = niv * njv;
	unsigned int mem_size_dataf = sizeof(float) * size_data;
	unsigned int mem_size_y = sizeof(float) * ny;
	unsigned int mem_size_res = sizeof(float) * nres;
	unsigned int mem_size_datai = sizeof(int) * size_data;
	// alloacte device memory
	float* d_vals;
	float* d_res;
	//float* d_y;
	int* d_cols;
	cudaMalloc((void**) &d_vals, mem_size_dataf);
	cudaMalloc((void**) &d_res, mem_size_res);
	//cudaMalloc((void**) &d_y, mem_size_y);
	cudaMalloc((void**) &d_cols, mem_size_datai);
	// copy host memory to device
	cudaMemcpy(d_vals, vals, mem_size_dataf, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_y, y, mem_size_res, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cols, cols, mem_size_datai, cudaMemcpyHostToDevice);
	// prepare texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    cudaMallocArray( &cu_array, &channelDesc, ny, 1 ); 
    cudaMemcpyToArray(cu_array, 0, 0, y, mem_size_y, cudaMemcpyHostToDevice);
	tex1.addressMode[0] = cudaAddressModeClamp;
    tex1.addressMode[1] = cudaAddressModeClamp;
    tex1.filterMode = cudaFilterModePoint;
    tex1.normalized = false;
    cudaBindTextureToArray(tex1, cu_array, channelDesc);

	// setup execution parameters
	int block_size, grid_size;
	block_size = 256;
	grid_size = (niv + block_size - 1) / block_size;
	dim3 threads(block_size);
	dim3 grid(grid_size);
	timeval start, end;
	double t1, t2, diff;
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	// spmv kernel
	matrix_ell_spmv<<< grid, threads >>>(d_vals, d_cols, d_res, niv, njv);
	cudaThreadSynchronize();
	// get back results to the host
	cudaMemcpy(res, d_res, mem_size_res, cudaMemcpyDeviceToHost);
	gettimeofday(&end, NULL);
	t2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = t2 - t1;
	printf("kernel time %f s\n", diff);
	// clean up memory
	cudaFree(d_vals);
	cudaFree(d_cols);
	//cudaFree(d_y);
	cudaFree(d_res);
}

void kernel_pet2D_LM_EMML_DDA_ELL_wrap_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, float* S, int ns, int wsrm, int wim, int maxite) {
	// select a GPU
	cudaSetDevice(0);
	// to time
	timeval start, end;
	double t1, t2, diff;
	// vars
	int ite;
	int block_size, grid_size;
	dim3 threads, grid;
	dim3 threads2, grid2;
	dim3 threads3, grid3;
	// allocate device memory
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	int size_SRM = nx1 * wsrm;
	unsigned int mem_size_iSRM = size_SRM * sizeof(int);
	unsigned int mem_size_fSRM = size_SRM * sizeof(float);
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_S = ns * sizeof(float);
	unsigned int mem_size_Q = nx1 * sizeof(float);
	unsigned int mem_size_F = nim * sizeof(float);
	unsigned int mem_size_point = nx1 * sizeof(int);
	printf("mem tot %i\n", 4*(size_SRM + size_SRM + nim + ns + nx1 + nim + 4*nx1));
	float* d_SRM_vals;
	int* d_SRM_cols;
	float* d_im;
	float* d_S;
	float* d_Q;
	float* d_F;
	int* d_x1;
	int* d_x2;
	int* d_y1;
	int* d_y2;
	cudaMalloc((void**) &d_SRM_vals, mem_size_fSRM);
	cudaMalloc((void**) &d_SRM_cols, mem_size_iSRM);
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_S, mem_size_S);
	cudaMalloc((void**) &d_Q, mem_size_Q);
	cudaMalloc((void**) &d_F, mem_size_F);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S, S, mem_size_S, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	gettimeofday(&end, NULL);
	t2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = t2 - t1;
	printf("prepare mem: %f s\n", diff);

	// Init kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	pet2D_SRM_ELL_init<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, wsrm, nx1);
	gettimeofday(&end, NULL);
	t2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = t2 - t1;
	printf("kernel SRM init: %f s\n", diff);
	// DDA kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	pet2D_SRM_DDA_ELL<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, d_x1, d_y1, d_x2, d_y2, wsrm, wim, nx1);
	gettimeofday(&end, NULL);
	t2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = t2 - t1;
	printf("kernel DDA: %f s\n", diff);
	// IM kernel
	block_size = 8;
	grid_size = (wsrm + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	matrix_ell_sumcol<<<grid, threads>>>(d_SRM_vals, nx1, wsrm, d_SRM_cols, d_im);
	gettimeofday(&end, NULL);
	t2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = t2 - t1;
	printf("kernel compute IM: %f s\n", diff);

	// Iteration loop
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;

	block_size = 8;
	grid_size = (wsrm + block_size - 1) / block_size;
	threads2.x = block_size;
	grid2.x = grid_size;
	
	block_size = 64;
	grid_size = (nim + block_size - 1) / block_size;
	threads3.x = block_size;
	grid3.x = grid_size;
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	for (ite=0; ite<maxite; ++ite) {
		// init F and Q to zeros
		pet2D_QF_init<<<grid, threads>>>(d_Q, d_F, nx1, nim);
		// compute Q
		pet2D_ell_spmv<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, d_Q, d_im, nx1, wsrm);
		// compute f = sum{SRMi / qi} for each i LOR
		pet2D_ell_F<<<grid2, threads2>>>(d_SRM_vals, d_SRM_cols, d_F, d_Q, nx1, wsrm);
		// update image
		pet2D_im_update<<< grid3, threads3 >>>(d_im, d_S, d_F, nim);

	}

	gettimeofday(&end, NULL);
	t2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = t2 - t1;
	printf("kernel iter: %f s\n", diff);
	

	// get back image
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	cudaMemcpy(im, d_im, mem_size_im, cudaMemcpyDeviceToHost);
	gettimeofday(&end, NULL);
	t2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = t2 - t1;
	printf("get image back: %f s\n", diff);

	cudaFree(d_SRM_vals);
	cudaFree(d_SRM_cols);
	cudaFree(d_im);
	cudaFree(d_S);
	cudaFree(d_Q);
	cudaFree(d_F);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	
}

// Compute the first image in LM 2D-OSEM algorithm (from x, y build SRM, then compute IM)
void kernel_pet2D_IM_SRM_DDA_ELL_wrap_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, int wsrm, int wim) {
	// select a GPU
	cudaSetDevice(0);
	// vars
	int block_size, grid_size;
	dim3 threads, grid;
	// allocate device memory
	int size_SRM = nx1 * wsrm;
	unsigned int mem_size_iSRM = size_SRM * sizeof(int);
	unsigned int mem_size_fSRM = size_SRM * sizeof(float);
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_point = nx1 * sizeof(int);
	float* d_SRM_vals;
	int* d_SRM_cols;
	float* d_im;
	int* d_x1;
	int* d_x2;
	int* d_y1;
	int* d_y2;
	cudaMalloc((void**) &d_SRM_vals, mem_size_fSRM);
	cudaMalloc((void**) &d_SRM_cols, mem_size_iSRM);
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	// Init kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	pet2D_SRM_ELL_init<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, wsrm, nx1);
	// DDA kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;
	pet2D_SRM_DDA_ELL<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, d_x1, d_y1, d_x2, d_y2, wsrm, wim, nx1);
	//pet2D_SRM_DDAA_ELL<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, d_x1, d_y1, d_x2, d_y2, wsrm, wim, nx1);
	// IM kernel
	block_size = 8;
	grid_size = (wsrm + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	matrix_ell_sumcol<<<grid, threads>>>(d_SRM_vals, nx1, wsrm, d_SRM_cols, d_im);
	// get back image
	cudaMemcpy(im, d_im, mem_size_im, cudaMemcpyDeviceToHost);
	// Free mem
	cudaFree(d_SRM_vals);
	cudaFree(d_SRM_cols);
	cudaFree(d_im);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_x2);
	cudaFree(d_y2);
}

// Update image for the 2D-LM-OSEM reconstruction (from x, y, IM and S, build SRM in ELL format then update IM)
void kernel_pet2D_IM_SRM_DDA_ELL_iter_wrap_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* S, int ns, float* im, int nim, int wsrm, int wim) {
	// select a GPU
	cudaSetDevice(0);
	// vars
	int block_size, grid_size;
	dim3 threads, grid;
	dim3 threads2, grid2;
	dim3 threads3, grid3;
	// allocate device memory
	int size_SRM = nx1 * wsrm;
	unsigned int mem_size_iSRM = size_SRM * sizeof(int);
	unsigned int mem_size_fSRM = size_SRM * sizeof(float);
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_S = ns * sizeof(float);
	unsigned int mem_size_Q = nx1 * sizeof(float);
	unsigned int mem_size_F = nim * sizeof(float);
	unsigned int mem_size_point = nx1 * sizeof(int);
	float* d_SRM_vals;
	int* d_SRM_cols;
	float* d_im;
	float* d_S;
	float* d_Q;
	float* d_F;
	int* d_x1;
	int* d_x2;
	int* d_y1;
	int* d_y2;
	cudaMalloc((void**) &d_SRM_vals, mem_size_fSRM);
	cudaMalloc((void**) &d_SRM_cols, mem_size_iSRM);
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_S, mem_size_S);
	cudaMalloc((void**) &d_Q, mem_size_Q);
	cudaMalloc((void**) &d_F, mem_size_F);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S, S, mem_size_S, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	// Init kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	pet2D_SRM_ELL_init<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, wsrm, nx1);
	// DDA kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;
	pet2D_SRM_DDA_ELL<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, d_x1, d_y1, d_x2, d_y2, wsrm, wim, nx1);
	//pet2D_SRM_DDAA_ELL<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, d_x1, d_y1, d_x2, d_y2, wsrm, wim, nx1);
	// One iteration
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;
	block_size = 8;
	grid_size = (wsrm + block_size - 1) / block_size;
	threads2.x = block_size;
	grid2.x = grid_size;
	block_size = 64;
	grid_size = (nim + block_size - 1) / block_size;
	threads3.x = block_size;
	grid3.x = grid_size;
	// init F and Q to zeros
	pet2D_QF_init<<<grid, threads>>>(d_Q, d_F, nx1, nim);
	// compute Q
	pet2D_ell_spmv<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, d_Q, d_im, nx1, wsrm);
	// compute f = sum{SRMi / qi} for each i LOR
	pet2D_ell_F<<<grid2, threads2>>>(d_SRM_vals, d_SRM_cols, d_F, d_Q, nx1, wsrm);
	// update image
	pet2D_im_update<<< grid3, threads3 >>>(d_im, d_S, d_F, nim);
	// get back image
	cudaMemcpy(im, d_im, mem_size_im, cudaMemcpyDeviceToHost);
	// Free mem
	cudaFree(d_SRM_vals);
	cudaFree(d_SRM_cols);
	cudaFree(d_im);
	cudaFree(d_S);
	cudaFree(d_Q);
	cudaFree(d_F);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_x2);
	cudaFree(d_y2);
}

// Compute the first image in LM 3D-OSEM algorithm (from x, y build SRM, then compute IM)
void kernel_pet3D_IM_SRM_DDA_ELL_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
										   unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										   float* im, int nim, int wsrm, int wim, int ID) {
	// select a GPU
	if (ID != -1) {cudaSetDevice(ID);}
	// vars
	int block_size, grid_size;
	dim3 threads, grid;
	// allocate device memory
	int size_SRM = nx1 * wsrm;
	unsigned int mem_size_iSRM = size_SRM * sizeof(int);
	unsigned int mem_size_fSRM = size_SRM * sizeof(float);
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_point = nx1 * sizeof(unsigned short int);
	float* d_SRM_vals;
	int* d_SRM_cols;
	float* d_im;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	cudaMalloc((void**) &d_SRM_vals, mem_size_fSRM);
	cudaMalloc((void**) &d_SRM_cols, mem_size_iSRM);
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_z1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	cudaMalloc((void**) &d_z2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z1, z1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z2, z2, mem_size_point, cudaMemcpyHostToDevice);
	// Init textures
	cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
	cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
	cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
	cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
	cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
	cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);
	// Init kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	pet2D_SRM_ELL_init<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, wsrm, nx1);
	// DDA kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;
	pet3D_SRM_DDA_ELL<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, wsrm, wim, nx1);
	// IM kernel
	block_size = 8;
	grid_size = (wsrm + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	matrix_ell_sumcol<<<grid, threads>>>(d_SRM_vals, nx1, wsrm, d_SRM_cols, d_im);
	// get back image
	cudaMemcpy(im, d_im, mem_size_im, cudaMemcpyDeviceToHost);
	// Free mem
	cudaFree(d_SRM_vals);
	cudaFree(d_SRM_cols);
	cudaFree(d_im);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_z1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_z2);
}

// Update image for the 3D-LM-OSEM reconstruction (from x, y, IM and S, build SRM in ELL format then return F)
void kernel_pet3D_IM_SRM_DDA_ELL_iter_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
												unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
												float* im, int nim, float* F, int nf, int wsrm, int wim, int ID){

	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}
	// vars
	int block_size, grid_size;
	dim3 threads, grid;
	dim3 threads2, grid2;
	dim3 threads3, grid3;
	// allocate device memory
	int size_SRM = nx1 * wsrm;
	unsigned int mem_size_iSRM = size_SRM * sizeof(int);
	unsigned int mem_size_fSRM = size_SRM * sizeof(float);
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_Q = nx1 * sizeof(float);
	unsigned int mem_size_F = nim * sizeof(float);
	unsigned int mem_size_point = nx1 * sizeof(unsigned short int);
	float* d_SRM_vals;
	int* d_SRM_cols;
	float* d_im;
	float* d_Q;
	float* d_F;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	cudaMalloc((void**) &d_SRM_vals, mem_size_fSRM);
	cudaMalloc((void**) &d_SRM_cols, mem_size_iSRM);
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_Q, mem_size_Q);
	cudaMalloc((void**) &d_F, mem_size_F);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_z1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	cudaMalloc((void**) &d_z2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F, mem_size_F, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z1, z1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z2, z2, mem_size_point, cudaMemcpyHostToDevice);
	// Init kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;
	pet2D_SRM_ELL_init<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, wsrm, nx1);
	// DDA kernel
	pet3D_SRM_DDA_ELL_Q<<<grid, threads>>>(d_SRM_vals, d_SRM_cols, d_im, d_Q, d_x1, d_y1, d_z1, d_x2, d_y2, d_z2, wsrm, wim, nx1);
	/*
	// init Q to zeros
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;
	pet2D_Q_init<<<grid, threads>>>(d_Q, nx1);
	// compute Q
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads2.x = block_size;
	grid2.x = grid_size;
	pet2D_ell_spmv<<<grid2, threads2>>>(d_SRM_vals, d_SRM_cols, d_Q, d_im, nx1, wsrm);
	*/
	// compute f = sum{SRMi / qi} for each i LOR
	block_size = 8;
	grid_size = (wsrm + block_size - 1) / block_size;
	threads3.x = block_size;
	grid3.x = grid_size;
	pet2D_ell_F<<<grid3, threads3>>>(d_SRM_vals, d_SRM_cols, d_F, d_Q, nx1, wsrm);
	// get back F
	cudaMemcpy(F, d_F, mem_size_F, cudaMemcpyDeviceToHost);

	// Free mem
	cudaFree(d_SRM_vals);
	cudaFree(d_SRM_cols);
	cudaFree(d_im);
	cudaFree(d_Q);
	cudaFree(d_F);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_z1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_z2);
}

/***********************************************
 * USED
 ***********************************************/

// Compute the first image in LM 3D-OSEM algorithm (from x, y build SRM, then compute IM)
void kernel_pet3D_IM_SRM_DDA_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
									   unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
									   unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
									   int* im, int nim1, int nim2, int nim3, int wim, int ID) {
	// select a GPU
	if (ID != -1) {cudaSetDevice(ID);}
	// vars
	int block_size, grid_size;
	dim3 threads, grid;
	// allocate device memory
	int nim = nim1 * nim2 * nim3;
	unsigned int mem_size_im = nim * sizeof(int);
	unsigned int mem_size_point = nx1 * sizeof(unsigned short int);
	int* d_im;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_z1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	cudaMalloc((void**) &d_z2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z1, z1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z2, z2, mem_size_point, cudaMemcpyHostToDevice);
	// texture
	cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
	cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
	cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
	cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
	cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
	cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);
	// IM kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16 Mlines
	threads.x = block_size;
	grid.x = grid_size;
	pet3D_SRM_DDA_ON<<<grid, threads>>>(d_im, wim, nx1, nim);
	//pet3D_SRM_DDA_fixed_ON<<<grid, threads>>>(d_im, wim, nx1, nim);
	// get back image
	cudaMemcpy(im, d_im, mem_size_im, cudaMemcpyDeviceToHost);
	// Free mem
	cudaFree(d_im);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_z1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_z2);
	cudaThreadExit();
}

// Compute update in LM 3D-OSEM algorithm on-line with DDA line drawing
void kernel_pet3D_IM_SRM_DDA_ON_iter_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
											   unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
											   unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
											   float* im, int nim1, int nim2, int nim3, float* F, int nf1, int nf2, int nf3,
											   int wim, int ID){

	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}
	// vars
	int block_size, grid_size, i;
	dim3 threads, grid;
	int nim = nim1 * nim2 * nim3;
	// Need to change
	int* Fi = (int*)calloc(nim, sizeof(int));
	// allocate device memory
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_F = nim * sizeof(unsigned int);
	unsigned int mem_size_point = nx1 * sizeof(unsigned short int);
	float* d_im;
	unsigned int* d_F;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_F, mem_size_F);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_z1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	cudaMalloc((void**) &d_z2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, Fi, mem_size_F, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z1, z1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z2, z2, mem_size_point, cudaMemcpyHostToDevice);
	// prepare texture
	cudaBindTexture(NULL, tex_im, d_im, mem_size_im);
	cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
	cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
	cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
	cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
	cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
	cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);
	// float to int scale
	float scale = 4000.0f;
	// kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16e6 lines
	threads.x = block_size;
	grid.x = grid_size;
	pet3D_SRM_DDA_F_ON<<<grid, threads>>>(d_F, wim, nx1, nim, scale);
	// get back F and convert
	cudaMemcpy(Fi, d_F, nim*sizeof(float), cudaMemcpyDeviceToHost);
	scale = 1 / scale;
	for (i=0; i<nim; ++i) {F[i] = (float)Fi[i] * scale;}
	// Free mem
	free(Fi);
	cudaFree(d_im);
	cudaFree(d_F);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_z1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_z2);
	cudaThreadExit();
}

// DEV Compute update in LM 3D-OSEM algorithm on-line with DDA line drawing and attenuation
void kernel_pet3D_IM_ATT_SRM_DDA_ON_iter_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
												   unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
												   unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
												   float* im, int nim1, int nim2, int nim3,
												   float* F, int nf1, int nf2, int nf3,
												   float* mumap, int nmu1, int nmu2, int nmu3, int wim, int ID){

	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}
	// vars
	int block_size, grid_size, i;
	dim3 threads, grid;
	int nim = nim1 * nim2 * nim3;
	// Need to change
	int* Fi = (int*)calloc(nim, sizeof(int));
	// allocate device memory
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_F = nim * sizeof(unsigned int);
	unsigned int mem_size_point = nx1 * sizeof(unsigned short int);
	float* d_im;
	float* d_mumap;
	unsigned int* d_F;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_mumap, mem_size_im);
	cudaMalloc((void**) &d_F, mem_size_F);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_z1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	cudaMalloc((void**) &d_z2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mumap, mumap, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, Fi, mem_size_F, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z1, z1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z2, z2, mem_size_point, cudaMemcpyHostToDevice);
	// prepare texture
	cudaBindTexture(NULL, tex_im, d_im, mem_size_im);
	cudaBindTexture(NULL, tex_mumap, d_mumap, mem_size_im);
	cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
	cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
	cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
	cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
	cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
	cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);
	// float to int scale
	float scale = 4000.0f;
	// kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16e6 lines
	threads.x = block_size;
	grid.x = grid_size;
	//pet3D_SRM_DDA_F_ON<<<grid, threads>>>(d_F, wim, nx1, nim, scale);
	pet3D_SRM_DDA_F_ATT_ON<<<grid, threads>>>(d_F, wim, nx1, nim, scale);
	// get back F and convert
	cudaMemcpy(Fi, d_F, nim*sizeof(float), cudaMemcpyDeviceToHost);
	for (i=0; i<nim; ++i) {F[i] = (float)Fi[i] / scale;}
	// Free mem
	free(Fi);
	cudaFree(d_im);
	cudaFree(d_mumap);
	cudaFree(d_F);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_z1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_z2);
	cudaThreadExit();
}


// 3D convolution (in Fourier)
void kernel_3Dconv_wrap_cuda(float* vol, int nz, int ny, int nx, float* H, int a, int b, int c) {
	int ID = 0;
	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}
	// prepare the filter
	int nc = (ny / 2) + 1;
	int size_H = c * b * a;
	int size_vol = nz * ny * nx;
	int size_fft = nz * nc * nx;
	
	cufftHandle plan_forward, plan_inverse;
	cufftReal* dvol;
	cufftComplex* dfft;
	float* dH;
	int status;

	// alloc mem GPU
	status = cudaMalloc((void**)&dvol, size_vol * sizeof(cufftReal));
	//printf("dvol %i\n", status);
	status = cudaMalloc((void**)&dfft, size_fft * sizeof(cufftComplex));
	//printf("dfft %i\n", status);
	status = cudaMalloc((void**)&dH, size_H * sizeof(float));
	//printf("dH %i\n", status);
	
	// tranfert to GPU
	status = cudaMemcpy(dvol, vol, size_vol * sizeof(cufftReal), cudaMemcpyHostToDevice);
	//printf("memcpy dvol %i\n", status);
	status = cudaMemcpy(dH, H, size_H * sizeof(float), cudaMemcpyHostToDevice);
	//printf("memcpy dH %i\n", status);
	
	// do fft
	status = cufftPlan3d(&plan_forward, nx, ny, nz, CUFFT_R2C);
	//printf("init plan %i\n", status);
	status = cufftExecR2C(plan_forward, dvol, dfft);
	//printf("fft %i\n", status);
	
	// do 3D convolution
	int block_size, grid_size;
	dim3 threads, grid;
	block_size = 128;
	grid_size = (size_fft + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	vector_complex_x_real<<<grid, threads>>>(dfft, dH, size_fft);

	// get inverse transform
	cufftPlan3d(&plan_inverse, nz, ny, nx, CUFFT_C2R);
	cufftExecC2R(plan_inverse, dfft, dvol);

	// Normalize values due to FFT theorem (1 / N)
	block_size = 128;
	grid_size = (size_vol + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	vector_real_x_cst<<<grid, threads>>>(dvol, 1 / float(size_vol), size_vol);

	// get back the volume
	cudaMemcpy(vol, dvol, size_vol * sizeof(float), cudaMemcpyDeviceToHost);
	
	// clean up
	cufftDestroy(plan_forward);
	cufftDestroy(plan_inverse);
	cudaFree(dvol);
	cudaFree(dH);
	cudaFree(dfft);
	
	cudaThreadExit();
}


////////////////////////////////////////////////////////////////////////
// 3D-OPLEM
////////////////////////////////////////////////////////////////////////

// DDA ray-projector
__global__ void pet3D_OPLEM_DDA_V0(unsigned int* d_F, float* d_im,
								   unsigned short int* d_x1, unsigned short int* d_y1, unsigned short int* d_z1,
								   unsigned short int* d_x2, unsigned short int* d_y2, unsigned short int* d_z2,
								   int sublor_start, int sublor_stop, int nim3, int nim, int nsublor, float scale) {

	int length, n, diffx, diffy, diffz, step;
	float flength, x, y, z, lx, ly, lz, xinc, yinc, zinc, Qi;
	unsigned short int x1, y1, z1;
	int idx = blockIdx.x * blockDim.x + threadIdx.x + sublor_start;
	step = nim3*nim3;
	
	if (idx < sublor_stop) {
		Qi = 0.0f;
		//x1 = tex1Dfetch(tex_x1, idx);
		//y1 = tex1Dfetch(tex_y1, idx);
		//z1 = tex1Dfetch(tex_z1, idx);
		//diffx = tex1Dfetch(tex_x2, idx)-x1;
		//diffy = tex1Dfetch(tex_y2, idx)-y1;
		//diffz = tex1Dfetch(tex_z2, idx)-z1;
		x1 = d_x1[idx];
		y1 = d_y1[idx];
		z1 = d_z1[idx];
		diffx = d_x2[idx] - x1;
		diffy = d_y2[idx] - y1;
		diffz = d_z2[idx] - z1;
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
			//Qi = Qi + tex1Dfetch(tex_im, (int)z * step + (int)y * wim + (int)x);
			Qi = Qi + d_im[(int)z * step + (int)y * nim3 + (int)x];
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}

		// compute F
		if (Qi==0.0f) {return;}
		Qi = 1 / Qi;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			//atomicFloatAdd(&d_F[(int)z * step + (int)y * wim + (int)x], Qi);
			atomicAdd(&d_F[(int)z * step + (int)y * nim3 + (int)x], (unsigned int)(Qi*scale));
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}
}

__global__ void toto(unsigned int* d_F, unsigned short int* d_x1) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_F[d_x1[idx]] = d_x1[idx];
}

// OPL-3D-OSEM algorithm with DDA-ELL
void kernel_pet3D_OPLEM_wrap_cuda_V0(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
									 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
									 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
									 float* im, int nim1, int nim2, int nim3,
									 float* NM, int NM1, int NM2, int NM3, int Nsub, int ID){
	
	// Constant according Graphical card
	int mem_max = 800000000; // only 800 MB on 1 GB required
	float scale = 4000.0f;
	
	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}
	// vars
	
	int block_size, grid_size, i;
	dim3 threads, grid;
	int nim = nim1 * nim2 * nim3;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	
	unsigned int mem_size_point;
	
	// memory managment 12 B / LORs and 8 B / Vox
	int max_sub, Nouter, mem_sub;
	mem_max -= (8 * nim);
	mem_sub = 12.0f * nx1 / float(Nsub);
	max_sub = int(mem_max / mem_sub);
	Nouter = (Nsub + max_sub - 1) / max_sub;

	printf("max_sub %i   Nouter %i\n", max_sub, Nouter);

	// device mem allocation
	float* d_im;
	unsigned int mem_size_im = nim * sizeof(float);
	cudaMalloc((void**) &d_im, mem_size_im);
	unsigned int* d_F;
	unsigned int mem_size_F = nim * sizeof(int);
	cudaMalloc((void**) &d_F, mem_size_F);
	unsigned int* F = (unsigned int*)malloc(mem_size_F);

	// Outer loop, avoid memory overflow
	int sub_start, sub_stop, nsub;
	int lor_start, lor_stop, nlor;
	int sublor_start, sublor_stop, nsublor;
	for (int iouter=0; iouter < Nouter; ++iouter) {
		printf("iouter %i\n", iouter);
		// split subset
		sub_start = int(float(Nsub) / Nouter * iouter + 0.5f);
		sub_stop = int(float(Nsub) / Nouter * (iouter+1) + 0.5f);
		nsub = sub_stop - sub_start;
		printf("   sub: %i to %i\n", sub_start, sub_stop);
		// load in memory subsets
		lor_start = int(float(nx1) / Nsub * sub_start + 0.5f);
		lor_stop = int(float(nx1) / Nsub * sub_stop + 0.5f);
		nlor = lor_stop - lor_start;
		printf("   lor: %i to %i\n", lor_start, lor_stop);
		mem_size_point = nlor * sizeof(short int);
		cudaMalloc((void**) &d_x1, mem_size_point);
		cudaMalloc((void**) &d_y1, mem_size_point);
		cudaMalloc((void**) &d_z1, mem_size_point);
		cudaMalloc((void**) &d_x2, mem_size_point);
		cudaMalloc((void**) &d_y2, mem_size_point);
		cudaMalloc((void**) &d_z2, mem_size_point);
		cudaMemcpy(d_x1, &x1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y1, &y1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_z1, &z1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_x2, &x2[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y2, &y2[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_z2, &z2[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		//cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
		//cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
		//cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
		//cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
		//cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
		//cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);

		cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
		// subset loop
		int c=0;
		for (int isub=0; isub < nsub; ++isub) {
			//isub=0;
			printf("   isub: %i\n", isub);
			sublor_start = int(float(nlor) / nsub * isub + 0.5f);
			sublor_stop = int(float(nlor) / nsub * (isub+1) + 0.5f);
			nsublor = sublor_stop - sublor_start;
			printf("      sublor: %i to %i\n", sublor_start, sublor_stop);

			/*
			// init F and load im to the GPU
			for (i=0; i<nim; ++i) {F[i] = 0;}
			cudaMemcpy(d_F, F, mem_size_F, cudaMemcpyHostToDevice);
			
			// kernel
			block_size = 256;
			grid_size = (nsublor + block_size - 1) / block_size; // CODE IS LIMITED TO < 16e6 lines
			threads.x = block_size;
			grid.x = grid_size;
			//pet3D_OPLEM_DDA_V0<<<grid, threads>>>(d_F, d_im, d_x1, d_y1, d_z1, d_x2, d_y2, d_z2,
			//									  sublor_start, sublor_stop, nim3, nim, nsublor, scale);
			toto<<<grid, threads>>>(d_F, d_x1);
			
			// get back F
			cudaMemcpy(F, d_F, mem_size_F, cudaMemcpyDeviceToHost);
			//cudaMemcpy(im, d_im, mem_size_im, cudaMemcpyDeviceToHost);
			// update volume
			scale = 1 / scale;
			int fmax=0;
			int imax=0;
			for (i=0; i<nim; ++i) {
				if (F[i] > fmax) {fmax = F[i];}
				//im[i] = im[i] * (float)F[i] * scale / NM[i];
				if (im[i] > imax) {imax = im[i];}
			}
			printf("      FMAX: %i\n", fmax);
			printf("      IMAX: %i\n", imax);
			// clean
			++c;
			if (c==2) {break;}
			*/

		} // isub
		// clean
		cudaFree(d_x1);
		cudaFree(d_y1);
		cudaFree(d_z1);
		cudaFree(d_x2);
		cudaFree(d_y2);
		cudaFree(d_z2);
		
	} // iouter
	free(F);
	cudaFree(d_im);
	cudaFree(d_F);
	cudaThreadExit();
}
