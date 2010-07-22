#include "kernel_cuda.h"
#include <stdio.h>
#include <cublas.h>
#include <sys/time.h>

// declare texture ref for 1D float texture
texture<float, 1, cudaReadModeElementType> tex;

// kernel to update image in pet2D EMML algorithm
__global__ void pet2D_im_update(float* im, float* S, float* F, int npix) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < npix && F[idx] != 0.0f) {
		im[idx] = im[idx] / S[idx] * F[idx];
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
			sum += (d_vals[ind+j] * tex1D(tex, vcol));
			++j;
			vcol = d_cols[ind+j];
		}
		d_res[idx] = sum;
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
	tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;
    cudaBindTextureToArray(tex, cu_array, channelDesc);

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
