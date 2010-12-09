void kernel_pet2D_EMML_wrap_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval, float* S, int ns, int maxit);
void kernel_pet2D_SRM_DDA_wrap_cuda(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);
void kernel_matrix_ell_spmv_wrap_cuda(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres);
void kernel_pet2D_LM_EMML_DDA_ELL_wrap_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, float* S, int ns, int wsrm, int wim, int maxite);
void kernel_pet2D_IM_SRM_DDA_ELL_wrap_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, int wsrm, int wim);
void kernel_pet2D_IM_SRM_DDA_ELL_iter_wrap_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* S, int ns, float* im, int nim, int wsrm, int wim);
void kernel_pet3D_IM_SRM_DDA_ELL_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
										   unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										   float* im, int nim, int wsrm, int wim, int ID);
void kernel_pet3D_IM_SRM_DDA_ELL_iter_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
												unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
												float* im, int nim, float* F, int nf, int wsrm, int wim, int ID);

void kernel_pet3D_IM_SRM_DDA_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
									   unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
									   unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
									   int* im, int nim1, int nim2, int nim3, int wim, int ID);

void kernel_pet3D_IM_SRM_DDA_ON_iter_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
											   unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
											   unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
											   float* im, int nim1, int nim2, int nim3, float* F, int nf1, int nf2, int nf3,
											   int wim, int ID);

void kernel_pet3D_IM_ATT_SRM_DDA_ON_iter_wrap_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
												   unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
												   unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
												   float* im, int nim1, int nim2, int nim3,
												   float* F, int nf1, int nf2, int nf3,
												   float* mumap, int nmu1, int nmu2, int nmu3, int wim, int ID);


void kernel_3Dconv_wrap_cuda(float* vol, int nz, int ny, int nx, float* H, int a, int b, int c);


void kernel_pet3D_OPLEM_wrap_cuda_V0(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
									 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
									 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
									 float* im, int nim1, int nim2, int nim3,
									 float* NM, int NM1, int NM2, int NM3, int Nsub, int ID);
