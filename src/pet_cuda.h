void kernel_pet3D_LMOSEM_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
							  unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
							  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
							  float* im, int nim1, int nim2, int nim3, float* F, int nf1, int nf2, int nf3,
							  int wim, int ID);

void kernel_pet3D_LMOSEM_att_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
								  unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
								  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
								  float* im, int nim1, int nim2, int nim3,
								  float* F, int nf1, int nf2, int nf3,
								  float* mumap, int nmu1, int nmu2, int nmu3, int wim, int ID);

void kernel_pet3D_OPLEM_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
							 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
							 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
							 float* im, int nim1, int nim2, int nim3,
							 float* NM, int NM1, int NM2, int NM3, int Nsub, int ID);

void kernel_pet3D_OPLEM_att_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
								 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
								 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
								 float* im, int nim1, int nim2, int nim3,
								 float* NM, int NM1, int NM2, int NM3,
								 float* at, int nat1, int nat2, int nat3,
								 int Nsub, int ID);
