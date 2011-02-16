// This file is part of FIREwork
// 
// FIREwork is free software: you can redistribute it and/or modify
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
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// FIREwork Copyright (C) 2008 - 2011 Julien Bert 

#include "mc_cuda.h"
#include "mc_cuda_cst.cu"
#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <cufft.h>
#include <sys/time.h>
#include <math_constants.h>

/***********************************************************
 * Utils
 ***********************************************************/
__constant__ const float pi = 3.14159265358979323846;
__constant__ const float twopi = 2*pi;
texture<float, 1, cudaReadModeElementType> tex_vol;

// Stack of gamma particles, format data is defined as SoA
struct StackGamma{
	float* E;
	float* dx;
	float* dy;
	float* dz;
	float* px;
	float* py;
	float* pz;
	int* seed;
	int* ct_eff;
	int* ct_Cpt;
	int* ct_PE;
	unsigned char* live;
	unsigned char* in;
	unsigned int size;
}; //

// Given by Hector doesn't work properly
__device__ float park_miller(long unsigned int *seed) {
	long unsigned int hi, lo;
	int const a = 16807;
	int const m = 2147483647;
	float const recm = __fdividef(1.0f, m);

	lo = a * (*seed & 0xFFFF);
	hi = a * (*seed >> 16);
	lo += (hi & 0x7FFF) << 16;
	lo += (hi >> 15);
	if (lo > 0x7FFFFFFF) {lo -= 0x7FFFFFFF;}
	*seed = (long)lo;

	return (*seed)*recm;
}

// Park-Miller from C numerical book
__device__ float park_miller_jb(int *seed) {
	int const a = 16807;
	int const m = 2147483647;
	int const iq = 127773;
	int const ir = 2836;
	int const mask = 123459876;
	float const recm = __fdividef(1.0f, m);
	int k;
	float ans;
	*seed ^= mask;
	k = (*seed)/iq;
	*seed = a * (*seed-k*iq) - ir*k;
	if (*seed < 0) {*seed += m;}
	ans = recm * (*seed);
	*seed ^= mask;
	return ans;
}

// Hamilton multiplication (quaternion)
__device__ float4 quat_mul(float4 p, float4 q) {
	return make_float4(
		   p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y,    // x
		   p.w*q.y + p.y*q.w + p.z*q.x - p.x*q.z,    // y
		   p.w*q.z + p.z*q.w + p.x*q.y - p.y*q.x,    // z
		   p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z);   // w
}

// Create quaternion for axis angle rotation
__device__ float4 quat_axis(float4 n, float theta) {
	theta /= 2.0f;
	float stheta = __sinf(theta);
	return make_float4(n.x*stheta, n.y*stheta, n.z*stheta, __cosf(theta));
}

// Conjugate quaternion
__device__ float4 quat_conj(float4 p) {
	return make_float4(-p.x, -p.y, -p.z, p.w);
}

// Normalize quaternion
__device__ float4 quat_norm(float4 p) {
	float norm = __fdividef(1.0f, __powf(p.w*p.w+p.x*p.x+p.y*p.y+p.z*p.z, 0.5f));
	return make_float4(p.x*norm, p.y*norm, p.z*norm, p.w*norm);
}

// Cross product
__device__ float4 quat_crossprod(float4 u, float4 v){
	return make_float4(u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x, 0.0f);
}

/***********************************************************
 * Physics
 ***********************************************************/
// Compton Cross Section Per Atom
__device__ float Compton_CSPA(float E, float Z) {
	float CrossSection = 0.0;
	if (Z<0.9999f || E < 1e-4f) {return CrossSection;}

	float p1Z = Z*(2.7965e-23f + 1.9756e-27f*Z + -3.9178e-29f*Z*Z);
	float p2Z = Z*(-1.8300e-23f + -1.0205e-24f*Z + 6.8241e-27f*Z*Z);
	float p3Z = Z*(6.7527e-22f + -7.3913e-24f*Z + 6.0480e-27f*Z*Z);
	float p4Z = Z*(-1.9798e-21f + 2.7079e-24f*Z + 3.0274e-26f*Z*Z);
	float T0 = (Z < 1.5f)? 40.0e-3f : 15.0e-3f;
	float d1, d2, d3, d4, d5;

	d1 = __fdividef(fmaxf(E, T0), 0.510998910f); // X
	CrossSection = __fdividef(p1Z*__logf(1.0f+2.0f*d1), d1) + __fdividef(p2Z + p3Z*d1 + p4Z*d1*d1, 1.0f + 20.0f*d1 + 230.0f*d1*d1 + 440.0f*d1*d1*d1);

	if (E < T0) {
		d1 = __fdividef(T0+1.0e-3f, 0.510998910f); // X
		d2 = __fdividef(p1Z*__logf(1.0f+2.0f*d1), d1) + __fdividef(p2Z + p3Z*d1 + p4Z*d1*d1, 1.0f + 20.0f*d1 + 230.0f*d1*d1 + 440.0f*d1*d1*d1); // sigma
		d3 = __fdividef(-T0 * (d2 - CrossSection), CrossSection*1.0e-3f); // c1
		d4 = (Z > 1.5f)? 0.375f-0.0556f*__logf(Z) : 0.15f; // c2
		d5 = __logf(__fdividef(E, T0)); // y
		CrossSection *= __expf(-d5 * (d3 + d4*d5));
	}

	return CrossSection;
}

// Compton Scatter (Klein-Nishina)
__device__ float Compton_scatter(StackGamma stackgamma, unsigned int id) {
	float E = stackgamma.E[id];
	int seed = stackgamma.seed[id];
	float E0 = __fdividef(E, 0.510998910f);

	float epszero = __fdividef(1.0f, (1.0f + 2.0f * E0));
	float eps02 = epszero*epszero;
	float a1 = -__logf(epszero);
	float a2 = __fdividef(a1, (a1 + 0.5f*(1.0f-eps02)));

	float greject, onecost, eps, eps2;
	do {
		if (a2 > park_miller_jb(&seed)) {
			eps = __expf(-a1 * park_miller_jb(&seed));
			eps2 = eps*eps;
		} else {
			eps2 = eps02 + (1.0f - eps02) * park_miller_jb(&seed);
			eps = sqrt(eps2);
		}
		onecost = __fdividef(1.0f - eps, eps * E0);
		greject = 1.0f - eps * onecost * __fdividef(2.0f - onecost, 1.0f + eps2);
	} while (greject < park_miller_jb(&seed));

	E *= eps;
	stackgamma.seed[id] = seed;
	stackgamma.E[id] = E;
	if (E <= 1.0e-6f) {
		stackgamma.live[id] = 0;
		return 0.0f;
	}
	
	return acos(1.0f - onecost);
}

// PhotoElectric Cross Section Per Atom, use Sandia data and load 21,236 Bytes on constant memory.
__device__ float PhotoElec_CSPA(float E, int Z) {
	float Emin = fmax(fIonizationPotentials[Z]*1e-6f, 0.01e-3f); // from Sandia, the same for all Z
	if (E < Emin) {return 0.0f;}
	
	int start = fCumulIntervals[Z];
	int stop = start + fNbOfIntervals[Z] - 1.0f;
	int pos;
	for (pos=stop; pos>start; --pos) {
		if (E < fSandiaTable[pos][0]*1.0e-3f) {break;}
	}
	float AoverAvo = 103.642688246e-10f * __fdividef((float)Z, fZtoAratio[Z]);
	float rE = __fdividef(1.0f, E);
	float rE2 = rE*rE;

	return rE * fSandiaTable[pos][1] * AoverAvo * 0.160217648e-22f
		+ rE2 * fSandiaTable[pos][2] * AoverAvo * 0.160217648e-25f
		+ rE * rE2 * fSandiaTable[pos][3] * AoverAvo * 0.160217648e-28f
		+ rE2 * rE2 * fSandiaTable[pos][4] * AoverAvo * 0.160217648e-31f;
}

__device__ float Compton_mu_Water(float E) {
	// H2O
	return (2*Compton_CSPA(E, 1) + Compton_CSPA(E, 8)) * 3.342796664e+19f; // Avogadro*H2O_density / (2*a_H+a_O)
}
__device__ float Compton_mu_Plastic(float E) {
	// 5C8H2O
	return (5*Compton_CSPA(E, 6) + 8*Compton_CSPA(E, 1) + 2*Compton_CSPA(E, 8)) * 7.096901340e17f;
}
__device__ float Compton_mu_Al(float E) {
	// Al
	return Compton_CSPA(E, 13) * 6.024030465e+19f; // Avogadro*Al_density/a_Al
}
__device__ float Compton_mu_Air(float E) {
	// O N Ar C
	return (0.231781f*Compton_CSPA(E, 8) + 0.755268f*Compton_CSPA(E, 7)
			+ 0.012827f*Compton_CSPA(E, 18) + 0.000124f*Compton_CSPA(E, 6)) * 5.247706935e17f;
}
__device__ float Compton_mu_Body(float E) {
	// H O
	return (0.112f*Compton_CSPA(E, 1) + 0.888f*Compton_CSPA(E, 8)) * 4.205077389e18f;
}
__device__ float Compton_mu_Lung(float E) {
	// H C N O Na P S Cl K
	return (0.103f*Compton_CSPA(E, 1)+ 0.105f*Compton_CSPA(E, 6) + 0.031f*Compton_CSPA(E, 7)
			+ 0.749f*Compton_CSPA(E, 8) + 0.002f*Compton_CSPA(E, 11) + 0.002f*Compton_CSPA(E, 15)
			+ 0.003f*Compton_CSPA(E, 16) + 0.003f*Compton_CSPA(E, 17) + 0.002f*Compton_CSPA(E, 19)) * 1.232299227e18f;
}
__device__ float Compton_mu_RibBone(float E) {
	// H C N O Na Mg P S Ca
	return (0.034f*Compton_CSPA(E, 1) + 0.155f*Compton_CSPA(E, 6) + 0.042f*Compton_CSPA(E, 7)
			+ 0.435f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.002f*Compton_CSPA(E, 12)
			+ 0.103f*Compton_CSPA(E, 15) + 0.003f*Compton_CSPA(E, 16) + 0.225f*Compton_CSPA(E, 20)) * 5.299038816e18f;
}
__device__ float Compton_mu_SpineBone(float E) {
	// H C N O Na Mg P S Cl K Ca
	return (0.063f*Compton_CSPA(E, 1) + 0.261f*Compton_CSPA(E, 6) + 0.039f*Compton_CSPA(E, 7)
			+ 0.436f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.001f*Compton_CSPA(E, 12)
			+ 0.061f*Compton_CSPA(E, 15) + 0.003f*Compton_CSPA(E, 16) + 0.001f*Compton_CSPA(E, 17)
			+ 0.001f*Compton_CSPA(E, 19) + 0.133f*Compton_CSPA(E, 20)) * 4.709337384e18f;
}
__device__ float Compton_mu_Heart(float E) {
	// H C N O Na P S Cl K
	return (0.104f*Compton_CSPA(E, 1) + 0.139f*Compton_CSPA(E, 6) + 0.029f*Compton_CSPA(E, 7)
			+ 0.718f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.002f*Compton_CSPA(E, 15)
			+ 0.002f*Compton_CSPA(E, 16) + 0.002f*Compton_CSPA(E, 17) + 0.003f*Compton_CSPA(E, 19)) * 4.514679219e18f;
}
__device__ float Compton_mu_Breast(float E) {
	// H C N O Na P S Cl
	return (0.106f*Compton_CSPA(E, 1) + 0.332f*Compton_CSPA(E, 6) + 0.03f*Compton_CSPA(E, 7)
			+ 0.527f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.001f*Compton_CSPA(E, 15)
			+ 0.002f*Compton_CSPA(E, 16) + 0.001f*Compton_CSPA(E, 17)) * 4.688916436e18f;
}

__device__ float PhotoElec_mu_Water(float E) {
	// H2O
	return (2*PhotoElec_CSPA(E, 1) + PhotoElec_CSPA(E, 8)) * 3.342796664e+19f; // Avogadro*H2O_density / (2*a_H+a_O)
}
__device__ float PhotoElec_mu_Plastic(float E) {
	// 5C8H2O
	return (5*PhotoElec_CSPA(E, 6) + 8*PhotoElec_CSPA(E, 1) + 2*PhotoElec_CSPA(E, 8)) * 7.096901340e17f;
}
__device__ float PhotoElec_mu_Al(float E) {
	// Al
	return PhotoElec_CSPA(E, 13) * 6.024030465e+19f; // Avogadro*Al_density/a_Al
}
__device__ float PhotoElec_mu_Air(float E) {
	// O N Ar C
	return (0.231781f*PhotoElec_CSPA(E, 8) + 0.755268f*PhotoElec_CSPA(E, 7)
			+ 0.012827f*PhotoElec_CSPA(E, 18) + 0.000124f*PhotoElec_CSPA(E, 6)) * 5.247706935e17f;
}
__device__ float PhotoElec_mu_Body(float E) {
	// H O
	return (0.112f*PhotoElec_CSPA(E, 1) + 0.888f*PhotoElec_CSPA(E, 8)) * 4.205077389e18f;
}
__device__ float PhotoElec_mu_Lung(float E) {
	// H C N O Na P S Cl K
	return (0.103f*PhotoElec_CSPA(E, 1)+ 0.105f*PhotoElec_CSPA(E, 6) + 0.031f*PhotoElec_CSPA(E, 7)
			+ 0.749f*PhotoElec_CSPA(E, 8) + 0.002f*PhotoElec_CSPA(E, 11) + 0.002f*PhotoElec_CSPA(E, 15)
			+ 0.003f*PhotoElec_CSPA(E, 16) + 0.003f*PhotoElec_CSPA(E, 17) + 0.002f*PhotoElec_CSPA(E, 19)) * 1.232299227e18f;
}
__device__ float PhotoElec_mu_RibBone(float E) {
	// H C N O Na Mg P S Ca
	return (0.034f*PhotoElec_CSPA(E, 1) + 0.155f*PhotoElec_CSPA(E, 6) + 0.042f*PhotoElec_CSPA(E, 7)
			+ 0.435f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.002f*PhotoElec_CSPA(E, 12)
			+ 0.103f*PhotoElec_CSPA(E, 15) + 0.003f*PhotoElec_CSPA(E, 16) + 0.225f*PhotoElec_CSPA(E, 20)) * 5.299038816e18f;
}
__device__ float PhotoElec_mu_SpineBone(float E) {
	// H C N O Na Mg P S Cl K Ca
	return (0.063f*PhotoElec_CSPA(E, 1) + 0.261f*PhotoElec_CSPA(E, 6) + 0.039f*PhotoElec_CSPA(E, 7)
			+ 0.436f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.001f*PhotoElec_CSPA(E, 12)
			+ 0.061f*PhotoElec_CSPA(E, 15) + 0.003f*PhotoElec_CSPA(E, 16) + 0.001f*PhotoElec_CSPA(E, 17)
			+ 0.001f*PhotoElec_CSPA(E, 19) + 0.133f*PhotoElec_CSPA(E, 20)) * 4.709337384e18f;
}
__device__ float PhotoElec_mu_Heart(float E) {
	// H C N O Na P S Cl K
	return (0.104f*PhotoElec_CSPA(E, 1) + 0.139f*PhotoElec_CSPA(E, 6) + 0.029f*PhotoElec_CSPA(E, 7)
			+ 0.718f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.002f*PhotoElec_CSPA(E, 15)
			+ 0.002f*PhotoElec_CSPA(E, 16) + 0.002f*PhotoElec_CSPA(E, 17) + 0.003f*PhotoElec_CSPA(E, 19)) * 4.514679219e18f;
}
__device__ float PhotoElec_mu_Breast(float E) {
	// H C N O Na P S Cl
	return (0.106f*PhotoElec_CSPA(E, 1) + 0.332f*PhotoElec_CSPA(E, 6) + 0.03f*PhotoElec_CSPA(E, 7)
			+ 0.527f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.001f*PhotoElec_CSPA(E, 15)
			+ 0.002f*PhotoElec_CSPA(E, 16) + 0.001f*PhotoElec_CSPA(E, 17)) * 4.688916436e18f;
}

// return attenuation according materials 
__device__ float att_from_mat(int mat, float E) {
	switch (mat) {
	case 0:     return Compton_mu_Air(E) + PhotoElec_mu_Air(E);
	case 1:     return Compton_mu_Body(E) + PhotoElec_mu_Body(E);
	case 2:     return Compton_mu_Lung(E) + PhotoElec_mu_Lung(E);
	case 3:     return Compton_mu_Breast(E) + PhotoElec_mu_Breast(E);
	case 4:     return Compton_mu_Heart(E) + PhotoElec_mu_Heart(E);
	case 5:     return Compton_mu_SpineBone(E) + PhotoElec_mu_SpineBone(E);
	case 6:     return Compton_mu_RibBone(E) + PhotoElec_mu_RibBone(E);
	case 98:    return Compton_mu_Plastic(E) + PhotoElec_mu_Plastic(E);
	case 99:	return Compton_mu_Water(E) + PhotoElec_mu_Water(E);
	case 100:	return Compton_mu_Al(E) + PhotoElec_mu_Al(E);
	}
	return 0.0f;
}


// Kernel interactions
__global__ void kernel_interactions(StackGamma stackgamma, float* ddose, int3 dimvol) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float theta, phi, dx, dy, dz, oldE, depdose;
	float Compton_CS, PhotoElec_CS, tot_CS, effect;
	int px, py, pz, jump, mat;
	int seed;
	if (id < stackgamma.size) {
		if (stackgamma.in[id] == 0) {return;} // if the particle is outside 
		
		seed = stackgamma.seed[id];
		dx = stackgamma.dx[id];
		dy = stackgamma.dy[id];
		dz = stackgamma.dz[id];
		px = int(stackgamma.px[id]);
		py = int(stackgamma.py[id]);
		pz = int(stackgamma.pz[id]);
		oldE = stackgamma.E[id];
		jump = dimvol.x * dimvol.y;
		//mat = int(dvol[pz*jump + py*dimvol.x + px]);
		mat = tex1Dfetch(tex_vol, pz*jump + py*dimvol.x + px);

		switch (mat) {
		case 0:     Compton_CS = Compton_mu_Air(oldE);       PhotoElec_CS = PhotoElec_mu_Air(oldE); break;
		case 1:     Compton_CS = Compton_mu_Body(oldE);      PhotoElec_CS = PhotoElec_mu_Body(oldE); break;
		case 2:     Compton_CS = Compton_mu_Lung(oldE);      PhotoElec_CS = PhotoElec_mu_Lung(oldE); break;
		case 3:     Compton_CS = Compton_mu_Breast(oldE);    PhotoElec_CS = PhotoElec_mu_Breast(oldE); break;
		case 4:     Compton_CS = Compton_mu_Heart(oldE);     PhotoElec_CS = PhotoElec_mu_Heart(oldE); break;
		case 5:     Compton_CS = Compton_mu_SpineBone(oldE); PhotoElec_CS = PhotoElec_mu_SpineBone(oldE); break;
		case 6:     Compton_CS = Compton_mu_RibBone(oldE);   PhotoElec_CS = PhotoElec_mu_RibBone(oldE); break;
		case 98:    Compton_CS = Compton_mu_Plastic(oldE);   PhotoElec_CS = PhotoElec_mu_Plastic(oldE); break;
		case 99:	Compton_CS = Compton_mu_Water(oldE);     PhotoElec_CS = PhotoElec_mu_Water(oldE); break;
		case 100:	Compton_CS = Compton_mu_Al(oldE);        PhotoElec_CS = PhotoElec_mu_Al(oldE); break;
		}

		// Select effect
		tot_CS = Compton_CS + PhotoElec_CS;
		PhotoElec_CS = __fdividef(PhotoElec_CS, tot_CS);
		Compton_CS = 1.0f;
		effect = park_miller_jb(&seed);

		if (effect <= PhotoElec_CS) {
			// PhotoElectric effect
			depdose = oldE;
			stackgamma.live[id] = 0;
			theta = 0.0f;
			phi = 0.0f;
			++stackgamma.ct_eff[id];
			++stackgamma.ct_PE[id];
		}
		if (effect > PhotoElec_CS && effect <= Compton_CS) {
			// Compton scattering
			theta = Compton_scatter(stackgamma, id);
			phi = park_miller_jb(&seed) * 2 * twopi;
			// !!!!! WARNING: should be 2*pi instead of 4*pi, it is to fix a pb with ParkMiller
			//                only uniform in half range ?! so the range must be twice.
			depdose = oldE - stackgamma.E[id];
			++stackgamma.ct_eff[id];
			++stackgamma.ct_Cpt[id];
		}

		// Dose depot
		ddose[pz*jump + py*dimvol.x + px] += depdose;
		// !!!!! WARNING: Atomic function is required (w/ ddose in uint)

		//*****************************************************
		// Apply new direction to the particle (use quaternion)
		//
		// create quaternion from particle and normalize it
		float4 p = make_float4(dx, dy, dz, 0.0f);
		p = quat_norm(p);
		// select best axis to compute the rotation axis
		float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if (dx<dy) {a.x=1.0f;} // choose x axis
		else {a.y=1.0f;}       // choose y axis
		// create virtual axis given by p^a
		a = quat_crossprod(p, a);
		a = quat_norm(a);
		// build rotation around p axis with phi (in order to rotate the next rotation axis a)
		float4 r = quat_axis(p, phi);
		// do rotation of a = rar*
		a = quat_mul(a, quat_conj(r)); // a = ar*
		a = quat_mul(r, a);            // a = ra
		// build rotation around a axis with theta (thus rotate p)
		r = quat_axis(a, theta);
		// do final rotation of p = rpr*
		p = quat_mul(p, quat_conj(r));
		p = quat_mul(r, p);
		// assign new values
		stackgamma.dx[id] = p.x;
		stackgamma.dy[id] = p.y;
		stackgamma.dz[id] = p.z;
		stackgamma.seed[id] = seed;
	}
}


/***********************************************************
 * Managment
 ***********************************************************/
__global__ void kernel_particle_rnd(StackGamma stackgamma, int3 dimvol) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	float phi, theta;
	int seed = stackgamma.seed[id];
	// warmpup to diverge
	park_miller_jb(&seed);
	park_miller_jb(&seed);
	park_miller_jb(&seed);
	if (id < stackgamma.size) {
		// position
		stackgamma.px[id] = park_miller_jb(&seed)*dimvol.x;
		stackgamma.py[id] = park_miller_jb(&seed)*dimvol.y;
		stackgamma.pz[id] = park_miller_jb(&seed)*dimvol.z;
		// direction
		phi = park_miller_jb(&seed) * twopi;
		theta = park_miller_jb(&seed) * pi - 0.5*pi;
		stackgamma.dx[id] = __cosf(theta) * __cosf(phi);
		stackgamma.dy[id] = __cosf(theta) * __sinf(phi);
		stackgamma.dz[id] = __sinf(theta);
		// enable particles
		stackgamma.live[id] = 1;
		stackgamma.in[id] = 1;
		stackgamma.seed[id] = seed;
	}
}

__global__ void kernel_particle_gun(StackGamma stackgamma, int3 dimvol,
									float posx, float posy, float posz,
									float dx, float dy, float dz, float E) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id < stackgamma.size) {
		if (stackgamma.in[id]==0 || stackgamma.live[id]==0) { 
			stackgamma.E[id] = E;
			stackgamma.px[id] = posx;
			stackgamma.py[id] = posy;
			stackgamma.pz[id] = posz;
			stackgamma.dx[id] = dx;
			stackgamma.dy[id] = dy;
			stackgamma.dz[id] = dz;
			stackgamma.live[id] = 1;
			stackgamma.in[id] = 1;
			stackgamma.ct_eff[id] = 0;
			stackgamma.ct_Cpt[id] = 0;
			stackgamma.ct_PE[id] = 0;
		}
	}
}

__global__ void kernel_particle_largegun(StackGamma stackgamma, int3 dimvol,
										 float posx, float posy, float posz,
										 float dx, float dy, float dz, float E, float rad) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id < stackgamma.size) {
		float phi, r;
		int seed;
		if (stackgamma.in[id]==0 || stackgamma.live[id]==0) {
			seed = stackgamma.seed[id];
			phi = park_miller_jb(&seed) * twopi;
			r   = park_miller_jb(&seed) * rad;
			stackgamma.seed[id] = seed;
			stackgamma.E[id] = E;
			stackgamma.px[id] = posx + r * __cosf(phi);
			stackgamma.py[id] = posy;
			stackgamma.pz[id] = posz + r * __sinf(phi);
			stackgamma.dx[id] = dx;
			stackgamma.dy[id] = dy;
			stackgamma.dz[id] = dz;
			stackgamma.live[id] = 1;
			stackgamma.in[id] = 1;
			stackgamma.ct_eff[id] = 0;
			stackgamma.ct_Cpt[id] = 0;
			stackgamma.ct_PE[id] = 0;
		}
	}

}

/***********************************************************
 * Tracking kernel
 ***********************************************************/
__global__ void kernel_siddon(int3 dimvol, StackGamma stackgamma, float* dtrack, float dimvox) {

	int3 u, i, e, stepi;
	float3 p0, pe, stept, astart, run, delta;
	float pq, oldv, newv, totv, val, E;
	float eps = 1.0e-5f;
	unsigned int id = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	int jump = dimvol.x*dimvol.y;
	int seed, inside, oldmat, mat;
	
	if (id < stackgamma.size) {
		p0.x = stackgamma.px[id];
		p0.y = stackgamma.py[id];
		p0.z = stackgamma.pz[id];
		delta.x = stackgamma.dx[id];
		delta.y = stackgamma.dy[id];
		delta.z = stackgamma.dz[id];
		seed = stackgamma.seed[id];
		E = stackgamma.E[id];

		// get free mean path
		//oldmat = dvol[int(p0.z)*jump + int(p0.y)*dimvol.x + int(p0.x)];
		oldmat = tex1Dfetch(tex_vol, int(p0.z)*jump + int(p0.y)*dimvol.x + int(p0.x));
		pq = -__fdividef(__logf(park_miller_jb(&seed)), att_from_mat(oldmat, E));
		pq = __fdividef(pq, dimvox);
		
		pe.x = p0.x + delta.x*pq;
		pe.y = p0.y + delta.y*pq;
		pe.z = p0.z + delta.z*pq;

		e.x = int(p0.x);
		e.y = int(p0.y);
		e.z = int(p0.z);

		if ((pe.x-p0.x) > 0) {stepi.x = 1; u.x = e.x + 1;}
		if ((pe.x-p0.x) < 0) {stepi.x = -1; u.x = e.x;}
		if ((pe.x-p0.x) == 0) {stepi.x = 0; u.x = e.x; pe.x = eps;}

		if ((pe.y-p0.y) > 0) {stepi.y = 1; u.y = e.y+1;}
		if ((pe.y-p0.y) < 0) {stepi.y = -1; u.y = e.y;}
		if ((pe.y-p0.y) == 0) {stepi.y = 0; u.y = e.y; pe.y = eps;}

		if ((pe.z-p0.z) > 0) {stepi.z = 1; u.z = e.z+1;}
		if ((pe.z-p0.z) < 0) {stepi.z = -1; u.z = e.z;}
		if ((pe.z-p0.z) == 0) {stepi.z = 0; u.z = e.z; pe.z = eps;}

		astart.x = (u.x - p0.x) / (pe.x - p0.x);
		astart.y = (u.y - p0.y) / (pe.y - p0.y);
		astart.z = (u.z - p0.z) / (pe.z - p0.z);

		run.x = astart.x * pq;
		run.y = astart.y * pq;
		run.z = astart.z * pq;
		oldv = run.x;
		if (run.y < oldv) {oldv = run.y;}
		if (run.z < oldv) {oldv = run.z;}

		stept.x = fabsf((pq / (pe.x - p0.x)));
		stept.y = fabsf((pq / (pe.y - p0.y)));
		stept.z = fabsf((pq / (pe.z - p0.z)));
		i.x = e.x;
		i.y = e.y;
		i.z = e.z;

		if (run.x == oldv) {run.x += stept.x; i.x += stepi.x;}
		if (run.y == oldv) {run.y += stept.y; i.y += stepi.y;}
		if (run.z == oldv) {run.z += stept.z; i.z += stepi.z;}

		// to debug
		//dtrack[e.z*jump + e.y*dimvol.x + e.x] += oldv;
		
		totv = 0.0f;
		inside = 1;
		while ((oldv < pq) & inside) {
			newv = run.x;
			if (run.y < newv) {newv=run.y;}
			if (run.z < newv) {newv=run.z;}
			val = (newv - oldv);

			// if mat change
			//mat = dvol[i.z*jump + i.y*dimvol.x + i.x];
			mat = tex1Dfetch(tex_vol, i.z*jump + i.y*dimvol.x + i.x);
			if (mat != oldmat) {
				pq = -__fdividef(__logf(park_miller_jb(&seed)), att_from_mat(oldmat, E));
				oldmat = mat;
			}

			//dtrack[i.z*jump + i.y*dimvol.x + i.x] += val;

			totv += val;
			oldv = newv;
			if (run.x==newv) {i.x += stepi.x; run.x += stept.x;}
			if (run.y==newv) {i.y += stepi.y; run.y += stept.y;}
			if (run.z==newv) {i.z += stepi.z; run.z += stept.z;}
			inside = (i.x >= 0) & (i.x < dimvol.x) & (i.y >= 0) & (i.y < dimvol.y) & (i.z >= 0) & (i.z < dimvol.z);
		}

		pe.x = p0.x + delta.x*oldv;
		pe.y = p0.y + delta.y*oldv;
		pe.z = p0.z + delta.z*oldv;
		stackgamma.seed[id] = seed;
		stackgamma.px[id] = pe.x;
		stackgamma.py[id] = pe.y;
		stackgamma.pz[id] = pe.z;

		if (!inside) {stackgamma.in[id] = 0;}

	} // id < nx

}

/*
__global__ void kernel_amanatides(float* dvol, float* dX0, float* dY0, float* dZ0,
								  float* dXe, float* dYe, float* dZe, int nx0, int jump, int nx) {

	int3 u, i, e, stepi;
	float3 p0, pe, stept, astart, run;
	float pq, oldv, totv, mu, val;
	float eps = 1.0e-5f;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < nx0) {
		p0.x = dX0[id];
		p0.y = dY0[id];
		p0.z = dZ0[id];
		pe.x = dXe[id];
		pe.y = dYe[id];
		pe.z = dZe[id];

		e.x = int(p0.x);
		e.y = int(p0.y);
		e.z = int(p0.z);

		if ((pe.x-p0.x) > 0) {stepi.x = 1; u.x = e.x + 1;}
		if ((pe.x-p0.x) < 0) {stepi.x = -1; u.x = e.x;}
		if ((pe.x-p0.x) == 0) {stepi.x = 0; u.x = e.x; pe.x = eps;}

		if ((pe.y-p0.y) > 0) {stepi.y = 1; u.y = e.y+1;}
		if ((pe.y-p0.y) < 0) {stepi.y = -1; u.y = e.y;}
		if ((pe.y-p0.y) == 0) {stepi.y = 0; u.y = e.y; pe.y = eps;}

		if ((pe.z-p0.z) > 0) {stepi.z = 1; u.z = e.z+1;}
		if ((pe.z-p0.z) < 0) {stepi.z = -1; u.z = e.z;}
		if ((pe.z-p0.z) == 0) {stepi.z = 0; u.z = e.z; pe.z = eps;}

		astart.x = (u.x - p0.x) / (pe.x - p0.x);
		astart.y = (u.y - p0.y) / (pe.y - p0.y);
		astart.z = (u.z - p0.z) / (pe.z - p0.z);
		
		pq = sqrtf((p0.x-pe.x)*(p0.x-pe.x)+(p0.y-pe.y)*(p0.y-pe.y)+(p0.z-pe.z)*(p0.z-pe.z));
		run.x = astart.x * pq;
		run.y = astart.y * pq;
		run.z = astart.z * pq;
		oldv = run.x;
		if (run.y < oldv) {oldv = run.y;}
		if (run.z < oldv) {oldv = run.z;}

		stept.x = fabsf((pq / (pe.x - p0.x)));
		stept.y = fabsf((pq / (pe.y - p0.y)));
		stept.z = fabsf((pq / (pe.z - p0.z)));
		i.x = e.x;
		i.y = e.y;
		i.z = e.z;

		mu = oldv*dvol[e.z*jump + e.y*nx + e.x];
		//dvol[e.z*jump + e.y*nx + e.x] += oldv;
		
		totv = 0.0f;
		while (totv < pq) {
			if (run.x < run.y) {
				if (run.x < run.z) {i.x += stepi.x; run.x += stept.x;}
				else {i.z += stepi.z; run.z += stept.z;}
			} else {
				if (run.y < run.z) {i.y += stepi.y; run.y += stept.y;}
				else {i.z += stepi.z; run.z += stept.z;}
			}
			totv = run.x;
			if (run.y < totv) {totv=run.y;}
			if (run.z < totv) {totv=run.z;}
			val = totv-oldv;
			mu = val * dvol[i.z*jump + i.y*nx + i.x];
			//dvol[i.z*jump + i.y*nx + i.x] += val;
			oldv = totv;
		}
		
		mu = (pq-totv)*dvol[i.z*jump + i.y*nx + i.x];
		//dvol[i.z*jump + i.y*nx + i.x] += (pq - totv);

	} // id < nx

}
*/
/*
__global__ void kernel_raypro(float* dvol, int3 dimvol, StackGamma stackgamma) {
	float3 xi, x0, d, rd, db, sd;
	int3 p, b, ob;
	float t, tn, tot_t, dist, mu, phi, theta;
	float eps = 1.0e-5f;
	int jump = dimvol.x*dimvol.y;

	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int seed = stackgamma.seed[id];
	int inside;
	int watchdog;
	if (id < stackgamma.size) {
		x0.x = stackgamma.px[id];
		x0.y = stackgamma.py[id];
		x0.z = stackgamma.pz[id];
		d.x = stackgamma.dx[id];
		d.y = stackgamma.dy[id];
		d.z = stackgamma.dz[id];

		dist = -__logf(park_miller_jb(&seed)) / 0.018f;

		if (d.x==0) {d.x=eps;}
		if (d.y==0) {d.y=eps;}
		if (d.z==0) {d.z=eps;}

		rd.x = __fdividef(1.0f, d.x);
		rd.y = __fdividef(1.0f, d.y);
		rd.z = __fdividef(1.0f, d.z);

		db.x = (d.x > 0) - (d.x < 0) * eps;
		db.y = (d.y > 0) - (d.y < 0) * eps;
		db.z = (d.z > 0) - (d.z < 0) * eps;

		b.x = int(x0.x+db.x);
		b.y = int(x0.y+db.y);
		b.z = int(x0.z+db.z);
		ob.x = b.x; ob.y = b.y; ob.z = b.z;

		t = (b.x - x0.x) * rd.x;
		tn = (b.y - x0.y) * rd.y;
		t = fminf(t, tn);
		tn = (b.z - x0.z) * rd.z;
		t = fminf(t, tn);

		xi.x = x0.x + (d.x * t);
		xi.y = x0.y + (d.y * t);
		xi.z = x0.z + (d.z * t);

		tn = 1.0f + int(xi.x) - xi.x;
		xi.x += (tn * (tn < eps));
		tn = 1.0f + int(xi.y) - xi.y;
		xi.y += (tn * (tn < eps));
		tn = 1.0f + int(xi.z) - xi.z;
		xi.z += (tn * (tn < eps));

		tot_t = t;
		p.x = int(x0.x);
		p.y = int(x0.y);
		p.z = int(x0.z);

		inside = 1;
		watchdog=0;
		while ((tot_t < dist) & inside) {
			mu = t * dvol[p.z*jump + p.y*dimvol.x + p.x];
			//dvol[p.z*jump + p.y*dimvol.x + p.x] += t;
			
			b.x = int(xi.x + db.x);
			b.y = int(xi.y + db.y);
			b.z = int(xi.z + db.z);

			t = (b.x - xi.x) * rd.x;
			tn = (b.y - xi.y) * rd.y;
			t = fminf(t, tn);
			tn = (b.z - xi.z) * rd.z;
			t = fminf(t, tn);
			
			xi.x = xi.x + (d.x * t);
			xi.y = xi.y + (d.y * t);
			xi.z = xi.z + (d.z * t);

			tot_t += t;
			p.x += (b.x - ob.x);
			p.y += (b.y - ob.y);
			p.z += (b.z - ob.z);
			ob.x = b.x; ob.y = b.y; ob.z = b.z;
			
			inside = (p.x >= 0) & (p.x < dimvol.x) & (p.y >= 0) & (p.y < dimvol.y) & (p.z >= 0) & (p.z < dimvol.z);
			dvol[watchdog] = p.x;
			++watchdog;
			if (watchdog > 500) {
				//dvol[0] = b.x;
				//dvol[1] = p.z;
				break;
			}

		}

		if (!inside) {
			stackgamma.in[id] = 0;
			return;
		}

		mu = (dist-tot_t) * dvol[p.z*jump + p.y*dimvol.x + p.x];
		//dvol[p.z*jump + p.y*dimvol.x + p.x] += (dist-tot_t);

		stackgamma.seed[id] = seed;
		
	} // id
	
}
*/

/***********************************************************
 * Main
 ***********************************************************/
void mc_cuda(float* vol, int nz, int ny, int nx,
			 float* E, int nE, float* dx, int ndx, float* dy, int ndy, float* dz, int ndz,
			 float* px, int npx, float* py, int npy, float* pz, int npz,
			 int nparticles) {
	cudaSetDevice(1);

    timeval start, end;
    double t1, t2, diff;
	int3 dimvol;
	int n, step;
	int countparticle=0;
	
	dimvol.x = nx;
	dimvol.y = ny;
	dimvol.z = nz;

	// Volume allocation
	unsigned int mem_vol = nz*ny*nx * sizeof(float);
	float* dvol;
	cudaMalloc((void**) &dvol, mem_vol);
	cudaMemcpy(dvol, vol, mem_vol, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_vol, dvol, mem_vol);
	float* dtrack;
	cudaMalloc((void**) &dtrack, mem_vol);
	cudaMemset(dtrack, 0, mem_vol);
	float* ddose;
	cudaMalloc((void**) &ddose, mem_vol);
	cudaMemset(ddose, 0, mem_vol);

	// Stacks
	StackGamma stackgamma;
	StackGamma collector;
	stackgamma.size = nparticles;
	//unsigned int mem_stack = stackgamma.size * sizeof(stackgamma);
	unsigned int mem_stack_float = stackgamma.size * sizeof(float);
	unsigned int mem_stack_int = stackgamma.size * sizeof(int);
	unsigned int mem_stack_char = stackgamma.size * sizeof(char);

	// Host stack allocation memory
	collector.E = (float*)malloc(mem_stack_float);
	collector.dx = (float*)malloc(mem_stack_float);
	collector.dy = (float*)malloc(mem_stack_float);
	collector.dz = (float*)malloc(mem_stack_float);
	collector.px = (float*)malloc(mem_stack_float);
	collector.py = (float*)malloc(mem_stack_float);
	collector.pz = (float*)malloc(mem_stack_float);
	collector.live = (unsigned char*)malloc(mem_stack_char);
	collector.in = (unsigned char*)malloc(mem_stack_char);
	collector.ct_eff = (int*)malloc(mem_stack_int);
	collector.ct_Cpt = (int*)malloc(mem_stack_int);
	collector.ct_PE = (int*)malloc(mem_stack_int);

	// Device stack allocation memory
	cudaMalloc((void**) &stackgamma.E, mem_stack_float);
	cudaMalloc((void**) &stackgamma.dx, mem_stack_float);
	cudaMalloc((void**) &stackgamma.dy, mem_stack_float);
	cudaMalloc((void**) &stackgamma.dz, mem_stack_float);
	cudaMalloc((void**) &stackgamma.px, mem_stack_float);
	cudaMalloc((void**) &stackgamma.py, mem_stack_float);
	cudaMalloc((void**) &stackgamma.pz, mem_stack_float);
	cudaMalloc((void**) &stackgamma.seed, mem_stack_int);
	cudaMalloc((void**) &stackgamma.live, mem_stack_char);
	cudaMalloc((void**) &stackgamma.in, mem_stack_char);
	cudaMalloc((void**) &stackgamma.ct_eff, mem_stack_int);
	cudaMalloc((void**) &stackgamma.ct_Cpt, mem_stack_int);
	cudaMalloc((void**) &stackgamma.ct_PE, mem_stack_int);
	cudaMemset(stackgamma.live, 0, mem_stack_char); // at beginning all particles are dead
	cudaMemset(stackgamma.in, 0, mem_stack_char);   // and outside the volume
	
	// Init seeds
	int* tmp = (int*)malloc(stackgamma.size * sizeof(int));
	srand(10);
	n=0;
	while (n<stackgamma.size) {tmp[n] = rand(); ++n;}
	cudaMemcpy(stackgamma.seed, tmp, mem_stack_int, cudaMemcpyHostToDevice);
	free(tmp);

	// Vars kernel
	dim3 threads, grid;
	int block_size = 256;
	int grid_size = (nparticles + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;

	// Outter loop
	for (step=0; step<2; ++step) {
		printf("Step %i\n", step);
		// Init particles
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		kernel_particle_largegun<<<grid, threads>>>(stackgamma, dimvol, 45.0, 0.0, 35.0, 0.0, 1.0, 0.0, 0.511, 5.0);
		cudaThreadSynchronize();
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("   Create gamma particles %f s\n", diff);
	
		// Propagation
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		kernel_siddon<<<grid, threads>>>(dimvol, stackgamma, dtrack, 4.0); // 4.0 mm3 voxel
		cudaThreadSynchronize();
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("   Track gamma particles %f s\n", diff);

		// Interactions
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		kernel_interactions<<<grid, threads>>>(stackgamma, ddose, dimvol);
		cudaThreadSynchronize();
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("   Interactions gamma particles %f s\n", diff);

		// Collector
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		cudaMemcpy(collector.E, stackgamma.E, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dx, stackgamma.dx, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dy, stackgamma.dy, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dz, stackgamma.dz, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.px, stackgamma.px, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.py, stackgamma.py, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.pz, stackgamma.pz, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.live, stackgamma.live, mem_stack_char, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.in, stackgamma.in, mem_stack_char, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.ct_eff, stackgamma.ct_eff, mem_stack_int, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.ct_Cpt, stackgamma.ct_Cpt, mem_stack_int, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.ct_PE, stackgamma.ct_PE, mem_stack_int, cudaMemcpyDeviceToHost);			
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("   Get back stack of gamma particles %f s\n", diff);

		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		int c1=0;
		int c2=0;
		int c3=0;
		int c4=0;
		n=0;
		while(n<nparticles && countparticle<nparticles) {
			if (collector.in[n] == 0) {
				E[countparticle] = collector.E[n];
				dx[countparticle] = collector.dx[n];
				dy[countparticle] = collector.dy[n];
				dz[countparticle] = collector.dz[n];
				px[countparticle] = collector.px[n];
				py[countparticle] = collector.py[n];
				pz[countparticle] = collector.pz[n];
				++countparticle;
			}
			if (collector.live[n] == 0) {++c1;}
			c2 += collector.ct_eff[n];
			c3 += collector.ct_Cpt[n];
			c4 += collector.ct_PE[n];
			++n;
		}
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		
		printf("   Store gamma particles %f s\n", diff);
		printf("   Nb particles outside %i absorbed %i\n", countparticle, c1);
		printf("   Tot interaction %i: %i Compton %i Photo-Electric\n", c2, c3, c4);

	} // outter loop (step)
	
	//cudaMemcpy(tmp, stackgamma.seed, mem_stack_int, cudaMemcpyDeviceToHost);
	//cudaMemcpy(vol, dtrack, mem_vol, cudaMemcpyDeviceToHost);
	cudaMemcpy(vol, ddose, mem_vol, cudaMemcpyDeviceToHost);

	// Clean memory
	free(collector.E);
	free(collector.dx);
	free(collector.dy);
	free(collector.dz);
	free(collector.px);
	free(collector.py);
	free(collector.pz);
	free(collector.live);
	free(collector.in);
	free(collector.ct_eff);
	free(collector.ct_Cpt);
	free(collector.ct_PE);
	
	cudaUnbindTexture(tex_vol);
	cudaFree(dvol);
	cudaFree(ddose);
	cudaFree(stackgamma.E);
	cudaFree(stackgamma.dx);
	cudaFree(stackgamma.dy);
	cudaFree(stackgamma.dz);
	cudaFree(stackgamma.px);
	cudaFree(stackgamma.py);
	cudaFree(stackgamma.pz);
	cudaFree(stackgamma.live);
	cudaFree(stackgamma.in);
	cudaFree(stackgamma.seed);
	cudaFree(stackgamma.ct_eff);
	cudaFree(stackgamma.ct_Cpt);
	cudaFree(stackgamma.ct_PE);
	cudaThreadExit();

}
