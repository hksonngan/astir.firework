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
texture<float, 1, cudaReadModeElementType> tex_small_vol;
texture<float, 1, cudaReadModeElementType> tex_tiny_vol;
texture<float, 1, cudaReadModeElementType> tex_rayl_cs;
texture<float, 1, cudaReadModeElementType> tex_rayl_ff;
texture<int, 1, cudaReadModeElementType> tex_ind;

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
	int* ct_Ray;
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

// PhotoElectric Cross Section Per Atom, use Sandia data and load 21,236 Bytes in constant memory.
__device__ float PhotoElec_CSPA(float E, int Z) {
	float Emin = fmax(fIonizationPotentials[Z]*1e-6f, 0.01e-3f); // from Sandia, the same for all Z
	if (E < Emin) {return 0.0f;}
	
	int start = fCumulIntervals[Z];
	int stop = start + fNbOfIntervals[Z] - 1;
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

// Rayleigh Cross Section Per Atom and Form Factor, use Rayleigh data table, and load 1,616 Bytes in constant memory,
// and 970,560 Bytes in texture memory
__device__ float Rayleigh_CSPA(float E, int Z) {
	if (E < 250e-6f || E > 100e3f) {return 0.0f;} // 250 eV < E < 100 GeV

	int start = Rayleigh_cs_CumulIntervals[Z];
	int stop  = start + 2 * (Rayleigh_cs_NbIntervals[Z] - 1);

	int pos;
	for (pos=start; pos<stop; pos+=2) {
		if (tex1Dfetch(tex_rayl_cs, pos) >= E) {break;}
	}

	//float hi_E = tex1Dfetch(tex_rayl_cs, pos);
	float lo_cs = tex1Dfetch(tex_rayl_cs, pos-1);
	if (E < 1e3f) { // 1 Gev
		float rlo_E = __fdividef(1.0f, tex1Dfetch(tex_rayl_cs, pos-2));
		float logcs = __log10f(lo_cs) + __fdividef(__log10f(__fdividef(tex1Dfetch(tex_rayl_cs, pos+1), lo_cs))
												   * __log10f(E * rlo_E), __log10f(tex1Dfetch(tex_rayl_cs, pos) * rlo_E));
		return __powf(10.0f, logcs) * 1.0e-22f;
	}
	else {return lo_cs * 1.0e-22f;}
}
__device__ float Rayleigh_FF(float E, int Z) {
	int start = Rayleigh_ff_CumulIntervals[Z];
	int stop  = start + 2 * (Rayleigh_ff_NbIntervals[Z] - 1);
	int pos;
	for (pos=start; pos<stop; pos+=2) {
		if (tex1Dfetch(tex_rayl_ff, pos) >= E) {break;}
	}
	float hi_E = tex1Dfetch(tex_rayl_ff, pos);
	float lo_cs = tex1Dfetch(tex_rayl_ff, pos-1);
	if (E < 1e3f) { // 1 Gev
		float rlo_E = __fdividef(1.0f, tex1Dfetch(tex_rayl_ff, pos-2));
		float logcs = __log10f(tex1Dfetch(tex_rayl_ff, pos-1)) * __log10f(__fdividef(hi_E, E))
			+ __log10f(tex1Dfetch(tex_rayl_ff, pos+1)) * __fdividef(__log10f(E * rlo_E), __log10f(hi_E * rlo_E));
		return __powf(10.0f, logcs);
	}
	else {return lo_cs;}
}

// Rayleigh Scatter
__device__ float Rayleigh_scatter(StackGamma stackgamma, unsigned int id, int Z) {
	float E = stackgamma.E[id];
	if (E <= 250.0e-6f) { // 250 eV
		stackgamma.live[id] = 0;
		return 0.0f;
	}
	int seed = stackgamma.seed[id];
	float wphot = __fdividef(123.984187539e-11f, E);
	float costheta, sinthetahalf, FF;
	do {
		if (E > 5.0f) {costheta = 1.0f;}
		else {
			do {
				costheta = 2.0f * park_miller_jb(&seed) - 1.0f;
			} while (((1.0f + costheta*costheta)*0.5f) < park_miller_jb(&seed));
		}
		sinthetahalf = sqrt((1.0f - costheta) * 0.5f);
		E = __fdividef(sinthetahalf, wphot * 0.1f);
		FF = (E > 1.0e5f)? Rayleigh_FF(E, Z) : Rayleigh_FF(0.0f, Z);
		// reuse costheta as sintheta
		costheta = sqrt(1.0f - costheta*costheta);
	} while (FF*FF < (park_miller_jb(&seed) * Z*Z));

	return asin(costheta);
}

/***********************************************************
 * Materials
 ***********************************************************/

// return attenuation according materials 
__device__ float att_from_mat(int mat, float E) {
	switch (mat) {
	case 0:     return Compton_mu_Air(E) + PhotoElec_mu_Air(E) + Rayleigh_mu_Air(E);
	case 1:     return Compton_mu_Water(E) + PhotoElec_mu_Water(E) + Rayleigh_mu_Water(E);
	case 2:     return Compton_mu_Body(E) + PhotoElec_mu_Body(E) + Rayleigh_mu_Body(E);
	case 3:     return Compton_mu_Lung(E) + PhotoElec_mu_Lung(E) + Rayleigh_mu_Lung(E);
	case 4:     return Compton_mu_Breast(E) + PhotoElec_mu_Breast(E) + Rayleigh_mu_Breast(E);
	case 5:     return Compton_mu_Heart(E) + PhotoElec_mu_Heart(E) + Rayleigh_mu_Heart(E);
	case 6:     return Compton_mu_SpineBone(E) + PhotoElec_mu_SpineBone(E) + Rayleigh_mu_SpineBone(E);
	case 7:     return Compton_mu_RibBone(E) + PhotoElec_mu_RibBone(E) + Rayleigh_mu_RibBone(E);
	case 8:     return Compton_mu_Intestine(E) + PhotoElec_mu_Intestine(E) + Rayleigh_mu_Intestine(E);
	case 9:     return Compton_mu_Spleen(E) + PhotoElec_mu_Spleen(E) + Rayleigh_mu_Spleen(E);
	case 10:     return Compton_mu_Blood(E) + PhotoElec_mu_Blood(E) + Rayleigh_mu_Blood(E);
	case 11:     return Compton_mu_Liver(E) + PhotoElec_mu_Liver(E) + Rayleigh_mu_Liver(E);
	case 12:     return Compton_mu_Kidney(E) + PhotoElec_mu_Kidney(E) + Rayleigh_mu_Kidney(E);
		//case 13:     return Compton_mu_Brain(E) + PhotoElec_mu_Brain(E) + Rayleigh_mu_Brain(E);
		//case 14:     return Compton_mu_Pancreas(E) + PhotoElec_mu_Pancreas(E) + Rayleigh_mu_Pancreas(E);
		//case 99:    return Compton_mu_Plastic(E) + PhotoElec_mu_Plastic(E) + Rayleigh_mu_Plastic(E);
		//case 100:	return Compton_mu_Al(E) + PhotoElec_mu_Al(E) + Rayleigh_mu_Al(E);
	}
	return 0.0f;
}

// return E element according materials and a random number (roulette-wheel method)
__device__ int rnd_Z_from_mat(int mat, float rnd) {
	int pos = 0;
	switch (mat) {
	case 0:     while(Air_cumul[pos] < rnd) {++pos;}; return Air_Z[pos];
	case 1:	    while(Water_cumul[pos] < rnd) {++pos;}; return Water_Z[pos];		
	case 2:     while(Body_cumul[pos] < rnd) {++pos;}; return Body_Z[pos];
	case 3:     while(Lung_cumul[pos] < rnd) {++pos;}; return Lung_Z[pos];
	case 4:     while(Breast_cumul[pos] < rnd) {++pos;}; return Breast_Z[pos];
	case 5:     while(Heart_cumul[pos] < rnd) {++pos;}; return Heart_Z[pos];
	case 6:     while(SpineBone_cumul[pos] < rnd) {++pos;}; return SpineBone_Z[pos];
	case 7:     while(RibBone_cumul[pos] < rnd) {++pos;}; return RibBone_Z[pos];
	case 8:     while(Intestine_cumul[pos] < rnd) {++pos;}; return Intestine_Z[pos];
	case 9:     while(Spleen_cumul[pos] < rnd) {++pos;}; return Spleen_Z[pos];
	case 10:    while(Blood_cumul[pos] < rnd) {++pos;}; return Blood_Z[pos];
	case 11:    while(Liver_cumul[pos] < rnd) {++pos;}; return Liver_Z[pos];
	case 12:    while(Kidney_cumul[pos] < rnd) {++pos;}; return Kidney_Z[pos];
		//case 99:    while(Plastic_cumul[pos] < rnd) {++pos;}; return Plastic_Z[pos];
		//case 100:	while(Al_cumul[pos] < rnd) {++pos;}; return Al_Z[pos];
	}
	return 0; // In order to avoid a warning during the compilation
}

/***********************************************************
 * Interactions
 ***********************************************************/

// Kernel interactions
__global__ void kernel_interactions(StackGamma stackgamma, float* ddose, int3 dimvol) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float theta, phi, dx, dy, dz, oldE, depdose;
	float Compton_CS, PhotoElec_CS, Rayleigh_CS, tot_CS, effect;
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
		case 0:
			Compton_CS = Compton_mu_Air(oldE);
			PhotoElec_CS = PhotoElec_mu_Air(oldE);
			Rayleigh_CS = Rayleigh_mu_Air(oldE);
			break;
		case 1:
			Compton_CS = Compton_mu_Water(oldE);
			PhotoElec_CS = PhotoElec_mu_Water(oldE);
			Rayleigh_CS = Rayleigh_mu_Water(oldE);
			break;
		case 2:
			Compton_CS = Compton_mu_Body(oldE);
			PhotoElec_CS = PhotoElec_mu_Body(oldE);
			Rayleigh_CS = Rayleigh_mu_Body(oldE); break;
		case 3:
			Compton_CS = Compton_mu_Lung(oldE);
			PhotoElec_CS = PhotoElec_mu_Lung(oldE);
			Rayleigh_CS = Rayleigh_mu_Lung(oldE); break;
		case 4:
			Compton_CS = Compton_mu_Breast(oldE);
			PhotoElec_CS = PhotoElec_mu_Breast(oldE);
			Rayleigh_CS = Rayleigh_mu_Breast(oldE); break;
		case 5:
			Compton_CS = Compton_mu_Heart(oldE);
			PhotoElec_CS = PhotoElec_mu_Heart(oldE);
			Rayleigh_CS = Rayleigh_mu_Heart(oldE); break;
		case 6:
			Compton_CS = Compton_mu_SpineBone(oldE);
			PhotoElec_CS = PhotoElec_mu_SpineBone(oldE);
			Rayleigh_CS = Rayleigh_mu_SpineBone(oldE); break;
		case 7:
			Compton_CS = Compton_mu_RibBone(oldE);
			PhotoElec_CS = PhotoElec_mu_RibBone(oldE);
			Rayleigh_CS = Rayleigh_mu_RibBone(oldE);
			break;
		case 8:
			Compton_CS = Compton_mu_Intestine(oldE);
			PhotoElec_CS = PhotoElec_mu_Intestine(oldE);
			Rayleigh_CS = Rayleigh_mu_Intestine(oldE);
			break;
		case 9:
			Compton_CS = Compton_mu_Spleen(oldE);
			PhotoElec_CS = PhotoElec_mu_Spleen(oldE);
			Rayleigh_CS = Rayleigh_mu_Spleen(oldE);
			break;
		case 10:
			Compton_CS = Compton_mu_Blood(oldE);
			PhotoElec_CS = PhotoElec_mu_Blood(oldE);
			Rayleigh_CS = Rayleigh_mu_Blood(oldE);
			break;
		case 11:
			Compton_CS = Compton_mu_Liver(oldE);
			PhotoElec_CS = PhotoElec_mu_Liver(oldE);
			Rayleigh_CS = Rayleigh_mu_Liver(oldE);
			break;
		case 12:
			Compton_CS = Compton_mu_Kidney(oldE);
			PhotoElec_CS = PhotoElec_mu_Kidney(oldE);
			Rayleigh_CS = Rayleigh_mu_Kidney(oldE);
			break;
			
			/*
		case 98:
			Compton_CS = Compton_mu_Plastic(oldE);
			PhotoElec_CS = PhotoElec_mu_Plastic(oldE);
			Rayleigh_CS = Rayleigh_mu_Plastic(oldE); break;
		case 100:
			Compton_CS = Compton_mu_Al(oldE);
			PhotoElec_CS = PhotoElec_mu_Al(oldE);
			Rayleigh_CS = Rayleigh_mu_Al(oldE); break;
			*/
		}

		// Select effect
		tot_CS = Compton_CS + PhotoElec_CS + Rayleigh_CS;
		PhotoElec_CS = __fdividef(PhotoElec_CS, tot_CS);
		Compton_CS = __fdividef(Compton_CS, tot_CS) + PhotoElec_CS;
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
			phi = park_miller_jb(&seed) * 2.0f * twopi;
			// !!!!! WARNING: should be 2*pi instead of 4*pi, it is to fix a pb with ParkMiller
			//                only uniform in half range ?! so the range must be twice.
			depdose = oldE - stackgamma.E[id];
			++stackgamma.ct_eff[id];
			++stackgamma.ct_Cpt[id];
		}
		if (effect > Compton_CS) {
			phi = park_miller_jb(&seed); // rnd number
			theta = Rayleigh_scatter(stackgamma, id, rnd_Z_from_mat(mat, phi)); // phi is a random number
			phi = park_miller_jb(&seed) * 2.0f * twopi;
			// !!!!! WARNING: should be 2*pi instead of 4*pi, it is to fix a pb with ParkMiller
			//                only uniform in half range ?! so the range must be twice.
			++stackgamma.ct_eff[id];
			++stackgamma.ct_Ray[id];
			depdose = 0.0f;
		}
			
		// Dose depot
		//ddose[pz*jump + py*dimvol.x + px] += depdose;
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
 * Sources
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
			stackgamma.ct_Ray[id] = 0;
		}
	}
}

__global__ void kernel_particle_largegun(StackGamma stackgamma, int3 dimvol,
										 float posx, float posy, float posz,
										 float dx, float dy, float dz, float E, float rad) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id < stackgamma.size) {
		float x, y;
		int seed;
		if (stackgamma.in[id]==0 || stackgamma.live[id]==0) {
			seed = stackgamma.seed[id];
			//phi = park_miller_jb(&seed) * twopi;
			//r   = park_miller_jb(&seed) * 5.0f;
			x = park_miller_jb(&seed) * 2.0f * rad;
			y = park_miller_jb(&seed) * 2.0f * rad;			
			stackgamma.seed[id] = seed;
			stackgamma.E[id] = E;
			
			//stackgamma.px[id] = posx + r * __cosf(phi);
			//stackgamma.py[id] = posy + r * __sinf(phi);
			stackgamma.px[id] = posx + x - rad;
			stackgamma.py[id] = posy + y - rad;
			stackgamma.pz[id] = posz;
			stackgamma.dx[id] = dx;
			stackgamma.dy[id] = dy;
			stackgamma.dz[id] = dz;
			stackgamma.live[id] = 1;
			stackgamma.in[id] = 1;
			stackgamma.ct_eff[id] = 0;
			stackgamma.ct_Cpt[id] = 0;
			stackgamma.ct_PE[id] = 0;
			stackgamma.ct_Ray[id] = 0;
		}
	}

}

__global__ void kernel_particle_isotrope(StackGamma stackgamma, float posx, float posy, float posz, float E, float* dtrack) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id < stackgamma.size) {
		float phi, theta, x, y, z;
		int seed;
		if (stackgamma.in[id]==0 || stackgamma.live[id]==0) {
			seed = stackgamma.seed[id];

			phi = park_miller_jb(&seed);
			theta = park_miller_jb(&seed);
			phi   = twopi * phi;
			theta = acosf(1.0f - 2.0f*theta);
			//theta = acosf(theta * 0.0136f + 0.9864f);
			//theta = acosf(theta * 0.165f + 0.835f);
			// convert to cartesian
			x = __cosf(phi)*__sinf(theta);
			y = __sinf(phi)*__sinf(theta);
			z = __cosf(theta);
			//if (z<0.0f) {z = -z;}

			//dtrack[id] = posz;
			
			stackgamma.dx[id] = x;
			stackgamma.dy[id] = y;
			stackgamma.dz[id] = z;
			stackgamma.seed[id] = seed;
			stackgamma.E[id] = E;
			stackgamma.px[id] = posx;
			stackgamma.py[id] = posy;
			stackgamma.pz[id] = posz;
			stackgamma.live[id] = 1;
			stackgamma.in[id] = 1;
			stackgamma.ct_eff[id] = 0;
			stackgamma.ct_Cpt[id] = 0;
			stackgamma.ct_PE[id] = 0;
			stackgamma.ct_Ray[id] = 0;
		}
	}

}

__global__ void kernel_particle_area(StackGamma stackgamma, int mina, int maxa, int minb, int maxb, float E) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id < stackgamma.size) {
		float x, y;
		int seed;
		if (stackgamma.in[id]==0 || stackgamma.live[id]==0) {
			seed = stackgamma.seed[id];

			x = park_miller_jb(&seed);
			y = park_miller_jb(&seed);
			x = x * float(maxa - mina) + (float)mina;
			y = y * float(maxb - minb) + (float)minb;

			stackgamma.dx[id] = 0;
			stackgamma.dy[id] = 0;
			stackgamma.dz[id] = 1;
			stackgamma.seed[id] = seed;
			stackgamma.E[id] = E;
			stackgamma.px[id] = x;
			stackgamma.py[id] = y;
			stackgamma.pz[id] = 0.0f;
			stackgamma.live[id] = 1;
			stackgamma.in[id] = 1;
			stackgamma.ct_eff[id] = 0;
			stackgamma.ct_Cpt[id] = 0;
			stackgamma.ct_PE[id] = 0;
			stackgamma.ct_Ray[id] = 0;
		}
	}

}

__global__ void kernel_particle_pet(StackGamma stackgamma, int nb, int small_nb, int tiny_nb,
									int nz, int ny, int nx, float E, int fact) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id < stackgamma.size) {
		float x, y, z, rnd, rx, ry, rz, phi, theta, dx, dy, dz;
		int seed, j;
		int jump = nx * ny;
		if (stackgamma.in[id]==0 || stackgamma.live[id]==0) {
			seed = stackgamma.seed[id];

			//// Get position
			rx = park_miller_jb(&seed);
			ry = park_miller_jb(&seed);
			rz = park_miller_jb(&seed);
			rnd = park_miller_jb(&seed);
			// first binary search
			j = int(rnd * tiny_nb);
			if (tex1Dfetch(tex_tiny_vol, j) < rnd) {
				while (tex1Dfetch(tex_tiny_vol, j) < rnd) {++j;}
			} else {
				while (tex1Dfetch(tex_tiny_vol, j) > rnd) {--j;}
				++j; // correct unershoot
			}
			// second binary search
			j *= fact;
			j = fminf(j, small_nb-1);
			if (tex1Dfetch(tex_small_vol, j) < rnd) {
				while (tex1Dfetch(tex_small_vol, j) < rnd) {++j;}
			} else {
				while (tex1Dfetch(tex_small_vol, j) > rnd) {--j;}
				++j; // correct undershoot
			}
			// final binary search
			j *= fact;
			j = fminf(j, nb-1);
			if (tex1Dfetch(tex_vol, j) < rnd) {
				while (tex1Dfetch(tex_vol, j) < rnd) {++j;}
			} else {
				while (tex1Dfetch(tex_vol, j) < rnd) {--j;}
				++j; // correct undershoot
			}
			// get the position inside the volume
			j = tex1Dfetch(tex_ind, j);  // look-up-table
			z = j / jump;
			j -= (z * jump);
			y = j / nx;
			x = j - y*nx;
			// get the pos inside the voxel
			x += rx;
			y += ry;
			z += rz;

			//// Get direction
			phi = park_miller_jb(&seed);
			theta = park_miller_jb(&seed);
			phi   = twopi * phi;
			theta = acosf(1.0f - 2.0f*theta);
			// convert to cartesian
			dx = __cosf(phi)*__sinf(theta);
			dy = __sinf(phi)*__sinf(theta);
			dz = __cosf(theta);

			// Assignment
			stackgamma.dx[id] = dx;
			stackgamma.dy[id] = dy;
			stackgamma.dz[id] = dz;
			stackgamma.seed[id] = seed;
			stackgamma.E[id] = E;
			stackgamma.px[id] = rx;
			stackgamma.py[id] = ry;
			stackgamma.pz[id] = rz;
			stackgamma.live[id] = 1;
			stackgamma.in[id] = 1;
			stackgamma.ct_eff[id] = 0;
			stackgamma.ct_Cpt[id] = 0;
			stackgamma.ct_PE[id] = 0;
			stackgamma.ct_Ray[id] = 0;
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

	// debug
	//int j=0;
	
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

		//dtrack[id]=pq;
		//++j;
		
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
				pq = oldv;
				pq += -__fdividef(__logf(park_miller_jb(&seed)), att_from_mat(mat, E));
				oldmat = mat;
				//dtrack[i.z*jump + i.y*dimvol.x + i.x] += 2.0f;
				//++j;
				//dtrack[id]=1.0f/att_from_mat(mat, E);
			}

			totv += val;
			oldv = newv;
			if (run.x==newv) {i.x += stepi.x; run.x += stept.x;}
			if (run.y==newv) {i.y += stepi.y; run.y += stept.y;}
			if (run.z==newv) {i.z += stepi.z; run.z += stept.z;}
			inside = (i.x >= 0) & (i.x < dimvol.x) & (i.y >= 0) & (i.y < dimvol.y) & (i.z >= 0) & (i.z < dimvol.z);
			// debug
			dtrack[i.z*jump + i.y*dimvol.x + i.x] += 1.0f;
		}

		pe.x = p0.x + delta.x*oldv;
		pe.y = p0.y + delta.y*oldv;
		pe.z = p0.z + delta.z*oldv;
		stackgamma.seed[id] = seed;
		stackgamma.px[id] = pe.x;
		stackgamma.py[id] = pe.y;
		stackgamma.pz[id] = pe.z;

		// to debug
		//dtrack[int(pe.z)*jump + int(pe.y)*dimvol.x + int(pe.x)] += 1.0f;

		if (!inside) {stackgamma.in[id] = 0;}

	} // id < nx

}

// Fictitious tracking (or delta-tracking)
__global__ void kernel_woodcock(int3 dimvol, StackGamma stackgamma, float dimvox, float* dtrack) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int jump = dimvol.x*dimvol.y;
	float3 p0, delta;
	int3 vox;
	float path, rec_mu_maj, E, cur_att;
	int mat, seed;
	dimvox = __fdividef(1.0f, dimvox);
	
	// to debug
	//int i=0;
	//float x, y, z;
		
	if (id < stackgamma.size) {
		p0.x = stackgamma.px[id];
		p0.y = stackgamma.py[id];
		p0.z = stackgamma.pz[id];
		delta.x = stackgamma.dx[id];
		delta.y = stackgamma.dy[id];
		delta.z = stackgamma.dz[id];
		seed = stackgamma.seed[id];
		E = stackgamma.E[id];
		
		__shared__ float CS[256][15];
		CS[threadIdx.x][0] = 0.0f;
		CS[threadIdx.x][1] = 0.0f;
		CS[threadIdx.x][2] = 0.0f;
		CS[threadIdx.x][3] = 0.0f;
		CS[threadIdx.x][4] = 0.0f;
		CS[threadIdx.x][5] = 0.0f;
		CS[threadIdx.x][6] = 0.0f;
		CS[threadIdx.x][7] = 0.0f;
		CS[threadIdx.x][8] = 0.0f;
		CS[threadIdx.x][9] = 0.0f;
		CS[threadIdx.x][10] = 0.0f;
		CS[threadIdx.x][11] = 0.0f;
		CS[threadIdx.x][12] = 0.0f;
		CS[threadIdx.x][13] = 0.0f;
		CS[threadIdx.x][14] = 0.0f;
		
		//CS[threadIdx.x][15] = 0.0f; ERROR, overflow!!!!!

		// Most attenuate material is RibBone (ID=6)
		rec_mu_maj = __fdividef(1.0f, att_from_mat(0, E));
		
		while (1) {
			// get mean path from the most attenuate material (RibBone)
			path = -__logf(park_miller_jb(&seed)) * rec_mu_maj * dimvox;
			
			/*
			// To debug the flight of the particle
			for (i=0; i<path*10.0f; ++i) {
				x = p0.x + delta.x * (float)i / 10.0f;
				y = p0.y + delta.y * (float)i / 10.0f;
				z = p0.z + delta.z * (float)i / 10.0f;
				dtrack[int(z)*jump + int(y)*dimvol.x + int(x)] += 1.0f;
			}
			*/
			
			// flight along the path
			p0.x = p0.x + delta.x * path;
			p0.y = p0.y + delta.y * path;
			p0.z = p0.z + delta.z * path;

			vox.x = int(p0.x);
			vox.y = int(p0.y);
			vox.z = int(p0.z);

			// Still inside the volume?			
			if (vox.x < 0 || vox.y < 0 || vox.z < 0
				|| vox.x >= dimvol.x || vox.y >= dimvol.y || vox.z >= dimvol.z) {
				stackgamma.in[id] = 0;
				break;
			}
			
			// Does the interaction is real?
			mat = tex1Dfetch(tex_vol, vox.z*jump + vox.y*dimvol.x + vox.x);

			if (CS[threadIdx.x][mat] == 0) {
				cur_att = att_from_mat(mat, E);
				CS[threadIdx.x][mat] = cur_att;
			} else {
				cur_att = CS[threadIdx.x][mat];
			}
			
			if (cur_att * rec_mu_maj > park_miller_jb(&seed)) {break;}

		}
		//if (id < 100) {dtrack[id] = id;}
		//dtrack[int(vox.z)*jump + int(vox.y)*dimvol.x + int(vox.x)] += 1.0f;
		stackgamma.seed[id] = seed;
		stackgamma.px[id] = p0.x;
		stackgamma.py[id] = p0.y;
		stackgamma.pz[id] = p0.z;
		
	}

}


/***********************************************************
 * Utils
 ***********************************************************/
__global__ void kernel_CS_from_mat(float* dCS, float* dE, int nE, int mat) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	if (id < nE) {
		dCS[id] = att_from_mat(mat, dE[id]);
	}
}

__global__ void kernel_test(float* dtrack, int max) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	if (id < max) {
		dtrack[id] = id;
	}
}


int mc_disk_detector(float* x, int nx, float* y, int ny, float* E, int nE, float* resE, int nrE,
					 int rad, int posx, int posy) {
	int i, ix, iy, iu, iv, c;
	c = 0;
	for (i=0; i<nx; ++i) {
		ix = x[i];
		iy = y[i];
		iu = ix - posx;
		iv = iy - posy;
		if ((iu*iu + iv*iv) < rad*rad) {
			resE[c] = E[i];
			++c;
		}
	}
	return c;
}

void mc_proj_detector(float* im, int ny, int nx, float* x, int sx, float* y, int sy) {
	int i=0;
	while (i<sx) {
		im[int(y[i])*nx + int(x[i])] += 1.0f;
		++i;
	}
}

#define SWAPf(a, b) {float tmp=(a); (a)=(b); (b)=tmp;}
#define SWAPi(a, b) {int tmp=(a); (a)=(b); (b)=tmp;}
// Quick sort O(n(log n))
void mc_quicksort(float* vec, int* ind, int l, int r) {
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
			SWAPf(vec[i], vec[j]);
			SWAPi(ind[i], ind[j]);
		}
		SWAPf(vec[l], vec[j]);
		SWAPi(ind[l], ind[j]);
		mc_quicksort(vec, ind, l, j-1);
		mc_quicksort(vec, ind, j+1, r);
	}
}
#undef SWAPf
#undef SWAPi

void mc_prepare_activity(float* vol, int* ind,
						 float* data, int nb, float* small_data, int small_nb,
						 float* tiny_data, int tiny_nb, int fact) {
	int i, c, k;
	float tot, sum;
	// build index
	i = 0; while (i<nb) {ind[i] = i; ++i;}
	// compute sum
	i = 0; while (i<nb) {tot += vol[i]; ++i;}
	tot = 1.0f / tot;
	// normalize
	i = 0; while (i<nb) {data[i] = vol[i] * tot; ++i;}
	// quicksort
	mc_quicksort(data, ind, 0, nb-1);
	// build small data
	c = 0; k = 0; i = 0; sum = 0.0f;
	while (i<nb) {
		sum += data[i];
		++c;
		if (c == fact) {
			small_data[k] = sum;
			sum = 0.0f;
			c = 0;
			++k;
		}
		++i;
	}
	if (k != small_nb) {small_data[k] = sum;}
	// build tiny data
	c = 0; k = 0; i = 0; sum = 0.0f;
	while (i<small_nb) {
		sum += small_data[i];
		++c;
		if (c == fact) {
			tiny_data[k] = sum;
			sum = 0.0f;
			c = 0;
			++k;
		}
		++i;
	}
	if (k != tiny_nb) {tiny_data[k] = sum;}
	// cumul values
	i = 1; while (i<nb) {data[i] += data[i-1]; ++i;}
	i = 1; while (i<small_nb) {small_data[i] += small_data[i-1]; ++i;}
	i = 1; while (i<tiny_nb) {tiny_data[i] += tiny_data[i-1]; ++i;}
}

/***********************************************************
 * Main
 ***********************************************************/
void mc_cuda(float* vol, int nz, int ny, int nx,
			 float* E, int nE, float* dx, int ndx, float* dy, int ndy, float* dz, int ndz,
			 float* px, int npx, float* py, int npy, float* pz, int npz,
			 int nparticles, int totparticles, int maxit, int seed) {
	cudaSetDevice(1);

    timeval start, end;
    double t1, t2, diff;
	timeval start_s, end_s;
	double ts1, ts2;
	int3 dimvol;
	int n, step;
	int countparticle=0;

	// time to init
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	
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

	/*
	// prepare volume activity
	int fact = 50;
	int nb = nx*ny*nz;
	int small_nb = int(nb / fact);
	if (nb % fact == 1) {++small_nb;}
	int tiny_nb = int(small_nb / fact);
	if (small_nb % fact == 1) {++tiny_nb;}
	float* data = (float*)malloc(nb * sizeof(float));
	float* small_data = (float*)malloc(small_nb * sizeof(float));
	float* tiny_data = (float*)malloc(tiny_nb * sizeof(float));
	int* ind = (int*)malloc(nb * sizeof(int));
	mc_prepare_activity(vol, ind, data, nb, small_data, small_nb, tiny_data, tiny_nb, fact);
	float* dvol;
	unsigned int mem_vol = nb * sizeof(float);
	cudaMalloc((void**) &dvol, mem_vol);
	cudaMemcpy(dvol, data, mem_vol, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_vol, dvol, mem_vol);
	float* small_dvol;
	unsigned int mem_small_vol = small_nb * sizeof(float);
	cudaMalloc((void**) &small_dvol, mem_small_vol);
	cudaMemcpy(small_dvol, small_data, mem_small_vol, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_small_vol, small_dvol, mem_small_vol);
	float* tiny_dvol;
	unsigned int mem_tiny_vol = tiny_nb * sizeof(float);
	cudaMalloc((void**) &tiny_dvol, mem_tiny_vol);
	cudaMemcpy(tiny_dvol, tiny_data, mem_tiny_vol, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_tiny_vol, tiny_dvol, mem_tiny_vol);
	int* dind;
	unsigned int mem_dind = nb * sizeof(int);
	cudaMalloc((void**) &dind, mem_dind);
	cudaMemcpy(dind, ind, mem_dind, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_ind, dind, mem_dind);
	*/

	// Load Rayleigh cross section to the texture mem
	FILE *pfile;
	const int ncs = 213816;  // CS file contains 213,816 floats
	const int nff = 28824;   // FF file contains  28,824 floats
	unsigned int mem_cs = ncs * sizeof(float);
	unsigned int mem_ff = nff * sizeof(float);
	float* raylcs = (float*)malloc(mem_cs);
	float* raylff = (float*)malloc(mem_ff);
	pfile = fopen("rayleigh_cs.bin", "rb");
	fread(raylcs, sizeof(float), ncs, pfile);
	fclose(pfile);
	pfile = fopen("rayleigh_ff.bin", "rb");
	fread(raylff, sizeof(float), nff, pfile);
	fclose(pfile);

	float* draylcs;
	cudaMalloc((void**) &draylcs, mem_cs);
	cudaMemcpy(draylcs, raylcs, mem_cs, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_rayl_cs, draylcs, mem_cs);
	free(raylcs);
	float* draylff;	
	cudaMalloc((void**) &draylff, mem_ff);
	cudaMemcpy(draylff, raylff, mem_ff, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_rayl_ff, draylff, mem_ff);
	free(raylff);
	
	// Defined Stacks
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
	collector.ct_Ray = (int*)malloc(mem_stack_int);

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
	cudaMalloc((void**) &stackgamma.ct_Ray, mem_stack_int);
	cudaMemset(stackgamma.live, 0, mem_stack_char); // at beginning all particles are dead
	cudaMemset(stackgamma.in, 0, mem_stack_char);   // and outside the volume
	
	// Init seeds
	int* tmp = (int*)malloc(stackgamma.size * sizeof(int));
	srand(seed);
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

	// Time to init
	gettimeofday(&end, NULL);
	t2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = t2 - t1;
	printf("Init GPU %f s\n", diff);

	// Outter loop
	gettimeofday(&start_s, NULL);
	ts1 = start_s.tv_sec + start_s.tv_usec / 1000000.0;
	step = 0;
	while (step < maxit) {
		printf("Step %i\n", step);
		// Init particles
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		//kernel_particle_largegun<<<grid, threads>>>(stackgamma, dimvol, 22.0, 22.0, 0.0, 0.0, 0.0, 1.0, 0.025, 5);
		kernel_particle_isotrope<<<grid, threads>>>(stackgamma, 5.0, 5.0, 5.0, 0.511, dtrack);
		//kernel_particle_area<<<grid, threads>>>(stackgamma, 0, 127, 0, 45, 0.080);
		cudaThreadSynchronize();
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("   Create gamma particles %f s\n", diff);
	
		// Propagation
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		//kernel_siddon<<<grid, threads>>>(dimvol, stackgamma, dtrack, 4.0); // 4.0 mm3 voxel
		kernel_woodcock<<<grid, threads>>>(dimvol, stackgamma, 4.0, dtrack);
		//kernel_test<<<grid, threads>>>(dtrack, nparticles);
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
		cudaMemcpy(collector.ct_Ray, stackgamma.ct_Ray, mem_stack_int, cudaMemcpyDeviceToHost);
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
		int c5=0;
		n=0;
		while(n<nparticles && countparticle<totparticles) {
			if (collector.in[n] == 0) {
				//&& collector.px[n] > 0 && collector.py[n] > 0
				//&& collector.px[n] < nx && collector.py[n] < ny
				//&& collector.pz[n] > 440.0f && collector.pz[n] < 550.0f) {
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
			c5 += collector.ct_Ray[n];
			++n;
		}
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		
		printf("   Store gamma particles %f s\n", diff);
		printf("   Nb particles saves %i/%i absorbed %i\n", countparticle, totparticles, c1);
		printf("   Tot interaction %i: %i Compton %i Photo-Electric %i Rayleigh\n", c2, c3, c4, c5);

		// loop control
		if (countparticle >= totparticles) {break;}
		++step;

	} // outter loop (step)

	gettimeofday(&end_s, NULL);
	ts2 = end_s.tv_sec + end_s.tv_usec / 1000000.0;
	diff = ts2 - ts1;
	printf("Step running time %f s\n", diff);
	
	//cudaMemcpy(tmp, stackgamma.seed, mem_stack_int, cudaMemcpyDeviceToHost);
	cudaMemcpy(vol, dtrack, mem_vol, cudaMemcpyDeviceToHost);
	//cudaMemcpy(vol, ddose, mem_vol, cudaMemcpyDeviceToHost);

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
	free(collector.ct_Ray);
	
	cudaUnbindTexture(tex_vol);
	cudaUnbindTexture(tex_rayl_cs);
	cudaUnbindTexture(tex_rayl_ff);
	cudaFree(draylcs);
	cudaFree(draylff);
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
	cudaFree(stackgamma.ct_Ray);
	cudaThreadExit();

}

void mc_get_cs_cuda(float* CS, int nncs, float* E, int nE, int mat) {
	cudaSetDevice(1);

	// Load Rayleigh cross section to the texture mem
	FILE *pfile;
	const int ncs = 213816;  // CS file contains 213,816 floats
	const int nff = 28824;   // FF file contains  28,824 floats
	unsigned int mem_cs = ncs * sizeof(float);
	unsigned int mem_ff = nff * sizeof(float);
	float* raylcs = (float*)malloc(mem_cs);
	float* raylff = (float*)malloc(mem_ff);
	pfile = fopen("rayleigh_cs.bin", "rb");
	fread(raylcs, sizeof(float), ncs, pfile);
	fclose(pfile);
	pfile = fopen("rayleigh_ff.bin", "rb");
	fread(raylff, sizeof(float), nff, pfile);
	fclose(pfile);
	float* draylcs;
	cudaMalloc((void**) &draylcs, mem_cs);
	cudaMemcpy(draylcs, raylcs, mem_cs, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_rayl_cs, draylcs, mem_cs);
	free(raylcs);
	float* draylff;	
	cudaMalloc((void**) &draylff, mem_ff);
	cudaMemcpy(draylff, raylff, mem_ff, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_rayl_ff, draylff, mem_ff);
	free(raylff);
	
	unsigned int mem_size = nE * sizeof(float);
	float* dE;
	cudaMalloc((void**) &dE, mem_size);
	cudaMemcpy(dE, E, mem_size, cudaMemcpyHostToDevice);
	float* dCS;
	cudaMalloc((void**) &dCS, mem_size);
	cudaMemset(dCS, 0, mem_size);

	dim3 threads, grid;
	int block_size = 256;
	int grid_size = (nE + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	kernel_CS_from_mat<<<grid, threads>>>(dCS, dE, nE, mat);
	
	cudaMemcpy(CS, dCS, mem_size, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(tex_vol);
	cudaUnbindTexture(tex_rayl_cs);
	cudaUnbindTexture(tex_rayl_ff);
	cudaFree(draylcs);
	cudaFree(draylff);
	cudaFree(dCS);
	cudaFree(dE);
	cudaThreadExit();
}
