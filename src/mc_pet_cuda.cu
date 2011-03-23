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

#include "mc_pet_cuda.h"
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
texture<unsigned short int, 1, cudaReadModeElementType> tex_phantom;
texture<float, 1, cudaReadModeElementType> tex_huge_act;
texture<float, 1, cudaReadModeElementType> tex_small_act;
texture<float, 1, cudaReadModeElementType> tex_tiny_act;
texture<int, 1, cudaReadModeElementType> tex_ind;
texture<float, 1, cudaReadModeElementType> tex_rayl_cs;
texture<float, 1, cudaReadModeElementType> tex_rayl_ff;

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
	unsigned char* interaction;
	unsigned char* live;
	unsigned char* in;
	unsigned int size;
}; //

// Park-Miller from C numerical book
__device__ float park_miller(int *seed) {
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
		if (a2 > park_miller(&seed)) {
			eps = __expf(-a1 * park_miller(&seed));
			eps2 = eps*eps;
		} else {
			eps2 = eps02 + (1.0f - eps02) * park_miller(&seed);
			eps = sqrt(eps2);
		}
		onecost = __fdividef(1.0f - eps, eps * E0);
		greject = 1.0f - eps * onecost * __fdividef(2.0f - onecost, 1.0f + eps2);
	} while (greject < park_miller(&seed));

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
	
	int start = fCumulIntervals[Z-1];
	int stop = start + fNbOfIntervals[Z];
	int pos=stop;
	while (E < fSandiaTable[pos][0]*1.0e-3f){--pos;}
	float AoverAvo = 0.0103642688246f * __fdividef((float)Z, fZtoAratio[Z]);
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
				costheta = 2.0f * park_miller(&seed) - 1.0f;
			} while (((1.0f + costheta*costheta)*0.5f) < park_miller(&seed));
		}
		sinthetahalf = sqrt((1.0f - costheta) * 0.5f);
		E = __fdividef(sinthetahalf, wphot * 0.1f);
		FF = (E > 1.0e5f)? Rayleigh_FF(E, Z) : Rayleigh_FF(0.0f, Z);
		// reuse costheta as sintheta
		costheta = sqrt(1.0f - costheta*costheta);
	} while (FF*FF < (park_miller(&seed) * Z*Z));

	return asin(costheta);
}

/***********************************************************
 * Materials
 ***********************************************************/
__device__  float Compton_mu_Water(float E) {
	// H2O
	return (2*Compton_CSPA(E, 1) + Compton_CSPA(E, 8)) * 3.342796664e+19f; // Avogadro*H2O_density / (2*a_H+a_O)
}
__device__  float PhotoElec_mu_Water(float E) {
	// H2O
	return (2*PhotoElec_CSPA(E, 1) + PhotoElec_CSPA(E, 8)) * 3.342796664e+19f; // Avogadro*H2O_density / (2*a_H+a_O)
}
__device__  float Rayleigh_mu_Water(float E) {
	// H2O
	return (2*Rayleigh_CSPA(E, 1) + Rayleigh_CSPA(E, 8)) * 3.342796664e+19f; // Avogadro*H2O_density / (2*a_H+a_O)
}

__device__  float Compton_mu_Plastic(float E) {
	// 5C8H2O
	return (5*Compton_CSPA(E, 6) + 8*Compton_CSPA(E, 1) + 2*Compton_CSPA(E, 8)) * 7.096901340e17f;
}
__device__  float PhotoElec_mu_Plastic(float E) {
	// 5C8H2O
	return (5*PhotoElec_CSPA(E, 6) + 8*PhotoElec_CSPA(E, 1) + 2*PhotoElec_CSPA(E, 8)) * 7.096901340e17f;
}
__device__  float Rayleigh_mu_Plastic(float E) {
	// 5C8H2O
	return (5*Rayleigh_CSPA(E, 6) + 8*Rayleigh_CSPA(E, 1) + 2*Rayleigh_CSPA(E, 8)) * 7.096901340e17f;
}

__device__  float Compton_mu_Al(float E) {
	// Al
	return Compton_CSPA(E, 13) * 6.024030465e+19f; // Avogadro*Al_density/a_Al
}
__device__  float PhotoElec_mu_Al(float E) {
	// Al
	return PhotoElec_CSPA(E, 13) * 6.024030465e+19f; // Avogadro*Al_density/a_Al
}
__device__  float Rayleigh_mu_Al(float E) {
	// Al
	return Rayleigh_CSPA(E, 13) * 6.024030465e+19f; // Avogadro*Al_density/a_Al
}

__device__  float Compton_mu_Air(float E) {
	// O N Ar C
	return (0.231781f*Compton_CSPA(E, 8) + 0.755268f*Compton_CSPA(E, 7)
			+ 0.012827f*Compton_CSPA(E, 18) + 0.000124f*Compton_CSPA(E, 6)) * 5.247706935e17f;
}
__device__  float PhotoElec_mu_Air(float E) {
	// O N Ar C
	return (0.231781f*PhotoElec_CSPA(E, 8) + 0.755268f*PhotoElec_CSPA(E, 7)
			+ 0.012827f*PhotoElec_CSPA(E, 18) + 0.000124f*PhotoElec_CSPA(E, 6)) * 5.247706935e17f;
}
__device__  float Rayleigh_mu_Air(float E) {
	// O N Ar C
	return (0.231781f*Rayleigh_CSPA(E, 8) + 0.755268f*Rayleigh_CSPA(E, 7)
			+ 0.012827f*Rayleigh_CSPA(E, 18) + 0.000124f*Rayleigh_CSPA(E, 6)) * 5.247706935e17f;
}

__device__  float Compton_mu_Body(float E) {
	// H O
	return (0.112f*Compton_CSPA(E, 1) + 0.888f*Compton_CSPA(E, 8)) * 4.205077389e18f;
}
__device__  float PhotoElec_mu_Body(float E) {
	// H O
	return (0.112f*PhotoElec_CSPA(E, 1) + 0.888f*PhotoElec_CSPA(E, 8)) * 4.205077389e18f;
}
__device__  float Rayleigh_mu_Body(float E) {
	// H O
	return (0.112f*Rayleigh_CSPA(E, 1) + 0.888f*Rayleigh_CSPA(E, 8)) * 4.205077389e18f;
}

__device__  float Compton_mu_Lung(float E) {
	// H C N O Na P S Cl K
	return (0.103f*Compton_CSPA(E, 1)+ 0.105f*Compton_CSPA(E, 6) + 0.031f*Compton_CSPA(E, 7)
			+ 0.749f*Compton_CSPA(E, 8) + 0.002f*Compton_CSPA(E, 11) + 0.002f*Compton_CSPA(E, 15)
			+ 0.003f*Compton_CSPA(E, 16) + 0.003f*Compton_CSPA(E, 17) + 0.002f*Compton_CSPA(E, 19)) * 1.232299227e18f;
}
__device__  float PhotoElec_mu_Lung(float E) {
	// H C N O Na P S Cl K
	return (0.103f*PhotoElec_CSPA(E, 1)+ 0.105f*PhotoElec_CSPA(E, 6) + 0.031f*PhotoElec_CSPA(E, 7)
			+ 0.749f*PhotoElec_CSPA(E, 8) + 0.002f*PhotoElec_CSPA(E, 11) + 0.002f*PhotoElec_CSPA(E, 15)
			+ 0.003f*PhotoElec_CSPA(E, 16) + 0.003f*PhotoElec_CSPA(E, 17) + 0.002f*PhotoElec_CSPA(E, 19)) * 1.232299227e18f;
}
__device__  float Rayleigh_mu_Lung(float E) {
	// H C N O Na P S Cl K
	return (0.103f*Rayleigh_CSPA(E, 1)+ 0.105f*Rayleigh_CSPA(E, 6) + 0.031f*Rayleigh_CSPA(E, 7)
			+ 0.749f*Rayleigh_CSPA(E, 8) + 0.002f*Rayleigh_CSPA(E, 11) + 0.002f*Rayleigh_CSPA(E, 15)
			+ 0.003f*Rayleigh_CSPA(E, 16) + 0.003f*Rayleigh_CSPA(E, 17) + 0.002f*Rayleigh_CSPA(E, 19)) * 1.232299227e18f;
}

__device__  float Compton_mu_RibBone(float E) {
	// H C N O Na Mg P S Ca
	return (0.034f*Compton_CSPA(E, 1) + 0.155f*Compton_CSPA(E, 6) + 0.042f*Compton_CSPA(E, 7)
			+ 0.435f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.002f*Compton_CSPA(E, 12)
			+ 0.103f*Compton_CSPA(E, 15) + 0.003f*Compton_CSPA(E, 16) + 0.225f*Compton_CSPA(E, 20)) * 5.299038816e18f;
}
__device__  float PhotoElec_mu_RibBone(float E) {
	// H C N O Na Mg P S Ca
	return (0.034f*PhotoElec_CSPA(E, 1) + 0.155f*PhotoElec_CSPA(E, 6) + 0.042f*PhotoElec_CSPA(E, 7)
			+ 0.435f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.002f*PhotoElec_CSPA(E, 12)
			+ 0.103f*PhotoElec_CSPA(E, 15) + 0.003f*PhotoElec_CSPA(E, 16) + 0.225f*PhotoElec_CSPA(E, 20)) * 5.299038816e18f;
}
__device__  float Rayleigh_mu_RibBone(float E) {
	// H C N O Na Mg P S Ca
	return (0.034f*Rayleigh_CSPA(E, 1) + 0.155f*Rayleigh_CSPA(E, 6) + 0.042f*Rayleigh_CSPA(E, 7)
			+ 0.435f*Rayleigh_CSPA(E, 8) + 0.001f*Rayleigh_CSPA(E, 11) + 0.002f*Rayleigh_CSPA(E, 12)
			+ 0.103f*Rayleigh_CSPA(E, 15) + 0.003f*Rayleigh_CSPA(E, 16) + 0.225f*Rayleigh_CSPA(E, 20)) * 5.299038816e18f;
}

__device__  float Compton_mu_SpineBone(float E) {
	// H C N O Na Mg P S Cl K Ca
	return (0.063f*Compton_CSPA(E, 1) + 0.261f*Compton_CSPA(E, 6) + 0.039f*Compton_CSPA(E, 7)
			+ 0.436f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.001f*Compton_CSPA(E, 12)
			+ 0.061f*Compton_CSPA(E, 15) + 0.003f*Compton_CSPA(E, 16) + 0.001f*Compton_CSPA(E, 17)
			+ 0.001f*Compton_CSPA(E, 19) + 0.133f*Compton_CSPA(E, 20)) * 4.709337384e18f;
}
__device__  float PhotoElec_mu_SpineBone(float E) {
	// H C N O Na Mg P S Cl K Ca
	return (0.063f*PhotoElec_CSPA(E, 1) + 0.261f*PhotoElec_CSPA(E, 6) + 0.039f*PhotoElec_CSPA(E, 7)
			+ 0.436f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.001f*PhotoElec_CSPA(E, 12)
			+ 0.061f*PhotoElec_CSPA(E, 15) + 0.003f*PhotoElec_CSPA(E, 16) + 0.001f*PhotoElec_CSPA(E, 17)
			+ 0.001f*PhotoElec_CSPA(E, 19) + 0.133f*PhotoElec_CSPA(E, 20)) * 4.709337384e18f;
}
__device__  float Rayleigh_mu_SpineBone(float E) {
	// H C N O Na Mg P S Cl K Ca
	return (0.063f*Rayleigh_CSPA(E, 1) + 0.261f*Rayleigh_CSPA(E, 6) + 0.039f*Rayleigh_CSPA(E, 7)
			+ 0.436f*Rayleigh_CSPA(E, 8) + 0.001f*Rayleigh_CSPA(E, 11) + 0.001f*Rayleigh_CSPA(E, 12)
			+ 0.061f*Rayleigh_CSPA(E, 15) + 0.003f*Rayleigh_CSPA(E, 16) + 0.001f*Rayleigh_CSPA(E, 17)
			+ 0.001f*Rayleigh_CSPA(E, 19) + 0.133f*Rayleigh_CSPA(E, 20)) * 4.709337384e18f;
}

__device__  float Compton_mu_Heart(float E) {
	// H C N O Na P S Cl K
	return (0.104f*Compton_CSPA(E, 1) + 0.139f*Compton_CSPA(E, 6) + 0.029f*Compton_CSPA(E, 7)
			+ 0.718f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.002f*Compton_CSPA(E, 15)
			+ 0.002f*Compton_CSPA(E, 16) + 0.002f*Compton_CSPA(E, 17) + 0.003f*Compton_CSPA(E, 19)) * 4.514679219e18f;
}
__device__  float PhotoElec_mu_Heart(float E) {
	// H C N O Na P S Cl K
	return (0.104f*PhotoElec_CSPA(E, 1) + 0.139f*PhotoElec_CSPA(E, 6) + 0.029f*PhotoElec_CSPA(E, 7)
			+ 0.718f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.002f*PhotoElec_CSPA(E, 15)
			+ 0.002f*PhotoElec_CSPA(E, 16) + 0.002f*PhotoElec_CSPA(E, 17) + 0.003f*PhotoElec_CSPA(E, 19)) * 4.514679219e18f;
}
__device__  float Rayleigh_mu_Heart(float E) {
	// H C N O Na P S Cl K
	return (0.104f*Rayleigh_CSPA(E, 1) + 0.139f*Rayleigh_CSPA(E, 6) + 0.029f*Rayleigh_CSPA(E, 7)
			+ 0.718f*Rayleigh_CSPA(E, 8) + 0.001f*Rayleigh_CSPA(E, 11) + 0.002f*Rayleigh_CSPA(E, 15)
			+ 0.002f*Rayleigh_CSPA(E, 16) + 0.002f*Rayleigh_CSPA(E, 17) + 0.003f*Rayleigh_CSPA(E, 19)) * 4.514679219e18f;
}

__device__  float Compton_mu_Breast(float E) {
	// H C N O Na P S Cl
	return (0.106f*Compton_CSPA(E, 1) + 0.332f*Compton_CSPA(E, 6) + 0.03f*Compton_CSPA(E, 7)
			+ 0.527f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.001f*Compton_CSPA(E, 15)
			+ 0.002f*Compton_CSPA(E, 16) + 0.001f*Compton_CSPA(E, 17)) * 4.688916436e18f;
}
__device__  float PhotoElec_mu_Breast(float E) {
	// H C N O Na P S Cl
	return (0.106f*PhotoElec_CSPA(E, 1) + 0.332f*PhotoElec_CSPA(E, 6) + 0.03f*PhotoElec_CSPA(E, 7)
			+ 0.527f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.001f*PhotoElec_CSPA(E, 15)
			+ 0.002f*PhotoElec_CSPA(E, 16) + 0.001f*PhotoElec_CSPA(E, 17)) * 4.688916436e18f;
}
__device__  float Rayleigh_mu_Breast(float E) {
	// H C N O Na P S Cl
	return (0.106f*Rayleigh_CSPA(E, 1) + 0.332f*Rayleigh_CSPA(E, 6) + 0.03f*Rayleigh_CSPA(E, 7)
			+ 0.527f*Rayleigh_CSPA(E, 8) + 0.001f*Rayleigh_CSPA(E, 11) + 0.001f*Rayleigh_CSPA(E, 15)
			+ 0.002f*Rayleigh_CSPA(E, 16) + 0.001f*Rayleigh_CSPA(E, 17)) * 4.688916436e18f;
}

__device__  float Compton_mu_Intestine(float E) {
	// H C N O Na P S Cl K
	return (0.106f*Compton_CSPA(E, 1) + 0.115f*Compton_CSPA(E, 6) + 0.022f*Compton_CSPA(E, 7)
			+ 0.751f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.001f*Compton_CSPA(E, 15)
			+ 0.001f*Compton_CSPA(E, 16) + 0.002f*Compton_CSPA(E, 17) + 0.001f*Compton_CSPA(E, 19)) * 4.427901925e18f;	
}
__device__  float PhotoElec_mu_Intestine(float E) {
	// H C N O Na P S Cl K
	return (0.106f*PhotoElec_CSPA(E, 1) + 0.115f*PhotoElec_CSPA(E, 6) + 0.022f*PhotoElec_CSPA(E, 7)
			+ 0.751f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.001f*PhotoElec_CSPA(E, 15)
			+ 0.001f*PhotoElec_CSPA(E, 16) + 0.002f*PhotoElec_CSPA(E, 17) + 0.001f*PhotoElec_CSPA(E, 19)) * 4.427901925e18f;	
}
__device__  float Rayleigh_mu_Intestine(float E) {
	// H C N O Na P S Cl K
	return (0.106f*Rayleigh_CSPA(E, 1) + 0.115f*Rayleigh_CSPA(E, 6) + 0.022f*Rayleigh_CSPA(E, 7)
			+ 0.751f*Rayleigh_CSPA(E, 8) + 0.001f*Rayleigh_CSPA(E, 11) + 0.001f*Rayleigh_CSPA(E, 15)
			+ 0.001f*Rayleigh_CSPA(E, 16) + 0.002f*Rayleigh_CSPA(E, 17) + 0.001f*Rayleigh_CSPA(E, 19)) * 4.427901925e18f;	
}

__device__  float Compton_mu_Spleen(float E) {
	// H C N O Na P S Cl K
	return (0.103f*Compton_CSPA(E, 1) + 0.113f*Compton_CSPA(E, 6) + 0.032f*Compton_CSPA(E, 7)
			+ 0.741f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.003f*Compton_CSPA(E, 15)
			+ 0.002f*Compton_CSPA(E, 16) + 0.002f*Compton_CSPA(E, 17) + 0.003f*Compton_CSPA(E, 19)) * 4.516487252e18f;	
}
__device__  float PhotoElec_mu_Spleen(float E) {
	// H C N O Na P S Cl K
	return (0.103f*PhotoElec_CSPA(E, 1) + 0.113f*PhotoElec_CSPA(E, 6) + 0.032f*PhotoElec_CSPA(E, 7)
			+ 0.741f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.003f*PhotoElec_CSPA(E, 15)
			+ 0.002f*PhotoElec_CSPA(E, 16) + 0.002f*PhotoElec_CSPA(E, 17) + 0.003f*PhotoElec_CSPA(E, 19)) * 4.516487252e18f;	
}
__device__  float Rayleigh_mu_Spleen(float E) {
	// H C N O Na P S Cl K
	return (0.103f*Rayleigh_CSPA(E, 1) + 0.113f*Rayleigh_CSPA(E, 6) + 0.032f*Rayleigh_CSPA(E, 7)
			+ 0.741f*Rayleigh_CSPA(E, 8) + 0.001f*Rayleigh_CSPA(E, 11) + 0.003f*Rayleigh_CSPA(E, 15)
			+ 0.002f*Rayleigh_CSPA(E, 16) + 0.002f*Rayleigh_CSPA(E, 17) + 0.003f*Rayleigh_CSPA(E, 19)) * 4.516487252e18f;	
}

__device__  float Compton_mu_Blood(float E) {
	// H C N O Na P S Cl K Fe
	return (0.102f*Compton_CSPA(E, 1) + 0.11f*Compton_CSPA(E, 6) + 0.033f*Compton_CSPA(E, 7)
			+ 0.745f*Compton_CSPA(E, 8) + 0.001f*Compton_CSPA(E, 11) + 0.001f*Compton_CSPA(E, 15)
			+ 0.002f*Compton_CSPA(E, 16) + 0.003f*Compton_CSPA(E, 17) + 0.002f*Compton_CSPA(E, 19)
			+ 0.001f*Compton_CSPA(E, 26)) * 4.506530526e18f;	
}
__device__  float PhotoElec_mu_Blood(float E) {
	// H C N O Na P S Cl K Fe
	return (0.102f*PhotoElec_CSPA(E, 1) + 0.11f*PhotoElec_CSPA(E, 6) + 0.033f*PhotoElec_CSPA(E, 7)
			+ 0.745f*PhotoElec_CSPA(E, 8) + 0.001f*PhotoElec_CSPA(E, 11) + 0.001f*PhotoElec_CSPA(E, 15)
			+ 0.002f*PhotoElec_CSPA(E, 16) + 0.003f*PhotoElec_CSPA(E, 17) + 0.002f*PhotoElec_CSPA(E, 19)
			+ 0.001f*PhotoElec_CSPA(E, 26)) * 4.506530526e18f;	
}
__device__  float Rayleigh_mu_Blood(float E) {
	// H C N O Na P S Cl K Fe
	return (0.102f*Rayleigh_CSPA(E, 1) + 0.11f*Rayleigh_CSPA(E, 6) + 0.033f*Rayleigh_CSPA(E, 7)
			+ 0.745f*Rayleigh_CSPA(E, 8) + 0.001f*Rayleigh_CSPA(E, 11) + 0.001f*Rayleigh_CSPA(E, 15)
			+ 0.002f*Rayleigh_CSPA(E, 16) + 0.003f*Rayleigh_CSPA(E, 17) + 0.002f*Rayleigh_CSPA(E, 19)
			+ 0.001f*Rayleigh_CSPA(E, 26)) * 4.506530526e18f;	
}

__device__  float Compton_mu_Liver(float E) {
	// H C N O Na P S Cl K
	return (0.102f*Compton_CSPA(E, 1) + 0.139f*Compton_CSPA(E, 6) + 0.03f*Compton_CSPA(E, 7)
			+ 0.716f*Compton_CSPA(E, 8) + 0.002f*Compton_CSPA(E, 11) + 0.003f*Compton_CSPA(E, 15)
			+ 0.003f*Compton_CSPA(E, 16) + 0.002f*Compton_CSPA(E, 17) + 0.003f*Compton_CSPA(E, 19)) * 4.536294717e18f;	
}
__device__  float PhotoElec_mu_Liver(float E) {
	// H C N O Na P S Cl K
	return (0.102f*PhotoElec_CSPA(E, 1) + 0.139f*PhotoElec_CSPA(E, 6) + 0.03f*PhotoElec_CSPA(E, 7)
			+ 0.716f*PhotoElec_CSPA(E, 8) + 0.002f*PhotoElec_CSPA(E, 11) + 0.003f*PhotoElec_CSPA(E, 15)
			+ 0.003f*PhotoElec_CSPA(E, 16) + 0.002f*PhotoElec_CSPA(E, 17) + 0.003f*PhotoElec_CSPA(E, 19)) * 4.536294717e18f;	
}
__device__  float Rayleigh_mu_Liver(float E) {
	// H C N O Na P S Cl K
	return (0.102f*Rayleigh_CSPA(E, 1) + 0.139f*Rayleigh_CSPA(E, 6) + 0.03f*Rayleigh_CSPA(E, 7)
			+ 0.716f*Rayleigh_CSPA(E, 8) + 0.002f*Rayleigh_CSPA(E, 11) + 0.003f*Rayleigh_CSPA(E, 15)
			+ 0.003f*Rayleigh_CSPA(E, 16) + 0.002f*Rayleigh_CSPA(E, 17) + 0.003f*Rayleigh_CSPA(E, 19)) * 4.536294717e18f;	
}

__device__  float Compton_mu_Kidney(float E) {
	// H C N O Na P S Cl K Ca
	return (0.103f*Compton_CSPA(E, 1) + 0.132f*Compton_CSPA(E, 6) + 0.03f*Compton_CSPA(E, 7)
			+ 0.724f*Compton_CSPA(E, 8) + 0.002f*Compton_CSPA(E, 11) + 0.002f*Compton_CSPA(E, 15)
			+ 0.002f*Compton_CSPA(E, 16) + 0.002f*Compton_CSPA(E, 17) + 0.002f*Compton_CSPA(E, 19)
			+ 0.001f*Compton_CSPA(E, 20)) * 4.498971018e18f;	
}
__device__  float PhotoElec_mu_Kidney(float E) {
	// H C N O Na P S Cl K Ca
	return (0.103f*PhotoElec_CSPA(E, 1) + 0.132f*PhotoElec_CSPA(E, 6) + 0.03f*PhotoElec_CSPA(E, 7)
			+ 0.724f*PhotoElec_CSPA(E, 8) + 0.002f*PhotoElec_CSPA(E, 11) + 0.002f*PhotoElec_CSPA(E, 15)
			+ 0.002f*PhotoElec_CSPA(E, 16) + 0.002f*PhotoElec_CSPA(E, 17) + 0.002f*PhotoElec_CSPA(E, 19)
			+ 0.001f*PhotoElec_CSPA(E, 20)) * 4.498971018e18f;	
}
__device__  float Rayleigh_mu_Kidney(float E) {
	// H C N O Na P S Cl K Ca
	return (0.103f*Rayleigh_CSPA(E, 1) + 0.132f*Rayleigh_CSPA(E, 6) + 0.03f*Rayleigh_CSPA(E, 7)
			+ 0.724f*Rayleigh_CSPA(E, 8) + 0.002f*Rayleigh_CSPA(E, 11) + 0.002f*Rayleigh_CSPA(E, 15)
			+ 0.002f*Rayleigh_CSPA(E, 16) + 0.002f*Rayleigh_CSPA(E, 17) + 0.002f*Rayleigh_CSPA(E, 19)
			+ 0.001f*Rayleigh_CSPA(E, 20)) * 4.498971018e18f;	
}

__device__  float Compton_mu_Brain(float E) {
	// H C N O Na P S Cl K
	return (0.107f*Compton_CSPA(E, 1) + 0.145f*Compton_CSPA(E, 6) + 0.022f*Compton_CSPA(E, 7)
			+ 0.712f*Compton_CSPA(E, 8) + 0.002f*Compton_CSPA(E, 11) + 0.004f*Compton_CSPA(E, 15)
			+ 0.002f*Compton_CSPA(E, 16) + 0.003f*Compton_CSPA(E, 17) + 0.003f*Compton_CSPA(E, 19)) * 4.471235341e18f;	
}
__device__  float PhotoElec_mu_Brain(float E) {
	// H C N O Na P S Cl K
	return (0.107f*PhotoElec_CSPA(E, 1) + 0.145f*PhotoElec_CSPA(E, 6) + 0.022f*PhotoElec_CSPA(E, 7)
			+ 0.712f*PhotoElec_CSPA(E, 8) + 0.002f*PhotoElec_CSPA(E, 11) + 0.004f*PhotoElec_CSPA(E, 15)
			+ 0.002f*PhotoElec_CSPA(E, 16) + 0.003f*PhotoElec_CSPA(E, 17) + 0.003f*PhotoElec_CSPA(E, 19)) * 4.471235341e18f;	
}
__device__  float Rayleigh_mu_Brain(float E) {
	// H C N O Na P S Cl K
	return (0.107f*Rayleigh_CSPA(E, 1) + 0.145f*Rayleigh_CSPA(E, 6) + 0.022f*Rayleigh_CSPA(E, 7)
			+ 0.712f*Rayleigh_CSPA(E, 8) + 0.002f*Rayleigh_CSPA(E, 11) + 0.004f*Rayleigh_CSPA(E, 15)
			+ 0.002f*Rayleigh_CSPA(E, 16) + 0.003f*Rayleigh_CSPA(E, 17) + 0.003f*Rayleigh_CSPA(E, 19)) * 4.471235341e18f;	
}

__device__  float Compton_mu_Pancreas(float E) {
	// H C N O Na P S Cl K
	return (0.106f*Compton_CSPA(E, 1) + 0.169f*Compton_CSPA(E, 6) + 0.022f*Compton_CSPA(E, 7)
			+ 0.694f*Compton_CSPA(E, 8) + 0.002f*Compton_CSPA(E, 11) + 0.002f*Compton_CSPA(E, 15)
			+ 0.001f*Compton_CSPA(E, 16) + 0.002f*Compton_CSPA(E, 17) + 0.002f*Compton_CSPA(E, 19)) * 4.525945892e18f;	
}
__device__  float PhotoElec_mu_Pancreas(float E) {
	// H C N O Na P S Cl K
	return (0.106f*PhotoElec_CSPA(E, 1) + 0.169f*PhotoElec_CSPA(E, 6) + 0.022f*PhotoElec_CSPA(E, 7)
			+ 0.694f*PhotoElec_CSPA(E, 8) + 0.002f*PhotoElec_CSPA(E, 11) + 0.002f*PhotoElec_CSPA(E, 15)
			+ 0.001f*PhotoElec_CSPA(E, 16) + 0.002f*PhotoElec_CSPA(E, 17) + 0.002f*PhotoElec_CSPA(E, 19)) * 4.525945892e18f;	
}
__device__  float Rayleigh_mu_Pancreas(float E) {
	// H C N O Na P S Cl K
	return (0.106f*Rayleigh_CSPA(E, 1) + 0.169f*Rayleigh_CSPA(E, 6) + 0.022f*Rayleigh_CSPA(E, 7)
			+ 0.694f*Rayleigh_CSPA(E, 8) + 0.002f*Rayleigh_CSPA(E, 11) + 0.002f*Rayleigh_CSPA(E, 15)
			+ 0.001f*Rayleigh_CSPA(E, 16) + 0.002f*Rayleigh_CSPA(E, 17) + 0.002f*Rayleigh_CSPA(E, 19)) * 4.525945892e18f;	
}

/***********************************************************
 * Materials
 ***********************************************************/

// return attenuation according materials 
__device__ float2 att_from_mat(int mat, float E) {
	switch (mat) {
	case 0:     return make_float2(Compton_mu_Air(E),       PhotoElec_mu_Air(E));
	case 1:     return make_float2(Compton_mu_Water(E),     PhotoElec_mu_Water(E));
	case 2:     return make_float2(Compton_mu_Body(E),      PhotoElec_mu_Body(E));
	case 3:     return make_float2(Compton_mu_Lung(E),      PhotoElec_mu_Lung(E));
	case 4:     return make_float2(Compton_mu_Breast(E),    PhotoElec_mu_Breast(E));
	case 5:     return make_float2(Compton_mu_Heart(E),     PhotoElec_mu_Heart(E));
	case 6:     return make_float2(Compton_mu_SpineBone(E), PhotoElec_mu_SpineBone(E));
	case 7:     return make_float2(Compton_mu_RibBone(E),   PhotoElec_mu_RibBone(E));
	case 8:     return make_float2(Compton_mu_Intestine(E), PhotoElec_mu_Intestine(E));
	case 9:     return make_float2(Compton_mu_Spleen(E),    PhotoElec_mu_Spleen(E));
	case 10:    return make_float2(Compton_mu_Blood(E),     PhotoElec_mu_Blood(E));
	case 11:    return make_float2(Compton_mu_Liver(E),     PhotoElec_mu_Liver(E));
	case 12:    return make_float2(Compton_mu_Kidney(E),    PhotoElec_mu_Kidney(E));
		//case 13:     return Compton_mu_Brain(E);
		//case 14:     return Compton_mu_Pancreas(E);
		//case 99:    return Compton_mu_Plastic(E);
		//case 100:	return Compton_mu_Al(E);
	}
	return make_float2(0.0f, 0.0f);
}

/***********************************************************
 * Interactions
 ***********************************************************/

// Kernel interactions
__global__ void kernel_interactions(StackGamma stackgamma, int3 dimvol) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float theta, phi, dx, dy;
	int seed;
	if (id < stackgamma.size) {
		if (stackgamma.in[id] == 0) {return;} // if the particle is outside do nothing...

		switch (stackgamma.interaction[id]) {
		case 0:
			// do nothing and release the thread (maybe the block if interactions are sorted)
			return;
		case 1:
			// PhotoElectric effect
			stackgamma.live[id] = 0; // kill the particle.
			return;
		case 2:
			seed = stackgamma.seed[id];
			// Compton scattering
			theta = Compton_scatter(stackgamma, id);
			phi = park_miller(&seed) * 2.0f * twopi; // TODO swap &seed directly by stackgamma.seed
			stackgamma.seed[id] = seed;
			break;
			// !!!!! WARNING: should be 2*pi instead of 4*pi, but I get only uniform random
			//                number on the half range?! pb with ParkMiller?
			//                So I double the range...
		}
		
		//*****************************************************
		// Apply new direction to the particle (use quaternion)
		//
		// create quaternion from particle and normalize it
		dx = stackgamma.dx[id];
		dy = stackgamma.dy[id];
		float4 d = make_float4(dx, dy, stackgamma.dz[id], 0.0f);
		d = quat_norm(d);
		// select best axis to compute the rotation axis
		float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if (dx<dy) {a.x=1.0f;} // choose x axis
		else {a.y=1.0f;}       // choose y axis
		// create virtual axis given by p^a
		a = quat_crossprod(d, a);
		a = quat_norm(a);
		// build rotation around p axis with phi (in order to rotate the next rotation axis a)
		float4 r = quat_axis(d, phi);
		// do rotation of a = rar*
		a = quat_mul(a, quat_conj(r)); // a = ar*
		a = quat_mul(r, a);            // a = ra
		// build rotation around a axis with theta (thus rotate p)
		r = quat_axis(a, theta);
		// do final rotation of p = rpr*
		d = quat_mul(d, quat_conj(r));
		d = quat_mul(r, d);
		// assign new values
		stackgamma.dx[id] = d.x;
		stackgamma.dy[id] = d.y;
		stackgamma.dz[id] = d.z;
		
	}
}


/***********************************************************
 * Sources
 ***********************************************************/
__global__ void kernel_particle_back2back(StackGamma stackgamma1,
										  StackGamma stackgamma2,
										  int tiny_nb, int3 dimvol, float E, int fact) {

	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id < stackgamma1.size) {
		float x, y, z, rnd, rx, ry, rz, phi, theta, dx, dy, dz;
		int seed, j;
		int jump = dimvol.x * dimvol.y;
		// Not optimize, we wait for a free coincidence (i.e. two particles) before loading a new one
		if ((stackgamma1.in[id]==0 || stackgamma1.live[id]==0) && (stackgamma2.in[id]==0 || stackgamma2.live[id])) {
			seed = stackgamma1.seed[id];

			//// Get position
			rx = park_miller(&seed);
			ry = park_miller(&seed);
			rz = park_miller(&seed);
			rnd = park_miller(&seed);

			// naive way
			//j = 0;
			//while (dact[j] < rnd) {++j;}
			//j = dind[j];

			// first search
			j = int(rnd * tiny_nb);
			if (tex1Dfetch(tex_tiny_act, j) < rnd) {
				while (tex1Dfetch(tex_tiny_act, j) < rnd) {++j;}
			} else {
				while (tex1Dfetch(tex_tiny_act, j) > rnd) {--j;}
				++j; // correct undershoot
			}
			// second search
			j *= fact;
			if (tex1Dfetch(tex_small_act, j) < rnd) {
				while (tex1Dfetch(tex_small_act, j) < rnd) {++j;}
			} else {
				while (tex1Dfetch(tex_small_act, j) > rnd) {--j;}
				++j; // correct undershoot
			}
			// final search
			j *= fact;
			if (tex1Dfetch(tex_huge_act, j) < rnd) {
				while (tex1Dfetch(tex_huge_act, j) < rnd) {++j;}
			} else {
				while (tex1Dfetch(tex_huge_act, j) > rnd) {--j;}
				++j; // correct undershoot
			}

			// get the position inside the volume
			j = tex1Dfetch(tex_ind, j);  // look-up-table
			z = __fdividef(j, jump);
			j -= (z * jump);
			y = __fdividef(j, dimvol.x);
			x = j - y*dimvol.x;
			// get the pos inside the voxel
			x += rx;
			y += ry;
			z += rz;

			//// Get direction
			phi = park_miller(&seed);
			theta = park_miller(&seed);
			phi   = twopi * phi;
			theta = acosf(1.0f - 2.0f*theta);
			// convert to cartesian
			dx = __cosf(phi)*__sinf(theta);
			dy = __sinf(phi)*__sinf(theta);
			dz = __cosf(theta);

			// Assignment
			stackgamma1.dx[id] = dx;
			stackgamma1.dy[id] = dy;
			stackgamma1.dz[id] = dz;
			stackgamma1.seed[id] = seed;
			stackgamma1.E[id] = E;
			stackgamma1.px[id] = x;
			stackgamma1.py[id] = y;
			stackgamma1.pz[id] = z;
			stackgamma1.live[id] = 1;
			stackgamma1.in[id] = 1;
			stackgamma1.interaction[id] = 0;

			stackgamma2.dx[id] = -dx;
			stackgamma2.dy[id] = -dy;  // back2back
			stackgamma2.dz[id] = -dz;
			stackgamma2.E[id] = E;
			stackgamma2.px[id] = x;
			stackgamma2.py[id] = y;
			stackgamma2.pz[id] = z;
			stackgamma2.live[id] = 1;
			stackgamma2.in[id] = 1;
			stackgamma2.interaction[id] = 0;

		} // if
	} // if

}

/***********************************************************
 * Tracking kernel
 ***********************************************************/

//
// STILL WORKING ON IT!!!!!
//   but enough stable to be used
//
// Fictitious tracking (or delta-tracking)
__global__ void kernel_woodcock(int3 dimvol, StackGamma stackgamma, float dimvox) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int jump = dimvol.x*dimvol.y;
	float3 p0, delta;
	int3 vox;
	float path, rec_mu_maj, E, sum_CS;
	float2 cur_att;
	int mat, seed;
	dimvox = __fdividef(1.0f, dimvox);
	
	if (id < stackgamma.size) {
		p0.x = stackgamma.px[id];
		p0.y = stackgamma.py[id];
		p0.z = stackgamma.pz[id];
		delta.x = stackgamma.dx[id];
		delta.y = stackgamma.dy[id];
		delta.z = stackgamma.dz[id];
		seed = stackgamma.seed[id];
		E = stackgamma.E[id];

		/*
		__shared__ float CS[256][15];
		CS[threadIdx.x][0] = 0.0f;
		CS[threadIdx.x][1] = 0.0f;
		*/
		//CS[threadIdx.x][15] = 0.0f; ERROR, overflow!!!!!

		// Most attenuate material is RibBone (ID=7)
		cur_att = att_from_mat(7, E);
		rec_mu_maj = __fdividef(1.0f, cur_att.x + cur_att.y);

		while (1) {
			// get mean path from the most attenuate material (RibBone)
			path = -__logf(park_miller(&seed)) * rec_mu_maj * dimvox;
			
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
				stackgamma.interaction[id] = 0;
				stackgamma.in[id] = 0;
				stackgamma.seed[id] = seed;
				stackgamma.px[id] = p0.x;
				stackgamma.py[id] = p0.y;
				stackgamma.pz[id] = p0.z;
				return;
			}
			
			// Does the interaction is real?
			mat = tex1Dfetch(tex_phantom, vox.z*jump + vox.y*dimvol.x + vox.x);
			cur_att = att_from_mat(mat, E); // cur_att = (Compton, PE)
			sum_CS = cur_att.x + cur_att.y;
			/*
			if (CS[threadIdx.x][mat] == 0) {
				cur_att = att_from_mat(mat, E);
				CS[threadIdx.x][mat] = cur_att;
			} else {
				cur_att = CS[threadIdx.x][mat];
			}
			*/
			
			if (sum_CS * rec_mu_maj > park_miller(&seed)) {break;}

		}

		// Select interaction
		rec_mu_maj = __fdividef(cur_att.y, sum_CS); // reuse rec_mu_maj variable

		if (park_miller(&seed) <= rec_mu_maj) {
			// PhotoElectric
			stackgamma.interaction[id] = 1;
		} else {
			// Compton
			stackgamma.interaction[id] = 2;
		}
		stackgamma.seed[id] = seed;
		stackgamma.px[id] = p0.x;
		stackgamma.py[id] = p0.y;
		stackgamma.pz[id] = p0.z;
	}

}


/***********************************************************
 * Utils
 ***********************************************************/
__global__ void kernel_test(float* dtrack, int max) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	if (id < max) {
		dtrack[id] = id;
		//dtrack[id] = tex1Dfetch(tex_phantom, id);
		//dtrack[id] = dact[id];
		//dtrack[id] = tex1Dfetch(tex_act, id);
	}
}


void mc_proj_detector(float* im, int nz, int ny, int nx, float* x, int sx, float* y, int sy, float* z, int sz) {
	int i=0;
	while (i<sx) {
		im[int(z[i])*nx*ny + int(y[i])*nx + int(x[i])] += 1.0f;
		//im[int(x[i])] += 1.0f;
		++i;
	}
}

/***********************************************************
 * Main
 ***********************************************************/
void mc_pet_cuda() {

	// Simulation parameters
	float E = 0.511; // MeV
	int totparticles = 10000000; // Nb of particules required by the simulation
	int stack_size = 5000000;    // note: during the simulation two stacks are running
	int seed = 10;
	int maxit = 5;
	char* output_name = "output.bin";
	cudaSetDevice(1); // Select a device, need to change accordingly (-1 select the most powerfull GPU on the computer)

	// Variables
    timeval start, end;
    double t1, t2, diff;
	timeval start_s, end_s;
	double ts1, ts2, te1, te2;
	int3 dim_phantom;
	int n, step;
	int countparticle = 0;

	// time to init
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;

	// Open the NCAT (46x63x128 voxels which represent ID material)
	FILE * pfile = fopen("ncat_12mat.bin", "rb");
	dim_phantom.z = 46;
	dim_phantom.y = 63;
	dim_phantom.x = 128;
	float size_voxel = 4.0f;  // used latter
	int nb = dim_phantom.z * dim_phantom.y * dim_phantom.x;
	unsigned int mem_phantom = nb * sizeof(unsigned short int);
	unsigned short int* phantom = (unsigned short int*)malloc(mem_phantom);
	fread(phantom, sizeof(unsigned short int), nb, pfile);
	fclose(pfile);

	// Load NCAT to texture
	unsigned short int* dphantom;
	cudaMalloc((void**) &dphantom, mem_phantom);
	cudaMemcpy(dphantom, phantom, mem_phantom, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_phantom, dphantom, mem_phantom);

	// Open voxelized activities
	int fact = 50; // scale factor to decimate activity map

	pfile = fopen("huge_act.bin", "rb");
	unsigned int mem_huge_act = nb * sizeof(float);
	float* huge_act = (float*)malloc(mem_huge_act);
	fread(huge_act, sizeof(float), nb, pfile);
	fclose(pfile);

	int small_nb = 7418; // nb / fact
	pfile = fopen("small_act.bin", "rb");
	unsigned int mem_small_act = small_nb * sizeof(float);
	float* small_act = (float*)malloc(mem_small_act);
	fread(small_act, sizeof(float), small_nb, pfile);
	fclose(pfile);
	
	int tiny_nb = 148; // nb / fact / fact
	pfile = fopen("tiny_act.bin", "rb");
	unsigned int mem_tiny_act = tiny_nb * sizeof(float);
	float* tiny_act = (float*)malloc(mem_tiny_act);
	fread(tiny_act, sizeof(float), tiny_nb, pfile);
	fclose(pfile);

	pfile = fopen("ind_act.bin", "rb");
	unsigned int mem_ind_act = nb * sizeof(int);
	int* ind_act = (int*)malloc(mem_ind_act);
	fread(ind_act, sizeof(int), nb, pfile);
	fclose(pfile);
	
	// Load voxelized actitivities to texture
	float* huge_dact;
	cudaMalloc((void**) &huge_dact, mem_huge_act);
	cudaMemcpy(huge_dact, huge_act, mem_huge_act, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_huge_act, huge_dact, mem_huge_act);

	float* small_dact;
	cudaMalloc((void**) &small_dact, mem_small_act);
	cudaMemcpy(small_dact, small_act, mem_small_act, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_small_act, small_dact, mem_small_act);

	float* tiny_dact;
	cudaMalloc((void**) &tiny_dact, mem_tiny_act);
	cudaMemcpy(tiny_dact, tiny_act, mem_tiny_act, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_tiny_act, tiny_dact, mem_tiny_act);

	int* ind_dact;
	cudaMalloc((void**) &ind_dact, mem_ind_act);
	cudaMemcpy(ind_dact, ind_act, mem_ind_act, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_ind, ind_dact, mem_ind_act);

	//////// Rayleigh is not used in this simulation //////////
	/*
	// Open Rayleigh cross section
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

	// Load Rayleigh data to texture
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
	*/

	// Host allocation memory to save the phase-space
	float* pE = (float*)malloc(totparticles * sizeof(float));
	float* px = (float*)malloc(totparticles * sizeof(float));
	float* py = (float*)malloc(totparticles * sizeof(float));
	float* pz = (float*)malloc(totparticles * sizeof(float));
	float* dx = (float*)malloc(totparticles * sizeof(float));
	float* dy = (float*)malloc(totparticles * sizeof(float));
	float* dz = (float*)malloc(totparticles * sizeof(float));

	// Defined Stacks
	StackGamma stackgamma1;
	StackGamma stackgamma2;
	StackGamma collector;
	stackgamma1.size = stack_size;
	stackgamma2.size = stack_size;
	collector.size = stack_size;
	unsigned int mem_stack_float = stackgamma1.size * sizeof(float);
	unsigned int mem_stack_int = stackgamma1.size * sizeof(int);
	unsigned int mem_stack_char = stackgamma1.size * sizeof(char);
	unsigned int mem_collector_float = collector.size * sizeof(float);
	unsigned int mem_collector_char = collector.size * sizeof(char);

	// Host stack allocation memory
	collector.E = (float*)malloc(mem_collector_float);
	collector.dx = (float*)malloc(mem_collector_float);
	collector.dy = (float*)malloc(mem_collector_float);
	collector.dz = (float*)malloc(mem_collector_float);
	collector.px = (float*)malloc(mem_collector_float);
	collector.py = (float*)malloc(mem_collector_float);
	collector.pz = (float*)malloc(mem_collector_float);
	collector.interaction = (unsigned char*)malloc(mem_collector_char);
	collector.live = (unsigned char*)malloc(mem_collector_char);
	collector.in = (unsigned char*)malloc(mem_collector_char);

	// Device stack allocation memory
	cudaMalloc((void**) &stackgamma1.E, mem_stack_float);
	cudaMalloc((void**) &stackgamma1.dx, mem_stack_float);
	cudaMalloc((void**) &stackgamma1.dy, mem_stack_float);
	cudaMalloc((void**) &stackgamma1.dz, mem_stack_float);
	cudaMalloc((void**) &stackgamma1.px, mem_stack_float);
	cudaMalloc((void**) &stackgamma1.py, mem_stack_float);
	cudaMalloc((void**) &stackgamma1.pz, mem_stack_float);
	cudaMalloc((void**) &stackgamma1.seed, mem_stack_int);
	cudaMalloc((void**) &stackgamma1.interaction, mem_stack_char);
	cudaMalloc((void**) &stackgamma1.live, mem_stack_char);
	cudaMalloc((void**) &stackgamma1.in, mem_stack_char);
	cudaMemset(stackgamma1.interaction, 0, mem_stack_char); // no interaction selected
	cudaMemset(stackgamma1.live, 0, mem_stack_char);        // at beginning all particles are dead
	cudaMemset(stackgamma1.in, 0, mem_stack_char);          // and outside the volume

	cudaMalloc((void**) &stackgamma2.E, mem_stack_float);
	cudaMalloc((void**) &stackgamma2.dx, mem_stack_float);
	cudaMalloc((void**) &stackgamma2.dy, mem_stack_float);
	cudaMalloc((void**) &stackgamma2.dz, mem_stack_float);
	cudaMalloc((void**) &stackgamma2.px, mem_stack_float);
	cudaMalloc((void**) &stackgamma2.py, mem_stack_float);
	cudaMalloc((void**) &stackgamma2.pz, mem_stack_float);
	cudaMalloc((void**) &stackgamma2.seed, mem_stack_int);
	cudaMalloc((void**) &stackgamma2.interaction, mem_stack_char);
	cudaMalloc((void**) &stackgamma2.live, mem_stack_char);
	cudaMalloc((void**) &stackgamma2.in, mem_stack_char);
	cudaMemset(stackgamma2.interaction, 0, mem_stack_char);	// no interaction selected
	cudaMemset(stackgamma2.live, 0, mem_stack_char);        // at beginning all particles are dead
	cudaMemset(stackgamma2.in, 0, mem_stack_char);          // and outside the volume

	// Init seeds
	int* tmp = (int*)malloc(stackgamma1.size * sizeof(int));
	srand(seed);
	n=0; while (n<stackgamma1.size) {tmp[n] = rand(); ++n;}
	cudaMemcpy(stackgamma1.seed, tmp, mem_stack_int, cudaMemcpyHostToDevice);
	n=0; while (n<stackgamma2.size) {tmp[n] = rand(); ++n;}
	cudaMemcpy(stackgamma2.seed, tmp, mem_stack_int, cudaMemcpyHostToDevice);
	free(tmp);

	// Usefull to debug
	//float* ddebug;
	//cudaMalloc((void**) &ddebug, mem_stack_float);
	//cudaMemset(ddebug, 0, mem_stack_float);

	// Vars kernel
	dim3 threads, grid;
	int block_size = 256;
	int grid_size = (stack_size + block_size - 1) / block_size;
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

		////////////////////////
		// Generation
		////////////////////////
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		kernel_particle_back2back<<<grid, threads>>>(stackgamma1, stackgamma2, tiny_nb, dim_phantom, E, fact);
		cudaThreadSynchronize();
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("   Generation %f s\n", diff);

		////////////////////////
		// Navigation
		////////////////////////
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		kernel_woodcock<<<grid, threads>>>(dim_phantom, stackgamma1, size_voxel);
		kernel_woodcock<<<grid, threads>>>(dim_phantom, stackgamma2, size_voxel);
		cudaThreadSynchronize();
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("   Navigation %f s\n", diff);
		
		////////////////////////
		// Interactions
		////////////////////////
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		kernel_interactions<<<grid, threads>>>(stackgamma1, dim_phantom);
		kernel_interactions<<<grid, threads>>>(stackgamma2, dim_phantom);		
		cudaThreadSynchronize();
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("   Interactions %f s\n", diff);
		
		////////////////////////
		// Extraction:
		//   This part have to change in order to gather the two host/device copies and the two picking.
		////////////////////////

		gettimeofday(&start, NULL);
		te1 = start.tv_sec + start.tv_usec / 1000000.0;
		printf("   Extraction\n");
		
		// first copy
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		cudaMemcpy(collector.E, stackgamma1.E, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dx, stackgamma1.dx, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dy, stackgamma1.dy, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dz, stackgamma1.dz, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.px, stackgamma1.px, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.py, stackgamma1.py, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.pz, stackgamma1.pz, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.interaction, stackgamma1.interaction, mem_stack_char, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.live, stackgamma1.live, mem_stack_char, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.in, stackgamma1.in, mem_stack_char, cudaMemcpyDeviceToHost);
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("      Get back the first stack %f s\n", diff);

		// first picking
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		int c0 = 0;	int c1 = 0;	int c2 = 0;
		n = 0;
		while(n < stack_size && countparticle < totparticles) {
			// pick only outsider particles
			if (collector.in[n] == 0) {
				pE[countparticle] = collector.E[n];
				dx[countparticle] = collector.dx[n];
				dy[countparticle] = collector.dy[n];
				dz[countparticle] = collector.dz[n];
				px[countparticle] = collector.px[n];
				py[countparticle] = collector.py[n];
				pz[countparticle] = collector.pz[n];
				++countparticle;
			}
			if (collector.interaction[n] == 0) {++c0;}
			if (collector.interaction[n] == 1) {++c1;}
			if (collector.interaction[n] == 2) {++c2;}
			++n;
		}
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		
		printf("      Picking particles %i/%i particles on the first stack %f s\n", countparticle, totparticles, diff);
		printf("         PE %i Cpt %i None %i\n", c1, c2, c0);

		// second copy
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		cudaMemcpy(collector.E, stackgamma2.E, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dx, stackgamma2.dx, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dy, stackgamma2.dy, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.dz, stackgamma2.dz, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.px, stackgamma2.px, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.py, stackgamma2.py, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.pz, stackgamma2.pz, mem_stack_float, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.interaction, stackgamma2.interaction, mem_stack_char, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.live, stackgamma2.live, mem_stack_char, cudaMemcpyDeviceToHost);
		cudaMemcpy(collector.in, stackgamma2.in, mem_stack_char, cudaMemcpyDeviceToHost);
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;
		printf("      Get back the second stack %f s\n", diff);

		// second picking
		gettimeofday(&start, NULL);
		t1 = start.tv_sec + start.tv_usec / 1000000.0;
		c0 = 0;	c1 = 0;	c2 = 0;
		n = 0;
		while(n < stack_size && countparticle < totparticles) {
			// pick only outsider particles
			if (collector.in[n] == 0) {
				pE[countparticle] = collector.E[n];
				dx[countparticle] = collector.dx[n];
				dy[countparticle] = collector.dy[n];
				dz[countparticle] = collector.dz[n];
				px[countparticle] = collector.px[n];
				py[countparticle] = collector.py[n];
				pz[countparticle] = collector.pz[n];
				++countparticle;
			}
			if (collector.interaction[n] == 0) {++c0;}
			if (collector.interaction[n] == 1) {++c1;}
			if (collector.interaction[n] == 2) {++c2;}
			++n;
		}
		gettimeofday(&end, NULL);
		t2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = t2 - t1;

		printf("      Picking particles %i/%i particles on the second stack %f s\n", countparticle, totparticles, diff);
		printf("         PE %i Cpt %i None %i\n", c1, c2, c0);

		gettimeofday(&end, NULL);
		te2 = end.tv_sec + end.tv_usec / 1000000.0;
		diff = te2 - te1;
		printf("      Extraction tot time %f s\n", diff);
		
		// loop control
		if (countparticle >= totparticles) {break;}
		++step;
		
	} // outter loop (step)

	gettimeofday(&end_s, NULL);
	ts2 = end_s.tv_sec + end_s.tv_usec / 1000000.0;
	diff = ts2 - ts1;
	printf("Step running time %f s\n", diff);
	
	// TO DEBUG
	//cudaMemcpy(px, ddebug, mem_stack_float, cudaMemcpyDeviceToHost);

	// Export phase-space
	//  TODO: export in phase-space format (IAEAphsp)
	gettimeofday(&start, NULL);
	t1 = start.tv_sec + start.tv_usec / 1000000.0;
	pfile = fopen(output_name, "wb");
	fwrite(pE, sizeof(float), totparticles, pfile);
	fwrite(px, sizeof(float), totparticles, pfile);
	fwrite(py, sizeof(float), totparticles, pfile);
	fwrite(pz, sizeof(float), totparticles, pfile);
	fwrite(dx, sizeof(float), totparticles, pfile);
	fwrite(dy, sizeof(float), totparticles, pfile);
	fwrite(dz, sizeof(float), totparticles, pfile);
	fclose(pfile);
	gettimeofday(&end, NULL);
	te2 = end.tv_sec + end.tv_usec / 1000000.0;
	diff = te2 - te1;
	printf("Save phase-space in %f s\n", diff);
	
	// Clean memory
	free(collector.E);
	free(collector.dx);
	free(collector.dy);
	free(collector.dz);
	free(collector.px);
	free(collector.py);
	free(collector.pz);
	free(collector.interaction);
	free(collector.live);
	free(collector.in);
	free(pE);
	free(px);
	free(py);
	free(pz);
	free(dx);
	free(dy);
	free(dz);
	free(phantom);
	free(huge_act);
	free(small_act);
	free(tiny_act);
	free(ind_act);
	
	cudaFree(stackgamma1.E);
	cudaFree(stackgamma1.dx);
	cudaFree(stackgamma1.dy);
	cudaFree(stackgamma1.dz);
	cudaFree(stackgamma1.px);
	cudaFree(stackgamma1.py);
	cudaFree(stackgamma1.pz);
	cudaFree(stackgamma1.interaction);
	cudaFree(stackgamma1.live);
	cudaFree(stackgamma1.in);
	cudaFree(stackgamma1.seed);
	
	cudaFree(stackgamma2.E);
	cudaFree(stackgamma2.dx);
	cudaFree(stackgamma2.dy);
	cudaFree(stackgamma2.dz);
	cudaFree(stackgamma2.px);
	cudaFree(stackgamma2.py);
	cudaFree(stackgamma2.pz);
	cudaFree(stackgamma2.interaction);
	cudaFree(stackgamma2.live);
	cudaFree(stackgamma2.in);
	cudaFree(stackgamma2.seed);

	cudaUnbindTexture(tex_phantom);
	cudaUnbindTexture(tex_huge_act);
	cudaUnbindTexture(tex_small_act);
	cudaUnbindTexture(tex_tiny_act);
	cudaUnbindTexture(tex_ind);

	cudaFree(dphantom);
	cudaFree(huge_dact);
	cudaFree(small_dact);
	cudaFree(tiny_dact);
	cudaFree(ind_dact);
	//cudaFree(ddebug);
	
	cudaThreadExit();

}
