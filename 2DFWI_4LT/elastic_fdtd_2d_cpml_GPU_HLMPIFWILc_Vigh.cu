#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BLOCK_SIZE 16

#define Lpo 4	//S_LC_space
#define Lso 6	//P_LC_space
#define Lcc (Lso>Lpo?Lso:Lpo)

#include "headcpu.h"
//#include "headstream.h"
#include "cufft.h"


struct Multistream
{
	cudaStream_t stream,stream_back;
};

extern "C"
void getdevice(int *GPU_N)
{
	
	cudaGetDeviceCount(GPU_N);	
//	GPU_N=6;//4;//2;//
}

__global__ void initialize_wavefields(
		float *vx, float *vz,
		float *sigmaxx, float *sigmaxxs, float *sigmazz, float *sigmaxz,
		float *phi_vx_x, float *phi_vxs_x, float *phi_vx_z,
		float *phi_vz_z, float *phi_vzs_z, float *phi_vz_x,
		float *phi_sigmaxx_x, float *phi_sigmaxxs_x, float *phi_sigmaxz_z,
		float *phi_sigmaxz_x, float *phi_sigmazz_z,
		float *phi_sigmaxx_z, float *phi_sigmaxxs_z, float *phi_sigmazz_x,
		int ntp, int ntx, int ntz 
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		vx[ip]=0.0;
		vz[ip]=0.0;

		sigmaxx[ip]=0.0;
		sigmaxxs[ip]=0.0;
		sigmazz[ip]=0.0;
		sigmaxz[ip]=0.0;

		phi_vx_x[ip]=0.0;
		phi_vxs_x[ip]=0.0;
		phi_vz_z[ip]=0.0;
		phi_vzs_z[ip]=0.0;
		phi_vx_z[ip]=0.0;
		phi_vz_x[ip]=0.0;

		phi_sigmaxx_x[ip]=0.0;
		phi_sigmaxxs_x[ip]=0.0;
		phi_sigmaxz_z[ip]=0.0;

		phi_sigmaxz_x[ip]=0.0;
		phi_sigmazz_z[ip]=0.0;

		phi_sigmaxx_z[ip]=0.0;
		phi_sigmaxxs_z[ip]=0.0;
		phi_sigmazz_x[ip]=0.0;
	}

	__syncthreads();
}

__global__ void fdtd_cpml_2d_GPU_kernel_vx(
		float *rho, int itmax,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *sigmaxx, float *sigmaxz,
		float *phi_sigmaxx_x, float *phi_sigmaxz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vx, int it, int pml, int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n, int inv_flag
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ic,ii;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxz_dz;
	float one_over_rho_half_x;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dsigmaxx_dx=0.0;
		dsigmaxz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxx_dx+=rc[ic]*(sigmaxx[ip+ic+1]-sigmaxx[ip-ic])*one_over_dx;
			dsigmaxz_dz+=rc[ic]*(sigmaxz[ip+ic*ntx]-sigmaxz[ip-(ic+1)*ntx])*one_over_dz;
		}

		phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
		phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
				+dsigmaxz_dz+phi_sigmaxz_z[ip])
			+vx[ip];
	}   

	if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				seismogram_vx[it*r_n+ii]=vx[ip];
			}
		}

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vx_shared(
		float *rho, int itmax,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *sigmaxx, float *sigmaxz,
		float *phi_sigmaxx_x, float *phi_sigmaxz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vx, int it, int pml,  int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n, int inv_flag
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ic,ii;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxz_dz;
	float one_over_rho_half_x;
	
	/***start the assignment of the shared memory***/
	__shared__ float s_vx[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_sigmaxx[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_sigmaxz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_vx[ty][tx]=vx[ip];
	s_sigmaxx[ty][s_tx]=sigmaxx[ip];
	s_sigmaxz[s_ty][tx]=sigmaxz[ip];	
		
	// in x_index
	if(tx<Lc)
		if(bx)
			s_sigmaxx[ty][tx]=sigmaxx[ip-Lc];
		else
			s_sigmaxx[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_sigmaxx[ty][tx+2*Lc]=sigmaxx[ip+Lc];
		else
			s_sigmaxx[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_sigmaxz[ty][tx]=sigmaxz[ip-Lc*ntx];
		else
			s_sigmaxz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_sigmaxz[ty+2*Lc][tx]=sigmaxz[ip+Lc*ntx];
		else
			s_sigmaxz[ty+2*Lc][tx]=0.0;
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/
	
	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dsigmaxx_dx=0.0;
		dsigmaxz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxx_dx+=s_rc[ic]*(s_sigmaxx[ty][s_tx+ic+1]-s_sigmaxx[ty][s_tx-ic])*one_over_dx;
			dsigmaxz_dz+=s_rc[ic]*(s_sigmaxz[s_ty+ic][tx]-s_sigmaxz[s_ty-ic-1][tx])*one_over_dz;
		}

		phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
		phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
				+dsigmaxz_dz+phi_sigmaxz_z[ip])
			+s_vx[ty][tx];
	} 
	
	if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				seismogram_vx[it*r_n+ii]=vx[ip];
			}
		}
	}
	__syncthreads();

}

__global__ void fdtd_cpml_2d_GPU_kernel_vx_4LT(
		float *rho, int itmax, float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *sigmaxx, float *sigmaxxs,float *sigmaxz,
		float *phi_sigmaxx_x, float *phi_sigmaxxs_x, float *phi_sigmaxz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vx, int it, int pml, int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n, int inv_flag
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ic,ii;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxxs_dx,dsigmaxz_dz;
	float one_over_rho_half_x;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	if(iz>=Lb&&iz<=ntz-Lb&&ix>=Lb-1&&ix<=ntx-Lb-1)
	{
		dsigmaxx_dx=0.0;
		dsigmaxxs_dx=0.0;
		dsigmaxz_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
			dsigmaxx_dx+=Gp[index1*Lp+ic]*(sigmaxx[ip+ic+1]-sigmaxx[ip-ic])*one_over_dx;

		dsigmaxx_dx+=Gp[index1*Lp+Lpo]*(sigmaxx[ip+ntx+1]-sigmaxx[ip+ntx]+sigmaxx[ip-ntx+1]-sigmaxx[ip-ntx])*one_over_dx;	//x direction

		for(ic=0;ic<Lso;ic++)
		{
			dsigmaxxs_dx+=Gs[index2*Ls+ic]*(sigmaxxs[ip+ic+1]-sigmaxxs[ip-ic])*one_over_dx;
			dsigmaxz_dz+=Gs[index2*Ls+ic]*(sigmaxz[ip+ic*ntx]-sigmaxz[ip-(ic+1)*ntx])*one_over_dz;
		}

		dsigmaxxs_dx+=Gs[index2*Ls+Lso]*(sigmaxxs[ip+ntx+1]-sigmaxxs[ip+ntx]+sigmaxxs[ip-ntx+1]-sigmaxxs[ip-ntx])*one_over_dx;	//x direction
		dsigmaxz_dz+=Gs[index2*Ls+Lso]*(sigmaxz[ip+1]-sigmaxz[ip-ntx+1]+sigmaxz[ip-1]-sigmaxz[ip-ntx-1])*one_over_dz;

		phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
		phi_sigmaxxs_x[ip]=b_x_half[ix]*phi_sigmaxxs_x[ip]+a_x_half[ix]*dsigmaxxs_dx;
		phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
				+dsigmaxz_dz+phi_sigmaxz_z[ip]+dsigmaxxs_dx+phi_sigmaxxs_x[ip])
			+vx[ip];
	}   

/*	if(iz>=pml&&iz<=pml+10*Lb&&ix>=Lb-1&&ix<=ntx-Lb-1)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				seismogram_vx[it*r_n+ii]=vx[ip];
			}
		}
*/
	__syncthreads();

}



__global__ void fdtd_cpml_2d_GPU_kernel_vz(
		float *rho, int itmax,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *sigmaxz, float *sigmazz, 
		float *phi_sigmaxz_x, float *phi_sigmazz_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vz, int it, int pml, int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n, int inv_flag
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmazz_dz;

	float one_over_rho_half_z;

	int ic;
	int ip,ii;
	//int pmlc=pml+Lc;
	ip=iz*ntx+ix;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc&&ix<=ntx-Lc)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxz_dx+=rc[ic]*(sigmaxz[ip+ic]-sigmaxz[ip-(ic+1)])*one_over_dx;
			dsigmazz_dz+=rc[ic]*(sigmazz[ip+(ic+1)*ntx]-sigmazz[ip-ic*ntx])*one_over_dz;
		}

		phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
		phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;


		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
				+dsigmazz_dz+phi_sigmazz_z[ip])
			+vz[ip];
	}

	if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc&&ix<=ntx-Lc)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				seismogram_vz[it*r_n+ii]=vz[ip];
			}
		}

	__syncthreads();

}

__global__ void fdtd_cpml_2d_GPU_kernel_vz_shared(
		float *rho, int itmax,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *sigmaxz, float *sigmazz, 
		float *phi_sigmaxz_x, float *phi_sigmazz_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vz, int it, int pml,   int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n, int inv_flag
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmazz_dz;

	float one_over_rho_half_z;

	int ic;
	int ip,ii;
	//int pmlc=pml+Lc;
	ip=iz*ntx+ix;
	
	/***start the assignment of the shared memory***/
	__shared__ float s_vz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_sigmazz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_sigmaxz[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_rc[Lcc];
		
	s_vz[ty][tx]=vz[ip];
	s_sigmazz[s_ty][tx]=sigmazz[ip];
	s_sigmaxz[ty][s_tx]=sigmaxz[ip];
		
	// in x_index
	if(tx<Lc)
		if(bx)
			s_sigmaxz[ty][tx]=sigmaxz[ip-Lc];
		else
			s_sigmaxz[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_sigmaxz[ty][tx+2*Lc]=sigmaxz[ip+Lc];
		else
			s_sigmaxz[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_sigmazz[ty][tx]=sigmazz[ip-Lc*ntx];
		else
			s_sigmazz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_sigmazz[ty+2*Lc][tx]=sigmazz[ip+Lc*ntx];
		else
			s_sigmazz[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/	
	
	
	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc&&ix<=ntx-Lc)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxz_dx+=s_rc[ic]*(s_sigmaxz[ty][s_tx+ic]-s_sigmaxz[ty][s_tx-ic-1])*one_over_dx;
			dsigmazz_dz+=s_rc[ic]*(s_sigmazz[s_ty+ic+1][tx]-s_sigmazz[s_ty-ic][tx])*one_over_dz;
		}

		phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
		phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;


		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
				+dsigmazz_dz+phi_sigmazz_z[ip])
			+s_vz[ty][tx];
	}

	if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc&&ix<=ntx-Lc)	
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				seismogram_vz[it*r_n+ii]=vz[ip];
			}
		}
	
	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vz_4LT(
		float *rho, int itmax,float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *sigmaxz, float *sigmaxx, float *sigmazz, 
		float *phi_sigmaxz_x, float *phi_sigmaxx_z,float *phi_sigmazz_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vz, int it, int pml, int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n, int inv_flag
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmaxx_dz,dsigmazz_dz;

	float one_over_rho_half_z;

	int ic;
	int ip,ii;
	//int pmlc=pml+Lc;
	ip=iz*ntx+ix;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	if(iz>=Lb-1&&iz<=ntz-Lb-1&&ix>=Lb&&ix<=ntx-Lb)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;
		dsigmaxx_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
		{
			dsigmaxx_dz+=Gp[index1*Lp+ic]*(sigmaxx[ip+(ic+1)*ntx]-sigmaxx[ip-ic*ntx])*one_over_dz;
		}
		dsigmaxx_dz+=Gp[index1*Lp+Lpo]*(sigmaxx[ip+ntx+1]-sigmaxx[ip+1]+sigmaxx[ip+ntx-1]-sigmaxx[ip-1])*one_over_dz;


		for(ic=0;ic<Lso;ic++)
		{
			dsigmaxz_dx+=Gs[index2*Ls+ic]*(sigmaxz[ip+ic]-sigmaxz[ip-(ic+1)])*one_over_dx;
			dsigmazz_dz+=Gs[index2*Ls+ic]*(sigmazz[ip+(ic+1)*ntx]-sigmazz[ip-ic*ntx])*one_over_dz;
		}
		dsigmaxz_dx+=Gs[index2*Ls+Lso]*(sigmaxz[ip+ntx]-sigmaxz[ip+ntx-1]+sigmaxz[ip-ntx]-sigmaxz[ip-ntx-1])*one_over_dx;
		dsigmazz_dz+=Gs[index2*Ls+Lso]*(sigmazz[ip+ntx+1]-sigmazz[ip+1]+sigmazz[ip+ntx-1]-sigmazz[ip-1])*one_over_dz;

		phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
		phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;
		phi_sigmaxx_z[ip]=b_z_half[iz]*phi_sigmaxx_z[ip]+a_z_half[iz]*dsigmaxx_dz;


		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
				+dsigmazz_dz+phi_sigmazz_z[ip]+dsigmaxx_dz+phi_sigmaxx_z[ip])
			+vz[ip];
	}

/*	if(iz>=pml&&iz<=pml+10*Lb&&ix>=Lb&&ix<=ntx-Lb)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				seismogram_vz[it*r_n+ii]=vz[ip];
			}
		}

*/	__syncthreads();

}

__global__ void fdtd_2d_GPU_kernel_borders_forward
(
 float *vx,
 float *vx_borders_up, float *vx_borders_bottom,
 float *vx_borders_left, float *vx_borders_right,
 float *vz,
 float *vz_borders_up, float *vz_borders_bottom,
 float *vz_borders_left, float *vz_borders_right,
 int ntp, int ntx, int ntz, int pml, int Lc, float *rc, int it, int itmax, int inv_flag
 )
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;

	int nx=ntx-2*pmlc;
	int nz=ntz-2*pmlc;

	// Borders...
	if(inv_flag==1)
	{
		if(iz>=pmlc&&iz<=pmlc+Lc-1&&ix>=pmlc-1&&ix<=ntx-pmlc-1)
		{
			vx_borders_up[it*Lc*(nx+1)+(iz-pmlc)*(nx+1)+ix-pmlc+1]=vx[ip];
		}
		if(iz>=ntz-pmlc-Lc&&iz<=ntz-pmlc-1&&ix>=pmlc-1&&ix<=ntx-pmlc-1)
		{
			vx_borders_bottom[it*Lc*(nx+1)+(iz-ntz+pmlc+Lc)*(nx+1)+ix-pmlc+1]=vx[ip];
		}

		if(ix>=pmlc-1&&ix<=pmlc+Lc-2&&iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1)
		{
			vx_borders_left[it*Lc*(nz-2*Lc)+(iz-pmlc-Lc)*Lc+ix-pmlc+1]=vx[ip];
		}
		if(ix>=ntx-pmlc-Lc&&ix<=ntx-pmlc-1&&iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1)
		{
			vx_borders_right[it*Lc*(nz-2*Lc)+(iz-pmlc-Lc)*Lc+ix-ntx+pmlc+Lc]=vx[ip];
		}

		//////////////////////////////////////////////////////////////

		if(iz>=pmlc-1&&iz<=pmlc+Lc-2&&ix>=pmlc&&ix<=ntx-pmlc-1)
		{
			vz_borders_up[it*Lc*nx+(iz-pmlc+1)*nx+ix-pmlc]=vz[ip];
		}
		if(iz>=ntz-pmlc-Lc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
		{
			vz_borders_bottom[it*Lc*nx+(iz-ntz+pmlc+Lc)*nx+ix-pmlc]=vz[ip];
		}

		if(ix>=pmlc&&ix<=pmlc+Lc-1&&iz>=pmlc+Lc-1&&iz<=ntz-pmlc-Lc-1)
		{
			vz_borders_left[it*Lc*(nz-2*Lc+1)+(iz-pmlc-Lc+1)*Lc+ix-pmlc]=vz[ip];
		}
		if(ix>=ntx-pmlc-Lc&&ix<=ntx-pmlc-1&&iz>=pmlc+Lc-1&&iz<=ntz-pmlc-Lc-1)
		{
			vz_borders_right[it*Lc*(nz-2*Lc+1)+(iz-pmlc-Lc+1)*Lc+ix-ntx+pmlc+Lc]=vz[ip];
		}
	}

	__syncthreads();

}

__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz(
		float *rick, 
		float *lambda, float *lambda_plus_two_mu,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *sigmaxx, float *sigmazz,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int s_ix, int s_iz, int it,
		int inv_flag, int Lc, float *rc
		)

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;


	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int ic;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc&&ix<=ntx-Lc)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=rc[ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=rc[ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;


		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip])+
				lambda[ip]*(dvz_dz+phi_vz_z[ip]))*dt+
			sigmaxx[ip];

		sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz+phi_vz_z[ip])+
				lambda[ip]*(dvx_dx+phi_vx_x[ip]))*dt+
			sigmazz[ip];
	}

	if(iz==s_iz&&ix==s_ix)
	{
		sigmaxx[ip]=sigmaxx[ip]+rick[it];
		sigmazz[ip]=sigmazz[ip]+rick[it];
	}

	__syncthreads();
}


__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_shared(
		float *rick, 
		float *lambda, float *lambda_plus_two_mu,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *sigmaxx, float *sigmazz,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int s_ix, int s_iz, int it,
		int inv_flag,  int Lc, float *rc
		)

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;		
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int ic;
	
	/***start the assignment of the shared memory***/
	__shared__ float s_vz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_vx[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_sigmazz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_sigmaxx[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_vz[s_ty][tx]=vz[ip];
	s_vx[ty][s_tx]=vx[ip];
	s_sigmazz[ty][tx]=sigmazz[ip];
	s_sigmaxx[ty][tx]=sigmaxx[ip];
		
	// in x_index
	if(tx<Lc)
		if(bx)
			s_vx[ty][tx]=vx[ip-Lc];
		else
			s_vx[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_vx[ty][tx+2*Lc]=vx[ip+Lc];
		else
			s_vx[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_vz[ty][tx]=vz[ip-Lc*ntx];
		else
			s_vz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_vz[ty+2*Lc][tx]=vz[ip+Lc*ntx];
		else
			s_vz[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/	
	
	
	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc&&ix<=ntx-Lc)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=s_rc[ic]*(s_vx[ty][s_tx+ic]-s_vx[ty][s_tx-ic-1])*one_over_dx;
			dvz_dz+=s_rc[ic]*(s_vz[s_ty+ic][tx]-s_vz[s_ty-ic-1][tx])*one_over_dz;
		}

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;


		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip])+
				lambda[ip]*(dvz_dz+phi_vz_z[ip]))*dt+
			s_sigmaxx[ty][tx];

		sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz+phi_vz_z[ip])+
				lambda[ip]*(dvx_dx+phi_vx_x[ip]))*dt+
			s_sigmazz[ty][tx];
	}

	if(iz==s_iz&&ix==s_ix)
	{
		sigmaxx[ip]=sigmaxx[ip]+rick[it];
		sigmazz[ip]=sigmazz[ip]+rick[it];
	}

	__syncthreads();
}


__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_4LT(
		float *rick, float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *lambda, float *lambda_plus_two_mu,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *sigmaxx, float *sigmaxxs,float *sigmazz,
		float *phi_vx_x, float *phi_vxs_x,float *phi_vz_z, float *phi_vzs_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int s_ix, int s_iz, int it,
		int inv_flag, int Lc, float *rc
		)

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;


	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;

	int ip=iz*ntx+ix;
	int ic;
     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;


	if(iz>=Lb&&iz<=ntz-Lb&&ix>=Lb&&ix<=ntx-Lb)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
		{
			dvx_dx+=Gp[index1*Lp+ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=Gp[index1*Lp+ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}
		dvx_dx+=Gp[index1*Lp+Lpo]*(vx[ip+ntx]-vx[ip+ntx-1]+vx[ip-ntx]-vx[ip-ntx-1])*one_over_dx;
		dvz_dz+=Gp[index1*Lp+Lpo]*(vz[ip+1]-vz[ip-ntx+1]+vz[ip-1]-vz[ip-ntx-1])*one_over_dz;

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;

		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip]+dvz_dz+phi_vz_z[ip]))*dt+
			sigmaxx[ip];

		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lso;ic++)
		{
			dvx_dx+=Gs[index2*Ls+ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=Gs[index2*Ls+ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}
		dvx_dx+=Gs[index2*Ls+Lso]*(vx[ip+ntx]-vx[ip+ntx-1]+vx[ip-ntx]-vx[ip-ntx-1])*one_over_dx;
		dvz_dz+=Gs[index2*Ls+Lso]*(vz[ip+1]-vz[ip-ntx+1]+vz[ip-1]-vz[ip-ntx-1])*one_over_dz;

		phi_vxs_x[ip]=b_x[ix]*phi_vxs_x[ip]+a_x[ix]*dvx_dx;
		phi_vzs_z[ip]=b_z[iz]*phi_vzs_z[ip]+a_z[iz]*dvz_dz;

		sigmaxxs[ip]=(lambda[ip]-lambda_plus_two_mu[ip])*(dvz_dz+phi_vzs_z[ip])*dt+
			sigmaxxs[ip];

		sigmazz[ip]=(lambda[ip]-lambda_plus_two_mu[ip])*(dvx_dx+phi_vxs_x[ip])*dt+
			sigmazz[ip];
	}

	/*if(iz==s_iz&&ix==s_ix)
	{
		sigmaxx[ip]=sigmaxx[ip]+rick[it];
		sigmaxxs[ip]=sigmaxxs[ip]+rick[it];
		sigmazz[ip]=sigmazz[ip]+rick[it];
	}*/

	__syncthreads();
}



__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz(
		float *mu,
		float *a_x_half, float *a_z_half,
		float *b_x_half, float *b_z_half,
		float *vx, float *vz, float *sigmaxz,
		float *phi_vx_z, float *phi_vz_x,
		int ntp, int ntx, int ntz,
		float dx, float dz, float dt,
		int inv_flag, int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int ic;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvz_dx+=rc[ic]*(vz[ip+ic+1]-vz[ip-ic])*one_over_dx;
			dvx_dz+=rc[ic]*(vx[ip+(ic+1)*ntx]-vx[ip-ic*ntx])*one_over_dz;
		}

		phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
		phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;


		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
				dvx_dz+phi_vx_z[ip])*dt+
			sigmaxz[ip];
	}

	__syncthreads();
}


__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_shared(
		float *mu,
		float *a_x_half, float *a_z_half,
		float *b_x_half, float *b_z_half,
		float *vx, float *vz, float *sigmaxz,
		float *phi_vx_z, float *phi_vz_x,
		int ntp, int ntx, int ntz,
		float dx, float dz, float dt,
		int inv_flag,  int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;		
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int ic;

	/***start the assignment of the shared memory***/
	__shared__ float s_vx[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_vz[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_sigmaxz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_vx[s_ty][tx]=vx[ip];
	s_vz[ty][s_tx]=vz[ip];
	s_sigmaxz[ty][tx]=sigmaxz[ip];
		
	// in x_index
	if(tx<Lc)
		if(bx)
			s_vz[ty][tx]=vz[ip-Lc];
		else
			s_vz[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_vz[ty][tx+2*Lc]=vz[ip+Lc];
		else
			s_vz[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_vx[ty][tx]=vx[ip-Lc*ntx];
		else
			s_vx[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_vx[ty+2*Lc][tx]=vx[ip+Lc*ntx];
		else
			s_vx[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/		
	
	
	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvz_dx+=s_rc[ic]*(s_vz[ty][s_tx+ic+1]-s_vz[ty][s_tx-ic])*one_over_dx;
			dvx_dz+=s_rc[ic]*(s_vx[s_ty+(ic+1)][tx]-s_vx[s_ty-ic][tx])*one_over_dz;
		}

		phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
		phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;


		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
				dvx_dz+phi_vx_z[ip])*dt+
			s_sigmaxz[ty][tx];
	}

	__syncthreads();
}


__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_4LT(
		float *mu,float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *a_x_half, float *a_z_half,
		float *b_x_half, float *b_z_half,
		float *vx, float *vz, float *sigmaxz,
		float *phi_vx_z, float *phi_vz_x,
		int ntp, int ntx, int ntz,
		float dx, float dz, float dt,
		int inv_flag, int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int ic;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	if(iz>=Lb-1&&iz<=ntz-Lb-1&&ix>=Lb-1&&ix<=ntx-Lb-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lso;ic++)
		{
			dvz_dx+=Gs[index2*Ls+ic]*(vz[ip+ic+1]-vz[ip-ic])*one_over_dx;
			dvx_dz+=Gs[index2*Ls+ic]*(vx[ip+(ic+1)*ntx]-vx[ip-ic*ntx])*one_over_dz;
		}

		dvz_dx+=Gs[index2*Ls+Lso]*(vz[ip+1+ntx]-vz[ip+ntx]+vz[ip+1-ntx]-vz[ip-ntx])*one_over_dx;
		dvx_dz+=Gs[index2*Ls+Lso]*(vx[ip+ntx+1]-vx[ip+1]+vx[ip+ntx-1]-vx[ip-1])*one_over_dz;

		phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
		phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;


		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
				dvx_dz+phi_vx_z[ip])*dt+
			sigmaxz[ip];
	}

	__syncthreads();
}



__global__ void fdtd_cpml_2d_GPU_kernel_ricker_and_seismogram(
		float *vx, float *vz, float *sigmaxx,
		float *sigmaxxs, float *sigmazz, float *sigmaxz,
		float *rick, float *seismogram_vx, float *seismogram_vz,
		int ntp, int ntx, int ntz,
		int it, int pml, int Lc,
		int s_ix, int s_iz, int *r_iz, int *r_ix, int r_n,
		int inv_flag
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;


	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ii;

	if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				seismogram_vx[it*r_n+ii]=vx[ip];
				seismogram_vz[it*r_n+ii]=vz[ip];
			}
		}
	}

	if(iz==s_iz&&ix==s_ix)
	{
		sigmaxx[ip]=sigmaxx[ip]+rick[it];
		sigmaxxs[ip]=sigmaxxs[ip]+rick[it];
		sigmazz[ip]=sigmazz[ip]+rick[it];
		//vz[ip]=vz[ip]+rick[it];
	}
	__syncthreads();

}


/*==========================================================

  This subroutine is used for calculating the forward wave 
  field of 2D in time domain.

  1.
  inv_flag==0----Calculate the observed seismograms of 
  Vx and Vz components...
  2.
  inv_flag==1----Calculate the synthetic seismograms of 
  Vx and Vz components and store the 
  borders of Vx and Vz used for constructing 
  the forward wavefields. 
  ===========================================================*/
extern "C"
void fdtd_cpml_2d_GPU_forward(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float *rc, float dx, float dz,
		float *rick, int itmax, float dt, int myid, float *vp, float *vs,
		float vp_min,float dvp,float vs_min,float dvs,float *Gp,float *Gs, int maxNp, int maxNs,
		int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, float *rho,
		float *lambda, float *mu, float *lambda_plus_two_mu,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half, 
		float *vx, float *vz, 
		float *sigmaxx, float *sigmaxxs, float *sigmazz, float *sigmaxz,
		int inv_flag)
{
	int it,ip;
	int ix,iz;
	int pmlc=pml+Lc;
	
	int i;
	Multistream plans[GPU_N];

	FILE *fp;
	char filename[50];

	size_t size_model=sizeof(float)*ntp;


	// allocate the memory for the device

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	
	}

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// Copy the vectors from the host to the device

		cudaMemcpyAsync(plan[i].d_r_ix,ss[is+i].r_ix,sizeof(int)*ss[is+i].r_n,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_r_iz,ss[is+i].r_iz,sizeof(int)*ss[is+i].r_n,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rc,rc,sizeof(float)*Lc,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_Gp,Gp,sizeof(float)*maxNp*(Lpo+1),cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_Gs,Gs,sizeof(float)*maxNs*(Lso+1),cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_lambda,lambda,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_mu,mu,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_lambda_plus_two_mu,lambda_plus_two_mu,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_vp,vp,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vs,vs,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rho,rho,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_a_x,a_x,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_x_half,a_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_z,a_z,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_z_half,a_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_b_x,b_x,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_x_half,b_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_z,b_z,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_z_half,b_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);

	}

	// =============================================================================

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

	//-----------------------------------------------------------------------//
	//=======================================================================//
	//-----------------------------------------------------------------------//
	////initialize the wavefields
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		initialize_wavefields<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
			 plan[i].d_vx, plan[i].d_vz,
			 plan[i].d_sigmaxx,  plan[i].d_sigmaxxs, plan[i].d_sigmazz, plan[i].d_sigmaxz,
			 plan[i].d_phi_vx_x, plan[i].d_phi_vxs_x, plan[i].d_phi_vx_z,
			 plan[i].d_phi_vz_z, plan[i].d_phi_vzs_z, plan[i].d_phi_vz_x,
			 plan[i].d_phi_sigmaxx_x, plan[i].d_phi_sigmaxxs_x, plan[i].d_phi_sigmaxz_z,
			 plan[i].d_phi_sigmaxz_x, plan[i].d_phi_sigmazz_z,
			 plan[i].d_phi_sigmaxx_z, plan[i].d_phi_sigmaxxs_z, plan[i].d_phi_sigmazz_x,
			 ntp, ntx, ntz 
			);
	}
	///////////////////////////////////
	for(it=0;it<itmax;it++)
	{
//		if(myid==0&&it%100==0)
//			printf("it == %d\n",it);

		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);

			fdtd_cpml_2d_GPU_kernel_vx_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, itmax, plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_a_x_half, plan[i].d_a_z, 
				 plan[i].d_b_x_half, plan[i].d_b_z, 
				 plan[i].d_vx, plan[i].d_sigmaxx, plan[i].d_sigmaxxs, plan[i].d_sigmaxz,
				 plan[i].d_phi_sigmaxx_x, plan[i].d_phi_sigmaxxs_x, plan[i].d_phi_sigmaxz_z, 
				 ntp, ntx, ntz, dx, dz, dt,
				 plan[i].d_seismogram_vx_syn, it, pml, Lc, plan[i].d_rc, plan[i].d_r_iz, plan[i].d_r_ix, ss[is+i].r_n,
				 inv_flag
				);

			fdtd_cpml_2d_GPU_kernel_vz_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, itmax,plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_a_x, plan[i].d_a_z_half,
				 plan[i].d_b_x, plan[i].d_b_z_half,
				 plan[i].d_vz, plan[i].d_sigmaxz,plan[i].d_sigmaxx, plan[i].d_sigmazz, 
				 plan[i].d_phi_sigmaxz_x, plan[i].d_phi_sigmaxx_z,plan[i].d_phi_sigmazz_z,
				 ntp, ntx, ntz, dx, dz, dt,
				 plan[i].d_seismogram_vz_syn, it, pml, Lc, plan[i].d_rc, plan[i].d_r_iz, plan[i].d_r_ix, ss[is+i].r_n,
				 inv_flag
				);

			fdtd_2d_GPU_kernel_borders_forward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_vx,
				 plan[i].d_vx_borders_up, plan[i].d_vx_borders_bottom,
				 plan[i].d_vx_borders_left, plan[i].d_vx_borders_right,
				 plan[i].d_vz,
				 plan[i].d_vz_borders_up, plan[i].d_vz_borders_bottom,
				 plan[i].d_vz_borders_left, plan[i].d_vz_borders_right,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, it, itmax, inv_flag
				);

			fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rick, plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_lambda, plan[i].d_lambda_plus_two_mu,
				 plan[i].d_a_x, plan[i].d_a_z, plan[i].d_b_x, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_sigmaxx, plan[i].d_sigmaxxs, plan[i].d_sigmazz,
				 plan[i].d_phi_vx_x, plan[i].d_phi_vxs_x, plan[i].d_phi_vz_z, plan[i].d_phi_vzs_z,
				 ntp, ntx, ntz, dx, dz, dt,
				 ss[is+i].s_ix, ss[is+i].s_iz, it,
				 inv_flag, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_sigmaxz_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_mu, plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_a_x_half, plan[i].d_a_z_half,
				 plan[i].d_b_x_half, plan[i].d_b_z_half,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_sigmaxz,
				 plan[i].d_phi_vx_z, plan[i].d_phi_vz_x,
				 ntp, ntx, ntz, dx, dz, dt,
				 inv_flag, Lc, plan[i].d_rc
				);
			
			fdtd_cpml_2d_GPU_kernel_ricker_and_seismogram<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_sigmaxx, plan[i].d_sigmaxxs, plan[i].d_sigmazz,plan[i].d_sigmaxz,
				 plan[i].d_rick, plan[i].d_seismogram_vx_syn,plan[i].d_seismogram_vz_syn,
				 ntp, ntx, ntz, it, pml, Lc,
				 ss[is+i].s_ix, ss[is+i].s_iz, plan[i].d_r_iz, plan[i].d_r_ix, ss[is+i].r_n,
				 inv_flag
				);

			/*if(i==0&&it%50==0)
			{
				cudaStreamSynchronize(plans[i].stream);
				cudaMemcpy(vx,plan[i].d_vx,sizeof(float)*ntp,cudaMemcpyDeviceToHost);

				sprintf(filename,"./output/%dvx%d.dat",it,myid);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{
						ip=iz*ntx+ix;
						fwrite(&vx[ip],sizeof(float),1,fp);
					}
				}
				fclose(fp);
			}*/
			
		}//end GPU_N

	}//end it

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		
		if(inv_flag==0)
		{
			cudaMemcpyAsync(plan[i].seismogram_vx_obs,plan[i].d_seismogram_vx_syn,sizeof(float)*rnmax*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].seismogram_vz_obs,plan[i].d_seismogram_vz_syn,sizeof(float)*rnmax*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
		}
		else
		{
			cudaMemcpyAsync(plan[i].seismogram_vx_syn,plan[i].d_seismogram_vx_syn,sizeof(float)*rnmax*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].seismogram_vz_syn,plan[i].d_seismogram_vz_syn,sizeof(float)*rnmax*itmax,cudaMemcpyDeviceToHost,plans[i].stream);

		}

		/////////Output The wavefields when Time=Itmax;////////
		if(inv_flag==1)
		{
		/*			cudaMemcpyAsync(plan[i].vx_borders_up,plan[i].d_vx_borders_up,sizeof(float)*Lc*itmax*(nx+1),cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].vx_borders_bottom,plan[i].d_vx_borders_bottom,sizeof(float)*Lc*itmax*(nx+1),cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].vx_borders_left,plan[i].d_vx_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc),cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].vx_borders_right,plan[i].d_vx_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc),cudaMemcpyDeviceToHost,plans[i].stream);

			cudaMemcpyAsync(plan[i].vz_borders_up,plan[i].d_vz_borders_up,sizeof(float)*Lc*itmax*nx,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].vz_borders_bottom,plan[i].d_vz_borders_bottom,sizeof(float)*Lc*itmax*nx,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].vz_borders_left,plan[i].d_vz_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc+1),cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].vz_borders_right,plan[i].d_vz_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc+1),cudaMemcpyDeviceToHost,plans[i].stream);*/
			/////////
			cudaMemcpyAsync(vx,plan[i].d_vx,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(vz,plan[i].d_vz,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(sigmaxx,plan[i].d_sigmaxx,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(sigmaxxs,plan[i].d_sigmaxxs,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(sigmazz,plan[i].d_sigmazz,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(sigmaxz,plan[i].d_sigmaxz,size_model,cudaMemcpyDeviceToHost,plans[i].stream);

			cudaStreamSynchronize(plans[i].stream);

			sprintf(filename,"./output/wavefield_itmax%d_%d.dat",i,myid);
			fp=fopen(filename,"wb");
			fwrite(&vx[0],sizeof(float),ntp,fp);
			fwrite(&vz[0],sizeof(float),ntp,fp);

			fwrite(&sigmaxx[0],sizeof(float),ntp,fp);
			fwrite(&sigmaxxs[0],sizeof(float),ntp,fp);
			fwrite(&sigmazz[0],sizeof(float),ntp,fp);
			fwrite(&sigmaxz[0],sizeof(float),ntp,fp);
			fclose(fp);

		}//end inv_flag
	}//end GPU

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}//end GPU

	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaStreamDestroy(plans[i].stream);
	}
}


/*==========================================================

  This subroutine is used for calculating the backward wave 
  field of 2D in time domain.

  ===========================================================*/

__global__ void fdtd_2d_GPU_kernel_sigmaxx_sigmazz_backward(
		float *rick, 
		float *lambda, float *lambda_plus_two_mu,
		float *vx, float *vz, float *sigmaxx, float *sigmazz,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt, int s_ix, int s_iz, int it
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=rc[ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=rc[ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}

		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx)+
				lambda[ip]*(dvz_dz))*dt+
			sigmaxx[ip];

		sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz)+
				lambda[ip]*(dvx_dx))*dt+
			sigmazz[ip];

	}

	if(iz==s_iz&&ix==s_ix)
	{
		sigmaxx[ip]=sigmaxx[ip]-rick[it+1];
		sigmazz[ip]=sigmazz[ip]-rick[it+1];
	}

	__syncthreads();

}

__global__ void fdtd_2d_GPU_kernel_sigmaxx_sigmazz_backward_shared(
		float *rick, 
		float *lambda, float *lambda_plus_two_mu,
		float *vx, float *vz, float *sigmaxx, float *sigmazz,
		int ntp, int ntx, int ntz, int pml,  int Lc, float *rc,
		float dx, float dz, float dt, int s_ix, int s_iz, int it
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;	
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

	/***start the assignment of the shared memory***/
	__shared__ float s_vz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_vx[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_sigmaxx[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_sigmazz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_vz[s_ty][tx]=vz[ip];
	s_vx[ty][s_tx]=vx[ip];
	s_sigmaxx[ty][tx]=sigmaxx[ip];
	s_sigmazz[ty][tx]=sigmazz[ip];		
	// in x_index
	if(tx<Lc)
		if(bx)
			s_vx[ty][tx]=vx[ip-Lc];
		else
			s_vx[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_vx[ty][tx+2*Lc]=vx[ip+Lc];
		else
			s_vx[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_vz[ty][tx]=vz[ip-Lc*ntx];
		else
			s_vz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_vz[ty+2*Lc][tx]=vz[ip+Lc*ntx];
		else
			s_vz[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/		
		
	
	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=s_rc[ic]*(s_vx[ty][s_tx+ic]-s_vx[ty][s_tx-(ic+1)])*one_over_dx;
			dvz_dz+=s_rc[ic]*(s_vz[s_ty+ic][tx]-s_vz[s_ty-(ic+1)][tx])*one_over_dz;
		}

		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx)+
				lambda[ip]*(dvz_dz))*dt+
			s_sigmaxx[ty][tx];

		sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz)+
				lambda[ip]*(dvx_dx))*dt+
			s_sigmazz[ty][tx];

	}

	if(iz==s_iz&&ix==s_ix)
	{
		sigmaxx[ip]=sigmaxx[ip]-rick[it+1];
		sigmazz[ip]=sigmazz[ip]-rick[it+1];
	}

	__syncthreads();

}


__global__ void fdtd_2d_GPU_kernel_sigmaxx_sigmazz_backward_4LT(
		float *rick, float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *lambda, float *lambda_plus_two_mu,
		float *vx, float *vz, float *sigmaxx, float *sigmaxxs, float *sigmazz,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt, int s_ix, int s_iz, int it
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;


	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
		{
			dvx_dx+=Gp[index1*Lp+ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=Gp[index1*Lp+ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}
		dvx_dx+=Gp[index1*Lp+Lpo]*(vx[ip+ntx]-vx[ip+ntx-1]+vx[ip-ntx]-vx[ip-ntx-1])*one_over_dx;
		dvz_dz+=Gp[index1*Lp+Lpo]*(vz[ip+1]-vz[ip-ntx+1]+vz[ip-1]-vz[ip-ntx-1])*one_over_dz;

		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+dvz_dz))*dt+
			sigmaxx[ip];

		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lso;ic++)
		{
			dvx_dx+=Gs[index2*Ls+ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=Gs[index2*Ls+ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}
		dvx_dx+=Gs[index2*Ls+Lso]*(vx[ip+ntx]-vx[ip+ntx-1]+vx[ip-ntx]-vx[ip-ntx-1])*one_over_dx;
		dvz_dz+=Gs[index2*Ls+Lso]*(vz[ip+1]-vz[ip-ntx+1]+vz[ip-1]-vz[ip-ntx-1])*one_over_dz;


		sigmaxxs[ip]=(lambda[ip]-lambda_plus_two_mu[ip])*(dvz_dz)*dt+
			sigmaxxs[ip];

		sigmazz[ip]=(lambda[ip]-lambda_plus_two_mu[ip])*(dvx_dx)*dt+
			sigmazz[ip];
	}

	/*if(iz==s_iz&&ix==s_ix)
	{
		sigmaxx[ip]=sigmaxx[ip]-rick[it+1];
		sigmaxxs[ip]=sigmaxxs[ip]-rick[it+1];
		sigmazz[ip]=sigmazz[ip]-rick[it+1];
	}*/


	__syncthreads();

}


__global__ void fdtd_2d_GPU_kernel_sigmaxz_backward(
		float *mu,
		float *vx, float *vz, float *sigmaxz,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvz_dx+=rc[ic]*(vz[ip+ic+1]-vz[ip-ic])*one_over_dx;
			dvx_dz+=rc[ic]*(vx[ip+(ic+1)*ntx]-vx[ip-ic*ntx])*one_over_dz;
		}

		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+
				dvx_dz)*dt+
			sigmaxz[ip];
	}

	__syncthreads();
}


__global__ void fdtd_2d_GPU_kernel_sigmaxz_backward_shared(
		float *mu,
		float *vx, float *vz, float *sigmaxz,
		int ntp, int ntx, int ntz, int pml,  int Lc, float *rc,
		float dx, float dz, float dt
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;	
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;


	/***start the assignment of the shared memory***/
	__shared__ float s_vx[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_vz[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_sigmaxz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_vx[s_ty][tx]=vx[ip];
	s_vz[ty][s_tx]=vz[ip];
	s_sigmaxz[ty][tx]=sigmaxz[ip];
	
	// in x_index
	if(tx<Lc)
		if(bx)
			s_vz[ty][tx]=vz[ip-Lc];
		else
			s_vz[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_vz[ty][tx+2*Lc]=vz[ip+Lc];
		else
			s_vz[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_vx[ty][tx]=vx[ip-Lc*ntx];
		else
			s_vx[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_vx[ty+2*Lc][tx]=vx[ip+Lc*ntx];
		else
			s_vx[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/			
	
	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvz_dx+=s_rc[ic]*(s_vz[ty][s_tx+ic+1]-s_vz[ty][s_tx-ic])*one_over_dx;
			dvx_dz+=s_rc[ic]*(s_vx[s_ty+(ic+1)][tx]-s_vx[s_ty-ic][tx])*one_over_dz;
		}

		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+
				dvx_dz)*dt+
			s_sigmaxz[ty][tx];
	}

	__syncthreads();
}


__global__ void fdtd_2d_GPU_kernel_sigmaxz_backward_4LT(
		float *mu, float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *vx, float *vz, float *sigmaxz,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lso;ic++)
		{
			dvz_dx+=Gs[index2*Ls+ic]*(vz[ip+ic+1]-vz[ip-ic])*one_over_dx;
			dvx_dz+=Gs[index2*Ls+ic]*(vx[ip+(ic+1)*ntx]-vx[ip-ic*ntx])*one_over_dz;
		}

		dvz_dx+=Gs[index2*Ls+Lso]*(vz[ip+1+ntx]-vz[ip+ntx]+vz[ip+1-ntx]-vz[ip-ntx])*one_over_dx;
		dvx_dz+=Gs[index2*Ls+Lso]*(vx[ip+ntx+1]-vx[ip+1]+vx[ip+ntx-1]-vx[ip-1])*one_over_dz;

		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+dvx_dz)*dt+
			sigmaxz[ip];
	}

	__syncthreads();
}

__global__ void fdtd_2d_GPU_kernel_vx_backward(
		float *rho,
		float *vx, float *sigmaxx, float *sigmaxz,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxz_dz;
	float one_over_rho_half_x;

	if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc-1&&ix<=ntx-pmlc-Lc-1)
	{   
		dsigmaxx_dx=0.0;
		dsigmaxz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxx_dx+=rc[ic]*(sigmaxx[ip+ic+1]-sigmaxx[ip-ic])*one_over_dx;
			dsigmaxz_dz+=rc[ic]*(sigmaxz[ip+ic*ntx]-sigmaxz[ip-(ic+1)*ntx])*one_over_dz;
		}

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx
				+dsigmaxz_dz)
			+vx[ip];
	}

	__syncthreads();
}

__global__ void fdtd_2d_GPU_kernel_vx_backward_shared(
		float *rho,
		float *vx, float *sigmaxx, float *sigmaxz,
		int ntp, int ntx, int ntz, int pml,  int Lc, float *rc,
		float dx, float dz, float dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;	
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxz_dz;
	float one_over_rho_half_x;


	/***start the assignment of the shared memory***/
	__shared__ float s_sigmaxz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_sigmaxx[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_vx[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_sigmaxz[s_ty][tx]=sigmaxz[ip];
	s_sigmaxx[ty][s_tx]=sigmaxx[ip];
	s_vx[ty][tx]=vx[ip];
	
	// in x_indexs
	if(tx<Lc)
		if(bx)
			s_sigmaxx[ty][tx]=sigmaxx[ip-Lc];
		else
			s_sigmaxx[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_sigmaxx[ty][tx+2*Lc]=sigmaxx[ip+Lc];
		else
			s_sigmaxx[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_sigmaxz[ty][tx]=sigmaxz[ip-Lc*ntx];
		else
			s_sigmaxz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_sigmaxz[ty+2*Lc][tx]=sigmaxz[ip+Lc*ntx];
		else
			s_sigmaxz[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/	
	
	if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc-1&&ix<=ntx-pmlc-Lc-1)
	{   
		dsigmaxx_dx=0.0;
		dsigmaxz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxx_dx+=s_rc[ic]*(s_sigmaxx[ty][s_tx+ic+1]-s_sigmaxx[ty][s_tx-ic])*one_over_dx;
			dsigmaxz_dz+=s_rc[ic]*(s_sigmaxz[s_ty+ic][tx]-s_sigmaxz[s_ty-(ic+1)][tx])*one_over_dz;
		}

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx
				+dsigmaxz_dz)
			+s_vx[ty][tx];
	}

	__syncthreads();
}

__global__ void fdtd_2d_GPU_kernel_vx_backward_4LT(
		float *rho,float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *vx, float *sigmaxx, float *sigmaxxs, float *sigmaxz,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxxs_dx,dsigmaxz_dz;
	float one_over_rho_half_x;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc-1&&ix<=ntx-pmlc-Lc-1)
	{
		dsigmaxx_dx=0.0;
		dsigmaxxs_dx=0.0;
		dsigmaxz_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
			dsigmaxx_dx+=Gp[index1*Lp+ic]*(sigmaxx[ip+ic+1]-sigmaxx[ip-ic])*one_over_dx;

		dsigmaxx_dx+=Gp[index1*Lp+Lpo]*(sigmaxx[ip+ntx+1]-sigmaxx[ip+ntx]+sigmaxx[ip-ntx+1]-sigmaxx[ip-ntx])*one_over_dx;	//x direction

		for(ic=0;ic<Lso;ic++)
		{
			dsigmaxxs_dx+=Gs[index2*Ls+ic]*(sigmaxxs[ip+ic+1]-sigmaxxs[ip-ic])*one_over_dx;
			dsigmaxz_dz+=Gs[index2*Ls+ic]*(sigmaxz[ip+ic*ntx]-sigmaxz[ip-(ic+1)*ntx])*one_over_dz;
		}

		dsigmaxxs_dx+=Gs[index2*Ls+Lso]*(sigmaxxs[ip+ntx+1]-sigmaxxs[ip+ntx]+sigmaxxs[ip-ntx+1]-sigmaxxs[ip-ntx])*one_over_dx;	//x direction
		dsigmaxz_dz+=Gs[index2*Ls+Lso]*(sigmaxz[ip+1]-sigmaxz[ip-ntx+1]+sigmaxz[ip-1]-sigmaxz[ip-ntx-1])*one_over_dz;

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+dsigmaxz_dz+dsigmaxxs_dx)
			+vx[ip];
	}

	__syncthreads();
}



__global__ void fdtd_2d_GPU_kernel_vz_backward(
		float *rho,
		float *vz, float *sigmaxz, float *sigmazz,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;


	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmazz_dz;

	float one_over_rho_half_z;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

	//if(iz>=pmlc&&iz<=ntz-pmlc-2&&ix>=pmlc+1&&ix<=ntx-pmlc-2)
	if(iz>=pmlc+Lc-1&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc&&ix<=ntx-pmlc-Lc-1)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxz_dx+=rc[ic]*(sigmaxz[ip+ic]-sigmaxz[ip-(ic+1)])*one_over_dx;
			dsigmazz_dz+=rc[ic]*(sigmazz[ip+(ic+1)*ntx]-sigmazz[ip-ic*ntx])*one_over_dz;
		}

		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx
				+dsigmazz_dz)
			+vz[ip];
	}

	__syncthreads();

}

__global__ void fdtd_2d_GPU_kernel_vz_backward_shared(
		float *rho,
		float *vz, float *sigmaxz, float *sigmazz,
		int ntp, int ntx, int ntz, int pml,  int Lc, float *rc,
		float dx, float dz, float dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;		
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;


	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmazz_dz;

	float one_over_rho_half_z;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;

	/***start the assignment of the shared memory***/
	__shared__ float s_sigmazz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_sigmaxz[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_vz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_sigmazz[s_ty][tx]=sigmazz[ip];
	s_sigmaxz[ty][s_tx]=sigmaxz[ip];
	s_vz[ty][tx]=vz[ip];
	
	// in x_index
	if(tx<Lc)
		if(bx)
			s_sigmaxz[ty][tx]=sigmaxz[ip-Lc];
		else
			s_sigmaxz[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_sigmaxz[ty][tx+2*Lc]=sigmaxz[ip+Lc];
		else
			s_sigmaxz[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_sigmazz[ty][tx]=sigmazz[ip-Lc*ntx];
		else
			s_sigmazz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_sigmazz[ty+2*Lc][tx]=sigmazz[ip+Lc*ntx];
		else
			s_sigmazz[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/	
		
	
	//if(iz>=pmlc&&iz<=ntz-pmlc-2&&ix>=pmlc+1&&ix<=ntx-pmlc-2)
	if(iz>=pmlc+Lc-1&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc&&ix<=ntx-pmlc-Lc-1)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxz_dx+=s_rc[ic]*(s_sigmaxz[ty][s_tx+ic]-s_sigmaxz[ty][s_tx-(ic+1)])*one_over_dx;
			dsigmazz_dz+=s_rc[ic]*(s_sigmazz[s_ty+(ic+1)][tx]-s_sigmazz[s_ty-ic][tx])*one_over_dz;
		}

		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx
				+dsigmazz_dz)
			+s_vz[ty][tx];
	}

	__syncthreads();

}


__global__ void fdtd_2d_GPU_kernel_vz_backward_4LT(
		float *rho,float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *vz, float *sigmaxz, float *sigmaxx,float *sigmazz,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;


	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmaxx_dz,dsigmazz_dz;

	float one_over_rho_half_z;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;
	int ic;


     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	//if(iz>=pmlc&&iz<=ntz-pmlc-2&&ix>=pmlc+1&&ix<=ntx-pmlc-2)
	if(iz>=pmlc+Lc-1&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc&&ix<=ntx-pmlc-Lc-1)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;
		dsigmaxx_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
		{
			dsigmaxx_dz+=Gp[index1*Lp+ic]*(sigmaxx[ip+(ic+1)*ntx]-sigmaxx[ip-ic*ntx])*one_over_dz;
		}
		dsigmaxx_dz+=Gp[index1*Lp+Lpo]*(sigmaxx[ip+ntx+1]-sigmaxx[ip+1]+sigmaxx[ip+ntx-1]-sigmaxx[ip-1])*one_over_dz;


		for(ic=0;ic<Lso;ic++)
		{
			dsigmaxz_dx+=Gs[index2*Ls+ic]*(sigmaxz[ip+ic]-sigmaxz[ip-(ic+1)])*one_over_dx;
			dsigmazz_dz+=Gs[index2*Ls+ic]*(sigmazz[ip+(ic+1)*ntx]-sigmazz[ip-ic*ntx])*one_over_dz;
		}
		dsigmaxz_dx+=Gs[index2*Ls+Lso]*(sigmaxz[ip+ntx]-sigmaxz[ip+ntx-1]+sigmaxz[ip-ntx]-sigmaxz[ip-ntx-1])*one_over_dx;
		dsigmazz_dz+=Gs[index2*Ls+Lso]*(sigmazz[ip+ntx+1]-sigmazz[ip+1]+sigmazz[ip+ntx-1]-sigmazz[ip-1])*one_over_dz;


		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+dsigmazz_dz+dsigmaxx_dz)
			+vz[ip];
	}

	__syncthreads();

}


__global__ void fdtd_2d_GPU_kernel_borders_backward
(
 float *vx,
 float *vx_borders_up, float *vx_borders_bottom,
 float *vx_borders_left, float *vx_borders_right,
 float *vz,
 float *vz_borders_up, float *vz_borders_bottom,
 float *vz_borders_left, float *vz_borders_right,
 int ntp, int ntx, int ntz, int pml, int Lc, float *rc, int it, int itmax
 )
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;

	int nx=ntx-2*pmlc;
	int nz=ntz-2*pmlc;

	if(iz>=pmlc&&iz<=pmlc+Lc-1&&ix>=pmlc-1&&ix<=ntx-pmlc-1)
	{
		vx[ip]=vx_borders_up[it*Lc*(nx+1)+(iz-pmlc)*(nx+1)+ix-pmlc+1];
	}
	if(iz>=ntz-pmlc-Lc&&iz<=ntz-pmlc-1&&ix>=pmlc-1&&ix<=ntx-pmlc-1)
	{
		vx[ip]=vx_borders_bottom[it*Lc*(nx+1)+(iz-ntz+pmlc+Lc)*(nx+1)+ix-pmlc+1];
	}

	if(ix>=pmlc-1&&ix<=pmlc+Lc-2&&iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1)
	{
		vx[ip]=vx_borders_left[it*Lc*(nz-2*Lc)+(iz-pmlc-Lc)*Lc+ix-pmlc+1];
	}
	if(ix>=ntx-pmlc-Lc&&ix<=ntx-pmlc-1&&iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1)
	{
		vx[ip]=vx_borders_right[it*Lc*(nz-2*Lc)+(iz-pmlc-Lc)*Lc+ix-ntx+pmlc+Lc];
	}

	//////////////////////////////////////////////////////////////

	if(iz>=pmlc-1&&iz<=pmlc+Lc-2&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		vz[ip]=vz_borders_up[it*Lc*nx+(iz-pmlc+1)*nx+ix-pmlc];
	}
	if(iz>=ntz-pmlc-Lc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		vz[ip]=vz_borders_bottom[it*Lc*nx+(iz-ntz+pmlc+Lc)*nx+ix-pmlc];
	}

	if(ix>=pmlc&&ix<=pmlc+Lc-1&&iz>=pmlc+Lc-1&&iz<=ntz-pmlc-Lc-1)
	{
		vz[ip]=vz_borders_left[it*Lc*(nz-2*Lc+1)+(iz-pmlc-Lc+1)*Lc+ix-pmlc];
	}
	if(ix>=ntx-pmlc-Lc&&ix<=ntx-pmlc-1&&iz>=pmlc+Lc-1&&iz<=ntz-pmlc-Lc-1)
	{
		vz[ip]=vz_borders_right[it*Lc*(nz-2*Lc+1)+(iz-pmlc-Lc+1)*Lc+ix-ntx+pmlc+Lc];
	}

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_ricker_backward(
		float *vx, float *vz, float *sigmaxx,
		float *sigmaxxs, float *sigmazz, float *sigmaxz,
		float *rick,
		int ntp, int ntx, int ntz, int it, int pml, int Lc,
		int s_ix, int s_iz
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;


	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ii;

	if(iz==s_iz&&ix==s_ix)
	{
		sigmaxx[ip]=sigmaxx[ip]-rick[it+1];
		sigmaxxs[ip]=sigmaxxs[ip]-rick[it+1];
		sigmazz[ip]=sigmazz[ip]-rick[it+1];
		//vz[ip]=vz[ip]+rick[it];
	}
	__syncthreads();

}



///////////////////////////////////////////////////////////////////////
/////////////////////////receiver wavefield////////////////////////////
///////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
/////////////////BACKWARD_RECEIVERS_WAVEFIELD///////////////////
////////////////////////////////////////////////////////////////


__global__ void fdtd_cpml_2d_GPU_kernel_vx_backward(
		float *rho,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *sigmaxx, float *sigmaxz,
		float *phi_sigmaxx_x, float *phi_sigmaxz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vx_rms, int it, int pml, int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ic,ii;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxz_dz;
	float one_over_rho_half_x;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dsigmaxx_dx=0.0;
		dsigmaxz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxx_dx+=rc[ic]*(sigmaxx[ip+ic+1]-sigmaxx[ip-ic])*one_over_dx;
			dsigmaxz_dz+=rc[ic]*(sigmaxz[ip+ic*ntx]-sigmaxz[ip-(ic+1)*ntx])*one_over_dz;
		}

		phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
		phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));


		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
				+dsigmaxz_dz+phi_sigmaxz_z[ip])
			+vx[ip];


	}

	if(iz>=pml&&iz<pml+10*Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				vx[ip]=seismogram_vx_rms[it*r_n+ii];
			}
		}
	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vx_backward_shared(
		float *rho,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *sigmaxx, float *sigmaxz,
		float *phi_sigmaxx_x, float *phi_sigmaxz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vx_rms, int it, int pml,  int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;	
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ic,ii;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxz_dz;
	float one_over_rho_half_x;


	/***start the assignment of the shared memory***/
	__shared__ float s_sigmaxz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_sigmaxx[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_vx[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_sigmaxz[s_ty][tx]=sigmaxz[ip];
	s_sigmaxx[ty][s_tx]=sigmaxx[ip];
	s_vx[ty][tx]=vx[ip];
	
	// in x_index
	if(tx<Lc)
		if(bx)
			s_sigmaxx[ty][tx]=sigmaxx[ip-Lc];
		else
			s_sigmaxx[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_sigmaxx[ty][tx+2*Lc]=sigmaxx[ip+Lc];
		else
			s_sigmaxx[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_sigmaxz[ty][tx]=sigmaxz[ip-Lc*ntx];
		else
			s_sigmaxz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_sigmaxz[ty+2*Lc][tx]=sigmaxz[ip+Lc*ntx];
		else
			s_sigmaxz[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/		
	
	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dsigmaxx_dx=0.0;
		dsigmaxz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxx_dx+=s_rc[ic]*(s_sigmaxx[ty][s_tx+ic+1]-s_sigmaxx[ty][s_tx-ic])*one_over_dx;
			dsigmaxz_dz+=s_rc[ic]*(s_sigmaxz[s_ty+ic][tx]-s_sigmaxz[s_ty-(ic+1)][tx])*one_over_dz;
		}

		phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
		phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));


		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
				+dsigmaxz_dz+phi_sigmaxz_z[ip])
			+s_vx[ty][tx];
	}

	if(iz>=pml&&iz<pml+10*Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)	
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				vx[ip]=seismogram_vx_rms[it*r_n+ii];
			}
		}
			
		
	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vx_backward_4LT(
		float *rho,float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *sigmaxx, float *sigmaxxs,float *sigmaxz,
		float *phi_sigmaxx_x, float *phi_sigmaxxs_x, float *phi_sigmaxz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vx_rms, int it, int pml, int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ic,ii;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmaxxs_dx,dsigmaxz_dz;
	float one_over_rho_half_x;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	if(iz>=Lb&&iz<=ntz-Lb&&ix>=Lb-1&&ix<=ntx-Lb-1)
	{
		dsigmaxx_dx=0.0;
		dsigmaxxs_dx=0.0;
		dsigmaxz_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
			dsigmaxx_dx+=Gp[index1*Lp+ic]*(sigmaxx[ip+ic+1]-sigmaxx[ip-ic])*one_over_dx;

		dsigmaxx_dx+=Gp[index1*Lp+Lpo]*(sigmaxx[ip+ntx+1]-sigmaxx[ip+ntx]+sigmaxx[ip-ntx+1]-sigmaxx[ip-ntx])*one_over_dx;	//x direction

		for(ic=0;ic<Lso;ic++)
		{
			dsigmaxxs_dx+=Gs[index2*Ls+ic]*(sigmaxxs[ip+ic+1]-sigmaxxs[ip-ic])*one_over_dx;
			dsigmaxz_dz+=Gs[index2*Ls+ic]*(sigmaxz[ip+ic*ntx]-sigmaxz[ip-(ic+1)*ntx])*one_over_dz;
		}

		dsigmaxxs_dx+=Gs[index2*Ls+Lso]*(sigmaxxs[ip+ntx+1]-sigmaxxs[ip+ntx]+sigmaxxs[ip-ntx+1]-sigmaxxs[ip-ntx])*one_over_dx;	//x direction
		dsigmaxz_dz+=Gs[index2*Ls+Lso]*(sigmaxz[ip+1]-sigmaxz[ip-ntx+1]+sigmaxz[ip-1]-sigmaxz[ip-ntx-1])*one_over_dz;

		phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
		phi_sigmaxxs_x[ip]=b_x_half[ix]*phi_sigmaxxs_x[ip]+a_x_half[ix]*dsigmaxxs_dx;
		phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
				+dsigmaxz_dz+phi_sigmaxz_z[ip]+dsigmaxxs_dx+phi_sigmaxxs_x[ip])
			+vx[ip];
	}

	/*if(iz>=pml&&iz<pml+10*Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				vx[ip]=seismogram_vx_rms[it*r_n+ii];
			}
		}*/
	__syncthreads();

}

__global__ void fdtd_cpml_2d_GPU_kernel_vz_backward(
		float *rho,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *sigmaxz, float *sigmazz, 
		float *phi_sigmaxz_x, float *phi_sigmazz_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vz_rms, int it, int pml, int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmazz_dz;

	float one_over_rho_half_z;

	int ic,ii;
	int ip=iz*ntx+ix;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc&&ix<=ntx-Lc)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxz_dx+=rc[ic]*(sigmaxz[ip+ic]-sigmaxz[ip-(ic+1)])*one_over_dx;
			dsigmazz_dz+=rc[ic]*(sigmazz[ip+(ic+1)*ntx]-sigmazz[ip-ic*ntx])*one_over_dz;
		}

		phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
		phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;


		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));


		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
				+dsigmazz_dz+phi_sigmazz_z[ip])
			+vz[ip];

	}
	

	if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc&&ix<=ntx-Lc)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				vz[ip]=seismogram_vz_rms[it*r_n+ii];
			}
		}
	__syncthreads();

}



__global__ void fdtd_cpml_2d_GPU_kernel_vz_backward_shared(
		float *rho,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *sigmaxz, float *sigmazz, 
		float *phi_sigmaxz_x, float *phi_sigmazz_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vz_rms, int it, int pml,  int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmazz_dz;

	float one_over_rho_half_z;

	int ic,ii;
	int ip=iz*ntx+ix;
	
	/***start the assignment of the shared memory***/
	__shared__ float s_vz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_sigmazz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_sigmaxz[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_rc[Lcc];
		
	s_vz[ty][tx]=vz[ip];
	s_sigmazz[s_ty][tx]=sigmazz[ip];
	s_sigmaxz[ty][s_tx]=sigmaxz[ip];
		
	// in x_index
	if(tx<Lc)
		if(bx)
			s_sigmaxz[ty][tx]=sigmaxz[ip-Lc];
		else
			s_sigmaxz[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_sigmaxz[ty][tx+2*Lc]=sigmaxz[ip+Lc];
		else
			s_sigmaxz[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_sigmazz[ty][tx]=sigmazz[ip-Lc*ntx];
		else
			s_sigmazz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_sigmazz[ty+2*Lc][tx]=sigmazz[ip+Lc*ntx];
		else
			s_sigmazz[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/	
	

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc&&ix<=ntx-Lc)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxz_dx+=s_rc[ic]*(s_sigmaxz[ty][s_tx+ic]-s_sigmaxz[ty][s_tx-(ic+1)])*one_over_dx;
			dsigmazz_dz+=s_rc[ic]*(s_sigmazz[s_ty+(ic+1)][tx]-s_sigmazz[s_ty-ic][tx])*one_over_dz;
		}

		phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
		phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;


		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));


		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
				+dsigmazz_dz+phi_sigmazz_z[ip])
			+s_vz[ty][tx];

	}

	if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc&&ix<=ntx-Lc)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				vz[ip]=seismogram_vz_rms[it*r_n+ii];
			}
		}
	__syncthreads();

}



__global__ void fdtd_cpml_2d_GPU_kernel_vz_backward_4LT(
		float *rho,float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *sigmaxz, float *sigmaxx, float *sigmazz, 
		float *phi_sigmaxz_x, float *phi_sigmaxx_z,float *phi_sigmazz_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vz_rms, int it, int pml, int Lc, float *rc,
		int *r_iz, int *r_ix, int r_n
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxz_dx,dsigmaxx_dz,dsigmazz_dz;

	float one_over_rho_half_z;

	int ic,ii;
	int ip=iz*ntx+ix;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	if(iz>=Lb-1&&iz<=ntz-Lb-1&&ix>=Lb&&ix<=ntx-Lb)
	{
		dsigmaxz_dx=0.0;
		dsigmazz_dz=0.0;
		dsigmaxx_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
		{
			dsigmaxx_dz+=Gp[index1*Lp+ic]*(sigmaxx[ip+(ic+1)*ntx]-sigmaxx[ip-ic*ntx])*one_over_dz;
		}
		dsigmaxx_dz+=Gp[index1*Lp+Lpo]*(sigmaxx[ip+ntx+1]-sigmaxx[ip+1]+sigmaxx[ip+ntx-1]-sigmaxx[ip-1])*one_over_dz;


		for(ic=0;ic<Lso;ic++)
		{
			dsigmaxz_dx+=Gs[index2*Ls+ic]*(sigmaxz[ip+ic]-sigmaxz[ip-(ic+1)])*one_over_dx;
			dsigmazz_dz+=Gs[index2*Ls+ic]*(sigmazz[ip+(ic+1)*ntx]-sigmazz[ip-ic*ntx])*one_over_dz;
		}
		dsigmaxz_dx+=Gs[index2*Ls+Lso]*(sigmaxz[ip+ntx]-sigmaxz[ip+ntx-1]+sigmaxz[ip-ntx]-sigmaxz[ip-ntx-1])*one_over_dx;
		dsigmazz_dz+=Gs[index2*Ls+Lso]*(sigmazz[ip+ntx+1]-sigmazz[ip+1]+sigmazz[ip+ntx-1]-sigmazz[ip-1])*one_over_dz;

		phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
		phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;
		phi_sigmaxx_z[ip]=b_z_half[iz]*phi_sigmaxx_z[ip]+a_z_half[iz]*dsigmaxx_dz;


		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
				+dsigmazz_dz+phi_sigmazz_z[ip]+dsigmaxx_dz+phi_sigmaxx_z[ip])
			+vz[ip];
	}
	

	/*if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc&&ix<=ntx-Lc)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				vz[ip]=seismogram_vz_rms[it*r_n+ii];
			}
		}*/
	__syncthreads();

}




__global__ void fdtd_cpml_2d_GPU_kernel_seismogram_backward(
		float *vx, float *vz, float *sigmaxx, float *sigmaxxs, float *sigmazz, float *sigmaxz, 
		float *seismogram_vx_rms, float *seismogram_vz_rms, 
		int ntp, int ntx, int ntz,
		int it, int pml, int Lc,int *r_iz, int *r_ix, int r_n
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ii;
	int ip=iz*ntx+ix;

	if(iz>=pml&&iz<=pml+10*Lc&&ix>=Lc&&ix<=ntx-Lc)
		for(ii=0;ii<r_n;ii++)
		{
			if(ix==r_ix[ii]&&iz==r_iz[ii])
			{
				vx[ip]=seismogram_vx_rms[it*r_n+ii];
				vz[ip]=seismogram_vz_rms[it*r_n+ii];
			}
		}
	__syncthreads();

}





__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_backward(
		float *lambda, float *lambda_plus_two_mu,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *sigmaxx, float *sigmazz,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int ic;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc&&ix<=ntx-Lc)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=rc[ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=rc[ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;


		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip])+
				lambda[ip]*(dvz_dz+phi_vz_z[ip]))*dt+
			sigmaxx[ip];

		sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz+phi_vz_z[ip])+
				lambda[ip]*(dvx_dx+phi_vx_x[ip]))*dt+
			sigmazz[ip];

	}

	__syncthreads();
}



__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_backward_shared(
		float *lambda, float *lambda_plus_two_mu,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *sigmaxx, float *sigmazz,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,  int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;
	
	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;	
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int ic;
	
	/***start the assignment of the shared memory***/
	__shared__ float s_vz[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_vx[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_sigmazz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_sigmaxx[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_vz[s_ty][tx]=vz[ip];
	s_vx[ty][s_tx]=vx[ip];
	s_sigmazz[ty][tx]=sigmazz[ip];
	s_sigmaxx[ty][tx]=sigmaxx[ip];
		
	// in x_index
	if(tx<Lc)
		if(bx)
			s_vx[ty][tx]=vx[ip-Lc];
		else
			s_vx[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_vx[ty][tx+2*Lc]=vx[ip+Lc];
		else
			s_vx[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_vz[ty][tx]=vz[ip-Lc*ntx];
		else
			s_vz[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_vz[ty+2*Lc][tx]=vz[ip+Lc*ntx];
		else
			s_vz[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/	
		
	
	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc&&ix<=ntx-Lc)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=s_rc[ic]*(s_vx[ty][s_tx+ic]-s_vx[ty][s_tx-(ic+1)])*one_over_dx;
			dvz_dz+=s_rc[ic]*(s_vz[s_ty+ic][tx]-s_vz[s_ty-(ic+1)][tx])*one_over_dz;
		}

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;


		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip])+
				lambda[ip]*(dvz_dz+phi_vz_z[ip]))*dt+
			s_sigmaxx[ty][tx];

		sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz+phi_vz_z[ip])+
				lambda[ip]*(dvx_dx+phi_vx_x[ip]))*dt+
			s_sigmazz[ty][tx];

	}

	__syncthreads();
}



__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_backward_4LT(
		float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *lambda, float *lambda_plus_two_mu,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *sigmaxx, float *sigmaxxs,float *sigmazz,
		float *phi_vx_x, float *phi_vxs_x,float *phi_vz_z, float *phi_vzs_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int ic;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;


	if(iz>=Lb&&iz<=ntz-Lb&&ix>=Lb&&ix<=ntx-Lb)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

                index1=(int)((vp[ip]-vpmin)/dvp+0.5);
                index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lpo;ic++)
		{
			dvx_dx+=Gp[index1*Lp+ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=Gp[index1*Lp+ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}
		dvx_dx+=Gp[index1*Lp+Lpo]*(vx[ip+ntx]-vx[ip+ntx-1]+vx[ip-ntx]-vx[ip-ntx-1])*one_over_dx;
		dvz_dz+=Gp[index1*Lp+Lpo]*(vz[ip+1]-vz[ip-ntx+1]+vz[ip-1]-vz[ip-ntx-1])*one_over_dz;

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;

		sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip]+dvz_dz+phi_vz_z[ip]))*dt+
			sigmaxx[ip];

		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lso;ic++)
		{
			dvx_dx+=Gs[index2*Ls+ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=Gs[index2*Ls+ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}
		dvx_dx+=Gs[index2*Ls+Lso]*(vx[ip+ntx]-vx[ip+ntx-1]+vx[ip-ntx]-vx[ip-ntx-1])*one_over_dx;
		dvz_dz+=Gs[index2*Ls+Lso]*(vz[ip+1]-vz[ip-ntx+1]+vz[ip-1]-vz[ip-ntx-1])*one_over_dz;

		phi_vxs_x[ip]=b_x[ix]*phi_vxs_x[ip]+a_x[ix]*dvx_dx;
		phi_vzs_z[ip]=b_z[iz]*phi_vzs_z[ip]+a_z[iz]*dvz_dz;

		sigmaxxs[ip]=(lambda[ip]-lambda_plus_two_mu[ip])*(dvz_dz+phi_vzs_z[ip])*dt+
			sigmaxxs[ip];

		sigmazz[ip]=(lambda[ip]-lambda_plus_two_mu[ip])*(dvx_dx+phi_vxs_x[ip])*dt+
			sigmazz[ip];

	}

	__syncthreads();
}




__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_backward(
		float *mu,
		float *a_x_half, float *a_z_half,
		float *b_x_half, float *b_z_half,
		float *vx, float *vz, float *sigmaxz,
		float *phi_vx_z, float *phi_vz_x,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int ic;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvz_dx+=rc[ic]*(vz[ip+ic+1]-vz[ip-ic])*one_over_dx;
			dvx_dz+=rc[ic]*(vx[ip+(ic+1)*ntx]-vx[ip-ic*ntx])*one_over_dz;
		}

		phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
		phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;


		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
				dvx_dz+phi_vx_z[ip])*dt+
			sigmaxz[ip];
	}

	__syncthreads();
}


__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_backward_shared(
		float *mu,
		float *a_x_half, float *a_z_half,
		float *b_x_half, float *b_z_half,
		float *vx, float *vz, float *sigmaxz,
		float *phi_vx_z, float *phi_vz_x,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,   int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	//thread's x-index in the shared memory tile
	int s_tx=tx+Lc;
	int s_ty=ty+Lc;		
	
	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int ic;

	/***start the assignment of the shared memory***/
	__shared__ float s_vx[BLOCK_SIZE+2*Lcc][BLOCK_SIZE];
	__shared__ float s_vz[BLOCK_SIZE][BLOCK_SIZE+2*Lcc];
	__shared__ float s_sigmaxz[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_rc[Lcc];
		
	s_vx[s_ty][tx]=vx[ip];
	s_vz[ty][s_tx]=vz[ip];
	s_sigmaxz[ty][tx]=sigmaxz[ip];
		
	// in x_index
	if(tx<Lc)
		if(bx)
			s_vz[ty][tx]=vz[ip-Lc];
		else
			s_vz[ty][tx]=0.0;
	if(tx>blockDim.x-Lc-1)
		if(bx<gridDim.x-1)
			s_vz[ty][tx+2*Lc]=vz[ip+Lc];
		else
			s_vz[ty][tx+2*Lc]=0.0;
	// in y_index		
	if(ty<Lc)
		if(by)
			s_vx[ty][tx]=vx[ip-Lc*ntx];
		else
			s_vx[ty][tx]=0.0;
	if(ty>blockDim.y-Lc-1)
		if(by<gridDim.y-1)
			s_vx[ty+2*Lc][tx]=vx[ip+Lc*ntx];
		else
			s_vx[ty+2*Lc][tx]=0.0;	
		
	//s_rc ---  rc	
	if(tx==0&&ty==0)
		for(ic=0;ic<Lc;ic++)
			s_rc[ic]=rc[ic];
		
	__syncthreads();
	/***finish the assignment of the shared memory***/		
		
	
	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvz_dx+=s_rc[ic]*(s_vz[ty][s_tx+ic+1]-s_vz[ty][s_tx-ic])*one_over_dx;
			dvx_dz+=s_rc[ic]*(s_vx[s_ty+(ic+1)][tx]-s_vx[s_ty-ic][tx])*one_over_dz;
		}

		phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
		phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;


		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
				dvx_dz+phi_vx_z[ip])*dt+
			s_sigmaxz[ty][tx];
	}

	__syncthreads();
}


__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_backward_4LT(
		float *mu,float *Gp, float *Gs,
		float *vp, float vpmin, float dvp,
		float *vs, float vsmin, float dvs,
		float *a_x_half, float *a_z_half,
		float *b_x_half, float *b_z_half,
		float *vx, float *vz, float *sigmaxz,
		float *phi_vx_z, float *phi_vz_x,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;
	float mu_half_x_half_z;

	int ip=iz*ntx+ix;
	int ic;

     	int Lb=(Lso>Lpo?Lso:Lpo);
     	int index1,index2;
	int Lp=Lpo+1;
	int Ls=Lso+1;

	if(iz>=Lb-1&&iz<=ntz-Lb-1&&ix>=Lb-1&&ix<=ntx-Lb-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		index2=(int)((vs[ip]-vsmin)/dvs+0.5); 

		for(ic=0;ic<Lso;ic++)
		{
			dvz_dx+=Gs[index2*Ls+ic]*(vz[ip+ic+1]-vz[ip-ic])*one_over_dx;
			dvx_dz+=Gs[index2*Ls+ic]*(vx[ip+(ic+1)*ntx]-vx[ip-ic*ntx])*one_over_dz;
		}

		dvz_dx+=Gs[index2*Ls+Lso]*(vz[ip+1+ntx]-vz[ip+ntx]+vz[ip+1-ntx]-vz[ip-ntx])*one_over_dx;
		dvx_dz+=Gs[index2*Ls+Lso]*(vx[ip+ntx+1]-vx[ip+1]+vx[ip+ntx-1]-vx[ip-1])*one_over_dz;

		phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
		phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;


		mu_half_x_half_z=0.25*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);

		sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
				dvx_dz+phi_vx_z[ip])*dt+
			sigmaxz[ip];
	}

	__syncthreads();
}

__global__ void fdtd_cpml_2d_GPU_kernel_vx_mine_backward(
		float *rho, float *lambda, float *mu, float *lambda_plus_two_mu,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *sigmaxx, float *sigmazz, float *sigmaxz,
		float *phi_sigmaxx_x, float *phi_sigmazz_x, float *phi_sigmaxz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vx_rms, int it, int pml, int Lc, float *rc,
		int r_iz, int *r_ix, int r_n
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ic,ii;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dx,dsigmazz_dx,dsigmaxz_dz;
	float one_over_rho_half_x;
	float lambda_half_x, mu_half_x,lambda_plus_two_mu_half_x;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dsigmaxx_dx=0.0;
		dsigmazz_dx=0.0;
		dsigmaxz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxx_dx+=rc[ic]*(sigmaxx[ip+ic+1]-sigmaxx[ip-ic])*one_over_dx;
			dsigmazz_dx+=rc[ic]*(sigmazz[ip+ic+1]-sigmazz[ip-ic])*one_over_dx;

			dsigmaxz_dz+=rc[ic]*(sigmaxz[ip+ic*ntx]-sigmaxz[ip-(ic+1)*ntx])*one_over_dz;

			/*dsigmaxx_dx+=rc[ic]*(lambda_plus_two_mu[ip+ic+1]*sigmaxx[ip+ic+1]-lambda_plus_two_mu[ip-ic]*sigmaxx[ip-ic])*one_over_dx;
			dsigmazz_dx+=rc[ic]*(lambda[ip+ic+1]*sigmazz[ip+ic+1]-lambda[ip-ic]*sigmazz[ip-ic])*one_over_dx;

			dsigmaxz_dz+=rc[ic]*(0.25*(mu[ip+ic*ntx]+mu[ip+ic*ntx+ntx]+mu[ip+ic*ntx+1]+mu[ip+ic*ntx+ntx+1])*sigmaxz[ip+ic*ntx]-0.25*(mu[ip-ic*ntx-ntx]+mu[ip-ic*ntx]+mu[ip-ic*ntx-ntx+1]+mu[ip-ic*ntx+1])*sigmaxz[ip-(ic+1)*ntx])*one_over_dz;
			*/
		}

		phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
		phi_sigmazz_x[ip]=b_x_half[ix]*phi_sigmazz_x[ip]+a_x_half[ix]*dsigmazz_dx;
		phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

		one_over_rho_half_x=1.0/(0.5*(rho[ip]+rho[ip+1]));
		lambda_half_x=(lambda[ip]+lambda[ip+1])/2.0;
		mu_half_x=(mu[ip]+mu[ip+1])/2.0;
		lambda_plus_two_mu_half_x=(lambda_plus_two_mu[ip]+lambda_plus_two_mu[ip+1])/2.0;

		vx[ip]=dt*one_over_rho_half_x*(lambda_plus_two_mu_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip])
				+lambda_half_x*(dsigmazz_dx+phi_sigmazz_x[ip])
				+mu_half_x*(dsigmaxz_dz+phi_sigmaxz_z[ip]))
			+vx[ip];
		/*vx[ip]=dt*one_over_rho_half_x*((dsigmaxx_dx+phi_sigmaxx_x[ip])
				+(dsigmazz_dx+phi_sigmazz_x[ip])
				+(dsigmaxz_dz+phi_sigmaxz_z[ip]))
			+vx[ip];
		*/
	}
	__syncthreads();


	// Seismogram...   
	for(ii=0;ii<r_n;ii++)
	{
		if(ix==r_ix[ii]&&iz==r_iz)
		{
			vx[ip]=seismogram_vx_rms[it*r_n+ii];
		}
	}
	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vz_mine_backward(
		float *rho, float *lambda, float *mu, float *lambda_plus_two_mu,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *sigmaxx, float *sigmazz, float *sigmaxz, 
		float *phi_sigmaxx_z, float *phi_sigmazz_z, float *phi_sigmaxz_x, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		float *seismogram_vz_rms, int it, int pml, int Lc, float *rc,
		int r_iz, int *r_ix, int r_n
		)

{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dsigmaxx_dz,dsigmazz_dz,dsigmaxz_dx;
	float one_over_rho_half_z;
	float lambda_half_z, mu_half_z,lambda_plus_two_mu_half_z;

	int ic,ii;
	int ip=iz*ntx+ix;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc&&ix<=ntx-Lc)
	{
		dsigmaxx_dz=0.0;
		dsigmazz_dz=0.0;
		dsigmaxz_dx=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dsigmaxx_dz+=rc[ic]*(sigmaxx[ip+(ic+1)*ntx]-sigmaxx[ip-ic*ntx])*one_over_dz;
			dsigmazz_dz+=rc[ic]*(sigmazz[ip+(ic+1)*ntx]-sigmazz[ip-ic*ntx])*one_over_dz;
			dsigmaxz_dx+=rc[ic]*(sigmaxz[ip+ic]-sigmaxz[ip-(ic+1)])*one_over_dx;

			/*dsigmaxx_dz+=rc[ic]*(lambda[ip+(ic+1)*ntx]*sigmaxx[ip+(ic+1)*ntx]-lambda[ip-ic*ntx]*sigmaxx[ip-ic*ntx])*one_over_dz;
			dsigmazz_dz+=rc[ic]*(lambda_plus_two_mu[ip+(ic+1)*ntx]*sigmazz[ip+(ic+1)*ntx]-lambda_plus_two_mu[ip-ic*ntx]*sigmazz[ip-ic*ntx])*one_over_dz;
			dsigmaxz_dx+=rc[ic]*(0.25*(mu[ip+ic]+mu[ip+ic+ntx]+mu[ip+ic+1]+mu[ip+ic+ntx+1])*sigmaxz[ip+ic]-0.25*(mu[ip-ic-1]+mu[ip-ic+ntx-1]+mu[ip-ic]+mu[ip-ic+ntx])*sigmaxz[ip-(ic+1)])*one_over_dx;
			*/
		}

		phi_sigmaxx_z[ip]=b_z_half[iz]*phi_sigmaxx_z[ip]+a_z_half[iz]*dsigmaxx_dz;
		phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;
		phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;

		one_over_rho_half_z=1.0/(0.5*(rho[ip]+rho[ip+ntx]));
		lambda_half_z=(lambda[ip]+lambda[ip+ntx])/2.0;
		mu_half_z=(mu[ip]+mu[ip+ntx])/2.0;
		lambda_plus_two_mu_half_z=(lambda_plus_two_mu[ip]+lambda_plus_two_mu[ip+ntx])/2.0;

		vz[ip]=dt*one_over_rho_half_z*(mu_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip])
				+lambda_half_z*(dsigmaxx_dz+phi_sigmaxx_z[ip])
				+lambda_plus_two_mu_half_z*(dsigmazz_dz+phi_sigmazz_z[ip]))
			+vz[ip];
	/*	vz[ip]=dt*one_over_rho_half_z*((dsigmaxz_dx+phi_sigmaxz_x[ip])
	 						+(dsigmaxx_dz+phi_sigmaxx_z[ip])
	 						+(dsigmazz_dz+phi_sigmazz_z[ip]))
			+vz[ip];
	*/
	}
	__syncthreads();

	for(ii=0;ii<r_n;ii++)
	{
		if(ix==r_ix[ii]&&iz==r_iz)
		{
			vz[ip]=seismogram_vz_rms[it*r_n+ii];
		}
	}
	__syncthreads();

}

__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_mine_backward(
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *sigmaxx, float *sigmazz,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;

	float dvx_dx,dvz_dz;
	int ip=iz*ntx+ix;
	int ic;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc&&ix<=ntx-Lc)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=rc[ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=rc[ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;


		sigmaxx[ip]=(dvx_dx+phi_vx_x[ip])*dt+sigmaxx[ip];

		sigmazz[ip]=(dvz_dz+phi_vz_z[ip])*dt+sigmazz[ip];

	}

	__syncthreads();
}

__global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_mine_backward(
		float *a_x_half, float *a_z_half,
		float *b_x_half, float *b_z_half,
		float *vx, float *vz, float *sigmaxz,
		float *phi_vx_z, float *phi_vz_x,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt, int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dx;
	float dvx_dz,dvz_dx;

	int ip=iz*ntx+ix;
	int ic;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dvz_dx=0.0;
		dvx_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvz_dx+=rc[ic]*(vz[ip+ic+1]-vz[ip-ic])*one_over_dx;
			dvx_dz+=rc[ic]*(vx[ip+(ic+1)*ntx]-vx[ip-ic*ntx])*one_over_dz;
		}

		phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
		phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;

		sigmaxz[ip]=(dvz_dx+phi_vz_x[ip]+
				dvx_dz+phi_vx_z[ip])*dt+
			sigmaxz[ip];
	}

	__syncthreads();
}


__global__ void sum_image_GPU_kernel
(
 float *lambda, float *mu, float *lambda_plus_two_mu,
 float *vx_inv, float *vz_inv, 
 float *vx, float *vz,
 float *sigmaxx_inv, float *sigmaxxs_inv,float *sigmazz_inv, float *sigmaxz_inv, 
 float *sigmaxx,float *sigmaxxs, float *sigmazz, float *sigmaxz, 
 float *image_lambda, float *image_mu, float *image_vp, float *image_vs, 
 int ntx, int ntz, int pml, int Lc, float *rc, float dx, float dz
 )
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;

	float dvx_dx,dvz_dz;
	//	float dvx_dx_b,dvz_dz_b;
	float dvx_dz,dvz_dx;
	//	float dvx_dz_b,dvz_dx_b;

	if(iz>=pmlc+1&&iz<=ntz-pmlc-1&&ix>=pmlc+1&&ix<=ntx-pmlc-1)
	{
		dvx_dx=(vx_inv[ip]-vx_inv[ip-1])/dx;
		dvz_dz=(vz_inv[ip]-vz_inv[ip-ntx])/dz;

		dvx_dz=0.25*(vx_inv[ip+ntx]-vx_inv[ip]+vx_inv[ip-1]-vx_inv[ip-ntx-1]
				+vx_inv[ip]-vx_inv[ip-ntx]+vx_inv[ip+ntx-1]-vx_inv[ip-1])/dz;
		dvz_dx=0.25*(vz_inv[ip+1]-vz_inv[ip]+vz_inv[ip-ntx]-vz_inv[ip-ntx-1]
				+vz_inv[ip]-vz_inv[ip-1]+vz_inv[ip-ntx+1]-vz_inv[ip-ntx])/dx;
		//vx_dz=(vx_inv[ip]-vx_inv[ip-ntx])/dz;
		//dvz_dx=(vz_inv[ip]-vz_inv[ip-1])/dx;

		//		dvx_dx_b=(vx[ip]-vx[ip-1])/dx;
		//		dvz_dz_b=(vz[ip]-vz[ip-ntx])/dz;

		//		dvx_dz_b=(vx[ip]-vx[ip-ntx])/dz;
		//		dvz_dx_b=(vz[ip]-vz[ip-1])/dx;

		//	image_lambda[ip]=image_lambda[ip]+(dvx_dx+dvz_dz)*(dvx_dx_b+dvz_dz_b);
		//	image_mu[ip]=image_mu[ip]+(dvx_dx+dvz_dz)*(dvz_dx_b-dvx_dz_b);
		image_lambda[ip]=image_lambda[ip]+(sigmaxx[ip]+sigmazz[ip])*(dvx_dx+dvz_dz);
		image_mu[ip]=image_mu[ip]+2.0*((sigmaxxs[ip])*dvx_dx+sigmazz[ip]*dvz_dz)
			+0.25*(sigmaxz[ip]+sigmaxz[ip-ntx-1]+sigmaxz[ip-ntx]+sigmaxz[ip-1])*(dvx_dz+dvz_dx);
			//+sigmaxz[ip]*(dvx_dz+dvz_dx);

		image_vp[ip]=image_vp[ip]+(2*lambda[ip]+2*mu[ip])*(dvx_dx+dvz_dz)*(sigmaxx[ip]+sigmaxxs[ip]+sigmazz[ip]+sigmaxx[ip]);
		image_vs[ip]=image_vs[ip]+(lambda_plus_two_mu[ip]*dvx_dx+lambda[ip]*dvz_dz)*(sigmaxx[ip]+sigmaxxs[ip])
			+(lambda_plus_two_mu[ip]*dvz_dz+lambda[ip]*dvx_dx)*(sigmazz[ip]+sigmaxx[ip])
			+2*mu[ip]*(dvz_dx+dvx_dz)*0.25*(sigmaxz[ip]+sigmaxz[ip-ntx-1]+sigmaxz[ip-ntx]+sigmaxz[ip-1]);

	}
	__syncthreads();
}


__global__ void sum_image_GPU_kernel_mu
(
 float *vx_inv, float *vz_inv, 
 float *sigmaxx, float *sigmazz, float *sigmaxz,
 float *image, 
 int ntx, int ntz, int pml, int Lc, float *rc, float dx, float dz
 )

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;

	float dvx_dx,dvz_dz;
	float dvz_dx,dvx_dz;

	if(iz>=pmlc+1&&iz<=ntz-pmlc-1&&ix>=pmlc+1&&ix<=ntx-pmlc-1)
	{
		dvx_dx=(vx_inv[ip]-vx_inv[ip-1])/dx;
		dvz_dz=(vz_inv[ip]-vz_inv[ip-ntx])/dz;
		dvx_dz=0.25*(vx_inv[ip+ntx]-vx_inv[ip]+vx_inv[ip-1]-vx_inv[ip-ntx-1]
				+vx_inv[ip]-vx_inv[ip-ntx]+vx_inv[ip+ntx-1]-vx_inv[ip-1])/dz;
		dvz_dx=0.25*(vz_inv[ip+1]-vz_inv[ip]+vz_inv[ip-ntx]-vz_inv[ip-ntx-1]
				+vz_inv[ip]-vz_inv[ip-1]+vz_inv[ip-ntx+1]-vz_inv[ip-ntx])/dx;
		//image[ip]=image[ip]+(sigmaxz[ip]*(dvx_dz+dvz_dx))-
		//          2.0*(sigmaxx[ip]*dvz_dz+sigmazz[ip]*dvx_dx);
		image[ip]=image[ip]-
			2.0*(sigmaxx[ip]*dvx_dx+sigmazz[ip]*dvz_dz)-
			sigmaxz[ip]*(dvx_dz+dvz_dx);
	}
	__syncthreads();
}

__global__ void sum_image_GPU_kernel_rho
(
 float *vx_inv, float *vz_inv,float *vx0_inv, float *vz0_inv,									
 float *vx, float *vz, float *image_rho,
 int ntx, int ntz, int pml, int Lc, float *rc, float dt
 )
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;

	float dvx_dt,dvz_dt;

	if(iz>=pmlc+1&&iz<=ntz-pmlc-1&&ix>=pmlc+1&&ix<=ntx-pmlc-1)
	{
		dvx_dt=(vx0_inv[ip]-vx_inv[ip])/dt;

		dvz_dt=(vz0_inv[ip]-vz_inv[ip])/dt;

		image_rho[ip]=image_rho[ip]+(vx[ip]*dvx_dt+vz[ip]*dvz_dt);

	}
	__syncthreads();
}

__global__ void fdtd_cpml_3d_GPU_kernel_vx0_change
(
 float *vx, float *vz,float *vx0, float *vz0,
 int ntx, int ntz, int pml, int Lc, float *rc
 )
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;

	if(iz>=pmlc+1&&iz<=ntz-pmlc-1&&ix>=pmlc+1&&ix<=ntx-pmlc-1)
	{
		vx0[ip]=vx[ip];

		vz0[ip]=vz[ip];
	}
	__syncthreads();
}

__global__ void sum_image_GPU_kernel_vp
(
 float *vx_inv, float *vz_inv, 
 float *image, 
 int ntx, int ntz, int pml, int Lc, float *rc, float dx, float dz
 )

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;

	float dvx_dx,dvz_dz;

	if(iz>=pmlc+1&&iz<=ntz-pmlc-1&&ix>=pmlc+1&&ix<=ntx-pmlc-1)
	{
		dvx_dx=(vx_inv[ip]-vx_inv[ip-1])/dx;
		dvz_dz=(vz_inv[ip]-vz_inv[ip-ntx])/dz;

		image[ip]=image[ip]+(dvx_dx+dvz_dz)*(dvx_dx+dvz_dz);
	}
	__syncthreads();
}


__global__ void sum_image_GPU_kernel_vs
(
 float *sigmaxx, float *sigmazz, float *image, 
 int ntx, int ntz, int pml, int Lc, float *rc, float dx, float dz
 )

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int pmlc=pml+Lc;

	if(iz>=pmlc+1&&iz<=ntz-pmlc-1&&ix>=pmlc+1&&ix<=ntx-pmlc-1)
	{
		image[ip]=image[ip]+
			(sigmaxx[ip]+sigmazz[ip])*(sigmaxx[ip]+sigmazz[ip]);
	}
	__syncthreads();
}

/*==========================================================

  This subroutine is used for calculating wave field in 2D.

  ===========================================================*/

extern "C"
void fdtd_cpml_2d_GPU_backward(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float *rc, float dx, float dz,
		float *rick, int itmax, float dt, int myid,float *vp, float *vs,
		float vp_min,float dvp,float vs_min,float dvs,float *Gp,float *Gs, int maxNp, int maxNs,
		int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, float *rho,
		float *lambda, float *mu, float *lambda_plus_two_mu,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half,
		float *vx, float *vz, 
		float *sigmaxx, float *sigmaxxs, float *sigmazz, float *sigmaxz
		)
{
	int it,ip;
	int ix,iz;
	int pmlc=pml+Lc;

	int i;
	Multistream plans[GPU_N];

	char filename[50];
	FILE *fp;

	size_t size_model=sizeof(float)*ntp;

	// allocate the memory for the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	
	}

	// =============================================================================

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

	//-----------------------------------------------------------------------//
	//=======================================================================//
	//-----------------------------------------------------------------------//

	for(i=0;i<GPU_N;i++)
	{
		sprintf(filename,"./output/wavefield_itmax%d_%d.dat",i,myid);
		fp=fopen(filename,"rb");
		fread(&vx[0],sizeof(float),ntp,fp);
		fread(&vz[0],sizeof(float),ntp,fp);

		fread(&sigmaxx[0],sizeof(float),ntp,fp);
		fread(&sigmaxxs[0],sizeof(float),ntp,fp);
		fread(&sigmazz[0],sizeof(float),ntp,fp);
		fread(&sigmaxz[0],sizeof(float),ntp,fp);
		fclose(fp);

		cudaSetDevice(i);

		cudaMemcpyAsync(plan[i].d_vx_inv,vx,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz_inv,vz,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_sigmaxx_inv,sigmaxx,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_sigmaxxs_inv,sigmaxxs,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_sigmazz_inv,sigmazz,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_sigmaxz_inv,sigmaxz,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaStreamSynchronize(plans[i].stream);
	}


	// Copy the vectors from the host to the device

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaMemcpyAsync(plan[i].d_seismogram_vx_rms,plan[i].seismogram_vx_rms,sizeof(float)*(rnmax)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_seismogram_vz_rms,plan[i].seismogram_vz_rms,sizeof(float)*(rnmax)*itmax,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_image_lambda,plan[i].image_lambda,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_image_mu,plan[i].image_mu,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_image_vp,plan[i].image_vp,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_image_vs,plan[i].image_vs,size_model,cudaMemcpyHostToDevice,plans[i].stream);
/*
		cudaMemcpyAsync(plan[i].d_r_ix,ss[is+i].r_ix,sizeof(int)*ss[is+i].r_n,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rc,rc,sizeof(float)*Lc,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_lambda,lambda,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_mu,mu,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_lambda_plus_two_mu,lambda_plus_two_mu,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rho,rho,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_a_x,a_x,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_x_half,a_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_z,a_z,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_z_half,a_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_b_x,b_x,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_x_half,b_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_z,b_z,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_z_half,b_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);


		cudaMemcpyAsync(plan[i].d_vx_borders_up,plan[i].vx_borders_up,sizeof(float)*Lc*itmax*(nx+1),cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vx_borders_bottom,plan[i].vx_borders_bottom,sizeof(float)*Lc*itmax*(nx+1),cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vx_borders_left,plan[i].vx_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc),cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vx_borders_right,plan[i].vx_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc),cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_vz_borders_up,plan[i].vz_borders_up,sizeof(float)*Lc*itmax*nx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz_borders_bottom,plan[i].vz_borders_bottom,sizeof(float)*Lc*itmax*nx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz_borders_left,plan[i].vz_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc+1),cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz_borders_right,plan[i].vz_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc+1),cudaMemcpyHostToDevice,plans[i].stream);
*/
	}//end GPU

	//==============================================================================
	//  THIS SECTION IS USED TO CONSTRUCT THE FORWARD WAVEFIELDS...           
	//==============================================================================

	////initialize the wavefields
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		initialize_wavefields<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
			 plan[i].d_vx, plan[i].d_vz,
			 plan[i].d_sigmaxx, plan[i].d_sigmaxxs, plan[i].d_sigmazz, plan[i].d_sigmaxz,
			 plan[i].d_phi_vx_x, plan[i].d_phi_vxs_x, plan[i].d_phi_vx_z,
			 plan[i].d_phi_vz_z, plan[i].d_phi_vzs_z, plan[i].d_phi_vz_x,
			 plan[i].d_phi_sigmaxx_x, plan[i].d_phi_sigmaxxs_x, plan[i].d_phi_sigmaxz_z,
			 plan[i].d_phi_sigmaxz_x, plan[i].d_phi_sigmazz_z,
			 plan[i].d_phi_sigmaxx_z, plan[i].d_phi_sigmaxxs_z, plan[i].d_phi_sigmazz_x,
			 ntp, ntx, ntz 
			);
	}
	///////////////////////////////////

	for(it=itmax-2;it>=0;it--)
	{

		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);

			fdtd_2d_GPU_kernel_sigmaxx_sigmazz_backward_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rick, plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_lambda, plan[i].d_lambda_plus_two_mu,
				 plan[i].d_vx_inv, plan[i].d_vz_inv, plan[i].d_sigmaxx_inv, plan[i].d_sigmaxxs_inv, plan[i].d_sigmazz_inv,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt,
				 ss[is+i].s_ix, ss[is+i].s_iz, it
				);

			fdtd_2d_GPU_kernel_sigmaxz_backward_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_mu,plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_vx_inv, plan[i].d_vz_inv, plan[i].d_sigmaxz_inv,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt
				);

			fdtd_2d_GPU_kernel_vx_backward_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho,plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs, 
				 plan[i].d_vx_inv, plan[i].d_sigmaxx_inv, plan[i].d_sigmaxxs_inv, plan[i].d_sigmaxz_inv, 
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt
				);

			fdtd_2d_GPU_kernel_vz_backward_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho,plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_vz_inv, plan[i].d_sigmaxz_inv, plan[i].d_sigmaxx_inv, plan[i].d_sigmazz_inv, 
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt
				);

			fdtd_2d_GPU_kernel_borders_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_vx_inv,
				 plan[i].d_vx_borders_up, plan[i].d_vx_borders_bottom,
				 plan[i].d_vx_borders_left, plan[i].d_vx_borders_right,
				 plan[i].d_vz_inv,
				 plan[i].d_vz_borders_up, plan[i].d_vz_borders_bottom,
				 plan[i].d_vz_borders_left, plan[i].d_vz_borders_right,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, it, itmax
				);

			fdtd_cpml_2d_GPU_kernel_ricker_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				plan[i].d_vx_inv, plan[i].d_vz_inv, plan[i].d_sigmaxx_inv,
				plan[i].d_sigmaxxs_inv, plan[i].d_sigmazz_inv, plan[i].d_sigmaxz_inv,
				plan[i].d_rick,
				ntp, ntx,ntz, it, pml, Lc,
				ss[is+i].s_ix, ss[is+i].s_iz
				);
			///////////////////////////////////////////////////////////////////////
			/////////////////////////receiver wavefield////////////////////////////
			///////////////////////////////////////////////////////////////////////
			fdtd_cpml_2d_GPU_kernel_vx_backward_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_a_x_half, plan[i].d_a_z, 
				 plan[i].d_b_x_half, plan[i].d_b_z, 
				 plan[i].d_vx, plan[i].d_sigmaxx, plan[i].d_sigmaxxs, plan[i].d_sigmaxz,
				 plan[i].d_phi_sigmaxx_x, plan[i].d_phi_sigmaxxs_x, plan[i].d_phi_sigmaxz_z, 
				 ntp, ntx, ntz, 
				 -dx, -dz, dt,
				 plan[i].d_seismogram_vx_rms, it, pml, Lc, plan[i].d_rc,
				 plan[i].d_r_iz, plan[i].d_r_ix, ss[is+i].r_n
				);

			fdtd_cpml_2d_GPU_kernel_vz_backward_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_a_x, plan[i].d_a_z_half,
				 plan[i].d_b_x, plan[i].d_b_z_half,
				 plan[i].d_vz, plan[i].d_sigmaxz, plan[i].d_sigmaxx,plan[i].d_sigmazz, 
				 plan[i].d_phi_sigmaxz_x, plan[i].d_phi_sigmaxx_z,plan[i].d_phi_sigmazz_z,
				 ntp, ntx, ntz, 
				 -dx, -dz, dt,
				 plan[i].d_seismogram_vz_rms, it, pml, Lc, plan[i].d_rc,
				 plan[i].d_r_iz, plan[i].d_r_ix, ss[is+i].r_n
				);

			fdtd_cpml_2d_GPU_kernel_seismogram_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				plan[i].d_vx, plan[i].d_vz, plan[i].d_sigmaxx, plan[i].d_sigmaxxs, plan[i].d_sigmazz, plan[i].d_sigmaxz, 
				plan[i].d_seismogram_vx_rms,  plan[i].d_seismogram_vz_rms, 
				ntp, ntx, ntz,
				it, pml, Lc, plan[i].d_r_iz, plan[i].d_r_ix, ss[is+i].r_n
				);

			fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_backward_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
 				 plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_lambda, plan[i].d_lambda_plus_two_mu,
				 plan[i].d_a_x, plan[i].d_a_z,
				 plan[i].d_b_x, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_sigmaxx, plan[i].d_sigmaxxs, plan[i].d_sigmazz,
				 plan[i].d_phi_vx_x, plan[i].d_phi_vxs_x,plan[i].d_phi_vz_z, plan[i].d_phi_vzs_z,
				 ntp, ntx, ntz, 
				 -dx, -dz, dt, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_sigmaxz_backward_4LT<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_mu, plan[i].d_Gp, plan[i].d_Gs,
				 plan[i].d_vp, vp_min, dvp, plan[i].d_vs, vs_min, dvs,
				 plan[i].d_a_x_half, plan[i].d_a_z_half,
				 plan[i].d_b_x_half, plan[i].d_b_z_half,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_sigmaxz,
				 plan[i].d_phi_vx_z, plan[i].d_phi_vz_x,
				 ntp, ntx, ntz, 
				 -dx, -dz, dt, Lc, plan[i].d_rc
				);

			sum_image_GPU_kernel<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_lambda, plan[i].d_mu, plan[i].d_lambda_plus_two_mu,
				 plan[i].d_vx_inv, plan[i].d_vz_inv, plan[i].d_vx, plan[i].d_vz, 
				 plan[i].d_sigmaxx_inv, plan[i].d_sigmaxxs_inv,plan[i].d_sigmazz_inv,plan[i].d_sigmaxz_inv,
				 plan[i].d_sigmaxx, plan[i].d_sigmaxxs,plan[i].d_sigmazz, plan[i].d_sigmaxz,
				 plan[i].d_image_lambda, plan[i].d_image_mu, plan[i].d_image_vp, plan[i].d_image_vs,
				 ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz
				);

			
			 /*  if(i==0&&it%50==0)
			   {
				   cudaStreamSynchronize(plans[i].stream);
				   cudaMemcpy(vx,plan[i].d_vx_inv,sizeof(float)*ntp,cudaMemcpyDeviceToHost);

				   sprintf(filename,"./output/%dvx_inv%d.dat",it,myid);     
				   fp=fopen(filename,"wb");
				   for(ix=pmlc;ix<ntx-pmlc;ix++)
				   {
					   for(iz=pmlc;iz<ntz-pmlc;iz++)
					   {
						   ip=iz*ntx+ix;
						   fwrite(&vx[ip],sizeof(float),1,fp);
					   }
				   }
				   fclose(fp);

				   cudaMemcpy(vx,plan[i].d_vx,sizeof(float)*ntp,cudaMemcpyDeviceToHost);

				   sprintf(filename,"./output/%dvxbak%d.dat",it,myid);     
				   fp=fopen(filename,"wb");
				   for(ix=pmlc;ix<ntx-pmlc;ix++)
				   {
					   for(iz=pmlc;iz<ntz-pmlc;iz++)
					   {
						   ip=iz*ntx+ix;
						   fwrite(&vx[ip],sizeof(float),1,fp);
					   }
				   }
				   fclose(fp);
			   }*/
			 
		}//end GPU_N

	}//end it


	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaMemcpyAsync(plan[i].image_lambda,plan[i].d_image_lambda,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].image_mu,plan[i].d_image_mu,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);

		cudaMemcpyAsync(plan[i].image_vp,plan[i].d_image_vp,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].image_vs,plan[i].d_image_vs,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
	/*	cudaStreamSynchronize(plans[i].stream);

		sprintf(filename,"./output/image_lambda%d.dat",is+i);
		fp=fopen(filename,"wb");
		fwrite(&plan[i].image_lambda[0],sizeof(float),ntp,fp);
		fclose(fp);

		sprintf(filename,"./output/image_mu%d.dat",is+i);
		fp=fopen(filename,"wb");
		fwrite(&plan[i].image_mu[0],sizeof(float),ntp,fp);
		fclose(fp);

		sprintf(filename,"./output/image_vp%d.dat",is+i);
		fp=fopen(filename,"wb");
		fwrite(&plan[i].image_vp[0],sizeof(float),ntp,fp);
		fclose(fp);

		sprintf(filename,"./output/image_vs%d.dat",is+i);
		fp=fopen(filename,"wb");
		fwrite(&plan[i].image_vs[0],sizeof(float),ntp,fp);
		fclose(fp);
*/	
	}//end GPU

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}//end GPU

	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaStreamDestroy(plans[i].stream);
	}
}

/*=====================================================================
  This function is used for calculating the convolution-based residuals
  =====================================================================*/

extern "C"
void congpu_fre(float *seismogram_syn, float *seismogram_obs, float *seismogram_rms, float *Misfit, int i, 
		float *ref_window, float *seis_window, int itmax, float dt, float dx, int is, int nx, int s_ix, int *r_ix, int pml)
{ 
	cudaSetDevice(i);

	int ix,it,itt,K,NX;
	float epsilon,rms,rmsmax;
	int BATCH=nx;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=(int)pow(2.0,K);	

	float vv=1950;//surface wave velocities
	float vt=vv*dt;

	int nnn=10;//1;//
	int reft;

	FILE *fp;
	char filename[30];

	int sx=s_ix-pml;//r_ix[0];

	if(sx+1+nnn>nx)
		reft=sx-1-nnn;
	else
		reft=sx+1;

	int NTP=NX*BATCH;

	cufftComplex *xx,*d,*h,*sh,*r,*ms,*rr,*temp,*temp1,*obs;

	cudaMallocHost((void **)&xx, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&d, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&h, sizeof(cufftComplex)*NX);
	cudaMallocHost((void **)&sh,sizeof(cufftComplex)*NX);
	cudaMallocHost((void **)&r, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&ms,sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&rr,sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&obs,sizeof(cufftComplex)*NX*BATCH);

	cudaMalloc((void **)&temp1,sizeof(cufftComplex)*NX);
	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*BATCH);

	for(it=0;it<NX;it++)
	{ 
		h[it].x=0.0;
		h[it].y=0.0; 

		sh[it].x=0.0;
		sh[it].y=0.0;    
	}

	for(it=0;it<itmax;it++)
	{
		/*for(ix=reft;ix<reft+nnn;ix++)
		{
			if(it-(int)ceil(fabs(ix-sx)*dx/vt)<0)
			{
				itt=0;
			}
			else
			{
				itt=it-(int)ceil(fabs(ix-sx)*dx/vt);
			}

			h[itt].x = h[itt].x+seismogram_obs[it*nx+ix]*seis_window[it]/nnn;
			sh[itt].x=sh[itt].x+seismogram_syn[it*nx+ix]*ref_window[it]/nnn;  
  
		}*/
		//h[it].x =ref_obs[it];// 
	    h[it].x+=seismogram_obs[it*nx+reft]*seis_window[it]/nnn;
		//sh[it].x=ref_syn[it];//
		sh[it].x+=seismogram_syn[it*nx+reft]*ref_window[it]/nnn;   
	}

	sprintf(filename,"./output/referenceobs%d.dat",is+1);
	fp=fopen(filename,"wb");
	fwrite(&h[0],sizeof(float),itmax,fp);
	fclose(fp);

	sprintf(filename,"./output/referencesyn%d.dat",is+1);
	fp=fopen(filename,"wb");
	fwrite(&sh[0],sizeof(float),itmax,fp);
	fclose(fp);

	cufftHandle plan1,plan2;
	cufftPlan1d(&plan1,NX,CUFFT_C2C,1);
	cufftPlan1d(&plan2,NX,CUFFT_C2C,BATCH);

	cudaMemcpy(temp1,h,sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
	cufftExecC2C(plan1,temp1,temp1,CUFFT_FORWARD);
	cudaMemcpy(h,temp1,sizeof(cufftComplex)*NX,cudaMemcpyDeviceToHost);

	cudaMemcpy(temp1,sh,sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
	cufftExecC2C(plan1,temp1,temp1,CUFFT_FORWARD);
	cudaMemcpy(sh,temp1,sizeof(cufftComplex)*NX,cudaMemcpyDeviceToHost);  	

	//    for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<NTP;it++)
		{ 
			xx[it].x=0.0;
			xx[it].y=0.0; 
			d[it].x=0.0;
			d[it].y=0.0;   
			r[it].x=0.0;
			r[it].y=0.0;
		}            
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				xx[ix*NX+it].x=seismogram_syn[it*nx+ix];
				d[ix*NX+it].x=seismogram_obs[it*nx+ix];	
			}
		}   

		cudaMemcpy(temp,xx,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(xx,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		cudaMemcpy(temp,d,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(d,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		for(it=0;it<NTP;it++)
		{
			obs[it].x=sh[it%NX].x*d[it].x-sh[it%NX].y*d[it].y;
			obs[it].y=sh[it%NX].x*d[it].y+sh[it%NX].y*d[it].x;
			r[it].x=xx[it].x*h[it%NX].x-xx[it].y*h[it%NX].y-obs[it].x;
			r[it].y=xx[it].x*h[it%NX].y+xx[it].y*h[it%NX].x-obs[it].y;
		}  

		// fft(r_real,r_imag,NFFT,-1);

		cudaMemcpy(temp,r,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(ms,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		cudaMemcpy(temp,obs,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(obs,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		epsilon=0.0;
		rms=0.0;
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				epsilon=epsilon+10*fabs(obs[ix*NX+it].x)/(itmax*nx);
			}
		}
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				rms=rms+ms[ix*NX+it].x*ms[ix*NX+it].x/(epsilon*epsilon);
			}
		}
		*Misfit+=sqrt(1+rms)-1;

		// Calculate the r of ( f= rXdref )	Right hide term of adjoint equation!!!
		for(it=0;it<NTP;it++)
		{
			ms[it].x=ms[it].x/(epsilon*epsilon*sqrt(1+ms[it].x*ms[it].x/(epsilon*epsilon)));  //Time domain ms.x==u*dref-vref*d
			ms[it].y=0.0;
		}

		cudaMemcpy(temp,ms,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(r,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		for(it=0;it<NTP;it++)
		{
			rr[it].x=h[it%NX].x*r[it].x+h[it%NX].y*r[it].y;

			rr[it].y=h[it%NX].x*r[it].y-h[it%NX].y*r[it].x;
		}   

		//fft(rr_real,rr_imag,NX*BATCH,-1);

		cudaMemcpy(temp,rr,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(rr,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		rmsmax=0.0;
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				seismogram_rms[it*nx+ix]=rr[ix*NX+it].x;
				if(rmsmax<fabs(seismogram_rms[it*nx+ix]))
				{
					rmsmax=fabs(seismogram_rms[it*nx+ix]);
				}
	//			if(it>=itmax-60)
				{
	//				seismogram_rms[it*nx+ix]=0.0;
				}            
			}
		}

		for(it=0;it<itmax*nx;it++)
		{
			seismogram_rms[it]/=rmsmax;
		}
	}

	cudaFreeHost(xx);
	cudaFreeHost(d);
	cudaFreeHost(h);
	cudaFreeHost(sh);
	cudaFreeHost(r);   
	cudaFreeHost(ms);   
	cudaFreeHost(rr);
	cudaFreeHost(obs); 

	cudaFree(temp);
	cudaFree(temp1);

	cufftDestroy(plan1);
	cufftDestroy(plan2);

	return;
}

/*=============================================
 * Allocate the memory for wavefield simulation
 * ===========================================*/
extern "C"
void variables_malloc(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax, int maxNp, int maxNs,
		struct MultiGPU plan[], int GPU_N, int rnmax, int NN
		)
{
	int i;

	size_t size_model=sizeof(float)*ntp;

	// ==========================================================
	// allocate the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,...
/*
	cudaMallocHost((void **)&rick,sizeof(float)*itmax);  
	cudaMallocHost((void **)&rc,sizeof(float)*Lc);  

	cudaMallocHost((void **)&lambda,sizeof(float)*ntp); 
	cudaMallocHost((void **)&mu,sizeof(float)*ntp); 
	cudaMallocHost((void **)&rho,sizeof(float)*ntp); 
	cudaMallocHost((void **)&lambda_plus_two_mu,sizeof(float)*ntp); 

	cudaMallocHost((void **)&a_x,sizeof(float)*ntx);
	cudaMallocHost((void **)&a_x_half,sizeof(float)*ntx);
	cudaMallocHost((void **)&a_z,sizeof(float)*ntz);
	cudaMallocHost((void **)&a_z_half,sizeof(float)*ntz);

	cudaMallocHost((void **)&b_x,sizeof(float)*ntx);
	cudaMallocHost((void **)&b_x_half,sizeof(float)*ntx);
	cudaMallocHost((void **)&b_z,sizeof(float)*ntz);
	cudaMallocHost((void **)&b_z_half,sizeof(float)*ntz);

	cudaMallocHost((void **)&vx,sizeof(float)*ntp); 
	cudaMallocHost((void **)&vz,sizeof(float)*ntp); 
	cudaMallocHost((void **)&sigmaxx,sizeof(float)*ntp);
	cudaMallocHost((void **)&sigmazz,sizeof(float)*ntp);
	cudaMallocHost((void **)&sigmaxz,sizeof(float)*ntp);
*/

	// allocate the memory for the device
	// allocate the memory for the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		///device//////////
		///device//////////
		///device//////////
		cudaMalloc((void**)&plan[i].d_seismogram_vx_syn,sizeof(float)*itmax*rnmax);
		cudaMalloc((void**)&plan[i].d_seismogram_vz_syn,sizeof(float)*itmax*rnmax);

		cudaMalloc((void**)&plan[i].d_seismogram_vx_rms,sizeof(float)*itmax*(rnmax));
		cudaMalloc((void**)&plan[i].d_seismogram_vz_rms,sizeof(float)*itmax*(rnmax));

		cudaMalloc((void**)&plan[i].d_r_ix,sizeof(int)*rnmax);
		cudaMalloc((void**)&plan[i].d_r_iz,sizeof(int)*rnmax);

		cudaMalloc((void**)&plan[i].d_rick,sizeof(float)*itmax);        // ricker wave 
		cudaMalloc((void**)&plan[i].d_rc,sizeof(float)*Lc);
		cudaMalloc((void**)&plan[i].d_Gp,sizeof(float)*maxNp*(Lpo+1));   
		cudaMalloc((void**)&plan[i].d_Gs,sizeof(float)*maxNs*(Lso+1));           
		cudaMalloc((void**)&plan[i].d_asr,sizeof(float)*NN);        

		cudaMalloc((void**)&plan[i].d_lambda,size_model);
		cudaMalloc((void**)&plan[i].d_mu,size_model);
		cudaMalloc((void**)&plan[i].d_muxz,size_model);
		cudaMalloc((void**)&plan[i].d_vp,size_model);
		cudaMalloc((void**)&plan[i].d_vs,size_model);
		cudaMalloc((void**)&plan[i].d_rho,size_model);
		cudaMalloc((void**)&plan[i].d_lambda_plus_two_mu,size_model);   // model 


		cudaMalloc((void**)&plan[i].d_a_x,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_a_x_half,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_a_z,sizeof(float)*ntz);
		cudaMalloc((void**)&plan[i].d_a_z_half,sizeof(float)*ntz);

		cudaMalloc((void**)&plan[i].d_b_x,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_b_x_half,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_b_z,sizeof(float)*ntz);
		cudaMalloc((void**)&plan[i].d_b_z_half,sizeof(float)*ntz);      // atten parameters


		cudaMalloc((void**)&plan[i].d_image_lambda,size_model);
		cudaMalloc((void**)&plan[i].d_image_mu,size_model);
		
		cudaMalloc((void**)&plan[i].d_image_vp,size_model);
		cudaMalloc((void**)&plan[i].d_image_vs,size_model);

		cudaMalloc((void**)&plan[i].d_vx,size_model);
		cudaMalloc((void**)&plan[i].d_vz,size_model);
		cudaMalloc((void**)&plan[i].d_sigmaxx,size_model);
		cudaMalloc((void**)&plan[i].d_sigmaxxs,size_model);
		cudaMalloc((void**)&plan[i].d_sigmazz,size_model);
		cudaMalloc((void**)&plan[i].d_sigmaxz,size_model);              // wavefields 


		cudaMalloc((void**)&plan[i].d_vx_inv,size_model);
		cudaMalloc((void**)&plan[i].d_vz_inv,size_model);
		cudaMalloc((void**)&plan[i].d_sigmaxx_inv,size_model);
		cudaMalloc((void**)&plan[i].d_sigmaxxs_inv,size_model);
		cudaMalloc((void**)&plan[i].d_sigmazz_inv,size_model);
		cudaMalloc((void**)&plan[i].d_sigmaxz_inv,size_model);          // constructed wavefields 


		cudaMalloc((void**)&plan[i].d_phi_vx_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vxs_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vzs_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vx_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_x,size_model);

		cudaMalloc((void**)&plan[i].d_phi_sigmaxx_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_sigmaxxs_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_sigmaxz_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_sigmaxz_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_sigmazz_z,size_model);

		cudaMalloc((void**)&plan[i].d_phi_sigmaxx_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_sigmaxxs_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_sigmazz_x,size_model);

		cudaMalloc((void**)&plan[i].d_vx_borders_up,sizeof(float)*Lc*itmax*(nx+1));
		cudaMalloc((void**)&plan[i].d_vx_borders_bottom,sizeof(float)*Lc*itmax*(nx+1));
		cudaMalloc((void**)&plan[i].d_vx_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc));
		cudaMalloc((void**)&plan[i].d_vx_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc));

		cudaMalloc((void**)&plan[i].d_vz_borders_up,sizeof(float)*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_vz_borders_bottom,sizeof(float)*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_vz_borders_left,sizeof(float)*Lc*itmax*(nz-2*Lc+1));
		cudaMalloc((void**)&plan[i].d_vz_borders_right,sizeof(float)*Lc*itmax*(nz-2*Lc+1));

	}
}

/*=============================================
 * Free the memory for wavefield simulation
 * ===========================================*/
extern "C"
void variables_free(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax,
		struct MultiGPU plan[], int GPU_N, int rnmax, int NN
		)
{
	int i;
/*
	cudaFreeHost(rick); 
	cudaFreeHost(rc); 
	
	//free the memory of lambda
	cudaFreeHost(lambda); 
	cudaFreeHost(mu); 
	cudaFreeHost(rho); 
	cudaFreeHost(lambda_plus_two_mu); 

	cudaFreeHost(a_x);
	cudaFreeHost(a_x_half);
	cudaFreeHost(a_z);
	cudaFreeHost(a_z_half);

	cudaFreeHost(b_x);
	cudaFreeHost(b_x_half);
	cudaFreeHost(b_z);
	cudaFreeHost(b_z_half);

	cudaFreeHost(vx);
	cudaFreeHost(vz);
	cudaFreeHost(sigmaxx);
	cudaFreeHost(sigmazz);
	cudaFreeHost(sigmaxz);
*/	 
	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		////device////
		////device////
		////device////
		cudaFree(plan[i].d_seismogram_vx_syn);
		cudaFree(plan[i].d_seismogram_vz_syn);

		cudaFree(plan[i].d_seismogram_vx_rms);
		cudaFree(plan[i].d_seismogram_vz_rms);

		cudaFree(plan[i].d_r_ix);
		cudaFree(plan[i].d_r_iz);

		cudaFree(plan[i].d_rick);
		cudaFree(plan[i].d_rc);
		cudaFree(plan[i].d_Gp);
		cudaFree(plan[i].d_Gs);
		cudaFree(plan[i].d_asr);

		cudaFree(plan[i].d_lambda);
		cudaFree(plan[i].d_mu);
		cudaFree(plan[i].d_muxz);
		cudaFree(plan[i].d_lambda_plus_two_mu);
		cudaFree(plan[i].d_vp);
		cudaFree(plan[i].d_vs);
		cudaFree(plan[i].d_rho);

		cudaFree(plan[i].d_a_x);
		cudaFree(plan[i].d_a_x_half);
		cudaFree(plan[i].d_a_z);
		cudaFree(plan[i].d_a_z_half);

		cudaFree(plan[i].d_b_x);
		cudaFree(plan[i].d_b_x_half);
		cudaFree(plan[i].d_b_z);
		cudaFree(plan[i].d_b_z_half);

		cudaFree(plan[i].d_image_lambda);
		cudaFree(plan[i].d_image_mu);

		cudaFree(plan[i].d_image_vp);
		cudaFree(plan[i].d_image_vs);

		cudaFree(plan[i].d_vx);
		cudaFree(plan[i].d_vz);
		cudaFree(plan[i].d_sigmaxx);
		cudaFree(plan[i].d_sigmaxxs);
		cudaFree(plan[i].d_sigmazz);
		cudaFree(plan[i].d_sigmaxz);

		cudaFree(plan[i].d_vx_inv);
		cudaFree(plan[i].d_vz_inv);
		cudaFree(plan[i].d_sigmaxx_inv);
		cudaFree(plan[i].d_sigmaxxs_inv);
		cudaFree(plan[i].d_sigmazz_inv);
		cudaFree(plan[i].d_sigmaxz_inv);

		cudaFree(plan[i].d_phi_vx_x);
		cudaFree(plan[i].d_phi_vz_z);
		cudaFree(plan[i].d_phi_vxs_x);
		cudaFree(plan[i].d_phi_vzs_z);
		cudaFree(plan[i].d_phi_vx_z);
		cudaFree(plan[i].d_phi_vz_x);

		cudaFree(plan[i].d_phi_sigmaxx_x);
		cudaFree(plan[i].d_phi_sigmaxxs_x);
		cudaFree(plan[i].d_phi_sigmaxz_z);
		cudaFree(plan[i].d_phi_sigmaxz_x);
		cudaFree(plan[i].d_phi_sigmazz_z);

		cudaFree(plan[i].d_phi_sigmaxx_z);
		cudaFree(plan[i].d_phi_sigmaxxs_z);
		cudaFree(plan[i].d_phi_sigmazz_x);

		cudaFree(plan[i].d_vx_borders_up);
		cudaFree(plan[i].d_vx_borders_bottom);
		cudaFree(plan[i].d_vx_borders_left);
		cudaFree(plan[i].d_vx_borders_right);

		cudaFree(plan[i].d_vz_borders_up);
		cudaFree(plan[i].d_vz_borders_bottom);
		cudaFree(plan[i].d_vz_borders_left);
		cudaFree(plan[i].d_vz_borders_right);
	}
}

