#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BLOCK_SIZE 16
#define WITH_SHARED_MEMORY 0

#define Lpo 4	//S_LC_space
#define Lso 6	//P_LC_space

#define Lcc (Lso>Lpo?Lso:Lpo)

#define PI 3.1415926
#define Df 10       //the length of high filter   
#define Rf 10       //the length of low filter 
#define J 8       //the level of decompose

#include "headcpu.h"
#include "mpi.h"

#define ricker_flag 0      //0--without 1--with
#define reference_window 1 //0--without 1--with

#define srms 1.0e-20


//High order 4T time FD modeling//
// P and S employ the different order to avoid the oversampling//


int main(int argc, char *argv[])
{

	int myid,numprocs,namelen,index;
	
	MPI_Comm comm=MPI_COMM_WORLD;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(comm,&myid);
	MPI_Comm_size(comm,&numprocs);
	MPI_Get_processor_name(processor_name,&namelen);

	if(myid==0)
		printf("Number of MPI thread is %d\n",numprocs);

	/*=========================================================
	  Parameters of the time of the system...
	  =========================================================*/

	time_t begin_time;
	//  time_t end_time;
	//  time_t last_time;

	clock_t start;
	clock_t end;

	//  float runtime=0.0;

	/*=========================================================
	  Parameters of Cartesian coordinate...
	  ========================================================*/ 

	int nz=170;//170;//180;//234;//126;//281;//2801;
	int nx=800;//645;//663;//1134;//1361;//13601;

   	int Lp=Lpo+1;	//P_LC_Space_time
    	int Ls=Lso+1;	//S_LC_space_time

	int Lc=Lcc;//5;

	int pml=18;
    	int pmlc=pml+(Lpo>Lso?Lpo:Lso);	

	int ntz=nz+2*pmlc;
	int ntx=nx+2*pmlc;
	int ntp=ntz*ntx;

	int np=nx*nz;
	int ip,iz,ix,it,ipp;

	float dz=12.5;
	float dx=dz;

	/*=========================================================
	  Parameters of ricker wave...
	  ========================================================*/

	int   itmax=2310;//2730
	float *rick;
	float f0=20.0,f1=20.0;
	float t0=1/f0;//0.2;//0.08;//
	float dt=1.3e-3;
	int ifreq,N_ifreq=1;//5;//8;//5
	float f[N_ifreq];
	f[0]=4.0;//.0;//5.0
	f[1]=7.0;
	f[2]=10.0;
	f[3]=14.0;
	f[4]=19.0;
	f[5]=21.0;

	float tmpwindow;
	int tc[N_ifreq];
/*	tc[0]=530;
	tc[1]=1000;	
	tc[2]=900;
	tc[3]=750;
	tc[4]=400;*/
	for(ifreq=0;ifreq<N_ifreq;ifreq++)
	{
		tc[ifreq]=(int)2.2/f[ifreq]/dt;
	}

	/*=========================================================
	  Iteration parameters...
	  ========================================================*/

	int iter,itn=1;//15;//20;//15//20

	/*=========================================================
	  File name....
	 *========================================================*/

	FILE *fp, *fs;
	char filename[50];
	/*=========================================================
	  Parameters of traditional LC...
	  ========================================================*/
	float *rc;
	rc=(float*)malloc(sizeof(float)*Lc);
	cal_xishu(Lc,rc);

	int ic;
	float tmprc=0.0;
	for(ic=0;ic<Lc;ic++)
	{
		tmprc+=fabs(rc[ic]);
	}
	if(myid==0)
		printf("Maximum velocity for stability is %f m/s\n",dx/(tmprc*dt*sqrt(2)));



	float *rpv,*rsv;
	int limit_vp=0;
	int limit_vs=0;
	rpv=(float*)malloc(sizeof(float)*Lpo);
	rsv=(float*)malloc(sizeof(float)*Lso);

	cal_xishu(Lpo,rpv);
	cal_xishu(Lso,rsv);

	tmprc=0.0;
	for(ic=0;ic<Lpo;ic++)
	{
		tmprc+=fabs(rpv[ic]);
	}
	limit_vp=(int)(dx/(tmprc*dt*sqrt(2))+0.5);
	if(myid==0)
		printf("P-wave maximum velocity for stability is %d m/s\n",limit_vp);

	tmprc=0.0;
	for(ic=0;ic<Lso;ic++)
	{
		tmprc+=fabs(rsv[ic]);
	}
	limit_vs=(int)(dx/(tmprc*dt*sqrt(2))+0.5);
	if(myid==0)
		printf("S-wave maximum velocity for stability is %d m/s\n",limit_vs);

	// set Gp and Gs //
    	float dvp=1.0;
    	float dvs=1.0;

    	int maxNp=(int)(limit_vp/dvp+1.5);		//四舍五入
    	int maxNs=(int)(limit_vs/dvs+1.5);
	/*=========================================================
	  Parameters of GPU...
	  ========================================================*/

	int i,GPU_N;
	getdevice(&GPU_N);
	//GPU_N=1;
	printf("The available Device number is %d on %s\n",GPU_N,processor_name);
	MPI_Barrier(comm);

	struct MultiGPU plan[GPU_N];

	/*=========================================================
	  Parameters of model...
	  ========================================================*/

	float *vp,*vs,*rho;
	float *lambda,*mu,*lambda_plus_two_mu;
	float vp_max,vs_max,rho_max;
	float vp_min,vs_min,rho_min;
	float *vp_n,*vs_n,*rho_n;

	float *vx,*vz;
	float *sigmaxx,*sigmaxxs,*sigmaxz,*sigmazz;

	/*=========================================================
	  Parameters of absorbing layers...
	  ========================================================*/

	float *d_x,*d_x_half,*d_z,*d_z_half;
	float *a_x,*a_x_half,*a_z,*a_z_half;
	float *b_x,*b_x_half,*b_z,*b_z_half;

	/*=========================================================
	  Parameters of Sources and Receivers...
	  ========================================================*/
	int is,rnmax=0;

	int NNmax=3;

	int nsid,modsr,prcs;
	int iss,eachsid,offsets;

	/*=========================================================
	  Calculate the sources' poisition...
	  ========================================================*/

	int ns=30;//30;//24;//6;//334;//88;//94;////126;//6;//124;//

	struct Source ss[ns];

	for(is=0;is<ns;is++)
	{
		ss[is].s_ix=pmlc+23+is*26;//30+is*20;//18+is*28;//

		ss[is].s_iz=pmlc+1;
		//ss[is].r_iz=pmlc+1;

		ss[is].r_n=nx;
	}

	for(is=0;is<ns;is++)
	{
		if(rnmax<ss[is].r_n)
			rnmax=ss[is].r_n;
	}
	if(myid==0)
		printf("The maximum trace number for source is %d\n",rnmax);

	for(is=0;is<ns;is++)
	{
		ss[is].r_ix=(int*)malloc(sizeof(int)*ss[is].r_n);
		ss[is].r_iz=(int*)malloc(sizeof(int)*ss[is].r_n);
	} 

	for(is=0;is<ns;is++)
	{
		for(ip=0;ip<ss[is].r_n;ip++)
		{
			ss[is].r_ix[ip]=pmlc+ip;
			ss[is].r_iz[ip]=pmlc+1;
		}
	}

	/*=========================================================
	  Parameters of the coefficients of the space...
	  ========================================================*/

//	float c[2]={9.0/8.0,-1.0/24.0};

	/*=========================================================
	  parameters of seismograms and borders...
	  ========================================================*/

	float *ref_window,*seis_window;

	/*=========================================================
	  Image / gradient ...
	 *========================================================*/

	float *Gradient_vp_pre;
	float *Gradient_vs_pre;
	float *dn_vp_pre;
	float *dn_vs_pre;
	float *Gradient_lambda_all,*Gradient_mu_all;
	float tmpnew,tmpold,*tmp1,*tmp2;

	float P[nz];
	float *Gradient_vp_all,*Gradient_vp;
	float *Gradient_vs_all,*Gradient_vs;

	float *dn_vp,*dn_vs;//,*dn_rho;
	float *un0_vp,*un0_vs;//,*un0_rho;

	/*=========================================================
	  Flags ....
	 *========================================================*/

	int inv_flag;

	//#######################################################################
	// NOW THE PROGRAM BEGIN
	//#######################################################################

	time(&begin_time);
	if(myid==0)
		printf("Today's data and time: %s",ctime(&begin_time));

	/*=========================================================
	  Allocate the memory of parameters of ricker wave...
	  ========================================================*/

	rick=(float*)malloc(sizeof(float)*itmax);  
	//cudaMallocHost((void **)&rick,sizeof(float)*itmax);  

	/*=========================================================
	  Allocate the memory of parameters of model...
	  ========================================================*/

	// allocate the memory of model parameters...

	vp                  = (float*)malloc(sizeof(float)*ntp);
	vs                  = (float*)malloc(sizeof(float)*ntp);
	rho                 = (float*)malloc(sizeof(float)*ntp);

	lambda=(float *)malloc(sizeof(float)*ntp); 
	mu=(float *)malloc(sizeof(float)*ntp); 
	lambda_plus_two_mu=(float *)malloc(sizeof(float)*ntp); 
	/*cudaMallocHost((void **)&lambda,sizeof(float)*ntp); 
	cudaMallocHost((void **)&mu,sizeof(float)*ntp); 
	cudaMallocHost((void **)&lambda_plus_two_mu,sizeof(float)*ntp);*/ 

	vp_n                = (float*)malloc(sizeof(float)*ntp);
	vs_n                = (float*)malloc(sizeof(float)*ntp);
	rho_n                = (float*)malloc(sizeof(float)*ntp);

	// ==========================================================
	// allocate the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,...

	vx=(float *)malloc(sizeof(float)*ntp);
	vz=(float *)malloc(sizeof(float)*ntp); 
	sigmaxx=(float *)malloc(sizeof(float)*ntp);
	sigmaxxs=(float *)malloc(sizeof(float)*ntp);
	sigmazz=(float *)malloc(sizeof(float)*ntp);
	sigmaxz=(float *)malloc(sizeof(float)*ntp); 


	/*=========================================================
	  Allocate the memory of parameters of absorbing layer...
	  ========================================================*/

	a_x=(float *)malloc(sizeof(float)*ntx);
	a_x_half=(float *)malloc(sizeof(float)*ntx);
	a_z=(float *)malloc(sizeof(float)*ntz);
	a_z_half=(float *)malloc(sizeof(float)*ntz);

	b_x=(float *)malloc(sizeof(float)*ntx);
	b_x_half=(float *)malloc(sizeof(float)*ntx);
	b_z=(float *)malloc(sizeof(float)*ntz);
	b_z_half=(float *)malloc(sizeof(float)*ntz);
/*
	cudaMallocHost((void **)&a_x,sizeof(float)*ntx);
	cudaMallocHost((void **)&a_x_half,sizeof(float)*ntx);
	cudaMallocHost((void **)&a_z,sizeof(float)*ntz);
	cudaMallocHost((void **)&a_z_half,sizeof(float)*ntz);

	cudaMallocHost((void **)&b_x,sizeof(float)*ntx);
	cudaMallocHost((void **)&b_x_half,sizeof(float)*ntx);
	cudaMallocHost((void **)&b_z,sizeof(float)*ntz);
	cudaMallocHost((void **)&b_z_half,sizeof(float)*ntz);
*/

	d_x      = (float*)malloc(ntx*sizeof(float));
	d_x_half = (float*)malloc(ntx*sizeof(float));    
	d_z      = (float*)malloc(ntz*sizeof(float));
	d_z_half = (float*)malloc(ntz*sizeof(float));


	/*=========================================================
	  Allocate the memory of Seismograms...
	  ========================================================*/

	for(i=0;i<GPU_N;i++)
	{
		plan[i].seismogram_vx_obs=(float *)malloc(sizeof(float)*rnmax*itmax);
		plan[i].seismogram_vx_syn=(float *)malloc(sizeof(float)*rnmax*itmax);
		plan[i].seismogram_vx_rms=(float *)malloc(sizeof(float)*rnmax*itmax);

		plan[i].seismogram_vz_obs=(float *)malloc(sizeof(float)*rnmax*itmax);
		plan[i].seismogram_vz_syn=(float *)malloc(sizeof(float)*rnmax*itmax);
		plan[i].seismogram_vz_rms=(float *)malloc(sizeof(float)*rnmax*itmax);

		plan[i].image_lambda=(float *)malloc(sizeof(float)*ntp);
		plan[i].image_mu=(float *)malloc(sizeof(float)*ntp);

		plan[i].image_vp=(float *)malloc(sizeof(float)*ntp);
		plan[i].image_vs=(float *)malloc(sizeof(float)*ntp);

		plan[i].image_normalize_vp=(float *)malloc(sizeof(float)*ntp);
		plan[i].image_normalize_vs=(float *)malloc(sizeof(float)*ntp);

		plan[i].image_source_vp=(float *)malloc(sizeof(float)*ntp);
		plan[i].image_source_vs=(float *)malloc(sizeof(float)*ntp);

		/*cudaMallocHost((void **)&plan[i].seismogram_vx_obs,sizeof(float)*rnmax*itmax);
		cudaMallocHost((void **)&plan[i].seismogram_vx_syn,sizeof(float)*rnmax*itmax);
		cudaMallocHost((void **)&plan[i].seismogram_vx_rms,sizeof(float)*rnmax*itmax);

		cudaMallocHost((void **)&plan[i].seismogram_vz_obs,sizeof(float)*rnmax*itmax);
		cudaMallocHost((void **)&plan[i].seismogram_vz_syn,sizeof(float)*rnmax*itmax);
		cudaMallocHost((void **)&plan[i].seismogram_vz_rms,sizeof(float)*rnmax*itmax);

		cudaMallocHost((void **)&plan[i].image_lambda,sizeof(float)*ntp);
		cudaMallocHost((void **)&plan[i].image_mu,sizeof(float)*ntp);

		cudaMallocHost((void **)&plan[i].image_vp,sizeof(float)*ntp);
		cudaMallocHost((void **)&plan[i].image_vs,sizeof(float)*ntp);

		plan[i].vx_borders_up=(float *)malloc(sizeof(float)*Lc*itmax*nx);
		plan[i].vx_borders_bottom=(float *)malloc(sizeof(float)*Lc*itmax*nx);
		plan[i].vx_borders_left=(float *)malloc(sizeof(float)*Lc*itmax*(nz-2*Lc));
		plan[i].vx_borders_right=(float *)malloc(sizeof(float)*Lc*itmax*(nz-2*Lc));

		plan[i].vz_borders_up=(float *)malloc(sizeof(float)*Lc*itmax*(nx));
		plan[i].vz_borders_bottom=(float *)malloc(sizeof(float)*Lc*itmax*(nx));
		plan[i].vz_borders_left=(float *)malloc(sizeof(float)*Lc*itmax*(nz-2*Lc));
		plan[i].vz_borders_right=(float *)malloc(sizeof(float)*Lc*itmax*(nz-2*Lc));
*/
	}

	ref_window       =(float*)malloc(sizeof(float)*itmax);
	seis_window      =(float*)malloc(sizeof(float)*itmax);


	/*=========================================================
	  Allocate the memory of image / gradient...
	  ========================================================*/

	Gradient_vp_all=(float*)malloc(sizeof(float)*ntp);
	Gradient_vs_all=(float*)malloc(sizeof(float)*ntp);

	tmp1=(float*)malloc(sizeof(float)*ntp);
	tmp2=(float*)malloc(sizeof(float)*ntp);

	Gradient_lambda_all=(float*)malloc(sizeof(float)*ntp);
	Gradient_mu_all=(float*)malloc(sizeof(float)*ntp);

	Gradient_vp_pre=(float*)malloc(sizeof(float)*np);
	Gradient_vs_pre=(float*)malloc(sizeof(float)*np);

	Gradient_vp    =(float*)malloc(sizeof(float)*np);
	Gradient_vs    =(float*)malloc(sizeof(float)*np);

	dn_vp_pre      =(float*)malloc(sizeof(float)*np);
	dn_vs_pre      =(float*)malloc(sizeof(float)*np);

	dn_vp      =(float*)malloc(sizeof(float)*np);
	dn_vs      =(float*)malloc(sizeof(float)*np);

	un0_vp         =(float*)malloc(sizeof(float)*1);
	un0_vs         =(float*)malloc(sizeof(float)*1);

	//Allocate the CUDA variables of wavefield// 

	variables_malloc(ntx, ntz, ntp, nx, nz,
			pml, Lc, dx, dz, itmax, maxNp,maxNs,
			plan, GPU_N, rnmax, NNmax
			);

	//========================================================
	//  Calculate the reference tiem window...
	//========================================================
	if(reference_window==1)
	{
		for(it=0;it<itmax;it++)
		{
			ref_window[it] =0.0;
			seis_window[it]=1.0;

			if(it<430)
			{
				ref_window[it]=1.0;//250 is choosed from the synthetic seismogram
			}
		}
	}
	else
	{
		for(it=0;it<itmax;it++)
		{
			ref_window[it] =1.0;
			seis_window[it]=1.0;	  	
		}
	}
	if(myid==0)
		printf("Set the reference time window\n");
	//========================================================
	//  Calculate the ricker wave...
	//========================================================

	if(myid==0)
	{
		ricker_wave(rick,itmax,f0,t0,dt,2);
		printf("Ricker wave is done\n");
	}

	MPI_Bcast(rick,itmax,MPI_FLOAT,0,comm);

	//=========================================================
	//  Calculate the ture model.../Or read in the true model
	//=========================================================

	inv_flag=0;
	if(myid==0)
	{
		get_acc_model(vp,vs,rho,ntp,ntx,ntz,pmlc,inv_flag);

		//*************************/*//
		for(ip=0;ip<ntp;ip++)
		{
	//		vs[ip]=vp[ip]/1.732;
		}
		//**************************//

		get_lame_constants(lambda,mu,lambda_plus_two_mu,vp,vs,rho,
				ntp);

		fp=fopen("./output/acc_vp.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		fp=fopen("./output/acc_vs.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&vs[iz*ntx+ix],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		fp=fopen("./output/acc_rho.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&rho[iz*ntx+ix],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		printf("The true model is done\n"); 
	}//end myid

	MPI_Bcast(vp,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(vs,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(rho,ntp,MPI_FLOAT,0,comm);

	MPI_Bcast(lambda,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(mu,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(lambda_plus_two_mu,ntp,MPI_FLOAT,0,comm);
	
	vp_max=0.0;vs_max=0.0;rho_max=0.0;
	vp_min=0.0;vs_min=0.0;rho_min=0.0;

    	get_max_min(vp,ntp,&vp_max,&vp_min);	
    	get_max_min(vs,ntp,&vs_max,&vs_min);
    	get_max_min(rho,ntp,&rho_max,&rho_min);

    	float *Gp,*Gs;
    	Gp=(float*)malloc(sizeof(float)*maxNp*Lp);		//LP_time_space_velocity
    	Gs=(float*)malloc(sizeof(float)*maxNs*Ls);		//LP_time_space_velocity

    	int Np=(int)((vp_max-vp_min)/dvp+1.5);		//四舍五入
    	int Ns=(int)((vs_max-vs_min)/dvs+1.5);
    
    	TE_2M4_2d(vp_min,dvp,Np,Lp,dt,dx,Gp);
    	TE_2M4_2d(vs_min,dvs,Ns,Ls,dt,dx,Gs);

    	fp=fopen("./output/acc_Gp.dat","wb");
    	fs=fopen("./output/acc_Gs.dat","wb");

    	fwrite(Gp,sizeof(float),Np*Lp,fp);
    	fwrite(Gs,sizeof(float),Ns*Ls,fs);
    	fclose(fp);fclose(fs);

    	printf("Np=%d Lp=%d Ns=%d Ls=%d\n",Np,Lp,Ns,Ls);

    	float delta1,delta2;
    	check_2M4_2d(vp_max,dt,dx,dx,Np,Lp,Lp,Gp,Gp,&delta1);
	printf("	Stable P-max=%f\n",1/(delta1/vp_max));
    	if(delta1-1.0>1.0e-4){printf("P-wave unstable!!! r=%f\n",delta1);exit(0);}

   
    	check_2M4_2d(vs_max,dt,dx,dx,Ns,Ls,Ls,Gs,Gs,&delta2);
	printf("	Stable S-max=%f\n",1/(delta2/vs_max));
    	if(delta2-1.0>1.0e-4){printf("S-wave unstable!!! r=%f\n",delta2);exit(0);}

	printf("%f %f\n",delta1,delta2);
	

	if(myid==0)
		printf("vp_max = %f\nvs_max = %f\nrho_max = %f\n",vp_max,vs_max,rho_max); 
	//=========================================================
	//  Calculate the parameters of absorbing layers...
	//========================================================
/*
	get_absorbing_parameters(
			d_x,d_x_half,d_z,d_z_half,
			a_x,a_x_half,a_z,a_z_half,
			b_x,b_x_half,b_z,b_z_half,
			ntz,ntx,nz,nx,pmlc,dx,f0,t0,
			dt,vp_max
			);
*/
	get_pml_parameters(a_x,b_x,a_z,b_z,
			a_x_half,b_x_half,a_z_half,b_z_half,
			ntx,ntz,f0,dx,dz,dt,pmlc);

	if(myid==0)
		printf("ABC parameters are done\n");

	start=clock();
	//
	//=====================================================
	//  Calculate the Observed seismograms...
	//=====================================================

	nsid=ns/(GPU_N*numprocs);
	modsr=ns%(GPU_N*numprocs);
	prcs=modsr/GPU_N;
	if(myid<prcs)
	{
		eachsid=nsid+1;

		offsets=myid*(nsid+1)*GPU_N;
	}
	else
	{
		eachsid=nsid;
		offsets=prcs*(nsid+1)*GPU_N+(myid-prcs)*nsid*GPU_N;
	}

	inv_flag=0;
	if(myid==0)
		printf("Obtain the observed seismograms !\n");

	for(iss=0;iss<eachsid;iss++)
	{
		is=offsets+iss*GPU_N;

		fdtd_cpml_2d_GPU_forward(ntx,ntz,ntp,nx,nz,pml,Lc,rc,dx,dz,
				rick, itmax, dt, myid,vp,vs,
				vp_min,dvp,vs_min,dvs,Gp,Gs,maxNp,maxNs,
				is, ss, plan, GPU_N, rnmax,
				rho,lambda,mu,lambda_plus_two_mu,
				a_x,a_x_half,a_z,a_z_half,
				b_x,b_x_half,b_z,b_z_half,
				vx, vz, 
				sigmaxx, sigmaxxs,sigmazz, sigmaxz,
				inv_flag
				);

		for(i=0;i<GPU_N;i++)
		{
			sprintf(filename,"./output/%dsource_seismogram_vx_obs.dat",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				for(it=0;it<itmax;it++)
				{
					fwrite(&plan[i].seismogram_vx_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);

			sprintf(filename,"./output/%dsource_seismogram_vz_obs.dat",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				for(it=0;it<itmax;it++)
				{
					fwrite(&plan[i].seismogram_vz_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);
		/*
			wavelet_Dec_rest(plan[i].seismogram_vx_obs, itmax, ss[is+i].r_n,plan[i].seismogram_vx_syn,plan[i].seismogram_vx_rms); 
			wavelet_Dec_rest(plan[i].seismogram_vz_obs, itmax, ss[is+i].r_n,plan[i].seismogram_vz_syn,plan[i].seismogram_vz_rms); 

			sprintf(filename,"./output/%dsource_seismogram_vx_obs_scal6.dat",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				for(it=0;it<itmax;it++)
				{
					fwrite(&plan[i].seismogram_vx_syn[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);

			sprintf(filename,"./output/%dsource_seismogram_vz_obs_scal6.dat",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				for(it=0;it<itmax;it++)
				{
					fwrite(&plan[i].seismogram_vz_syn[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);
			sprintf(filename,"./output/%dsource_seismogram_vx_obs_scal5.dat",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				for(it=0;it<itmax;it++)
				{
					fwrite(&plan[i].seismogram_vx_rms[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);

			sprintf(filename,"./output/%dsource_seismogram_vz_obs_scal5.dat",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				for(it=0;it<itmax;it++)
				{
					fwrite(&plan[i].seismogram_vz_rms[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);
		*/
		}//end GPU

	}//end is

	end=clock();
	if(myid==0)
			printf("The cost of the run time is %f seconds\n",
			(double)(end-start)/CLOCKS_PER_SEC);
    return 0;

	if(myid==0)
	{
		////////////////////////////////////////////////
		ini_model_mine(vp,vp_n,ntp,ntz,ntx,pmlc,1);
		ini_model_mine(vs,vs_n,ntp,ntz,ntx,pmlc,2);
		ini_model_mine(rho,rho_n,ntp,ntz,ntx,pmlc,3);
		
		get_lame_constants(lambda,mu,lambda_plus_two_mu,vp,vs,rho,
				ntp);

		fp=fopen("./output/ini_vp.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
			}
		}
		fclose(fp);

		fp=fopen("./output/ini_vs.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&vs[iz*ntx+ix],sizeof(float),1,fp);
			}
		}
		fclose(fp);

		fp=fopen("./output/ini_rho.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&rho[iz*ntx+ix],sizeof(float),1,fp);
			}
		}
		fclose(fp);
	}//end myid

	MPI_Bcast(vp,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(vs,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(rho,ntp,MPI_FLOAT,0,comm);

	MPI_Bcast(vp_n,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(vs_n,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(rho_n,ntp,MPI_FLOAT,0,comm);

	MPI_Bcast(lambda,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(mu,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(lambda_plus_two_mu,ntp,MPI_FLOAT,0,comm);

	// obtain parameters for 4T
	vp_max=0.0;vs_max=0.0;rho_max=0.0;
	vp_min=0.0;vs_min=0.0;rho_min=0.0;

    	get_max_min(vp,ntp,&vp_max,&vp_min);	
    	get_max_min(vs,ntp,&vs_max,&vs_min);
    	get_max_min(rho,ntp,&rho_max,&rho_min);

    	Np=(int)((vp_max-vp_min)/dvp+1.5);		//四舍五入
    	Ns=(int)((vs_max-vs_min)/dvs+1.5);

    	TE_2M4_2d(vp_min,dvp,Np,Lp,dt,dx,Gp);
    	TE_2M4_2d(vs_min,dvs,Ns,Ls,dt,dx,Gs);

    	fp=fopen("./output/ini_Gp.dat","wb");
    	fs=fopen("./output/ini_Gs.dat","wb");

    	fwrite(Gp,sizeof(float),Np*Lp,fp);
    	fwrite(Gs,sizeof(float),Ns*Ls,fs);
    	fclose(fp);fclose(fs);

    	printf("Np=%d Lp=%d Ns=%d Ls=%d\n",Np,Lp,Ns,Ls);

    	check_2M4_2d(vp_max,dt,dx,dx,Np,Lp,Lp,Gp,Gp,&delta1);
    	if(delta1-1.0>1.0e-4){printf("P-wave unstable!!! r=%f\n",delta1);exit(0);}

   
    	check_2M4_2d(vs_max,dt,dx,dx,Ns,Ls,Ls,Gs,Gs,&delta2);
    	if(delta2-1.0>1.0e-4){printf("S-wave unstable!!! r=%f\n",delta2);exit(0);}

	printf("%f %f\n",delta1,delta2);
	
	if(myid==0)
		printf("vp_max = %f\nvs_max = %f\nrho_max = %f\n",vp_max,vs_max,rho_max); 

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//  !	        ITERATION OF FWI IN TIME DOMAIN BEGINS...                      !
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	float sum1p,sum2p,betap;
	float sum1s,sum2s,betas;
//	float sum1r,sum2r,betar;
	float amp_scale=1.0;      
	float Misfit_old,Misfit_new;
	float *Misfit_vx,*Misfit_vz;
	float misfit[itn];
	int m;

	float max_grad_vp=0.0;
	float max_grad_vs=0.0;

	Misfit_vx=(float *)malloc(sizeof(float));
	Misfit_vz=(float *)malloc(sizeof(float));

	Preprocess(nz,nx,dx,dz,P);

	///=======================================================
	//  Back-propagate the RMS wavefields and Construct 
	//  the forward wavefield..Meanwhile the gradients 
	//  of lambda and mu are computed... 
	// ========================================================/

	for(ifreq=0;ifreq<N_ifreq;ifreq++)
	{
		if(myid==0)
		{
			printf("====================\n");
			printf("   FREQUENCY == f[%d]-->%f\n",ifreq+1,f[ifreq]);
			printf("====================\n");

			t0=1/f[ifreq];
			ricker_wave(rick,itmax,f[ifreq],t0,dt,2);
			printf("Ricker wave is done again\n");
		}
		MPI_Barrier(comm);
		MPI_Bcast(rick,itmax,MPI_FLOAT,0,comm);

		////reference window length///
		for(it=0;it<itmax;it++)
		{
			tmpwindow=(float)it/tc[ifreq];
			ref_window[it]=1.0/(1.0+pow(tmpwindow,40));//2n=10,n=5

			tmpwindow=(float)it/(2.2/f0/dt);
			seis_window[it]=1.0/(1.0+pow(tmpwindow,40));//;
		}

		for(ip=0;ip<np;ip++)
		{
			dn_vp[ip]=0.0;
			dn_vs[ip]=0.0;
		}

		for(iter=0;iter<itn;iter++)
		{
			if(myid==0)
			{
				printf("====================\n");
				printf("ITERATION == %d\n",iter+1);
			}

			for(ip=0;ip<ntp;ip++)
			{
				Gradient_lambda_all[ip]=0.0;
				Gradient_mu_all[ip]=0.0;

				Gradient_vp_all[ip]=0.0;
				Gradient_vs_all[ip]=0.0;
			}

			for(i=0;i<GPU_N;i++)
			{
				for(ip=0;ip<ntp;ip++)
				{
					plan[i].image_lambda[ip]=0.0;
					plan[i].image_mu[ip]=0.0;

					//plan[i].image_vp[ip]=0.0;
					//plan[i].image_vs[ip]=0.0;
					plan[i].image_normalize_vp[ip]=0.0;
					plan[i].image_normalize_vs[ip]=0.0;
				}
			}
			for(ip=0;ip<ntp;ip++)
			{
				tmp1[ip]=0.0;
				tmp2[ip]=0.0;
			}

			*Misfit_vx=0.0;
			*Misfit_vz=0.0;

			inv_flag=1;

			Misfit_old=0.0;

			////////   FORWARD   /////////
			for(iss=0;iss<eachsid;iss++)
			{
				is=offsets+iss*GPU_N;
			//for(is=0;is<ns;is=is+GPU_N*numprocs)
			//	
				fdtd_cpml_2d_GPU_forward(ntx,ntz,ntp,nx,nz,pml,Lc,rc,dx,dz,
						rick, itmax, dt, myid,vp,vs,
						vp_min,dvp,vs_min,dvs,Gp,Gs,maxNp,maxNs,
						is, ss, plan, GPU_N, rnmax,
						rho,lambda,mu,lambda_plus_two_mu,
						a_x,a_x_half,a_z,a_z_half,
						b_x,b_x_half,b_z,b_z_half,
						vx, vz, 
						sigmaxx, sigmaxxs, sigmazz, sigmaxz,
						inv_flag
						);

				// READ IN OBSERVED SEISMOGRAMS...  
				for(i=0;i<GPU_N;i++)
				{
					/*if(ifreq<1)
					{

						sprintf(filename,"./output/%dsource_seismogram_vx_obs_scal6.dat",is+i+1);
						fp=fopen(filename,"rb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fread(&plan[i].seismogram_vx_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);

						sprintf(filename,"./output/%dsource_seismogram_vz_obs_scal6.dat",is+i+1);
						fp=fopen(filename,"rb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fread(&plan[i].seismogram_vz_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);

					}
					else if(ifreq<3)
					{

						sprintf(filename,"./output/%dsource_seismogram_vx_obs_scal5.dat",is+i+1);
						fp=fopen(filename,"rb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fread(&plan[i].seismogram_vx_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);

						sprintf(filename,"./output/%dsource_seismogram_vz_obs_scal5.dat",is+i+1);
						fp=fopen(filename,"rb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fread(&plan[i].seismogram_vz_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);

					}
					else
					*/{
						sprintf(filename,"./output/%dsource_seismogram_vx_obs.dat",is+i+1);
						fp=fopen(filename,"rb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fread(&plan[i].seismogram_vx_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);

						sprintf(filename,"./output/%dsource_seismogram_vz_obs.dat",is+i+1);
						fp=fopen(filename,"rb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fread(&plan[i].seismogram_vz_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);
					}


					////convolution-based residuals////
					congpu_fre(plan[i].seismogram_vx_syn,plan[i].seismogram_vx_obs,plan[i].seismogram_vx_rms,Misfit_vx,i,ref_window,seis_window,itmax,dt,dx,is+i,ss[is+i].r_n,ss[is+i].s_ix,ss[is+i].r_ix,pmlc);
					congpu_fre(plan[i].seismogram_vz_syn,plan[i].seismogram_vz_obs,plan[i].seismogram_vz_rms,Misfit_vz,i,ref_window,seis_window,itmax,dt,dx,is+i,ss[is+i].r_n,ss[is+i].s_ix,ss[is+i].r_ix,pmlc);

					////output seismogram////
					if(iter==0||iter==itn-1||iter%5==0)
					{
						///output synthetic seismogram////
						sprintf(filename,"./output/%dsource_seismogram_vx_%dsyn.dat",is+i+1,iter+1);
						fp=fopen(filename,"wb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fwrite(&plan[i].seismogram_vx_syn[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);

						sprintf(filename,"./output/%dsource_seismogram_vz_%dsyn.dat",is+i+1,iter+1);
						fp=fopen(filename,"wb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fwrite(&plan[i].seismogram_vz_syn[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);

						////output rms seismogram////
						sprintf(filename,"./output/%dsource_seismogram_vx_%drms.dat",is+i+1,iter+1);
						fp=fopen(filename,"wb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fwrite(&plan[i].seismogram_vx_rms[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);

						sprintf(filename,"./output/%dsource_seismogram_vz_%drms.dat",is+i+1,iter+1);
						fp=fopen(filename,"wb");
						for(ix=0;ix<ss[is+i].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								fwrite(&plan[i].seismogram_vz_rms[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
							}
						}
						fclose(fp);
					}
				}//end GPU

				//=======================================================//
				//  Calculate the IMAGE/GRADIENT OF RTM/FWI...
				// ======================================================//

				for(i=0;i<GPU_N;i++)
				{
					for(ip=0;ip<ntp;ip++)
					{
						plan[i].image_lambda[ip]=0.0;
						plan[i].image_mu[ip]=0.0;

						plan[i].image_vp[ip]=0.0;
						plan[i].image_vs[ip]=0.0;

						plan[i].image_source_vp[ip]=0.0;
						plan[i].image_source_vs[ip]=0.0;
					}
				}

				fdtd_cpml_2d_GPU_backward(ntx,ntz,ntp,nx,nz,pml,Lc,rc,dx,dz,
						rick, itmax, dt, myid,vp,vs,
						vp_min,dvp,vs_min,dvs,Gp,Gs,maxNp,maxNs,
						is, ss, plan, GPU_N, rnmax,
						rho,lambda,mu,lambda_plus_two_mu,
						a_x,a_x_half,a_z,a_z_half,
						b_x,b_x_half,b_z,b_z_half,
						vx, vz, 
						sigmaxx, sigmaxxs,sigmazz, sigmaxz
						);


				//normalize
				for(i=0;i<GPU_N;i++)
				{
					max_grad_vp=0.0;
					max_grad_vs=0.0;
					//maximum_vector(plan[i].image_vp, ntp, max_grad_vp);
					//maximum_vector(plan[i].image_vs, ntp, max_grad_vs);

					/*for(ip=0;ip<ntp;ip++)
					{
						if((plan[i].image_vs[ip])>max_grad_vs)
							max_grad_vs=(plan[i].image_vs[ip]);
						if((plan[i].image_vp[ip])>max_grad_vp)
							max_grad_vp=(plan[i].image_vp[ip]);
					}*/

					for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
					{
						for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
						{
							ip=iz*ntx+ix;
							if(fabs(plan[i].image_vs[ip])>max_grad_vs)
								max_grad_vs=fabs(plan[i].image_vs[ip]);
							if(fabs(plan[i].image_vp[ip])>max_grad_vp)
								max_grad_vp=fabs(plan[i].image_vp[ip]);
						}
					}


					for(ip=0;ip<ntp;ip++)
					{
						//plan[i].image_normalize_vp[ip]+=plan[i].image_vp[ip];
						//plan[i].image_normalize_vs[ip]+=plan[i].image_vs[ip];

						plan[i].image_normalize_vp[ip]+=plan[i].image_vp[ip]/(plan[i].image_source_vp[ip]+max_grad_vp*1.0e-5);
						plan[i].image_normalize_vs[ip]+=plan[i].image_vs[ip]/(plan[i].image_source_vp[ip]+max_grad_vp*1.0e-5);
					}
				}

			}//end is

			for(i=0;i<GPU_N;i++)
			{
				for(ip=0;ip<ntp;ip++)
				{
					tmp1[ip]+=plan[i].image_normalize_vp[ip];
					tmp2[ip]+=plan[i].image_normalize_vs[ip];
				}
			}//end GPU
			tmpold=*Misfit_vx+*Misfit_vz;

			MPI_Barrier(comm);

			MPI_Allreduce(tmp1,Gradient_lambda_all,ntp,MPI_FLOAT,MPI_SUM,comm);
			MPI_Allreduce(tmp2,Gradient_mu_all,ntp,MPI_FLOAT,MPI_SUM,comm);
			MPI_Allreduce(&tmpold,&Misfit_old,1,MPI_FLOAT,MPI_SUM,comm);

			misfit[iter]=Misfit_old;      

			if(myid==0)
				printf("== ** Misfit_old ** == %e\n",Misfit_old);

			for(ip=0;ip<ntp;ip++)
			{
				Gradient_vp_all[ip]=2.0*vp[ip]/(rho[ip]*pow((3*pow(vp[ip],2)-4*pow(vs[ip],2)),2))*Gradient_lambda_all[ip]*amp_scale;

				Gradient_vs_all[ip]=1.0/(rho[ip]*pow(vs[ip],3))*(Gradient_mu_all[ip]
						+(8.0*pow(vp[ip],2)*pow(vs[ip],2)-8.0*pow(vs[ip],4)-3.0*pow(vp[ip],4))/pow((4.0*pow(vs[ip],2)-3.0*pow(vp[ip],2)),2)*Gradient_lambda_all[ip])*amp_scale;

				//			Gradient_rho_all[ip]=(image_rho[ip]+(vp_n[ip]*vp_n[ip]-2.0*vs_n[ip]*vs_n[ip])*image_lambda[ip]
				//					+vs_n[ip]*vs_n[ip]*image_mu[ip])*amp_scale;
			}

			//=========================================================
			//  Applied the conjugate gradient method in FWI
			//=========================================================//

			//  ---------------------------------------------------------------
			//	Gradient of P wave velocity...
			//	-------------------------------------------------------------//

			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				{
					ip=iz*ntx+ix;
					ipp=(iz-pmlc)*nx+ix-pmlc;

					Gradient_vp[ipp]=Gradient_vp_all[ip];    // inner gradient...        
					Gradient_vs[ipp]=Gradient_vs_all[ip];    // inner gradient...        
					//				Gradient_rho[ipp]=Gradient_rho_all[ip];    // inner gradient...
				}
			}

			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				{
					ip=iz*ntx+ix;
					ipp=(iz-pmlc)*nx+ix-pmlc;

					if(iz>ntz-pmlc-3)
					{
						//				Gradient_vp[ipp]=Gradient_vp[(nz-3)*nx+ix-pmlc];
						Gradient_vs[ipp]=Gradient_vs[(nz-3)*nx+ix-pmlc];
					}
					if(ix<pmlc+1)
					{
						//				Gradient_vp[ipp]=Gradient_vp[(iz-pmlc)*nx+1];
						Gradient_vs[ipp]=Gradient_vs[(iz-pmlc)*nx+1];
					}
					if(ix>ntx-pmlc-3)
					{
						//				Gradient_vp[ipp]=Gradient_vp[(iz-pmlc)*nx+nx-3];
						Gradient_vs[ipp]=Gradient_vs[(iz-pmlc)*nx+nx-3];
					}

				}
			}

			for(ip=0;ip<np;ip++)
			{
				iz=ip/nx;

				Gradient_vp[ip]=Gradient_vp[ip]*P[iz];
				Gradient_vs[ip]=Gradient_vs[ip]*P[iz];
				//			Gradient_rho[ip]=Gradient_rho[ip]*P[iz];
			}

			//==========================================================
			//  Applying the conjugate gradient method...
			//==========================================================//

			if(iter==0)
			{
				for(ip=0;ip<np;ip++)
				{
					dn_vp[ip]=-Gradient_vp[ip];
					dn_vs[ip]=-Gradient_vs[ip];
					//				dn_rho[ip]=-Gradient_rho[ip];
				}
			}

			if(iter>=1)
			{
				sum1p=0.0;sum1s=0.0;//sum1r=0.0;
				sum2p=0.0;sum2s=0.0;//sum2r=0.0;

				for(ip=0;ip<np;ip++)
				{
					sum1p=sum1p+Gradient_vp[ip]*Gradient_vp[ip];
					sum1s=sum1s+Gradient_vs[ip]*Gradient_vs[ip];
					//				sum1r=sum1s+Gradient_rho[ip]*Gradient_rho[ip];

					sum2p=sum2p+Gradient_vp_pre[ip]*Gradient_vp_pre[ip];
					sum2s=sum2s+Gradient_vs_pre[ip]*Gradient_vs_pre[ip];
					//				sum2r=sum2s+Gradient_rho_pre[ip]*Gradient_rho_pre[ip];
				}

				betap=sum1p/sum2p;
				betas=sum1s/sum2s;
				//			betar=sum1r/sum2r;

				for(ip=0;ip<np;ip++)
				{
					dn_vp[ip]=-Gradient_vp[ip]+betap*dn_vp_pre[ip];
					dn_vs[ip]=-Gradient_vs[ip]+betas*dn_vs_pre[ip];
					//				dn_rho[ip]=-Gradient_rho[ip]+betar*dn_rho_pre[ip];
				}
			}

			for(ip=0;ip<np;ip++)
			{
				Gradient_vp_pre[ip]=Gradient_vp[ip];
				dn_vp_pre[ip]=dn_vp[ip];

				Gradient_vs_pre[ip]=Gradient_vs[ip];
				dn_vs_pre[ip]=dn_vs[ip];
				//
				//			Gradient_rho_pre[ip]=Gradient_rho[ip];
				//			dn_rho_pre[ip]=dn_rho[ip];
			}

			//	---------------------------------------------------------------
			//	------------calculate the step --------------------------------
			//	---------------------------------------------------------------

			ini_step(dn_vp,np,un0_vp,vp_max,1);
			ini_step(dn_vs,np,un0_vs,vs_max,2);
			//		ini_step(dn_rho,np,un0_rho,rho_max,3);

			if(myid==0)
				printf("   an_vp == %e , an_vs == %e\n",*un0_vp,*un0_vs);

			update_model(vp,vp_n,dn_vp,un0_vp,ntp, ntz, ntx, pmlc,1);
			update_model(vs,vs_n,dn_vs,un0_vs,ntp, ntz, ntx, pmlc,2);
			//			update_model(rho,rho_n,dn_rho,un0_rho,ntp, ntz, ntx, pmlc);

			get_lame_constants(lambda,mu,lambda_plus_two_mu,vp,vs,rho,ntp);


			// obtain parameters for 4T
			vp_max=0.0;vs_max=0.0;rho_max=0.0;
			vp_min=0.0;vs_min=0.0;rho_min=0.0;

		    	get_max_min(vp,ntp,&vp_max,&vp_min);	
		    	get_max_min(vs,ntp,&vs_max,&vs_min);
		    	get_max_min(rho,ntp,&rho_max,&rho_min);

		    	Np=(int)((vp_max-vp_min)/dvp+1.5);		//四舍五入
		    	Ns=(int)((vs_max-vs_min)/dvs+1.5);

		    	TE_2M4_2d(vp_min,dvp,Np,Lp,dt,dx,Gp);
		    	TE_2M4_2d(vs_min,dvs,Ns,Ls,dt,dx,Gs);

			sprintf(filename,"./output/%dGp.dat",iter+1);
			fp=fopen(filename,"wb");
		    	fwrite(Gp,sizeof(float),Np*Lp,fp);
		    	fclose(fp);

			sprintf(filename,"./output/%dGs.dat",iter+1);
			fp=fopen(filename,"wb");
		    	fwrite(Gs,sizeof(float),Np*Lp,fp);
		    	fclose(fp);

		    	printf("Np=%d Lp=%d Ns=%d Ls=%d\n",Np,Lp,Ns,Ls);

		    	check_2M4_2d(vp_max,dt,dx,dx,Np,Lp,Lp,Gp,Gp,&delta1);
		    	if(delta1-1.0>1.0e-4){printf("P-wave unstable!!! r=%f\n",delta1);exit(0);}

		   
		    	check_2M4_2d(vs_max,dt,dx,dx,Ns,Ls,Ls,Gs,Gs,&delta2);
		    	if(delta2-1.0>1.0e-4){printf("S-wave unstable!!! r=%f\n",delta2);exit(0);}

			printf("%f %f\n",delta1,delta2);
	
			if(myid==0)
				printf("vp_max = %f\nvs_max = %f\nrho_max = %f\n",vp_max,vs_max,rho_max); 

			for(ip=0;ip<ntp;ip++)
			{
				vp_n[ip]=vp[ip];
				vs_n[ip]=vs[ip];
				//			rho_n[ip]=rho[ip];
			}

			//==========================================================
			//  Output the updated model such as vp,vs,...
			//==========================================================
			if(myid==0)
			{
				/*
				   sprintf(filename,"./output/%dimage_lambda.dat",iter+1);
				   fp=fopen(filename,"wb");
				   fwrite(&Gradient_lambda_all[0],sizeof(float),ntp,fp);
				   fclose(fp);

				   sprintf(filename,"./output/%dimage_mu.dat",iter+1);
				   fp=fopen(filename,"wb");
				   fwrite(&Gradient_mu_all[0],sizeof(float),ntp,fp);
				   fclose(fp);
				   */
				sprintf(filename,"./output/%dimage_vp.dat",iter+1);
				fp=fopen(filename,"wb");
				fwrite(&Gradient_vp[0],sizeof(float),np,fp);
				fclose(fp);

				sprintf(filename,"./output/%dimage_vs.dat",iter+1);
				fp=fopen(filename,"wb");
				fwrite(&Gradient_vs[0],sizeof(float),np,fp);
				fclose(fp);

				sprintf(filename,"./output/%dGradient_vpf.dat",iter+1);
				fp=fopen(filename,"wb");
				fwrite(&dn_vp[0],sizeof(float),np,fp);
				fclose(fp);

				sprintf(filename,"./output/%dGradient_vsf.dat",iter+1);
				fp=fopen(filename,"wb");
				fwrite(&dn_vs[0],sizeof(float),np,fp);
				fclose(fp);

				sprintf(filename,"./output/%dvpf.dat",iter+1);
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				{
					for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
					{
						fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				sprintf(filename,"./output/%dvsf.dat",iter+1);
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				{
					for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
					{
						fwrite(&vs[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				if(iter==0)
				{
					sprintf(filename,"./output/%d_%dGradient_vpf.dat",ifreq+1,iter+1);
					fp=fopen(filename,"wb");
					fwrite(&dn_vp[0],sizeof(float),np,fp);
					fclose(fp);

					sprintf(filename,"./output/%d_%dGradient_vsf.dat",ifreq+1,iter+1);
					fp=fopen(filename,"wb");
					fwrite(&dn_vs[0],sizeof(float),np,fp);
					fclose(fp);
				}


			}

			MPI_Barrier(comm);

		}//end iter

		if(myid==0)
		{
			sprintf(filename,"./output/%difreq_vp.dat",ifreq+1);
			fp=fopen(filename,"wb");
			for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
			{
				for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
				{
					fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);

			sprintf(filename,"./output/%difreq_vs.dat",ifreq+1);
			fp=fopen(filename,"wb");
			for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
			{
				for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
				{
					fwrite(&vs[iz*ntx+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);

			sprintf(filename,"./output/misfit_%difreq.txt",ifreq+1);
			fp=fopen(filename,"w");
			for(iter=0;iter<itn;iter++)
			{
				fprintf(fp,"%e\r\n",misfit[iter]);
			}
			fclose(fp);
		}

		MPI_Barrier(comm);
	}//end frequency

	end=clock();
	if(myid==0)
			printf("The cost of the run time is %f seconds\n",
			(double)(end-start)/CLOCKS_PER_SEC);
	/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	  !	        ITERATION OF FWI IN TIME DOMAIN ENDS...                        !
	  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

	variables_free(ntx, ntz, ntp, nx, nz,
		pml, Lc, dx, dz, itmax,
		plan, GPU_N, rnmax, NNmax
		);

	free(rc); 

	for(is=0;is<ns;is++)
	{
		free(ss[is].r_ix);
		free(ss[is].r_iz);
	} 
	
	free(rick); 
	//cudaFreeHost(rick); 

	//free the memory of P velocity
	free(vp);
	free(vp_n);
	//free the memory of S velocity
	free(vs); 
	free(vs_n);
	//free the memory of Density
	free(rho); 
	free(rho_n);
	//free the memory of lambda
	free(lambda); 
	free(mu); 
	free(lambda_plus_two_mu); 

	free(a_x);
	free(a_x_half);
	free(a_z);
	free(a_z_half);

	free(b_x);
	free(b_x_half);
	free(b_z);
	free(b_z_half);
/*
	//free the memory of lambda
	cudaFreeHost(lambda); 
	cudaFreeHost(mu); 
	cudaFreeHost(lambda_plus_two_mu); 

	cudaFreeHost(a_x);
	cudaFreeHost(a_x_half);
	cudaFreeHost(a_z);
	cudaFreeHost(a_z_half);

	cudaFreeHost(b_x);
	cudaFreeHost(b_x_half);
	cudaFreeHost(b_z);
	cudaFreeHost(b_z_half);
	*/
	free(d_x);
	free(d_x_half);
	free(d_z);
	free(d_z_half);

	for(i=0;i<GPU_N;i++)
	{
		free(plan[i].seismogram_vx_obs);
		free(plan[i].seismogram_vx_syn); 
		free(plan[i].seismogram_vx_rms);

		free(plan[i].seismogram_vz_obs);
		free(plan[i].seismogram_vz_syn); 
		free(plan[i].seismogram_vz_rms);

		free(plan[i].image_lambda);
		free(plan[i].image_mu);

		free(plan[i].image_vp);
		free(plan[i].image_vs);

		free(plan[i].image_normalize_vp);
		free(plan[i].image_normalize_vs);

		free(plan[i].image_source_vp);
		free(plan[i].image_source_vs);
		/*
		cudaFreeHost(plan[i].seismogram_vx_obs);
		cudaFreeHost(plan[i].seismogram_vx_syn); 
		cudaFreeHost(plan[i].seismogram_vx_rms);

		cudaFreeHost(plan[i].seismogram_vz_obs);
		cudaFreeHost(plan[i].seismogram_vz_syn); 
		cudaFreeHost(plan[i].seismogram_vz_rms);

		cudaFreeHost(plan[i].image_lambda);
		cudaFreeHost(plan[i].image_mu);

		cudaFreeHost(plan[i].image_vp);
		cudaFreeHost(plan[i].image_vs);

		free(plan[i].vx_borders_up);
		free(plan[i].vx_borders_bottom);
		free(plan[i].vx_borders_left);
		free(plan[i].vx_borders_right);

		free(plan[i].vz_borders_up);
		free(plan[i].vz_borders_bottom);
		free(plan[i].vz_borders_left);
		free(plan[i].vz_borders_right);
		*/
	}

	free(ref_window);
	free(seis_window);

	free(Gradient_vp_all);
	free(Gradient_vs_all);

	free(Gradient_lambda_all);
	free(Gradient_mu_all);

	free(tmp1);
	free(tmp2);

	free(Gradient_vp_pre);
	free(Gradient_vs_pre);

	free(Gradient_vp);
	free(Gradient_vs);

	free(dn_vp_pre);
	free(dn_vs_pre);

	free(dn_vp);
	free(dn_vs);

	free(un0_vp);
	free(un0_vs);

	free(Misfit_vx);
	free(Misfit_vz);

	MPI_Barrier(comm);
	MPI_Finalize();
}


/*==========================================================
  This subroutine is used for calculating the parameters of 
  absorbing layers
  ===========================================================*/

void get_absorbing_parameters(
		float *d_x, float *d_x_half, 
		float *d_z, float *d_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half,
		int ntz, int ntx, int nz, int nx,
		int pml, float dx, float f0, float t0, float dt, float vp_max)

{
	int   N=2;
	int   iz,ix;

	float thickness_of_pml;
	float Rc=1.0e-5;

	float d0;
	float pi=3.1415927;
	float alpha_max=pi*15;

	float Vpmax;

	float k_x,k_x_half;
	float k_z,k_z_half;

	float *alpha_x,*alpha_x_half;
	float *alpha_z,*alpha_z_half;

	float x_start,x_end,delta_x;
	float z_start,z_end,delta_z;
	float x_current,z_current;

	Vpmax=vp_max;

	thickness_of_pml=(pml-1)*dx;

	d0=-(N+1)*Vpmax*log(Rc)/(2.0*thickness_of_pml);

	alpha_x      = (float*)malloc(ntx*sizeof(float));
	alpha_x_half = (float*)malloc(ntx*sizeof(float));

	alpha_z      = (float*)malloc(ntz*sizeof(float));
	alpha_z_half = (float*)malloc(ntz*sizeof(float));

	//--------------------initialize the vectors--------------

	for(ix=0;ix<ntx;ix++)
	{
		a_x[ix]          = 0.0;
		a_x_half[ix]     = 0.0;
		b_x[ix]          = 0.0;
		b_x_half[ix]     = 0.0;
		d_x[ix]          = 0.0;
		d_x_half[ix]     = 0.0;
		alpha_x[ix]      = 0.0;
		alpha_x_half[ix] = 0.0;
	}
	k_x          = 1.0;
	k_x_half     = 1.0;

	for(iz=0;iz<ntz;iz++)
	{
		a_z[iz]          = 0.0;
		a_z_half[iz]     = 0.0;
		b_z[iz]          = 0.0;
		b_z_half[iz]     = 0.0;
		d_z[iz]          = 0.0;
		d_z_half[iz]     = 0.0;

		alpha_z[iz]      = 0.0;
		alpha_z_half[iz] = 0.0;
	}
	k_z          = 1.0;
	k_z_half     = 1.0;

	// X direction

	x_start=pml*dx;
	x_end=(ntx-pml-1)*dx;

	// Integer points
	for(ix=0;ix<ntx;ix++)
	{ 
		x_current=ix*dx;

		// LEFT EDGE
		if(x_current<=x_start)
		{
			delta_x=x_start-x_current;
			d_x[ix]=d0*pow(delta_x/thickness_of_pml,2);
			alpha_x[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}

		// RIGHT EDGE      
		if(x_current>=x_end)
		{
			delta_x=x_current-x_end;
			d_x[ix]=d0*pow(delta_x/thickness_of_pml,2);
			alpha_x[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}
	}


	// Half Integer points
	for(ix=0;ix<ntx;ix++)
	{
		x_current=(ix+0.5)*dx;

		if(x_current<=x_start)
		{
			delta_x=x_start-x_current;
			d_x_half[ix]=d0*pow(delta_x/thickness_of_pml,2);
			alpha_x_half[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}

		if(x_current>=x_end)
		{
			delta_x=x_current-x_end;
			d_x_half[ix]=d0*pow(delta_x/thickness_of_pml,2);
			alpha_x_half[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}
	}

	for (ix=0;ix<ntx;ix++)
	{
		if(alpha_x[ix]<0.0)
		{
			alpha_x[ix]=0.0;
		}
		if(alpha_x_half[ix]<0.0)
		{
			alpha_x_half[ix]=0.0;
		}

		b_x[ix]=exp(-(d_x[ix]/k_x+alpha_x[ix])*dt);

		if(d_x[ix] > 1.0e-6)
		{
			a_x[ix]=d_x[ix]/(k_x*(d_x[ix]+k_x*alpha_x[ix]))*(b_x[ix]-1.0);
		}

		b_x_half[ix]=exp(-(d_x_half[ix]/k_x_half+alpha_x_half[ix])*dt);

		if(d_x_half[ix] > 1.0e-6)
		{
			a_x_half[ix]=d_x_half[ix]/(k_x_half*(d_x_half[ix]+k_x_half*alpha_x_half[ix]))*(b_x_half[ix]-1.0);
		}
	}

	// Z direction

	z_start=pml*dx;
	z_end=(ntz-pml-1)*dx;

	// Integer points
	for(iz=0;iz<ntz;iz++)
	{ 
		z_current=iz*dx;

		// LEFT EDGE
		if(z_current<=z_start)
		{
			delta_z=z_start-z_current;
			d_z[iz]=d0*pow(delta_z/thickness_of_pml,2);
			alpha_z[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}

		// RIGHT EDGE      
		if(z_current>=z_end)
		{
			delta_z=z_current-z_end;
			d_z[iz]=d0*pow(delta_z/thickness_of_pml,2);
			alpha_z[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}
	}

	// Half Integer points
	for(iz=0;iz<ntz;iz++)
	{
		z_current=(iz+0.5)*dx;

		if(z_current<=z_start)
		{
			delta_z=z_start-z_current;
			d_z_half[iz]=d0*pow(delta_z/thickness_of_pml,2);
			alpha_z_half[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}

		if(z_current>=z_end)
		{
			delta_z=z_current-z_end;
			d_z_half[iz]=d0*pow(delta_z/thickness_of_pml,2);
			alpha_z_half[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}
	}

	for (iz=0;iz<ntz;iz++)
	{
		if(alpha_z[iz]<0.0)
		{
			alpha_z[iz]=0.0;
		}
		if(alpha_z_half[iz]<0.0)
		{
			alpha_z_half[iz]=0.0;
		}

		b_z[iz]=exp(-(d_z[iz]/k_z+alpha_z[iz])*dt);

		if(d_z[iz]>1.0e-6)
		{
			a_z[iz]=d_z[iz]/(k_z*(d_z[iz]+k_z*alpha_z[iz]))*(b_z[iz]-1.0);
		}

		b_z_half[iz]=exp(-(d_z_half[iz]/k_z_half+alpha_z_half[iz])*dt);

		if(d_z_half[iz]>1.0e-6)
		{
			a_z_half[iz]=d_z_half[iz]/(k_z_half*(d_z_half[iz]+k_z_half*alpha_z_half[iz]))*(b_z_half[iz]-1.0);
		}
	}

	free(alpha_x);
	free(alpha_x_half);
	free(alpha_z);
	free(alpha_z_half);

	return;

}

void get_pml_parameters(float *i_a_x,float *i_b_x,float *i_a_z,float *i_b_z,
                        float *h_a_x,float *h_b_x,float *h_a_z,float *h_b_z,
                        int nxx,int nzz,float fm,float dx,float dz,float dt,
                        int pmlc)
{
        float pi=3.1415926;
		float alpha_max=pi*fm,dmaxx,dmaxz,widthx,widthz;
        float Re=1e-5,vmax=8000;
        int n1=2,n2=1,n3=1,i,j;
        float temp,temp1,temp2,temp3,temp4;
        
        float *int_dx,*int_dz,*int_alphax,*int_alphaz;
        float *half_dx,*half_dz,*half_alphax,*half_alphaz;
        
        int_dx=(float*)malloc(sizeof(float)*nxx);
        half_dx=(float*)malloc(sizeof(float)*nxx);
        int_alphax=(float*)malloc(sizeof(float)*nxx);
        half_alphax=(float*)malloc(sizeof(float)*nxx);
        int_dz=(float*)malloc(sizeof(float)*nzz);
        half_dz=(float*)malloc(sizeof(float)*nzz);
        int_alphaz=(float*)malloc(sizeof(float)*nzz);
        half_alphaz=(float*)malloc(sizeof(float)*nzz);
        
        int pmlx=pmlc,pmlz=pmlc;
          
        widthx=pmlx*dx;widthz=pmlz*dz;
        dmaxx=(1+n1+n2)*vmax*log(1.0/Re)/(2.0*widthx);
        dmaxz=(1+n1+n2)*vmax*log(1.0/Re)/(2.0*widthz);
     
        // integer absorbing parameters
        for(i=0;i<pmlx;i++)
        {
            temp1=pow(1.0*(pmlx-1-i)/pmlx,n1+n2);
            int_dx[i]=dmaxx*temp1;
         
            temp3=pow(1.0*i/(pmlx-1),n3);
            int_alphax[i]=alpha_max*temp3;

            int_dx[nxx-1-i]=int_dx[i];
            int_alphax[nxx-1-i]=int_alphax[i];
        }
        for(i=pmlx;i<nxx-pmlx;i++)
        {
           int_dx[i]=0.0;
           int_alphax[i]=int_alphax[pmlx-1];
        }
        
        for(j=0;j<pmlz;j++)
        {
            temp1=pow(1.0*(pmlz-1-j)/pmlz,n1+n2);
            int_dz[j]=dmaxz*temp1;
         
            temp3=pow(1.0*j/(pmlz-1),n3);
            int_alphaz[j]=alpha_max*temp3;

            int_dz[nzz-1-j]=int_dz[j];
            int_alphaz[nzz-1-j]=int_alphaz[j];
        }
        for(j=pmlz;j<nzz-pmlz;j++)
        {
            int_dz[j]=0.0;
            int_alphaz[j]=int_alphaz[pmlz-1];
        }
     
        // half absorbing parameters
        for(i=0;i<pmlx-1;i++)
        {
            temp2=pow(1.0*(pmlx-1.5-i)/pmlx,n1+n2);
            half_dx[i]=dmaxx*temp2;
            half_dx[nxx-2-i]=half_dx[i];
         
            temp4=pow(1.0*(i+0.5)/(pmlx-1),n3);
            half_alphax[i]=alpha_max*temp4;
            half_alphax[nxx-2-i]=half_alphax[i];
        }
        for(i=pmlx-1;i<nxx-pmlx;i++)
        {
             half_dx[i]=0.0;
             half_alphax[i]=half_alphax[pmlx-2];
        }
        half_dx[nxx-1]=0.0;
        half_alphax[nxx-1]=half_alphax[nxx-2];
        
        for(j=0;j<pmlz-1;j++)
        {
            temp2=pow(1.0*(pmlz-1.5-j)/pmlz,n1+n2);
            half_dz[j]=dmaxz*temp2;
            half_dz[nzz-2-j]=half_dz[j];
         
            temp4=pow(1.0*(j+0.5)/(pmlz-1),n3);
            half_alphaz[j]=alpha_max*temp4;
            half_alphaz[nzz-2-j]=half_alphaz[j];
        }
        for(j=pmlz-1;j<nzz-pmlz;j++)
        {
            half_dz[j]=0.0;
            half_alphaz[j]=half_alphaz[pmlz-2];;
        }
        half_dz[nzz-1]=0.0;
        half_alphaz[nzz-1]=half_alphaz[nzz-2];
    
        for(i=0;i<nxx;i++)
        {
            temp=int_dx[i]+int_alphax[i];
            i_b_x[i]=exp(-dt*temp);
            i_a_x[i]=int_dx[i]/temp*(i_b_x[i]-1.0);
        
            temp=half_dx[i]+half_alphax[i];
            h_b_x[i]=exp(-dt*temp);
            h_a_x[i]=half_dx[i]/temp*(h_b_x[i]-1.0);
        }
     
        for(j=0;j<nzz;j++)
        {
            temp=int_dz[j]+int_alphaz[j];
            i_b_z[j]=exp(-dt*temp);
            i_a_z[j]=int_dz[j]/temp*(i_b_z[j]-1.0);
        
            temp=half_dz[j]+half_alphaz[j];
            h_b_z[j]=exp(-dt*temp);
            h_a_z[j]=half_dz[j]/temp*(h_b_z[j]-1.0);
        }
        
        free(int_dx);free(int_dz);free(int_alphax);free(int_alphaz);
        free(half_dx);free(half_dz);free(half_alphax);free(half_alphaz);
        
        return;
}

/*==========================================================
  This subroutine is used for initializing the true model...
  ===========================================================*/

void get_acc_model(float *vp, float *vs, float *rho, int ntp, int ntx, int ntz, int pml, int inv_flag)
{
	int ip,ipp,iz,ix;
	// THE MODEL    
	FILE *fp;

	fp=fopen("./input/overthrust_acc_vp_170_800.dat","rb");
	for(ix=pml;ix<ntx-pml;ix++)
	{
		for(iz=pml;iz<ntz-pml;iz++)
		{
			ip=iz*ntx+ix;
			fread(&vp[ip],sizeof(float),1,fp);       
		}
		vp[ip]=3000;
	}
	fclose(fp);

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}
	///////////
	fp=fopen("./input/overthrust_acc_vs_170_800.dat","rb");
	for(ix=pml;ix<ntx-pml;ix++)
	{
		for(iz=pml;iz<ntz-pml;iz++)
		{
			ip=iz*ntx+ix;
			fread(&vs[ip],sizeof(float),1,fp);
		}
		vs[ip]=2000;
	}
	fclose(fp);

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vs[ip]=vs[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vs[ip]=vs[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vs[ip]=vs[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vs[ip]=vs[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vs[ip]=vs[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vs[ip]=vs[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vs[ip]=vs[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vs[ip]=vs[ipp];
		}
	}
	////////////
/*	fp=fopen("./input/acc_rho.dat","rb");
	for(ix=pml;ix<ntx-pml;ix++)
	{
		for(iz=pml;iz<ntz-pml;iz++)
		{
			ip=iz*ntx+ix;
			fread(&rho[ip],sizeof(float),1,fp);

			rho[ip]=rho[ip]*1000.0;
		}
	}
	fclose(fp);

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			rho[ip]=rho[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			rho[ip]=rho[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			rho[ip]=rho[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			rho[ip]=rho[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			rho[ip]=rho[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			rho[ip]=rho[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			rho[ip]=rho[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			rho[ip]=rho[ipp];
		}
	}
*/
	for(ip=0;ip<ntp;ip++)
	{
		rho[ip]=2000.0;    
	}

	return;
}


/*==========================================================
  This subroutine is used for initializing the initial model...
  ===========================================================*/

void get_ini_model(float *vp, float *vs, float *rho, 
		float *vp_n, float *vs_n,
		int ntp, int ntx, int ntz, int pml)
{
	int ip,ix,iz,ipp;
	FILE *fp;

	//fp=fopen("./input/acc_vp.dat","rb");
	fp=fopen("./input/3ifreq_vp.dat","rb");
	for(ix=pml;ix<=ntx-pml-1;ix++)
	{
		for(iz=pml;iz<=ntz-pml-1;iz++)
		{ 
			fread(&vp[iz*ntx+ix],sizeof(float),1,fp); 
			if(vp[iz*ntx+ix]<600.0)
				vp[iz*ntx+ix]=600.0;
		}
	}
	fclose(fp);

	//  Model in PML..............

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	fp=fopen("./input/3ifreq_vs.dat","rb");
	for(ix=pml;ix<=ntx-pml-1;ix++)
	{
		for(iz=pml;iz<=ntz-pml-1;iz++)
		{ 
			fread(&vs[iz*ntx+ix],sizeof(float),1,fp);  
		}
	}
	fclose(fp);

	//  Model in PML..............

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vs[ip]=vs[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vs[ip]=vs[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vs[ip]=vs[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vs[ip]=vs[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vs[ip]=vs[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vs[ip]=vs[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vs[ip]=vs[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vs[ip]=vs[ipp];
		}
	}

	for(ip=0;ip<ntp;ip++)
	{
//		vs[ip]=vp[ip]/1.732;
//		rho[ip]=1000.0;
	}

	return;
}

/*==========================================================
  This subroutine is used for finding the maximum and min value of 

  a vector.
  ===========================================================*/
void get_max_min(float *vpo,int N,float *vpmax,float *vpmin)
{
     int i;
     *vpmax=-1.0;
     *vpmin=8000;
     for(i=0;i<N;i++)
     {
         if(vpo[i]>*vpmax){*vpmax=vpo[i];}
         if(vpo[i]<*vpmin&&vpo[i]!=0.0){*vpmin=vpo[i];}
     }
     return;
} 

/*==========================================================
  This subroutine is used for finding the maximum value of 
  a vector.
  ===========================================================*/ 
void maximum_vector(float *vector, int n, float *maximum_value)
{
	int i;

	*maximum_value=1.0e-20;
	for(i=0;i<n;i++)
	{
		if(vector[i]>*maximum_value);
		{
			*maximum_value=vector[i];
		}
	}
	return;
}


/*==========================================================
  This subroutine is used for calculating the Lame constants...
  ===========================================================*/

void get_lame_constants(float *lambda, float *mu, 
		float *lambda_plus_two_mu, float *vp, 
		float * vs, float * rho, int ntp)
{
	int ip;

	// Lambda_plus_two_mu
	for(ip=0;ip<ntp;ip++)
	{
		lambda_plus_two_mu[ip]=vp[ip]*vp[ip]*rho[ip];
	}

	// Mu
	for(ip=0;ip<ntp;ip++)
	{
		mu[ip]=vs[ip]*vs[ip]*rho[ip];
	}

	// Lambda
	for(ip=0;ip<ntp;ip++)
	{
		lambda[ip]=lambda_plus_two_mu[ip]-2.0*mu[ip];
	}
	return;
}

/*==========================================================
  This subroutine is used for calculating the sum of two 
  vectors!
  ===========================================================*/

void add(float *a,float *b,float *c,int n)
{
	int i;
	for(i=0;i<n;i++)
	{
		c[i]=a[i]-b[i];
	}

}

/*==========================================================

  This subroutine is used for calculating the ricker wave

  ===========================================================*/

void ricker_wave(float *rick, int itmax, float f0, float t0, float dt, int flag)
{
	float pi=3.1415927;
	int   it;
	float temp,max=0.0;

	FILE *fp;

	if(flag==3)
	{	
		for(it=0;it<itmax;it++)
		{
			temp=1.5*pi*f0*(it*dt-t0);
			temp=temp*temp;
			rick[it]=exp(-temp);  

			if(max<fabs(rick[it]))
			{
				max=fabs(rick[it]);
			}
		}

		for(it=0;it<itmax;it++)
		{
			rick[it]=rick[it]/max;
		}

		fp=fopen("./output/rick_third_derive.dat","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}

	if(flag==2)
	{
		for(it=0;it<itmax;it++)
		{
			temp=pi*f0*(it*dt-t0);
			temp=temp*temp;
			rick[it]=(1.0-2.0*temp)*exp(-temp);
		}

		fp=fopen("./output/rick_second_derive.dat","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}
	if(flag==1)
	{
		for(it=0;it<itmax;it++)
		{
			temp=pi*f0*(it*dt-t0);
			temp=temp*temp;         
			rick[it]=(it*dt-t0)*exp(-temp);

			if(max<fabs(rick[it]))
			{
				max=fabs(rick[it]);
			}
		}

		for(it=0;it<itmax;it++)
		{
			rick[it]=rick[it]/max;
		}

		fp=fopen("./output/rick_first_derive.dat","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}

	return;
}

//*************************************************************************
//*******un0*cnmax=vmax*0.01
//************************************************************************

void ini_step(float *dn, int np, float *un0, float max, int flag)
{
	float dnmax=-1.0e20;
	int ip;

	for(ip=0;ip<np;ip++)
	{
		if(dnmax<fabs(dn[ip]))
		{
			dnmax=fabs(dn[ip]);
		}
	}   

	if(flag==1)
	{
		*un0=max*0.012/dnmax;  
	}
	else if(flag==2)
	{
		*un0=max*0.01/dnmax;  
	}
	else
	{
		*un0=max*0.003/dnmax;  
	}

	return;
}


/*=========================================================================
  To calculate the updated model...
  ========================================================================*/

void update_model(float *vp, float *vp_n,
		float *dn_vp, float *un_vp,
		int ntp, int ntz, int ntx, int pml, int flag)
{
	int ip,ipp;
	int iz,ix;
	int nx=ntx-2*pml;

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(iz-pml)*nx+ix-pml;
			vp[ip]=vp_n[ip]+*un_vp*dn_vp[ipp];
			
/*			if(flag==1)
				if(vp[ip]<=1930)
					vp[ip]=1930;
				else if(vp[ip]>=4750)
					vp[ip]=4750;
					
			if(flag==2)
				if(vp[ip]<=1150)
					vp[ip]=1150;
				else if(vp[ip]>=2650)
					vp[ip]=2650;				
*/

			if(flag==1)
				if(vp[ip]<=2360)
					vp[ip]=2360;
				else if(vp[ip]>=6000)
					vp[ip]=6000.;
					
			if(flag==2)
				if(vp[ip]<=1360)
					vp[ip]=1360;
				else if(vp[ip]>=3470)
					vp[ip]=3470.;	
		}
	}

	//  Model in PML..............

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;
			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}
	return;
}


/***********************************************************************
  !                initial model
  !***********************************************************************/
void ini_model_mine(float *vp, float *vp_n, int ntp, int ntz, int ntx, int pml, int flag)
{
	/*  flag == 1 :: P velocity
		flag == 2 :: S velocity
		flag == 3 :: Density
		*/
	int window;
	if(flag==1)
	{
		window=30;
	}
	else if(flag==2)
	{
		window=30;
	}
	else
	{
		window=30;
	}

	float *vp_old1;

	float sum;
	int number;

	int iz,ix;
	int izw,ixw,iz1,ix1;
	int ip,ipp;

	vp_old1=(float*)malloc(sizeof(float)*ntp);


	for(ip=0;ip<ntp;ip++)
	{
		vp_old1[ip]=vp[ip];
	}

	//-----smooth in the x direction---------

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			sum=0.0;
			number=0;

			for(izw=iz-window;izw<iz+window;izw++)
			{
				for(ixw=ix-window;ixw<ix+window;ixw++)
				{
					if(izw<0)
					{
						iz1=0;                		
					}
					else if(izw>ntz-1)
					{
						iz1=ntz-1;
					}
					else
					{
						iz1=izw;
					}

					if(ixw<0)
					{
						ix1=0;
					}
					else if(ixw>ntx-1)
					{
						ix1=ntx-1;
					}
					else
					{
						ix1=ixw;
					}

					ip=iz1*ntx+ix1;
					sum=sum+vp_old1[ip];
					number=number+1;
				}
			}
			ip=iz*ntx+ix;
			vp[ip]=sum/number;

			if(iz<pml+5)
			{
				vp[ip]=vp_old1[ip];
			}
		}
	}    

	//  Model in PML..............

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(ip=0;ip<ntp;ip++)
	{
		vp_n[ip]=vp[ip];
	}

	free(vp_old1);
}


/*=======================================================================

  subroutine preprocess(nz,nx,dx,dz,P)

  !=======================================================================*/
// in this program Precondition P is computed

void Preprocess(int nz, int nx, float dx, float dz, float *P)
{
	int iz,iz_depth_one,iz_depth_two;
	float z,delta1,a,temp,z1,z2;

	a=3.0;
	iz_depth_one=3;
	iz_depth_two=9;
	delta1=(iz_depth_two-iz_depth_one)*dx;
	z1=(iz_depth_one-1)*dz;
	z2=(iz_depth_two-1)*dz;

	for(iz=0;iz<nz;iz++)
	{ 
		z=iz*dz;
		if(z>=0.0&&z<=z1)
		{
			P[iz]=0.0;
		}

		if(z>z1&&z<=z2)
		{
			temp=z-z1-delta1;
			temp=a*temp*2/delta1;
			temp=temp*temp;
			P[iz]=1.0*exp(-0.5*temp);
		}

		if(z>z2)
		{
			P[iz]=0.5+float(z)/float(z2)*0.5;//1.0;//
		}
	}
}

/*=====================================================================

  =====================================================================*/


/*===========================================================

  This subroutine is used for FFT/IFFT

  ===========================================================*/
void fft(float *xreal,float *ximag,int n,int sign)
{
	int i,j,k,m,temp;
	int h,q,p;
	float t;
	float *a,*b;
	float *at,*bt;
	int *r;

	a=(float*)malloc(n*sizeof(float));
	b=(float*)malloc(n*sizeof(float));
	r=(int*)malloc(n*sizeof(int));
	at=(float*)malloc(n*sizeof(float));
	bt=(float*)malloc(n*sizeof(float));

	m=(int)(log(n-0.5)/log(2.0))+1; //2çå¹ïŒ?çmæ¬¡æ¹ç­äºnïŒ?	for(i=0;i<n;i++)
	{
		a[i]=xreal[i];
		b[i]=ximag[i];
		r[i]=i;
	}
	for(i=0,j=0;i<n-1;i++)  //0å°nçååºïŒ
	{
		if(i<j)
		{
			temp=r[i];
			r[i]=j;
			r[j]=temp;
		}
		k=n/2;
		while(k<(j+1))
		{
			j=j-k;
			k=k/2;
		}
		j=j+k;
	}

	t=2*PI/n;
	for(h=m-1;h>=0;h--)
	{
		p=(int)pow(2.0,h);
		q=n/p;
		for(k=0;k<n;k++)
		{
			at[k]=a[k];
			bt[k]=b[k];
		}

		for(k=0;k<n;k++)
		{
			if(k%p==k%(2*p))
			{

				a[k]=at[k]+at[k+p];
				b[k]=bt[k]+bt[k+p];
				a[k+p]=(at[k]-at[k+p])*cos(t*(q/2)*(k%p))-(bt[k]-bt[k+p])*sign*sin(t*(q/2)*(k%p));
				b[k+p]=(bt[k]-bt[k+p])*cos(t*(q/2)*(k%p))+(at[k]-at[k+p])*sign*sin(t*(q/2)*(k%p));
			}
		}

	}

	for(i=0;i<n;i++)
	{
		if(sign==1)
		{
			xreal[r[i]]=a[i];
			ximag[r[i]]=b[i];
		}
		else if(sign==-1)
		{
			xreal[r[i]]=a[i]/n;
			ximag[r[i]]=b[i]/n;
		}
	}

	free(a);
	free(b);
	free(r);
	free(at);
	free(bt);
}

void cal_xishu(int Lx,float *rx)
{
	int m,i;
	float s1,s2;
	for(m=1;m<=Lx;m++)
	{
		s1=1.0;s2=1.0;
		for(i=1;i<m;i++)
		{
			s1=s1*(2.0*i-1)*(2.0*i-1);
			s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
		}
		for(i=m+1;i<=Lx;i++)
		{
			s1=s1*(2.0*i-1)*(2.0*i-1);
			s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
		}
		s2=fabs(s2);
		rx[m-1]=pow(-1.0,m+1)*s1/(s2*(2.0*m-1));
	}
}
float ExtendOdd(float a[],int m,int n)   
{   
      int temp;   
      if(n<0)   
        n=abs(n);   
      temp=(int)fmod(n+1,2*m-2);                   //fmod(x,y)º¯ÊýÓÃÀŽÇóÓàÊýx/yµÄÓàÊý   
      if(temp>m)   
        n=2*m-1-temp;   
      else   
      {   
        if(temp==0)   
            n=1;   
        else   
            n=temp-1;   
      }   
      return(a[n]);   
}   
   
//ÓÃÀŽ¶ÔÊý×éœøÐÐ¶Ô³ÆÖÜÆÚÑÓÍØ£šÂË²šÆ÷ÏµÊýÎªÅŒÊýžöÊ±£©   
//ÑÓÍØÎª¡­¡­, a[m-1], a[m-2], ¡­¡­, a[1],a[0], a[0],a[1],¡­¡­,a[m-2],a[m-1], a[m-1],a[m-2],¡­¡­, a[1], a[0],a[0], a[1], ¡­¡­   
float ExtendEven(float a[],int m,int n)   
{   
      int temp;   
      if(n<0)   
        n=abs(n)-1;   
      temp=(int)fmod(n+1,2*m);   
      if(temp>m)   
        n=2*m-temp;   
      else   
      {   
        if(temp==0)   
            n=0;   
        else   
            n=temp-1;   
      }   
      return(a[n]);   
}   
   
//Õâžöº¯ÊýÓÃÀŽŒÆËãŸí»ý²¢ÊµÏÖ¶þÔªÏÂ²ÉÑù   
//Coeff[]ÓÃÀŽŽæ·ÅÂË²šÆ÷ÏµÊý£¬mÎªËüµÄ³€¶È£¬floor1ÎªÊµŒÊÖÐÂË²šÆ÷ÏµÊýµÄÆðÊŒÏÂ±êÖµ£¬   
//Ein_lengthÓÃÀŽÖž³öinput[]ÖÐµÄÇ°Ein_lengthžöÔªËØÎªÂË²šÆ÷µÄÊäÈë£¬œá¹û·ÅÔÚoutput[]ÖÐ   
void ConvolutionDownS(float Coeff[],int f_length,float *input,int Ein_length,float *output,int N)   
{   
     int i;   
     int k;   
   
     float *inter;  
     float acc; 
  
     inter= (float*)malloc(sizeof(float)*(N+Df-1));

     for(i=0;i<=Ein_length+f_length-2;i++)   
     {   
         acc=0.0;   
         for(k=0;k<f_length;k++)   
         {   
            if((int)fmod(f_length,2)==0)   
               acc=acc+Coeff[k]*ExtendEven(input,Ein_length,i-k);   
            else   
               acc=acc+Coeff[k]*ExtendOdd(input,Ein_length,i-k);   
         }   
        inter[i]=acc;   
     }   
     for(i=0;i<=(Ein_length+f_length-3)/2;i++)   
         output[i]=inter[2*i+1];   
}   
   
//ŽËº¯ÊýÓÃÀŽŽÓÊý×éa[]µÄÖÐÑë³éÈ¡³ö³€¶ÈÎªlength_ExµÄÒ»ÁÐÊý   
   
//length_a±íÊŸa[]µÄ³€¶È£¬nÓÃÓÚ¶Ô³éÈ¡³öÀŽµÄÊýœøÐÐ·ÃÎÊ,È¡Öµ·¶Î§Îª0¡«length_Ex-1   
   
/*³éÈ¡µÄÀý×ÓÎª£º¢Ù length_a=5:{0 1 2 3 4}  
                   length_Ex=4:{0 1 2 3}  
  
                ¢Ú length_a=4:{0 1 2 3}  
                   length_Ex=3:{0 1 2}  
*/   
float Extract(float a[],int length_a,int length_Ex,int n)   
{   
    int i=length_a-length_Ex;   
    if((int)fmod(length_a,2)==(int)fmod(length_Ex,2))   
        return(a[n+i/2]);   
    else   
        return(a[n+(i-1)/2]);   
}   
   
//ŽËº¯ÊýÓÃÀŽ¶ÔÊý×éœøÐÐ¶þÔªÉÏ²ÉÑù£¬ÔÚÆæÊýÏî²åÈë0Öµ   
float UpSample(float a[],int n)   
{   
    if((int)fmod(n,2)==0)   
        return(a[n/2]);   
    else   
        return(0.0);   
}   
   
//ŽËº¯ÊýÓÃÓÚ¶ÔÊý×éÁœ¶Ë²¹ÁãÑÓÍØ   
float Zero_padded(float a[],int length_a,int n)   
{   
    if(n>=0&&n<length_a)   
        return(a[n]);   
    else   
        return(0.0);   
}   
   
//ŽËº¯ÊýÓÃÀŽÍê³É(¶þÔªÉÏ²ÉÑù£«²¹ÁãÑÓÍØ£«ÓëÂË²šÆ÷Ÿí»ý£«³éÈ¡ÊýŸÝ)µÄÈÎÎñ   
//filter[]ÎªÂË²šÆ÷£¬length_fÎªÂË²šÆ÷³€¶È£¬fminÎªÊµŒÊµ±ÖÐ£¬ÂË²šÆ÷µÄÆðÊŒÏÂ±êÖµ   
//input[]ÎªÊäÈëÊýŸÝ£¬length_inÎªÆä³€¶È   
//output[]Îª×îºóµÄÊä³ö£¬length_ExÎªÆä³€¶È   
void UpSConvEx(float filter[],int length_f,float input[],int length_in,float output[],int length_Ex)   
{   
    int i,k;   
    int length_Ups=2*length_in-1;   
    int length_temp=length_Ups+length_f-1;   
   
    float temp[length_temp],Ups[length_Ups];   
    float acc;   
    for(i=0;i<length_Ups;i++)              //Íê³É¶þÔªÉÏ²ÉÑù   
        Ups[i]=UpSample(input,i);   
   
    for(i=0;i<length_temp;i++)             //Íê³ÉŸí»ý   
    {   
        acc=0.0;   
        for(k=0;k<length_f;k++)   
        {   
            acc=acc+Zero_padded(Ups,length_Ups,i-k)*filter[k];   
        }   
        temp[i]=acc;   
    }   
    for(i=0;i<length_Ex;i++)   
        output[i]=Extract(temp,length_temp,length_Ex,i);   
}   
   
/*
this function is used for 1D signal wavelet transform

bior2.4 is adopted

intput is the adress of vetor

N is the length  of input data (nt)

J is the level to decomption

*/

void wavelet_Dec_rest(float *input_obs, int N, int nx, float *out_scal6,float *out_scal5)   
{   

     int i,j,k,ix;   
     int no_use;      //临时变量，没用实际用途   
     int end;      //后面可以看到，此变量用来指出数组Decomp[]中存放有小波分解结果的总的数据长度   
   
     float *input,*outA,*outD;   //outA[]暂时存放近似，outD[]暂时存放细节   
     float *inputA,*inputD;   
     float *tempA,*tempD;   
     float *Decomp;          //最后的小波分解结果放在此数组中   
     int L[J+2];                       //此数组用于表示数组Decomp[]中的前L[1]个元素为第1层细节，紧接着的L[2]个元素为第2层细节   
        
     float CoeffDh[Df]={0,0.0331,-0.0663,-0.1768,0.4198,0.9944,0.4198,-0.1768,-0.0663,0.0331};//分解时用到的低通滤波器的滤波器系数   
     float CoeffDg[Df]={0,0,0,0.3536,-0.7071,0.3536,0,0,0,0};   //分解时用到的高通滤波器系数   
     float CoeffRh[Rf]={0,0,0,0.3536,0.7071,0.3536,0,0,0,0};   //重构时用到的低通滤波器系数   
     float CoeffRg[Rf]={0,-0.0331,-0.0663,0.1768,0.4198,-0.9944,0.4198,0.1768,-0.0663,-0.0331};//重构时用到的高通滤波器的滤波器系数   

     int first[J+1];   //每一层细节及最后一层近似在Decomp[]中的起始位置      
 
    //malloc memory 
     input                  = (float*)malloc(sizeof(float)*N);
     inputA                  = (float*)malloc(sizeof(float)*N);
     inputD                  = (float*)malloc(sizeof(float)*N);

     outA                  = (float*)malloc(sizeof(float)*N);
     outD                  = (float*)malloc(sizeof(float)*N);

     tempA                  = (float*)malloc(sizeof(float)*N);
     tempD                  = (float*)malloc(sizeof(float)*N);

     Decomp                  = (float*)malloc(sizeof(float)*2*N);
     
     for(i=0;i<N;i++)
	{
		input[i]=0.0;
		inputA[i]=0.0;		inputD[i]=0.0;
		outA[i]=0.0;		outD[i]=0.0;
		tempA[i]=0.0;		tempD[i]=0.0;
		Decomp[i]=0.0;		Decomp[i+N]=0.0;
	}


     for(ix=0;ix<=nx;ix++)
	{
	     end=0;
             L[0]=N;                           //照此继续下去，数组Decomp[]的最后L[J＋1]个元素为第J层近似，其前面的L[J]个元素为第J层细节  
	     first[0]=0;
   
	     for(i=0;i<N;i++)   
		   input[i]=input_obs[i*nx+ix];
	   
	     //以下实现分解   
	     for(j=0;j<J;j++)   
	     {   
		 ConvolutionDownS(CoeffDh,Df,input,L[j],outA,N);   
		 ConvolutionDownS(CoeffDg,Df,input,L[j],outD,N);   
		 L[j+1]=(int)floor((L[j]+Df-1)/2.0);   
		 for(i=0;i<L[j+1];i++)   
		     input[i]=outA[i];   
		 for(i=0;i<L[j+1];i++)   
		     Decomp[i+first[j]]=outD[i];   
		 first[j+1]=first[j]+L[j+1];   
	     }   
	   
	     L[J+1]=L[J];   
	     for(i=0;i<L[J+1];i++)   
		 Decomp[i+first[J]]=outA[i];   
	   
	     for(i=1;i<J+2;i++)   
		 end=end+L[i];   
	   
	     //以下实现重构   
	     for(i=0;i<L[J+1];i++)   //将第J层即最后一层近似存放到input中，为重构近似做准备   
	     {   
		 input[i]=Decomp[i+first[J]];   
		 tempA[i]=input[i];   
	     }   
   
	      //以下求第J层即最后一层N点近似   
	     for(i=J-1;i>=0;i--)   
	     {   
		 UpSConvEx(CoeffRh,Rf,tempA,L[i+1],inputA,L[i]);   
		 for(k=0;k<L[i];k++)   
		     tempA[k]=inputA[k];   
	     }   
	     /*for(i=0;i<N;i++)   
	 	fprintf(fpNapp[8],"%d,%f\n",i,inputA[i]);   
	     fclose(fpNapp[8]); */  
	   
	   
	   
	     //以下求第0～J-1层N点近似和1～J层细节   
	     for(j=J-1;j>=0;j--)   
	     {   
		 for(i=0;i<L[j+1];i++)  //抽取第(j+1)层细节   
		     tempD[i]=Decomp[i+first[j]];   
	   
		 UpSConvEx(CoeffRh,Rf,input,L[j+1],outA,L[j]);   
		 UpSConvEx(CoeffRg,Rf,tempD,L[j+1],outD,L[j]);   
	   
		 for(i=0;i<L[j];i++)   //第j层近似，到j=0时，就是原始数据   
		 {   
		     input[i]=outA[i]+outD[i];   
		     tempA[i]=input[i];   
		 }    
	   
		 //以下求j层N点近似   
		 for(i=j-1;i>=0;i--)   
		 {   
		     UpSConvEx(CoeffRh,Rf,tempA,L[i+1],inputA,L[i]);   
		     for(k=0;k<L[i];k++)   
		         tempA[k]=inputA[k];   
		 }   
		 /*for(i=0;i<N;i++)   
		     fprintf(fpNapp[j],"%d,%f\n",i,tempA[i]);   
		 fclose(fpNapp[j]); */ 

		 if(j==J-2)
		 	for(i=0;i<N;i++)
			   out_scal6[i*nx+ix]=tempA[i];
		 if(j==J-3)
		 	for(i=0;i<N;i++)
			   out_scal5[i*nx+ix]=tempA[i];
  	      
		 //以下求第j层N点细节   
		 UpSConvEx(CoeffRg,Rf,tempD,L[j+1],inputD,L[j]);   
		 for(k=0;k<L[j];k++)   
		     tempD[k]=inputD[k];   
		 for(i=j-1;i>=0;i--)   
		 {   
		     UpSConvEx(CoeffRh,Rf,tempD,L[i+1],inputD,L[i]);   
		     for(k=0;k<L[i];k++)   
		         tempD[k]=inputD[k];   
		 }   
		 /*for(i=0;i<N;i++)   
		     fprintf(fpNdet[j],"%d,%f\n",i,fabs(inputD[i]));   
		 fclose(fpNdet[j]);*/
		 
/*		 if(j==J-2)
		 	for(i=0;i<N;i++)
			   out_scal8[i*nx+ix]=inputD[i];
		 if(j==J-3)
		 	for(i=0;i<N;i++)
			   out_scal7[i*nx+ix]=inputD[i];
		 if(j==J-4)
		 	for(i=0;i<N;i++)
			   out_scal6[i*nx+ix]=inputD[i];
		 if(j==J-5)
		 	for(i=0;i<N;i++)
			   out_scal5[i*nx+ix]=inputD[i];
*/   
	    }   
      }
}   


void check_2M4_2d(float vpmax,float dt,float dx,float dz,int Nf,int Lpx,int Lpz,float *Gpx,float *Gpz,float *delta)
{
      int i;
      float sumx=0.0,sumz=0.0;
      float vdt=vpmax*dt;
      float temp1,temp2;

      for(i=0;i<Lpx-1;i++)
      {
          sumx=sumx+fabs(Gpx[(Nf-1)*Lpx+i]);
      }
      temp1=sumx-2.0*Gpx[(Nf-1)*Lpx+Lpx-1];
      temp1=pow(temp1/dx,2.0);

      for(i=0;i<Lpz-1;i++)
      {
          sumz=sumz+fabs(Gpz[(Nf-1)*Lpz+i]);
      }
      temp2=sumz-2.0*Gpz[(Nf-1)*Lpz+Lpz-1];
      temp2=pow(temp2/dz,2.0);
      
      *delta=vdt*sqrt(temp1+temp2);

      return;
}

// TE-based time-space domain coefficients 1D 2D 3D the same
void cal_xishur(int Lx,float *rx,float gam)
{
     int m,i;
     float s1,s2;
     for(m=1;m<=Lx;m++)
     {
        s1=1.0;s2=1.0;
        for(i=1;i<m;i++)
        {
            s1=s1*((2.0*i-1)*(2.0*i-1)-gam*gam);
            s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
        }
        for(i=m+1;i<=Lx;i++)
        {
            s1=s1*((2.0*i-1)*(2.0*i-1)-gam*gam);
            s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
        }
        s2=fabs(s2);

        rx[m-1]=pow(-1.0,m+1)*s1/(s2*(2.0*m-1));
     }
     
     return;
}


// TE based 2M-4 coefficients: 2D case  Lxo+1
void TE_2M4_2d(float vpmin,float dv,int N,int Lx,float dt,float dx,float *Gp)
{   
     int i,ii,Lxo=Lx-1;
     float dtdx=dt/dx;
     float *rp;
     float v,gam;
     rp=(float*)malloc(sizeof(float)*Lxo);

     float temp1,sumx1;
     
     
     for(i=0;i<N;i++)
     {
         v=vpmin+i*dv;
         gam=v*dtdx;

         cal_xishur(Lxo,rp,gam);
         temp1=gam*gam/24.0;

         sumx1=0.0;
         for(ii=1;ii<Lxo;ii++)
         {
                sumx1=sumx1+(2*ii+1)*rp[ii];
         }
         Gp[i*Lx+0]=1.0-2.0*temp1-sumx1;    //first value is m=1

         for(ii=1;ii<Lx-1;ii++)
         {
                Gp[i*Lx+ii]=rp[ii];			//then start from m=2-Lx-1
         }
         Gp[i*Lx+Lx-1]=temp1;				//last is the corner 
         
     }

     free(rp);
     return;    
}


