extern "C"
struct Source
{
	int s_iz,s_ix,r_n;
	int *r_ix, *r_iz;
};

extern "C"
struct MultiGPU
{
	float *seismogram_vx_obs;
	float *seismogram_vx_syn;
	float *seismogram_vx_rms;

	float *seismogram_vz_obs;
	float *seismogram_vz_syn;
	float *seismogram_vz_rms;

	float *vx_borders_up,*vx_borders_bottom;
	float *vx_borders_left,*vx_borders_right;
	float *vz_borders_up,*vz_borders_bottom;
	float *vz_borders_left,*vz_borders_right;

	float *image_lambda,*image_mu,*image_rho;
	float *image_vp,*image_vs;
	float *image_source_vp,*image_source_vs;
	float *image_normalize_vp,*image_normalize_vs;
	// vectors for the devices
	int *d_r_ix;
	int *d_r_iz;
	
	float *d_rc;
	float *d_Gp;
	float *d_Gs;

	float *d_asr;

	float *d_muxz;

	float *d_rick;
	float *d_lambda, *d_mu;
	float *d_vp, *d_vs, *d_rho;
	float *d_lambda_plus_two_mu;

	float *d_a_x, *d_a_x_half;
	float *d_a_z, *d_a_z_half;
	float *d_b_x, *d_b_x_half;
	float *d_b_z, *d_b_z_half;

	float *d_vx,*d_vz;
	float *d_sigmaxx,*d_sigmaxxs,*d_sigmaxz,*d_sigmazz;

	//  Wavefields of the constructed by using the storage of the borders...
	float *d_vx_inv,*d_vz_inv;
	float *d_sigmaxx_inv,*d_sigmaxxs_inv,*d_sigmaxz_inv,*d_sigmazz_inv;  

	float *d_phi_vx_x,*d_phi_vx_z,*d_phi_vz_z,*d_phi_vz_x;
	float *d_phi_vxs_x,*d_phi_vzs_z;

	float *d_phi_sigmaxx_x,*d_phi_sigmaxxs_x,*d_phi_sigmaxz_z;
	float *d_phi_sigmaxz_x,*d_phi_sigmazz_z;
	float *d_phi_sigmaxx_z,*d_phi_sigmaxxs_z,*d_phi_sigmazz_x;

	// =======================================================
	float *d_seismogram_vx_syn;
	float *d_seismogram_vx_rms;

	float *d_seismogram_vz_syn;
	float *d_seismogram_vz_rms;

	float *d_vx_borders_up,*d_vx_borders_bottom;
	float *d_vx_borders_left,*d_vx_borders_right;

	float *d_vz_borders_up,*d_vz_borders_bottom;
	float *d_vz_borders_left,*d_vz_borders_right;

	float *d_image_lambda,*d_image_mu,*d_image_rho;
	float *d_image_vp,*d_image_vs;
	float *d_image_source_vp,*d_image_source_vs;
};

void cal_xishu(int Lx,float *rx);

void ricker_wave(float *rick, int itmax, float f0, float t0, float dt, int flag);

void get_acc_model(float *vp, float *vs, float *rho, int ntp, int ntx, int ntz, int pml, int inv_flag);

void get_ini_model(float *vp, float *vs, float *rho, 
		float *vp_n, float *vs_n,
		int ntp, int ntx, int ntz, int pml);

void ini_model_mine(float *vp, float *vp_n, int ntp, int ntz, int ntx, int pml, int flag);

void maximum_vector(float *vector, int n, float *maximum_value);

void get_max_min(float *vpo,int N,float *vpmax,float *vpmin);

void get_lame_constants(float *lambda, float *mu, 
		float *lambda_plus_two_mu, float *vp, 
		float * vs, float * rho, int ntp);

void get_absorbing_parameters(
		float *d_x, float *d_x_half, 
		float *d_z, float *d_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half,
		int ntz, int ntx, int nz, int nx,
		int pml, float dx, float f0, 
		float t0, float dt, float vp_max);

void get_pml_parameters(float *i_a_x,float *i_b_x,float *i_a_z,float *i_b_z,
                        float *h_a_x,float *h_b_x,float *h_a_z,float *h_b_z,
                        int nxx,int nzz,float fm,float dx,float dz,float dt,
                        int pmlc);

extern "C"
void fdtd_cpml_2d_GPU_forward(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float *rc, float dx, float dz,
		float *rick, int itmax, float dt, int myid, float *vp, float *vs,
		float vp_min,float dvp,float vs_min,float dvs,float *Gp,float *Gs,int maxNp, int maxNs,
		int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, float *rho,
		float *lambda, float *mu, float *lambda_plus_two_mu,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half, 
		float *vx, float *vz, 
		float *sigmaxx, float *sigmaxxs,float *sigmazz, float *sigmaxz,
		int inv_flag);

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
		);

void ini_step(float *dn, int np, float *un0, float max, int flag);

void update_model(float *vp, float *vp_n,
		float *dn_vp, float *un_vp,
		int ntp, int ntz, int ntx, int pml, int flag);

void Preprocess(int nz, int nx, float dx, float dz, float *P);

extern "C"
void congpu_fre(float *seismogram_syn, float *seismogram_obs, float *seismogram_rms, float *Misfit, int i, 
		float *ref_window, float *seis_window, int itmax, float dt, float dx, int is, int nx, int s_ix, int *r_ix, int pml);

extern "C"
void getdevice(int *GPU_N);

extern "C"
void variables_malloc(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax, int maxNp, int maxNs,
		struct MultiGPU plan[], int GPU_N, int rnmax, int NN
		);

extern "C"
void variables_free(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax,
		struct MultiGPU plan[], int GPU_N, int rnmax, int NN
		);
extern "C"
void wavelet_Dec_rest(float *input_obs, int N, int nx, float *out_scal6,float *out_scal5) ;

void check_2M4_2d(float vpmax,float dt,float dx,float dz,int Nf,int Lpx,int Lpz,float *Gpx,float *Gpz,float *delta);

void cal_xishur(int Lx,float *rx,float gam);

void TE_2M4_2d(float vpmin,float dv,int N,int Lx,float dt,float dx,float *Gp);
