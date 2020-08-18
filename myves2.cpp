#include "ves3d_simulation.h"
#include "GLIntegrator.h"
#include "Surface.h"
#include "StokesVelocity.h"
#include "OperatorsMats.h"
#include "Streamable.h"
#include "SHTrans.h"
#include "Enums.h"
#include "anyoption.h"
#include <vector>
#include <math.h>
#include "SHTrans.h"
#include <petscksp.h>
#include <petscmat.h>
#include <cassert>
#include <cstring>
#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <string>
#include <sstream>
#include <iomanip>

typedef Device<CPU> Dev;
extern const Dev cpu(0);
typedef Simulation<Dev, cpu> Sim_t;
typedef Sim_t::Param_t Param_t;


typedef double T;
typedef Scalars<T, Dev, cpu> Sca_t;
typedef Vectors<T, Dev, cpu> Vec_t;
typedef Surface<Sca_t, Vec_t> Sur_t;
typedef typename Sca_t::array_type Arr_t;
typedef OperatorsMats<Arr_t> Mats_t;
typedef typename pvfmm::Vector<T> PVFMMVec;
typedef ParallelLinSolver<T> PSolver_t;
typedef typename PSolver_t::vec_type vec_type;
typedef Parameters<T> Params_t;
typedef StokesVelocity<T> SVel;

static char help[] = "Appends to an ASCII file.\n\n";


struct staticargs
{
	int sh_order,sh_order_up,M_gl,Mg,Ng, nVes;
  	T box_size;
  	void *sctx, *rctx;
	PVFMMVec sphere_v_,sphere_unitv_,DLMatrix,sphere_centers, ves_centers, ves_v_, ves_unitv_, En_far, q_n, sigma_n, r ;
	Vec_t sphere_x0, ves_x0;
	Mats_t *mats;
	Params_t params_;
	Sur_t *sphere_S_,*sphere_unitS_, *ves_S_, *ves_unitS_;
	SVel *sphere_stokesdl,*sphere_unitdl, *ves_stokesdl, *ves_unitdl;
	T rsc;
  	Mat P;
  	MPI_Comm comm;
};




void SetGeometry(int shape, int M_gl, int nVes, staticargs &sargs, MPI_Comm comm)
{
  int np,myrank,Mg=sargs.Mg,Ng=sargs.Ng;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);
  PVFMMVec theta,thetaunit,shc,grid,sphere_tmpv_, ves_tempv_;
  theta.ReInit(3*Mg);
  thetaunit.ReInit(3);
  theta    .SetZero();
  thetaunit.SetZero();
  if(shape==1)
 	{
    	shc.ReInit(24);
		shc.SetZero();
		shc[ 1]  = -1.0/sqrt(1.5);
		shc[13] = -sqrt(2)/sqrt(1.5);
		shc[19] = -sqrt(2)/sqrt(1.5);
		SphericalHarmonics<T>::SHC2Grid(shc, 2, sargs.sh_order, sphere_tmpv_);
		if(M_gl == 1)
		{
			sargs.rsc=1;
			T inx = 0.5,
        	iny = 0.5,
        	inz = 0.5,
        	inr = 0.15;
			sargs.sphere_centers[0]=inx;
      		sargs.sphere_centers[1]=iny;
      		sargs.sphere_centers[2]=inz;
			for(long i=0;i<Ng;i++)
			{
				sargs.sphere_v_[0*Ng+i]=inx+inr*sphere_tmpv_[0*Ng+i];
				sargs.sphere_v_[1*Ng+i]=iny+inr*sphere_tmpv_[1*Ng+i];
				sargs.sphere_v_[2*Ng+i]=inz+inr*sphere_tmpv_[2*Ng+i];
        		std::cout<<sphere_tmpv_[0*Ng+i]<<" "<<sphere_tmpv_[1*Ng+i]<<" "<<sphere_tmpv_[2*Ng+i]<<'\n';
			}
		}
		SphericalHarmonics<T>::SHC2Grid(shc, 2, sargs.sh_order, ves_tmpv_);
		if(nVes == 1)
		{
			sargs.rsc=1;
			T inx = 0.0,
        	iny = 0.0,
        	inz = 0.0,
        	inr = 0.1;
			sargs.ves_centers[0]=inx;
      		sargs.ves_centers[1]=iny;
      		sargs.ves_centers[2]=inz;
			for(long i=0;i<Ng;i++)
			{
				sargs.ves_v_[0*Ng+i]=inx+inr*ves_tmpv_[0*Ng+i];
				sargs.ves_v_[1*Ng+i]=iny+inr*ves_tmpv_[1*Ng+i];
				sargs.ves_v_[2*Ng+i]=inz+inr*ves_tmpv_[2*Ng+i];
        		std::cout<<ves_tmpv_[0*Ng+i]<<" "<<ves_tmpv_[1*Ng+i]<<" "<<ves_tmpv_[2*Ng+i]<<'\n';
			}
		}
	}


	sargs.stokesdl = new SVel(sargs.sh_order,sargs.sh_order_up,sargs.box_size, 1e-03, comm);
	sargs.unitdl   = new SVel(sargs.sh_order,sargs.sh_order_up,            -1, 1e-03, comm);


 	sargs.sphere_unitv_ = sphere_tmpv_;
 	sargs.ves_unitv_ = ves_tmpv_;
 	sargs.sphere_x0.resize(Mg,sargs.sh_order);
 	args.ves_x0.resize(Mg,sargs.sh_order);
	for(long i=0;i<sargs.sphere_v_.Dim();i++) sargs.sphere_x0.begin()[i]=sargs.sphere_v_[i];
	for(long i=0;i<sargs.ves_v_.Dim();i++) sargs.ves_x0.begin()[i]=sargs.ves_v_[i];
	sargs.sphere_S_ = new Sur_t(sargs.sh_order, *sargs.mats, &sargs.sphere_x0);
	sargs.ves_S_ = new Sur_t(sargs.sh_order, *sargs.mats, &sargs.ves_x0);

	sargs.sphere_x0.resize(1,sargs.sh_order);
	sargs.ves_x0.resize(1,sargs.sh_order);
	for(long i=0;i<sphere_tmpv_.Dim();i++) sargs.sphere_x0.begin()[i]=sphere_tmpv_[i];
	for(long i=0;i<ves_tmpv_.Dim();i++) sargs.ves_x0.begin()[i]=ves_tmpv_[i];
	sargs.sphere_unitS_ = new Sur_t(sargs.sh_order, *sargs.mats, &sargs.sphere_x0);
    sargs.ves_unitS_ = new Sur_t(sargs.sh_order, *sargs.mats, &sargs.ves_x0);


 	sargs.sphere_stokesdl->SetSrcCoord(sargs.sphere_v_);
 	sargs.sphere_unitdl->SetSrcCoord(sphere_tmpv_);
 	sargs.ves_stokesdl->SetSrcCoord(sargs.ves_v_);
 	sargs.ves_unitdl->SetSrcCoord(ves_tmpv_);
 	sargs.stokesdl->SetThetaRot(&theta);
 	sargs.unitdl->SetThetaRot(&thetaunit);
}



int main()
{
	//Initialise MPI 
	MPI_Init(NULL, NULL);
	PetscErrorCode ierr;
	PetscInitialize(&argc,&argv,0,help);
	MPI_Comm comm_self, comm = MPI_COMM_WORLD;

	int np,myrank;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_split(comm, myrank, myrank, &comm_self);

	//Initialise static arguments  
	staticargs sargs;
  	sargs.            comm = comm;
  	sargs.            M_gl = 1;  //number of spheres
  	sargs.nVes             = 1; //number of vesicles
	sargs.        sh_order = 16;
	sargs.     sh_order_up = 32;
  	sargs.        box_size = 10;
  	sargs.params_.sh_order = sargs.sh_order;
	sargs.              Mg = sargs.M_gl/np + (myrank < sargs.M_gl % np ? 1 : 0);
  	sargs.              Ng = 2*sargs.sh_order*(sargs.sh_order+1);
  	sargs.            sctx = PVFMMCreateContext<T>(sargs.box_size,1000,10,MAX_DEPTH,&StokesKernel<T>::Kernel(),comm);
  	sargs.            rctx = PVFMMCreateContext<T>(sargs.box_size,1000,10,MAX_DEPTH,   &rotlet_kernel(),comm);
	sargs.sphere_centers.Resize(3*sargs.Mg);
	sargs.ves_centers.Resize(3*sargs.Mg);
  	sargs.v_.Resize(3*sargs.Mg*sargs.Ng);
  	sargs.ves_v_.Resize(3*sargs.Mg*sargs.Ng);
  	std::cout<<"proc "<<myrank<<" "<<sargs.Mg<<'\n';
	PetscInt m,n;
  	int M_gl = sargs.M_gl, Ng = sargs.Ng, Mg = sargs.Mg, shape=atoi(argv[8]), testcase=atoi(argv[9]), precondition=atoi(argv[5]);
  	T box_size  = sargs.box_size;
  	T grid_size = atof(argv[7]);
  	crat        = atof(argv[10]);


  	//TestPeriodic();
	m = 3*Mg*Ng+6*Mg; // local rows
	n = m; // local columns
	sargs.mats = new Mats_t(true, sargs.params_);


	int ngrid[3] = {10,10,10};

	int nVes = 1; // number of vesicles 
	int nRigidBodies = 1; // number of rigid bodies 
	int shape=1;  // shape number , different numbers correspond to different geometries of rigid body

	//Now we have the domain filled with fluid and with one vesicle and a rigid body, so we have to perform vesicle simulation
	// We set the geometry of vesicle and rigid bodies . Rigid bodies will have fixed geometry while vesicle shape and position will evolve with time

	//-----------------------Set the shape of vesicle and get the spherical harmonics coordinates for both rigid bodies and vesicles---------------
	SetGeometry(shape, M_gl, nVes, sargs, comm);
	//-------------------------We have set the geometry now------------------------------------------------------------------------

	/* Once we have the geometry, we can get started with time step*/
	//Initialize x_0, x_e , beta, alpha, En_far, 
	T dt = 0.1; //time step
	T g = 0.0; // Calculate P(y_e)x_0 , change this according to initial shape and velocity, used for surface divergence zero

	//Now set En_far and q 
	/*sargs.En_far = ;
	sargs.q_n = ;
	sargs.sigma_n = ;*/
	// Time stepping loop for the scheme
	for(t=0; t < 0.2; t = t+dt)
	{
		std::cout<<"Hello";
	}
}