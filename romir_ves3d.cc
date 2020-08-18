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
	int sh_order,sh_order_up,M_gl,Mg,Ng;
  T box_size;
  void *sctx, *rctx;
	PVFMMVec v_,unitv_,DLMatrix,centers;
	Vec_t x0;
	Mats_t *mats;
	Params_t params_;
	Sur_t *S_,*unitS_;
	SVel *stokesdl,*unitdl;
	T rsc;
  Mat P;
  MPI_Comm comm;
};

void PrintVec(PVFMMVec x)
{
	for(long i=0;i<x.Dim();i++) std::cout<<x[i]<<'\n';
}

void printnormU(Vec U)
{
	PetscErrorCode ierr;

	PetscInt U_size;
	T norm2=0;
	const PetscScalar *U_ptr;
	ierr = VecGetLocalSize(U, &U_size);
	ierr = VecGetArrayRead(U, &U_ptr);
	for(long i=0;i<U_size;i++) 
	{
		norm2+=(U_ptr[i])*(U_ptr[i]);
	}
	std::cout<<"2 norm : "<<sqrt(norm2)<<'\n';
}

void savesolution(Vec U, staticargs *sargs,std::ostringstream &ss, MPI_Comm comm)
{
  int np, myrank, msgsend, msgrecv;
  int Mg = sargs->Mg, Ng = sargs->Ng;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);
  MPI_Status stat;
  PetscErrorCode ierr;

	PetscInt U_size;
	PetscScalar *U_ptr;
	ierr = VecGetLocalSize(U, &U_size);
	ierr = VecGetArray(U, &U_ptr);

	std::string s;
	s = "./solution_" + ss.str() + ".txt";
	std::ofstream myfile;
  if(!myrank)
  {
    myfile.open(s.c_str());
	  for(long i=0;i<3*Mg*Ng;i++)	myfile<<std::scientific<<std::setprecision(15)<<U_ptr[i]<<'\n';
	  myfile.close();
    if(np>1)
    {
      MPI_Send(&msgsend, 1, MPI_INT, (myrank+1)%np, 0, comm);
      MPI_Recv(&msgrecv, 1, MPI_INT, np - 1, 0, comm, &stat);
    }
  }
  else
  {
    MPI_Recv(&msgrecv, 1, MPI_INT, (myrank-1)%np, 0, comm, &stat);
    myfile.open(s.c_str(), std::ofstream::app);
	  for(long i=0;i<3*Mg*Ng;i++)	myfile<<std::scientific<<std::setprecision(15)<<U_ptr[i]<<'\n';
	  myfile.close();
    MPI_Send(&msgsend, 1, MPI_INT, (myrank+1)%np, 0, comm);
  }

  if(!myrank)
  {
    myfile.open(s.c_str(), std::ofstream::app);
	  for(long i=3*Mg*Ng;i<3*Mg*Ng+3*Mg;i++)	myfile<<std::scientific<<std::setprecision(15)<<U_ptr[i]<<'\n';
	  myfile.close();
    if(np>1)
    {
      MPI_Send(&msgsend, 1, MPI_INT, (myrank+1)%np, 0, comm);
      MPI_Recv(&msgrecv, 1, MPI_INT, np - 1, 0, comm, &stat);
    }
  }
  else
  {
    MPI_Recv(&msgrecv, 1, MPI_INT, (myrank-1)%np, 0, comm, &stat);
    myfile.open(s.c_str(), std::ofstream::app);
	  for(long i=3*Mg*Ng;i<3*Mg*Ng+3*Mg;i++)	myfile<<std::scientific<<std::setprecision(15)<<U_ptr[i]<<'\n';
	  myfile.close();
    MPI_Send(&msgsend, 1, MPI_INT, (myrank+1)%np, 0, comm);
  }

  if(!myrank)
  {
    myfile.open(s.c_str(), std::ofstream::app);
	  for(long i=3*Mg*Ng+3*Mg;i<3*Mg*Ng+6*Mg;i++)	myfile<<std::scientific<<std::setprecision(15)<<U_ptr[i]<<'\n';
	  myfile.close();
    if(np>1)
    {
      MPI_Send(&msgsend, 1, MPI_INT, (myrank+1)%np, 0, comm);
      MPI_Recv(&msgrecv, 1, MPI_INT, np - 1, 0, comm, &stat);
    }
  }
  else
  {
    MPI_Recv(&msgrecv, 1, MPI_INT, (myrank-1)%np, 0, comm, &stat);
    myfile.open(s.c_str(), std::ofstream::app);
	  for(long i=3*Mg*Ng+3*Mg;i<3*Mg*Ng+6*Mg;i++)	myfile<<std::scientific<<std::setprecision(15)<<U_ptr[i]<<'\n';
	  myfile.close();
    MPI_Send(&msgsend, 1, MPI_INT, (myrank+1)%np, 0, comm);
  }
  ierr = VecRestoreArray(U, &U_ptr);
}

void RelativeError(PVFMMVec x, PVFMMVec y)
{
	T rel_error = 0.0, ymax = 0.0;
	for(long i=0;i<x.Dim();i++) 
	{
		rel_error = fmax(rel_error,fabs(x[i]-y[i]));
    ymax = fmax(ymax,fabs(y[i]));
	}
	std::cout<<"relative_error : "<<rel_error/ymax<<'\n';
}

template <class T>
void stokes_vel(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* v_trg, pvfmm::mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  pvfmm::Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(28*dof));
#endif

  const T mu=1.0;
  const T OOEPMU = 1.0/(8.0*pvfmm::const_pi<T>()*mu);
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[3]={0,0,0};
      for(int s=0;s<src_cnt;s++){
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R!=0){
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T v_src[3]={v_src_[(s*dof+i)*3  ],
                      v_src_[(s*dof+i)*3+1],
                      v_src_[(s*dof+i)*3+2]};
          T inner_prod=(v_src[0]*dR[0] +
                        v_src[1]*dR[1] +
                        v_src[2]*dR[2])* invR2;
          p[0] += (v_src[0] + dR[0]*inner_prod)*invR;
          p[1] += (v_src[1] + dR[1]*inner_prod)*invR;
          p[2] += (v_src[2] + dR[2]*inner_prod)*invR;
        }
      }
      v_trg[(t*dof+i)*3+0] += p[0]*OOEPMU;
      v_trg[(t*dof+i)*3+1] += p[1]*OOEPMU;
      v_trg[(t*dof+i)*3+2] += p[2]*OOEPMU;
    }
  }
}

template <class T>
void stokes_sym_dip(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, pvfmm::mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  pvfmm::Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(47*dof));
#endif

  const T mu=1.0;
  const T OOEPMU = -1.0/(8.0*pvfmm::const_pi<T>()*mu);
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[3]={0,0,0};
      for(int s=0;s<src_cnt;s++){
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R!=0){
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;

          T* f=&v_src[(s*dof+i)*6+0];
          T* n=&v_src[(s*dof+i)*6+3];

          T r_dot_n=(n[0]*dR[0]+n[1]*dR[1]+n[2]*dR[2]);
          T r_dot_f=(f[0]*dR[0]+f[1]*dR[1]+f[2]*dR[2]);
          T n_dot_f=(f[0]* n[0]+f[1]* n[1]+f[2]* n[2]);

          p[0] += dR[0]*(n_dot_f - 3*r_dot_n*r_dot_f*invR2)*invR3;
          p[1] += dR[1]*(n_dot_f - 3*r_dot_n*r_dot_f*invR2)*invR3;
          p[2] += dR[2]*(n_dot_f - 3*r_dot_n*r_dot_f*invR2)*invR3;
        }
      }
      v_trg[(t*dof+i)*3+0] += p[0]*OOEPMU;
      v_trg[(t*dof+i)*3+1] += p[1]*OOEPMU;
      v_trg[(t*dof+i)*3+2] += p[2]*OOEPMU;
    }
  }
}

const pvfmm::Kernel<T>& Velocity(){
  static const pvfmm::Kernel<T> ker=pvfmm::BuildKernel<T, stokes_vel<T>, stokes_sym_dip<T> >("stokes_velo", 3, std::pair<int,int>(3,3));
  return ker;
}

void rotlet_sl(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, pvfmm::mem::MemoryManager* mem_mgr)
{
  const T mu = 1;
  const T OOEPMU = 1/(8.0*M_PI*mu);
  for(int t=0;t<trg_cnt;t++)
  {
    for(int i=0;i<dof;i++)
    {
      T p[3]={0,0,0};
      for(int s=0;s<src_cnt;s++)
      {
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR [1]+dR[2]*dR[2]);
        if (R!=0)
        {
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;

          T* f=&v_src[(s*dof+i)*3+0];

          p[0] += (f[1]*dR[2] - f[2]*dR[1])*invR3;
          p[1] += (f[2]*dR[0] - f[0]*dR[2])*invR3;
          p[2] += (f[0]*dR[1] - f[1]*dR[0])*invR3;
        }
      }
      v_trg[(t*dof+i)*3+0] += p[0]*OOEPMU;
      v_trg[(t*dof+i)*3+1] += p[1]*OOEPMU;
      v_trg[(t*dof+i)*3+2] += p[2]*OOEPMU;
    }
  }
}

const pvfmm::Kernel<T>& rotlet_kernel()
{
  static const pvfmm::Kernel<T> ker=pvfmm::BuildKernel<T, rotlet_sl>("rotlet", 3, std::pair<int,int>(3,3));
  return ker;
}

void stokeslet_sl(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, pvfmm::mem::MemoryManager* mem_mgr)
{
  const T mu = 1.0;
  const T OOEPMU = 1/(8.0*M_PI*mu);
  for(int t=0;t<trg_cnt;t++)
  {
    for(int i=0;i<dof;i++)
    {
      T p[3]={0,0,0};
      for(int s=0;s<src_cnt;s++)
      {
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR [1]+dR[2]*dR[2]);
        if (R!=0)
        {
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;

          T* f=&v_src[(s*dof+i)*3+0];
          T r_dot_f=(f[0]*dR[0]+f[1]*dR[1]+f[2]*dR[2])*invR2;

          p[0] += (f[0] + dR[0]*r_dot_f)*invR;
          p[1] += (f[1] + dR[1]*r_dot_f)*invR;
          p[2] += (f[2] + dR[2]*r_dot_f)*invR;
        }
      }
      v_trg[(t*dof+i)*3+0] += p[0]*OOEPMU;
      v_trg[(t*dof+i)*3+1] += p[1]*OOEPMU;
      v_trg[(t*dof+i)*3+2] += p[2]*OOEPMU;
    }
  }
}

const pvfmm::Kernel<T>& stokeslet_kernel()
{
  static const pvfmm::Kernel<T> ker=pvfmm::BuildKernel<T, stokeslet_sl>("stokeslet", 3, std::pair<int,int>(3,3));
  return ker;
}

void stokes_press_sl(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, pvfmm::mem::MemoryManager* mem_mgr)
{
  const T OOFP = 1/(4.0*M_PI);
  for(int t=0;t<trg_cnt;t++)
  {
    for(int i=0;i<dof;i++)
    {
      T p=0;
      for(int s=0;s<src_cnt;s++)
      {
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR [1]+dR[2]*dR[2]);
        if (R!=0)
        {
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;

          T* f=&v_src[(s*dof+i)*3+0];
          T r_dot_f=(f[0]*dR[0]+f[1]*dR[1]+f[2]*dR[2]);

          p += (r_dot_f)*invR3;
        }
      }
      v_trg[(t*dof+i)+0] += p*OOFP;
    }
  }
}

void stokes_press_dl(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, pvfmm::mem::MemoryManager* mem_mgr)
{
  const T mu=1.0;
  const T MUOTP = mu/(2.0*M_PI);
  for(int t=0;t<trg_cnt;t++)
  {
    for(int i=0;i<dof;i++)
    {
      T p=0;
      for(int s=0;s<src_cnt;s++)
      {
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R!=0)
        {
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;

          T* f=&v_src[(s*dof+i)*6+0];
          T* n=&v_src[(s*dof+i)*6+3];

          T r_dot_n=(n[0]*dR[0]+n[1]*dR[1]+n[2]*dR[2]);
          T r_dot_f=(f[0]*dR[0]+f[1]*dR[1]+f[2]*dR[2]);
          T n_dot_f=(f[0]* n[0]+f[1]* n[1]+f[2]* n[2]);

          p += (n_dot_f - 3*r_dot_n*r_dot_f*invR2)*invR3;
        }
      }
      v_trg[(t*dof+i)+0] += p*MUOTP;
    }
  }
}

const pvfmm::Kernel<T>& pressure_kernel()
{
  static const pvfmm::Kernel<T> ker=pvfmm::BuildKernel<T, stokes_press_sl, stokes_press_dl>("stokes_pressure", 3, std::pair<int,int>(3,1));//,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,true);
  return ker;
}

void get_pressure(PVFMMVec src_pos, PVFMMVec trg_pos, PVFMMVec src_den,PVFMMVec &trg_press, staticargs *sargs)
{
  T box_size = sargs->box_size;
  MPI_Comm comm = sargs->comm;
  void *pvfmm_ctx=PVFMMCreateContext<T>(box_size,1000,10,MAX_DEPTH,&pressure_kernel(),comm); //&StokesKernel<T>::Kernel()
  trg_press.ReInit(trg_pos.Dim()/3);
  std::cout<<"trg_press Dim: "<<trg_press.Dim()<<" "<<trg_press[trg_press.Dim()]<<'\n';
  PVFMMVec sl_den(0);
  PVFMMEval(&src_pos[0],sl_den.Dim()?&src_pos[0]:NULL,&src_den[0],src_pos.Dim()/3,&trg_pos[0],&trg_press[0],trg_pos.Dim()/3,&pvfmm_ctx,1);
}

int setupsr = 1;
int mult(Mat M, Vec U, Vec Y)
{
  T t0,t1,t2,t3;
  t0 = MPI_Wtime();
  t2 = MPI_Wtime();
  MPI_Comm comm;
  int np, myrank;
  PetscObjectGetComm((PetscObject) M, &comm);
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);

  void* dt=NULL;
  MatShellGetContext(M, &dt);
  staticargs *data = (staticargs *)dt;
  int Ng = data->Ng, Mg = data->Mg, M_gl = data->M_gl;
  T rsc = data->rsc;
  T box_size = data->box_size;
  long i,j,k,lx,ly,lz;

	PetscErrorCode ierr;
	PetscScalar val[1];

	PetscInt U_size;
	const PetscScalar *U_ptr;
	ierr = VecGetLocalSize(U, &U_size);
	ierr = VecGetArrayRead(U, &U_ptr);

	PetscInt Y_size;
	PetscScalar *Y_ptr;
	ierr = VecGetLocalSize(Y, &Y_size);
	ierr = VecGetArray(Y, &Y_ptr);

  //Eqs 3mn+1 to 3mn+3m
	GaussLegendreIntegrator<Sca_t> integrator_;
	Sca_t scw,gl_out; //,area_elements;		
	//area_elements.resize(Mg,data->sh_order);
	scw.resize(Mg,data->sh_order);	//allocating more than needed
	gl_out.resize(Mg,data->sh_order);
	//for(long i=0;i<Mg*Ng;i++)  area_elements.begin()[i] = data->S_->getAreaElement().begin()[i];
  const Sca_t& area_elements=data->S_->getAreaElement();

	for(j=0;j<3;j++)
	{
    //#pragma omp parallel for private(i,k)
		for(k=0;k<Mg;k++)
		{
			for(i=0;i<Ng;i++)
			{
				scw.begin()[k*Ng+i]=U_ptr[j*Ng+3*k*Ng+i];
			}
		}
		integrator_(scw, area_elements, gl_out);
    //#pragma omp parallel for
		for(i=0;i<Mg;i++)
		{
			Y_ptr[3*Mg*Ng+Mg*j+i]=gl_out.begin()[i]/(rsc*rsc);
		}
	}

  //Eqs 3mn+3m+1 to 3mn+6m

  int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
  T cx,cy,cz;
  #pragma omp parallel for private(i,j,cx,cy,cz)
	for(j=0;j<Mg;j++)
	{
		cx=data->centers[3*j]; cy=data->centers[3*j+1]; cz=data->centers[3*j+2];
		for(i=0;i<Ng;i++)
		{
			scw.begin()[j*Ng+i]=(data->v_[1*Ng+3*j*Ng+i]-cy)*U_ptr[2*Ng+3*j*Ng+i]-(data->v_[2*Ng+3*j*Ng+i]-cz)*U_ptr[1*Ng+3*j*Ng+i];
		}
	}
	integrator_(scw, area_elements, gl_out);
  #pragma omp parallel for private(i,j,cx,cy,cz)
	for(i=0;i<Mg;i++)
	{
		Y_ptr[3*Mg*Ng+3*Mg+i]=gl_out.begin()[i]/(rsc*rsc*rsc);
	}

  #pragma omp parallel for private(i,j,cx,cy,cz)
	for(j=0;j<Mg;j++)
	{
		cx=data->centers[3*j]; cy=data->centers[3*j+1]; cz=data->centers[3*j+2];
		for(i=0;i<Ng;i++)
		{
			scw.begin()[j*Ng+i]=(data->v_[2*Ng+3*j*Ng+i]-cz)*U_ptr[0*Ng+3*j*Ng+i]-(data->v_[0*Ng+3*j*Ng+i]-cx)*U_ptr[2*Ng+3*j*Ng+i];
		}
	}
	integrator_(scw, area_elements, gl_out);
  #pragma omp parallel for
	for(i=0;i<Mg;i++)
	{
		Y_ptr[3*Mg*Ng+4*Mg+i]=gl_out.begin()[i]/(rsc*rsc*rsc);
	}

  #pragma omp parallel for private(i,j,cx,cy,cz)
	for(j=0;j<Mg;j++)
	{
		cx=data->centers[3*j]; cy=data->centers[3*j+1]; cz=data->centers[3*j+2];
		for(i=0;i<Ng;i++)
		{
			scw.begin()[j*Ng+i]=(data->v_[0*Ng+3*j*Ng+i]-cx)*U_ptr[1*Ng+3*j*Ng+i]-(data->v_[1*Ng+3*j*Ng+i]-cy)*U_ptr[0*Ng+3*j*Ng+i];
		}
	}
	integrator_(scw, area_elements, gl_out);
  #pragma omp parallel for
	for(i=0;i<Mg;i++)
	{
		Y_ptr[3*Mg*Ng+5*Mg+i]=gl_out.begin()[i]/(rsc*rsc*rsc);
	}

  t3 = MPI_Wtime();
  if(!myrank) std::cout<<"Time taken by force and torque = "<<t3-t2<<'\n';
	PVFMMVec tmpG, tmpS, tmpDphiS, tmpDphiG,tmpDphiG2, X_theta, X_phi;

  tmpG.ReInit(3*Mg*Ng);
	tmpDphiG2.ReInit(3*Mg*Ng);
  #pragma omp parallel for private(i)
	for(i=0;i<3*Mg*Ng;i++)
	{
		tmpG[i] = U_ptr[i];
	}

  data->stokesdl->SetDensityDL(&tmpG);
  MPI_Barrier(comm);
	tmpDphiG=data->stokesdl->operator()();

  #pragma omp parallel for private(i)
	for(i=0;i<3*Mg*Ng;i++)
	{
		tmpDphiG[i] += U_ptr[i]/2.0;
	}

  t2 = MPI_Wtime();
//  T sbuf[6*Mg],rbuf[6*M_gl];
//  int send_count = 6*Mg, recv_counts[np],disps[np];
//  sbuf = (T*) malloc(6*Mg*sizeof(T));
//  rbuf = (T*) malloc(6*M_gl*sizeof(T));

/*
  for(i=0;i<np;i++) recv_counts[i] = 6*(M_gl/np + (M_gl%np>i?1:0));
  disps[0] = 0;
  for(i=1;i<np;i++) disps[i] = disps[i-1] + recv_counts[i-1];
  #pragma omp parallel for private(i)
  for(i=0;i<Mg;i++)
  {
    sbuf[6*i+0] = U_ptr[3*Mg*Ng+3*i+0];
    sbuf[6*i+1] = U_ptr[3*Mg*Ng+3*i+1];
    sbuf[6*i+2] = U_ptr[3*Mg*Ng+3*i+2];
    sbuf[6*i+3] = U_ptr[3*Mg*Ng+3*Mg+3*i+0];
    sbuf[6*i+4] = U_ptr[3*Mg*Ng+3*Mg+3*i+1];
    sbuf[6*i+5] = U_ptr[3*Mg*Ng+3*Mg+3*i+2];
  }

  MPI_Allgatherv(sbuf,send_count,MPI_DOUBLE,rbuf,recv_counts,disps,MPI_DOUBLE,comm);
*/

  PVFMMVec tar(3*Mg*Ng),stokes_den(3*Mg),rot_den(3*Mg),trg_pot_stokes(3*Mg*Ng),trg_pot_rot(3*Mg*Ng);


  for(j=0;j<Mg;j++)
  {
    for(i=0;i<Ng;i++)
    {
      tar[3*Ng*j+3*i+0] = data->v_[0*Ng+3*Ng*j+i];
      tar[3*Ng*j+3*i+1] = data->v_[1*Ng+3*Ng*j+i];
      tar[3*Ng*j+3*i+2] = data->v_[2*Ng+3*Ng*j+i];
    }
  }

  for(long i=0;i<3*Mg;i++)
  {
    stokes_den[i] = U_ptr[3*Mg*Ng     +i];
    rot_den   [i] = U_ptr[3*Mg*Ng+3*Mg+i];
  }

  MPI_Barrier(comm);
  PVFMMEval(&data->centers[0],&stokes_den[0],(T*) NULL,Mg,&tar[0],&trg_pot_stokes[0],trg_pot_stokes.Dim()/3,&data->sctx,setupsr);
  PVFMMEval(&data->centers[0],&rot_den   [0],(T*) NULL,Mg,&tar[0],&trg_pot_rot   [0],trg_pot_rot   .Dim()/3,&data->rctx,setupsr);
  setupsr=0;
  for(long j=0;j<Mg;j++)
  {
    for(long i=0;i<Ng;i++)
    {
      tmpDphiG[0*Ng+3*Ng*j+i] += rsc*trg_pot_stokes[3*Ng*j+3*i+0] + rsc*rsc*trg_pot_rot[3*Ng*j+3*i+0];
      tmpDphiG[1*Ng+3*Ng*j+i] += rsc*trg_pot_stokes[3*Ng*j+3*i+1] + rsc*rsc*trg_pot_rot[3*Ng*j+3*i+1];
      tmpDphiG[2*Ng+3*Ng*j+i] += rsc*trg_pot_stokes[3*Ng*j+3*i+2] + rsc*rsc*trg_pot_rot[3*Ng*j+3*i+2];
    }
  }




/*

  int stp =0;
  if(box_size>=0) stp = 5;
  //stokeslet
  #pragma omp parallel for private(i,j,k,lx,ly,lz,cx,cy,cz)
  for(i=0;i<Ng;i++)
  {
    for(k=0;k<Mg;k++)
    {
      for(j=0;j<M_gl;j++)
      {
        for(lx=-stp;lx<=stp;lx++)
        {
          for(ly=-stp;ly<=stp;ly++)
          {
            for(lz=-stp;lz<=stp;lz++)
            {
              cx=data->centers[3*j]+lx*box_size; cy=data->centers[3*j+1]+ly*box_size; cz=data->centers[3*j+2]+lz*box_size;
              T rx=data->v_[0*Ng+3*k*Ng+i]-cx;
              T ry=data->v_[1*Ng+3*k*Ng+i]-cy;
              T rz=data->v_[2*Ng+3*k*Ng+i]-cz;
              T ri=sqrt(rx*rx+ry*ry+rz*rz);
              tmpDphiG[0*Ng+3*k*Ng+i]+=rsc*(1.0/(8*M_PI))*(rbuf[6*j+0]/ri+(rx*rx*rbuf[6*j]+rx*ry*rbuf[6*j+1]+rx*rz*rbuf[6*j+2])/(ri*ri*ri));
              tmpDphiG[1*Ng+3*k*Ng+i]+=rsc*(1.0/(8*M_PI))*(rbuf[6*j+1]/ri+(ry*rx*rbuf[6*j]+ry*ry*rbuf[6*j+1]+ry*rz*rbuf[6*j+2])/(ri*ri*ri));
              tmpDphiG[2*Ng+3*k*Ng+i]+=rsc*(1.0/(8*M_PI))*(rbuf[6*j+2]/ri+(rz*rx*rbuf[6*j]+rz*ry*rbuf[6*j+1]+rz*rz*rbuf[6*j+2])/(ri*ri*ri));
            }
          }
        }
      }
    }
  }
	//rotlets 
  #pragma omp parallel for private(i,j,k,lx,ly,lz,cx,cy,cz)
  for(i=0;i<Ng;i++)
  {
    for(k=0;k<Mg;k++)
    {
      for(j=0;j<M_gl;j++)
      {
        for(lx=-stp;lx<=stp;lx++)
        {
          for(ly=-stp;ly<=stp;ly++)
          {
            for(lz=-stp;lz<=stp;lz++)
            {
              cx=data->centers[3*j]+lx*box_size; cy=data->centers[3*j+1]+ly*box_size; cz=data->centers[3*j+2]+lz*box_size;
              T rx=data->v_[0*Ng+3*k*Ng+i]-cx;
              T ry=data->v_[1*Ng+3*k*Ng+i]-cy;
              T rz=data->v_[2*Ng+3*k*Ng+i]-cz;
              T ri=sqrt(rx*rx+ry*ry+rz*rz);
              tmpDphiG[0*Ng+3*k*Ng+i]+=rsc*rsc*(1.0/(8*M_PI))*(rz*rbuf[6*j+4]-ry*rbuf[6*j+5])/(ri*ri*ri);
              tmpDphiG[1*Ng+3*k*Ng+i]+=rsc*rsc*(1.0/(8*M_PI))*(rx*rbuf[6*j+5]-rz*rbuf[6*j+3])/(ri*ri*ri);
              tmpDphiG[2*Ng+3*k*Ng+i]+=rsc*rsc*(1.0/(8*M_PI))*(ry*rbuf[6*j+3]-rx*rbuf[6*j+4])/(ri*ri*ri);
            }
          }
        }
      }
    }
  }
*/
  t3 = MPI_Wtime();
  if(!myrank) std::cout<<"Time taken by stokeslets and rotlets = "<<t3-t2<<'\n';

  #pragma omp parallel for private(i)
  for(i=0;i<3*Mg*Ng;i++)
	{
		Y_ptr[i]=tmpDphiG[i];
	}
	ierr = VecRestoreArray(Y, &Y_ptr);

  t1 = MPI_Wtime();
  if(!myrank) std::cout<<"Time taken by mult() = "<<t1-t0<<'\n';


	//std::ostringstream ss;
	//ss << "n_" <<data->Mg << "_sh_" << data->sh_order << "_" << data->sh_order_up;
  //savesolution(U,ss);
	return 0;
}

void press_eval(staticargs &sargs,T *trg,PVFMMVec normal_,PetscScalar* x_ptr,T *p,bool flag)
{
	int Ng=sargs.Ng,Mg=sargs.Mg;
	T rsc = sargs.rsc;
  T box_size = sargs.box_size;
	p[0]=0;

	if(flag)
	{
		PVFMMVec r_(3*Mg*Ng);
		for(long i=0;i<Ng;i++)
		{
			for(long j=0;j<Mg;j++)
			{
				r_[0*Ng+3*Ng*j+i]=trg[0]-sargs.v_[0*Ng+3*Ng*j+i];
				r_[1*Ng+3*Ng*j+i]=trg[1]-sargs.v_[1*Ng+3*Ng*j+i];
				r_[2*Ng+3*Ng*j+i]=trg[2]-sargs.v_[2*Ng+3*Ng*j+i];
			}
		}

		Sca_t scw,gl_out;
		GaussLegendreIntegrator<Sca_t> integrator_;
		scw.resize(Mg,sargs.sh_order);	//allocating more than needed
		gl_out.resize(Mg,sargs.sh_order);

		for(long j=0;j<Mg;j++)
		{
			for(long i=0;i<Ng;i++)
			{
				T ri = sqrt(r_[3*Ng*j+i]*r_[3*Ng*j+i]+r_[Ng+3*Ng*j+i]*r_[Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*r_[2*Ng+3*Ng*j+i]);
				T rdotn =    r_[0*Ng+3*Ng*j+i]*normal_[0*Ng+3*Ng*j+i]+   r_[1*Ng+3*Ng*j+i]*normal_[1*Ng+3*Ng*j+i]+   r_[2*Ng+3*Ng*j+i]*normal_[2*Ng+3*Ng*j+i];
        T xdotn = x_ptr[0*Ng+3*Ng*j+i]*normal_[0*Ng+3*Ng*j+i]+x_ptr[1*Ng+3*Ng*j+i]*normal_[1*Ng+3*Ng*j+i]+x_ptr[2*Ng+3*Ng*j+i]*normal_[2*Ng+3*Ng*j+i];
        T xdotr = x_ptr[0*Ng+3*Ng*j+i]*     r_[0*Ng+3*Ng*j+i]+x_ptr[1*Ng+3*Ng*j+i]*     r_[1*Ng+3*Ng*j+i]+x_ptr[2*Ng+3*Ng*j+i]     *r_[2*Ng+3*Ng*j+i];
				scw.begin()[j*Ng+i]=(1.0/(2.0*M_PI*ri*ri*ri))*(xdotn - 3*rdotn*xdotr/(ri*ri));
			}
		}
		integrator_(scw, sargs.S_->getAreaElement(), gl_out);
		for(long j=0;j<Mg;j++)
		{
			p[0]+=gl_out.begin()[j];
		}
	}

	T cx,cy,cz;
  int stp = 0;
  if(box_size>=0) stp = 5;

	for(long j=0;j<Mg;j++)
	{
    for(long lx=-stp;lx<=stp;lx++)
    {
      for(long ly=-stp;ly<=stp;ly++)
      {
        for(long lz=-stp;lz<=stp;lz++)
        {
          cx=sargs.centers[3*j]+lx*box_size; cy=sargs.centers[3*j+1]+ly*box_size; cz=sargs.centers[3*j+2]+lz*box_size;
          T rx=trg[0]-cx;
          T ry=trg[1]-cy;
          T rz=trg[2]-cz;
          T ri=sqrt(rx*rx+ry*ry+rz*rz);
          p[0]+=rsc*(1.0/(4*M_PI))*(rx*x_ptr[3*Mg*Ng+3*j]+ry*x_ptr[3*Mg*Ng+3*j+1]+rz*x_ptr[3*Mg*Ng+3*j+2])/(ri*ri*ri);
        }
      }
    }
	}
}

void vel_eval(staticargs &sargs,T *trg,PVFMMVec normal_,PetscScalar* x_ptr,T *vel,bool flag)
{
	int Ng=sargs.Ng,Mg=sargs.Mg;
	T rsc = sargs.rsc;
  T box_size = sargs.box_size;
	vel[0]=0; vel[1]=0; vel[2]=0;
	if(flag)
	{
		PVFMMVec r_(3*Mg*Ng);
		for(long i=0;i<Ng;i++)
		{
			for(long j=0;j<Mg;j++)
			{
				r_[0*Ng+3*Ng*j+i]=trg[0]-sargs.v_[0*Ng+3*Ng*j+i];
				r_[1*Ng+3*Ng*j+i]=trg[1]-sargs.v_[1*Ng+3*Ng*j+i];
				r_[2*Ng+3*Ng*j+i]=trg[2]-sargs.v_[2*Ng+3*Ng*j+i];
			}
		}

		Sca_t scw,gl_out;
		GaussLegendreIntegrator<Sca_t> integrator_;
		scw.resize(Mg,sargs.sh_order);	//allocating more than needed
		gl_out.resize(Mg,sargs.sh_order);

		for(long j=0;j<Mg;j++)
		{
			for(long i=0;i<Ng;i++)
			{
				T ri = sqrt(r_[3*Ng*j+i]*r_[3*Ng*j+i]+r_[Ng+3*Ng*j+i]*r_[Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*r_[2*Ng+3*Ng*j+i]);
				T rdotn = r_[3*Ng*j+i]*normal_[3*Ng*j+i]+r_[Ng+3*Ng*j+i]*normal_[Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*normal_[2*Ng+3*Ng*j+i];
				scw.begin()[j*Ng+i]=(-3.0/(4.0*M_PI*(ri*ri*ri*ri*ri)))*(rdotn)*(r_[3*Ng*j+i]*r_[3*Ng*j+i]*x_ptr[0*Ng+3*Ng*j+i]+r_[3*Ng*j+i]*r_[Ng+3*Ng*j+i]*x_ptr[1*Ng+3*Ng*j+i]+r_[3*Ng*j+i]*r_[2*Ng+3*Ng*j+i]*x_ptr[2*Ng+3*Ng*j+i]);
			}
		}
		integrator_(scw, sargs.S_->getAreaElement(), gl_out);
		for(long j=0;j<Mg;j++)
		{
			vel[0]+=gl_out.begin()[j];
		}

		for(long j=0;j<Mg;j++)
		{
			for(long i=0;i<Ng;i++)
			{
				T ri = sqrt(r_[3*Ng*j+i]*r_[3*Ng*j+i]+r_[Ng+3*Ng*j+i]*r_[Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*r_[2*Ng+3*Ng*j+i]);
				T rdotn = r_[3*Ng*j+i]*normal_[3*Ng*j+i]+r_[Ng+3*Ng*j+i]*normal_[Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*normal_[2*Ng+3*Ng*j+i];
				scw.begin()[j*Ng+i]=(-3.0/(4.0*M_PI*(ri*ri*ri*ri*ri)))*(rdotn)*(r_[Ng+3*Ng*j+i]*r_[3*Ng*j+i]*x_ptr[0*Ng+3*Ng*j+i]+r_[Ng+3*Ng*j+i]*r_[Ng+3*Ng*j+i]*x_ptr[1*Ng+3*Ng*j+i]+r_[Ng+3*Ng*j+i]*r_[2*Ng+3*Ng*j+i]*x_ptr[2*Ng+3*Ng*j+i]);
			}
		}
		integrator_(scw, sargs.S_->getAreaElement(), gl_out);
		for(long j=0;j<Mg;j++)
		{
			vel[1]+=gl_out.begin()[j];
		}

		for(long j=0;j<Mg;j++)
		{
			for(long i=0;i<Ng;i++)
			{
				T ri = sqrt(r_[3*Ng*j+i]*r_[3*Ng*j+i]+r_[Ng+3*Ng*j+i]*r_[Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*r_[2*Ng+3*Ng*j+i]);
				T rdotn = r_[3*Ng*j+i]*normal_[3*Ng*j+i]+r_[Ng+3*Ng*j+i]*normal_[Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*normal_[2*Ng+3*Ng*j+i];
				scw.begin()[j*Ng+i]=(-3.0/(4.0*M_PI*(ri*ri*ri*ri*ri)))*(rdotn)*(r_[2*Ng+3*Ng*j+i]*r_[3*Ng*j+i]*x_ptr[0*Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*r_[Ng+3*Ng*j+i]*x_ptr[1*Ng+3*Ng*j+i]+r_[2*Ng+3*Ng*j+i]*r_[2*Ng+3*Ng*j+i]*x_ptr[2*Ng+3*Ng*j+i]);
			}
		}
		integrator_(scw, sargs.S_->getAreaElement(), gl_out);
		for(long j=0;j<Mg;j++)
		{
			vel[2]+=gl_out.begin()[j];
		}
	}

	T cx,cy,cz;
  int stp = 0;
  if(box_size>=0) stp = 5;

	for(long j=0;j<Mg;j++)
	{
    for(long lx=-stp;lx<=stp;lx++)
    {
      for(long ly=-stp;ly<=stp;ly++)
      {
        for(long lz=-stp;lz<=stp;lz++)
        {
          cx=sargs.centers[3*j]+lx*box_size; cy=sargs.centers[3*j+1]+ly*box_size; cz=sargs.centers[3*j+2]+lz*box_size;
          T rx=trg[0]-cx;
          T ry=trg[1]-cy;
          T rz=trg[2]-cz;
          T ri=sqrt(rx*rx+ry*ry+rz*rz);
          vel[0]+=rsc*(1.0/(8*M_PI))*(x_ptr[3*Mg*Ng+3*j+0]/ri+(rx*rx*x_ptr[3*Mg*Ng+3*j]+rx*ry*x_ptr[3*Mg*Ng+3*j+1]+rx*rz*x_ptr[3*Mg*Ng+3*j+2])/(ri*ri*ri));
          vel[1]+=rsc*(1.0/(8*M_PI))*(x_ptr[3*Mg*Ng+3*j+1]/ri+(ry*rx*x_ptr[3*Mg*Ng+3*j]+ry*ry*x_ptr[3*Mg*Ng+3*j+1]+ry*rz*x_ptr[3*Mg*Ng+3*j+2])/(ri*ri*ri));
          vel[2]+=rsc*(1.0/(8*M_PI))*(x_ptr[3*Mg*Ng+3*j+2]/ri+(rz*rx*x_ptr[3*Mg*Ng+3*j]+rz*ry*x_ptr[3*Mg*Ng+3*j+1]+rz*rz*x_ptr[3*Mg*Ng+3*j+2])/(ri*ri*ri));
        }
      }
    }
	}

	for(long j=0;j<Mg;j++)
	{
    for(long lx=-stp;lx<=stp;lx++)
    {
      for(long ly=-stp;ly<=stp;ly++)
      {
        for(long lz=-stp;lz<=stp;lz++)
        {
          cx=sargs.centers[3*j]+lx*box_size; cy=sargs.centers[3*j+1]+ly*box_size; cz=sargs.centers[3*j+2]+lz*box_size;
          T rx=trg[0]-cx;
          T ry=trg[1]-cy;
          T rz=trg[2]-cz;
          T ri=sqrt(rx*rx+ry*ry+rz*rz);
          vel[0]+=rsc*rsc*(1.0/(8*M_PI))*(rz*x_ptr[3*Mg*Ng+3*Mg+3*j+1]-ry*x_ptr[3*Mg*Ng+3*Mg+3*j+2])/(ri*ri*ri);
          vel[1]+=rsc*rsc*(1.0/(8*M_PI))*(rx*x_ptr[3*Mg*Ng+3*Mg+3*j+2]-rz*x_ptr[3*Mg*Ng+3*Mg+3*j+0])/(ri*ri*ri);
          vel[2]+=rsc*rsc*(1.0/(8*M_PI))*(ry*x_ptr[3*Mg*Ng+3*Mg+3*j+0]-rx*x_ptr[3*Mg*Ng+3*Mg+3*j+1])/(ri*ri*ri);
        }
      }
    }
  }
}

void stokrot_eval(staticargs &sargs, PVFMMVec sstok, PVFMMVec srot, PVFMMVec flowcenters, T *trg, int trg_cnt, T *vel)
{
  int Mg=sargs.Mg,
      Ng=sargs.Ng;
  MPI_Comm comm = sargs.comm;
  T box_size    = sargs.box_size;

  PVFMMVec src_pos(3*Mg),stokes_den(3*Mg),rot_den(3*Mg),trg_pot_stokes(3*trg_cnt), trg_pot_rot(3*trg_cnt);


  for(long i=0;i<3*Mg;i++)
  {
    src_pos   [i] = flowcenters[i];
    stokes_den[i] = sstok      [i];
    rot_den   [i] = srot       [i];
  }

  PVFMMEval(&src_pos[0],&stokes_den[0],(T*) NULL,src_pos.Dim()/3,&trg[0],&trg_pot_stokes[0],trg_pot_stokes.Dim()/3,&sargs.sctx,1);
  PVFMMEval(&src_pos[0],   &rot_den[0],(T*) NULL,src_pos.Dim()/3,&trg[0],   &trg_pot_rot[0],   trg_pot_rot.Dim()/3,&sargs.rctx,1);

  for(long i=0;i<3*trg_cnt;i++) vel[i] = trg_pot_stokes[i] + trg_pot_rot[i];
/*
	T cx,cy,cz;
	vel[0] = 0,vel[1] = 0,vel[2] = 0;
	for(long j=0;j<Mg;j++)
	{
		cx=flowcenters[3*j]; cy=flowcenters[3*j+1]; cz=flowcenters[3*j+2];
		T rx = trg[0] - cx;
		T ry = trg[1] - cy;
		T rz = trg[2] - cz;
		T ri = sqrt(rx * rx + ry * ry + rz * rz);
		vel[0] += (1.0/(8*M_PI))*(sstok[3*j+0]/ri+(rx*rx*sstok[3*j]+rx*ry*sstok[3*j+1]+rx*rz*sstok[3*j+2])/(ri*ri*ri));
		vel[1] += (1.0/(8*M_PI))*(sstok[3*j+1]/ri+(ry*rx*sstok[3*j]+ry*ry*sstok[3*j+1]+ry*rz*sstok[3*j+2])/(ri*ri*ri));
		vel[2] += (1.0/(8*M_PI))*(sstok[3*j+2]/ri+(rz*rx*sstok[3*j]+rz*ry*sstok[3*j+1]+rz*rz*sstok[3*j+2])/(ri*ri*ri));

		vel[0] += (1.0/(8*M_PI))*(rz*srot[3*j+1]-ry*srot[3*j+2])/(ri*ri*ri);
		vel[1] += (1.0/(8*M_PI))*(rx*srot[3*j+2]-rz*srot[3*j+0])/(ri*ri*ri);
		vel[2] += (1.0/(8*M_PI))*(ry*srot[3*j+0]-rx*srot[3*j+1])/(ri*ri*ri);
	}
*/
}

void linearitycheck(Mat M, Vec U, Vec Y)
{
		MPI_Comm comm=MPI_COMM_WORLD;
		void* dt=NULL;
		MatShellGetContext(M, &dt);
		staticargs *data = (staticargs *)dt;
		int Ng = data->Ng,Mg = data->Mg;
    int n=3*Mg*Ng+6*Mg;
		Vec f1,f2,f3,b1,b2,b3;
	 // Create vectors
		VecCreateMPI(comm,n,PETSC_DETERMINE,&f1);
		VecCreateMPI(comm,n,PETSC_DETERMINE,&f2);
		VecCreateMPI(comm,n,PETSC_DETERMINE,&f3);
		VecCreateMPI(comm,n,PETSC_DETERMINE,&b1);
		VecCreateMPI(comm,n,PETSC_DETERMINE,&b2);
		VecCreateMPI(comm,n,PETSC_DETERMINE,&b3);

		PetscErrorCode ierr;
		PetscScalar *f1_ptr,*f2_ptr,*f3_ptr;

		ierr = VecGetArray(f1, &f1_ptr);
		ierr = VecGetArray(f2, &f2_ptr);
		ierr = VecGetArray(f3, &f3_ptr);

		for(long i=0;i<n;i++)
		{
			f1_ptr[i]=drand48();	
			f2_ptr[i]=drand48();
			f3_ptr[i]=f1_ptr[i]+f2_ptr[i];
		}

		ierr = VecRestoreArray(f1, &f1_ptr);
		ierr = VecRestoreArray(f2, &f2_ptr);
		ierr = VecRestoreArray(f3, &f3_ptr);

		MatMult(M,f1,b1);
		MatMult(M,f2,b2);
		MatMult(M,f3,b3);

		ierr = VecGetArray(b1, &f1_ptr);
		ierr = VecGetArray(b2, &f2_ptr);

		for(long i=0;i<n;i++)
		{
			f1_ptr[i]=f1_ptr[i]+f2_ptr[i];
		}	
		ierr = VecRestoreArray(b1, &f1_ptr);
		ierr = VecRestoreArray(b2, &f2_ptr);
		//RelativeError(b1,b3);
}

void saveMat(Mat M, Vec U, Vec Y)
{
		PetscErrorCode ierr;
		PetscInt U_size;
		PetscScalar *U_ptr;
		PetscScalar *Y_ptr;
		ierr = VecGetLocalSize(U, &U_size);
		ierr = VecGetArray(U, &U_ptr);

		for(long i=0;i<U_size;i++) U_ptr[i]=0;
		ierr = VecRestoreArray(U, &U_ptr);

		std::ofstream myfile;
		myfile.open ("/scratch/romir/Mat_sh32.txt");
		std::cout<<"Started"<<'\n';
		std::cout<<"n = "<<U_size<<'\n';
		for(long i=0;i<U_size;i++)
		{
			std::cout<<"i = "<<i<<'\n';
			ierr = VecGetArray(U, &U_ptr);
			U_ptr[i]=1;
			ierr = VecRestoreArray(U, &U_ptr);
			MatMult(M,U,Y);
			ierr = VecGetArray(U, &U_ptr);
			U_ptr[i]=0;
			ierr = VecRestoreArray(U, &U_ptr);
			ierr = VecGetArray(Y, &Y_ptr);
			for(long j=0;j<U_size;j++) myfile<<std::scientific<<std::setprecision(15)<<Y_ptr[j]<<'\n';
			ierr = VecRestoreArray(Y, &Y_ptr);
		}
		myfile.close();
}

int setDiagonalPreconditionerMatrix(staticargs *s, MPI_Comm comm_self)
{
  T t0,t1;
  t0 = MPI_Wtime();
	int i,j,k,seti,setj,Ng = s->Ng;
  int Ncoef = s->sh_order*(s->sh_order + 2);
  PVFMMVec x,DLMatrix_shc,testdensity,testout;
  testdensity.ReInit(3*Ng);
  for(i=0;i<3*Ng;i++) testdensity[i]=0;
  testdensity[0]=1;
  s->unitdl->SetDensityDL(&testdensity);
	testout=s->unitdl->operator()();  // To setup DLMatrix
  DLMatrix_shc=s->unitdl->GetDLMatrix();

  Mat P_inv, Iden;
  MatCreate(comm_self,&Iden);
  MatCreate(comm_self,&P_inv);
  MatCreate(comm_self,&(s->P));
  MatSetSizes(Iden,3*Ng+6,3*Ng+6,3*Ng+6,3*Ng+6);
  MatSetSizes(P_inv,3*Ng+6,3*Ng+6,3*Ng+6,3*Ng+6);
  MatSetSizes(s->P,3*Ng+6,3*Ng+6,3*Ng+6,3*Ng+6);
  MatSetType(Iden,MATSEQDENSE);
  MatSetType(P_inv,MATSEQDENSE);
  MatSetType(s->P,MATSEQDENSE);
  MatSeqDenseSetPreallocation(Iden,NULL);
  MatSeqDenseSetPreallocation(P_inv,NULL);
  MatSeqDenseSetPreallocation(s->P,NULL);

  PVFMMVec PF,F(3 * Ng * 3 * Ng),DLPMatrix(3 * Ncoef * 3 * Ng),DLMatrix(3 * Ng * 3 * Ng);
  F.SetZero();
  for(i=0;i<3*Ng;i++) F[i+3*Ng*i]=1;
  SphericalHarmonics<T>::Grid2SHC(F, s->sh_order, s->sh_order, PF);
  pvfmm::Matrix<T> Mv(3*Ng,3*Ncoef,&DLPMatrix[0],false);
  pvfmm::Matrix<T> Mf(3*Ng,3*Ncoef,&PF[0],false);
  pvfmm::Matrix<T> M(3*Ncoef,3*Ncoef,&DLMatrix_shc[0],false);
  pvfmm::Matrix<T>::GEMM(Mv,Mf,M);
  SphericalHarmonics<T>::SHC2Grid(DLPMatrix, s->sh_order, s->sh_order, DLMatrix);

  //Initialization
  for(i=0;i<3*Ng+6;i++)
  {
    for(j=0;j<3*Ng+6;j++)
    {
      MatSetValue(P_inv,i,j,0,INSERT_VALUES);
      MatSetValue(Iden,i,j,0,INSERT_VALUES);
      if(i==j) MatSetValue(Iden,i,j,1,INSERT_VALUES);
    }
  }
  MatAssemblyBegin(Iden,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Iden,MAT_FINAL_ASSEMBLY);

  //DL
  for(j=0;j<3*Ng;j++)
  {
    for(i=0;i<3*Ng;i++)
    {
      MatSetValue(P_inv,i,j,DLMatrix[i+j*3*Ng],INSERT_VALUES);
    }
  }

  //1/2I
  for(i=0;i<3*Ng;i++) MatSetValue(P_inv,i,i,0.5,ADD_VALUES);

  //Net Force
	GaussLegendreIntegrator<Sca_t> integrator_;
	Sca_t scw,gl_out;
	scw.resize(1,s->sh_order);
	gl_out.resize(1,s->sh_order);
  x.ReInit(3*Ng);
  const Sca_t& area_elements=s->unitS_->getAreaElement();
	for(i=0;i<Ng;i++) scw.begin()[i]=0;
  for(i=0;i<Ng;i++)
  {
    scw.begin()[i]=1;
	  integrator_(scw, area_elements, gl_out);
    seti=3*Ng+0; setj=0*Ng+i;
    MatSetValue(P_inv,seti,setj,gl_out.begin()[0],INSERT_VALUES);
    seti=3*Ng+1; setj=1*Ng+i;
    MatSetValue(P_inv,seti,setj,gl_out.begin()[0],INSERT_VALUES);
    seti=3*Ng+2; setj=2*Ng+i;
    MatSetValue(P_inv,seti,setj,gl_out.begin()[0],INSERT_VALUES);
    scw.begin()[i]=0;
  }

  //Net Torque
  for(i=0;i<3*Ng;i++) x[i]=0;
  for(k=0;k<3*Ng;k++)
  {
    x[k]=1;
    for(i=0;i<Ng;i++)
	  {
      scw.begin()[i]=s->unitv_[1*Ng+i]*x[2*Ng+i]-s->unitv_[2*Ng+i]*x[1*Ng+i];
	  }
    integrator_(scw, area_elements, gl_out);
    seti=3*Ng+3; setj=k;
    MatSetValue(P_inv,seti,setj,gl_out.begin()[0],INSERT_VALUES);

		for(i=0;i<Ng;i++)
		{
			scw.begin()[i]=s->unitv_[2*Ng+i]*x[0*Ng+i]-s->unitv_[0*Ng+i]*x[2*Ng+i];
		}
    integrator_(scw, area_elements, gl_out);
    seti=3*Ng+4; setj=k;
    MatSetValue(P_inv,seti,setj,gl_out.begin()[0],INSERT_VALUES);

		for(i=0;i<Ng;i++)
		{
			scw.begin()[i]=s->unitv_[0*Ng+i]*x[1*Ng+i]-s->unitv_[1*Ng+i]*x[0*Ng+i];
		}
	  integrator_(scw, area_elements, gl_out);
    seti=3*Ng+5; setj=k;
    MatSetValue(P_inv,seti,setj,gl_out.begin()[0],INSERT_VALUES);

    x[k]=0;
  }

  T val;
  //Stokeslets
	for(i=0;i<Ng;i++)
	{
		T rx=s->unitv_[0*Ng+i];
		T ry=s->unitv_[1*Ng+i];
		T rz=s->unitv_[2*Ng+i];
		T ri=sqrt(rx*rx+ry*ry+rz*rz);

    val=(1.0/(8*M_PI))*(1/ri+(rx*rx)/(ri*ri*ri));
    seti=i; setj=3*Ng+0;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=(1.0/(8*M_PI))*((rx*ry)/(ri*ri*ri));
    seti=i; setj=3*Ng+1;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=(1.0/(8*M_PI))*((rx*rz)/(ri*ri*ri));
    seti=i; setj=3*Ng+2;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
  }

	for(i=Ng;i<2*Ng;i++)
  {
		T rx=s->unitv_[-1*Ng+i];
		T ry=s->unitv_[ 0*Ng+i];
		T rz=s->unitv_[ 1*Ng+i];
		T ri=sqrt(rx*rx+ry*ry+rz*rz);

    val=(1.0/(8*M_PI))*((ry*rx)/(ri*ri*ri));
    seti=i; setj=3*Ng+0;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=(1.0/(8*M_PI))*(1/ri+(ry*ry)/(ri*ri*ri));
    seti=i; setj=3*Ng+1;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=(1.0/(8*M_PI))*((ry*rz)/(ri*ri*ri));
    seti=i; setj=3*Ng+2;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
  }

	for(i=2*Ng;i<3*Ng;i++)
	{
		T rx=s->unitv_[-2*Ng+i];
		T ry=s->unitv_[-1*Ng+i];
		T rz=s->unitv_[ 0*Ng+i];
		T ri=sqrt(rx*rx+ry*ry+rz*rz);

    val=(1.0/(8*M_PI))*((rz*rx)/(ri*ri*ri));
    seti=i; setj=3*Ng+0;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=(1.0/(8*M_PI))*((rz*ry)/(ri*ri*ri));
    seti=i; setj=3*Ng+1;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=(1.0/(8*M_PI))*(1/ri+(rz*rz)/(ri*ri*ri));
    seti=i; setj=3*Ng+2;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
	}

	//Rotlets
  for(i=0;i<Ng;i++)
	{
		T rx=s->unitv_[0*Ng+i];
		T ry=s->unitv_[1*Ng+i];
		T rz=s->unitv_[2*Ng+i];
		T ri=sqrt(rx*rx+ry*ry+rz*rz);

    val=0;
    seti=i; setj=3*Ng+3;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=(1.0/(8*M_PI))*(rz)/(ri*ri*ri);
    seti=i; setj=3*Ng+4;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=-(1.0/(8*M_PI))*(ry)/(ri*ri*ri);
    seti=i; setj=3*Ng+5;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
	}

  for(i=Ng;i<2*Ng;i++)
	{
		T rx=s->unitv_[-1*Ng+i];
		T ry=s->unitv_[ 0*Ng+i];
		T rz=s->unitv_[ 1*Ng+i];
		T ri=sqrt(rx*rx+ry*ry+rz*rz);

    val=-(1.0/(8*M_PI))*(rz)/(ri*ri*ri);
    seti=i; setj=3*Ng+3;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=0;
    seti=i; setj=3*Ng+4;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=(1.0/(8*M_PI))*(rx)/(ri*ri*ri);
    seti=i; setj=3*Ng+5;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
	}

  for(i=2*Ng;i<3*Ng;i++)
	{
		T rx=s->unitv_[-2*Ng+i];
		T ry=s->unitv_[-1*Ng+i];
		T rz=s->unitv_[ 0*Ng+i];
		T ri=sqrt(rx*rx+ry*ry+rz*rz);

    val=(1.0/(8*M_PI))*(ry)/(ri*ri*ri);
    seti=i; setj=3*Ng+3;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=-(1.0/(8*M_PI))*(rx)/(ri*ri*ri);
    seti=i; setj=3*Ng+4;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
    val=0;
    seti=i; setj=3*Ng+5;
    MatSetValue(P_inv,seti,setj,val,INSERT_VALUES);
  }

  MatAssemblyBegin(P_inv,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(P_inv,MAT_FINAL_ASSEMBLY);

  MatLUFactor(P_inv,NULL,NULL,NULL);
  MatMatSolve(P_inv,Iden,s->P);
  t1 = MPI_Wtime();
  std::cout<<"Time taken to build preconditioner = "<<t1-t0<<'\n';
  return 0;
}

int diagonalPreconditioner(PC pc, Vec U, Vec Y)
{
  T t0, t1;
  t0 = MPI_Wtime();
  void* dt = NULL;
	PCShellGetContext(pc, &dt);
	staticargs *data = (staticargs *)dt;
	int Ng = data->Ng, Mg = data->Mg, myrank;
	T rsc = data->rsc;
	long i,j,k;

  PetscErrorCode ierr;

	PetscInt U_size;
	const PetscScalar *U_ptr;
	ierr = VecGetLocalSize(U, &U_size);
	ierr = VecGetArrayRead(U, &U_ptr);

	PetscInt Y_size;
	PetscScalar *Y_ptr;
	ierr = VecGetLocalSize(Y, &Y_size);
	ierr = VecGetArray(Y, &Y_ptr);

  MPI_Comm comm_self, comm = data->comm;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_split(comm, myrank, myrank, &comm_self);
  Vec tmpU,tmpY;
  VecCreateMPI(comm_self,3*Ng+6,PETSC_DETERMINE,&tmpU);
  VecCreateMPI(comm_self,3*Ng+6,PETSC_DETERMINE,&tmpY);
  PetscScalar *tmpU_ptr,*tmpY_ptr;
  for(i=0;i<Mg;i++)
  {
    ierr = VecGetArray(tmpU, &tmpU_ptr);
    #pragma omp parallel for
    for(j=0;j<3*Ng;j++) tmpU_ptr[j]=U_ptr[3*Ng*i+j];
    tmpU_ptr[3*Ng+0]=U_ptr[3*Ng*Mg+3*i+0];
    tmpU_ptr[3*Ng+1]=U_ptr[3*Ng*Mg+3*i+1];
    tmpU_ptr[3*Ng+2]=U_ptr[3*Ng*Mg+3*i+2];
    tmpU_ptr[3*Ng+3]=U_ptr[3*Ng*Mg+3*Mg+3*i+0];
    tmpU_ptr[3*Ng+4]=U_ptr[3*Ng*Mg+3*Mg+3*i+1];
    tmpU_ptr[3*Ng+5]=U_ptr[3*Ng*Mg+3*Mg+3*i+2];
    ierr = VecRestoreArray(tmpU, &tmpU_ptr);
    MatMult(data->P,tmpU,tmpY);

    ierr = VecGetArray(tmpY, &tmpY_ptr);
    #pragma omp parallel for
    for(j=0;j<3*Ng;j++) Y_ptr[3*Ng*i+j]=tmpY_ptr[j];
    Y_ptr[3*Ng*Mg+3*i+0]=tmpY_ptr[3*Ng+0];
    Y_ptr[3*Ng*Mg+3*i+1]=tmpY_ptr[3*Ng+1];
    Y_ptr[3*Ng*Mg+3*i+2]=tmpY_ptr[3*Ng+2];
    Y_ptr[3*Ng*Mg+3*Mg+3*i+0]=tmpY_ptr[3*Ng+3];
    Y_ptr[3*Ng*Mg+3*Mg+3*i+1]=tmpY_ptr[3*Ng+4];
    Y_ptr[3*Ng*Mg+3*Mg+3*i+2]=tmpY_ptr[3*Ng+5];
    ierr = VecRestoreArray(tmpY, &tmpY_ptr);
  }
  ierr = VecRestoreArrayRead(U, &U_ptr);
  ierr = VecRestoreArray(Y, &Y_ptr);
  t1 = MPI_Wtime();
  if(!myrank) std::cout<<"Time taken by preconditioner = "<<t1-t0<<'\n';
  return 0;
}

void TimeStepper(T t_final, T dt, T n_tracers, staticargs &sargs, PVFMMVec &position, T* v_back, PetscScalar* x_ptr, int flag)
{
  T t_steps = t_final/dt;
  T vel_rs[3];
  PVFMMVec pos_current(3*n_tracers),k1,k2,k3,k4;
  for(int i=0;i<t_steps;i++)
  {
    if(flag == 0)
    {
      for(int j=0;j<3*n_tracers;j++)  pos_current[j] = position[i*3*n_tracers + j];
	    sargs.stokesdl->SetTrgCoord(&pos_current);
      k1=sargs.stokesdl->operator()();
      for(int j=0;j<n_tracers;j++)
      {
        vel_eval(sargs,&position[i*3*n_tracers + 3*j],NULL,x_ptr,vel_rs,0);
        position[(i+1)*3*n_tracers + 3*j + 0] = position[i*3*n_tracers + 3*j + 0] + dt * (k1[3*j + 0] + vel_rs[0] + v_back[0]);
        position[(i+1)*3*n_tracers + 3*j + 1] = position[i*3*n_tracers + 3*j + 1] + dt * (k1[3*j + 1] + vel_rs[1] + v_back[1]);
        position[(i+1)*3*n_tracers + 3*j + 2] = position[i*3*n_tracers + 3*j + 2] + dt * (k1[3*j + 2] + vel_rs[2] + v_back[2]);
      }
    }/*
    else if(flag == 1)
    {
      for(int j=0;j<3*n_tracers;j++)  pos_current[j] = position[i*3*n_tracers + j];
	    sargs->stokesdl->SetTrgCoord(&pos_current);
      k1=sargs->stokesdl->operator()();
      for(int j=0;j<n_tracers;j++)
      {
        k1[3*j + 0] = dt * (k1[3*j + 0] + v_back[0]);
        k1[3*j + 1] = dt * (k1[3*j + 1] + v_back[1]);
        k1[3*j + 2] = dt * (k1[3*j + 2] + v_back[2]);
        pos_current[3*j + 0] += k1[3*j + 0]/2;
        pos_current[3*j + 1] += k1[3*j + 1]/2;
        pos_current[3*j + 2] += k1[3*j + 2]/2;
      }
	    sargs->stokesdl->SetTrgCoord(&pos_current);
      k2=sargs->stokesdl->operator()();
      for(int j=0;j<n_tracers;j++)
      {
        k2[3*j + 0] = dt * (k2[3*j + 0] + v_back[0]);
        k2[3*j + 1] = dt * (k2[3*j + 1] + v_back[1]);
        k2[3*j + 2] = dt * (k2[3*j + 2] + v_back[2]);
        pos_current[3*j + 0] += k2[3*j + 0]/2;
        pos_current[3*j + 1] += k2[3*j + 1]/2;
        pos_current[3*j + 2] += k2[3*j + 2]/2;
      }
	    sargs->stokesdl->SetTrgCoord(&pos_current);
      k3=sargs->stokesdl->operator()();
      for(int j=0;j<n_tracers;j++)
      {
        k3[3*j + 0] = dt * (k3[3*j + 0] + v_back[0]);
        k3[3*j + 1] = dt * (k3[3*j + 1] + v_back[1]);
        k3[3*j + 2] = dt * (k3[3*j + 2] + v_back[2]);
        pos_current[3*j + 0] += k3[3*j + 0];
        pos_current[3*j + 1] += k3[3*j + 1];
        pos_current[3*j + 2] += k3[3*j + 2];
      }
	    sargs->stokesdl->SetTrgCoord(&pos_current);
      k4=sargs->stokesdl->operator()();
      for(int j=0;j<n_tracers;j++)
      {
        position[(i+1)*3*n_tracers + 3*j + 0] = position[i*3*n_tracers + 3*j + 0] + k1[3*j+0]/6 + k2[3*j+0]/3 + k3[3*j+0]/3 + dt * (k4[3*j + 0] + v_back[0])/6;
        position[(i+1)*3*n_tracers + 3*j + 1] = position[i*3*n_tracers + 3*j + 1] + k1[3*j+1]/6 + k2[3*j+1]/3 + k3[3*j+1]/3 + dt * (k4[3*j + 1] + v_back[1])/6;
        position[(i+1)*3*n_tracers + 3*j + 2] = position[i*3*n_tracers + 3*j + 2] + k1[3*j+2]/6 + k2[3*j+2]/3 + k3[3*j+2]/3 + dt * (k4[3*j + 2] + v_back[2])/6;
      }
    }*/
  }
}

void single_sphere_uniform_flow(staticargs &sargs,Vec x, T *v_back,T r_tar, T r_ref=10.0)
{
  //spherical grid for unit sphere case
  long ti=0;
  int Mg = sargs.Mg, Ng = sargs.Ng;
  T rel_error_vel=0.0, rel_error_press=0.0, max_press_exact=0.0, theta_tar, phi_tar,vel[3];
  PVFMMVec tmpG, tmpDphiG, tmpKphiG(50*50), tmpKphiG_ref(50*50), tar, tar_ref, tar_vel, tar_press, tar_press_ref, tar_vel_exact, tar_press_exact;
	tar.ReInit(3*50*50);
  tar_ref.ReInit(3*50*50);
	tar_vel.ReInit(3*50*50);
  tar_press.ReInit(50*50);
  tar_press_ref.ReInit(50*50);
  tar_vel_exact.ReInit(3);
  tar_press_exact.ReInit(1);

  PetscScalar *x_ptr;
  VecGetArray(x, &x_ptr);
	tmpG.ReInit(3*Mg*Ng);
	for(long i=0;i<3*Mg*Ng;i++) tmpG[i]=x_ptr[i];

  PVFMMVec normal_(3*Mg*Ng);
  for(long i=0;i<3*Mg*Ng;i++)  normal_[i] = -1*sargs.S_->getNormal().begin()[i];


  for(theta_tar=0;theta_tar<M_PI/2;theta_tar+=M_PI/100)
  {
    for(phi_tar=0;phi_tar<M_PI/2;phi_tar+=M_PI/100)
    {
      T ax = r_tar*sin(theta_tar)*cos(phi_tar);
      T ay = r_tar*sin(theta_tar)*sin(phi_tar);
      T az = r_tar*cos(theta_tar);
      tar    [ti+0] = ax;
      tar    [ti+1] = ay;
      tar    [ti+2] = az;
      tar_ref[ti+0] = ax*r_ref/r_tar;
      tar_ref[ti+1] = ay*r_ref/r_tar;
      tar_ref[ti+2] = az*r_ref/r_tar;
			vel_eval(sargs,&tar[ti],NULL,x_ptr,&tar_vel[ti],0);
			press_eval(sargs,&tar[ti],normal_,x_ptr,&tar_press[ti/3],0);
			press_eval(sargs,&tar_ref[ti],normal_,x_ptr,&tar_press_ref[ti/3],0);
      ti+=3;
    }
  }

  VecRestoreArray(x, &x_ptr);

  sargs.stokesdl->SetKernel(sargs.box_size,1000,10,MAX_DEPTH,&StokesKernel<T>::Kernel(),sargs.comm);
  sargs.stokesdl->SetSrcCoord(sargs.v_);
	sargs.stokesdl->SetTrgCoord(&tar);
	sargs.stokesdl->SetDensityDL(&tmpG);

  tmpDphiG=sargs.stokesdl->operator()();

  sargs.stokesdl->SetKernel(sargs.box_size,1000,10,MAX_DEPTH,&pressure_kernel(),sargs.comm);
  sargs.stokesdl->SetSrcCoord(sargs.v_);
	sargs.stokesdl->SetTrgCoord(&tar);
	sargs.stokesdl->SetDensityDL(&tmpG);
	tmpKphiG=sargs.stokesdl->operator()();

  sargs.stokesdl->SetTrgCoord(&tar_ref);
	tmpKphiG_ref=sargs.stokesdl->operator()();
  std::cout<<"1"<<'\n';
  ti=0;
  for(theta_tar=0;theta_tar<M_PI/2;theta_tar+=M_PI/100)
  {
    for(phi_tar=0;phi_tar<M_PI/2;phi_tar+=M_PI/100)
    {
      tar_vel[ti+0] += tmpDphiG[ti+0] + v_back[0];
      tar_vel[ti+1] += tmpDphiG[ti+1] + v_back[1];
      tar_vel[ti+2] += tmpDphiG[ti+2] + v_back[2];

      tar_press    [ti/3] += tmpKphiG    [ti/3];
      tar_press_ref[ti/3] += tmpKphiG_ref[ti/3];

      T ax = tar[ti+0];
      T ay = tar[ti+1];
      T az = tar[ti+2];

      T v_r = cos(theta_tar)*(1+1/(2*pow(r_tar,3))-3/(2*r_tar));
      T v_theta = -sin(theta_tar)*(1-1/(4*pow(r_tar,3))-3/(4*r_tar));

      tar_vel_exact  [0] = v_r*sin(theta_tar)*cos(phi_tar) + v_theta*cos(theta_tar)*cos(phi_tar);
      tar_vel_exact  [1] = v_r*sin(theta_tar)*sin(phi_tar) + v_theta*cos(theta_tar)*sin(phi_tar);
      tar_vel_exact  [2] = v_r*cos(theta_tar) - v_theta*sin(theta_tar);
      tar_press_exact[0] = (-3.0/(2.0*(pow(r_tar,2))))*cos(theta_tar) - (-3.0/(2.0*(pow(r_ref,2))))*cos(theta_tar);

      vel[0] = tar_vel[ti+0];
      vel[1] = tar_vel[ti+1];
      vel[2] = tar_vel[ti+2];

      T tmp=sqrt((tar_vel_exact[0]-vel[0])*(tar_vel_exact[0]-vel[0])+(tar_vel_exact[1]-vel[1])*(tar_vel_exact[1]-vel[1])+(tar_vel_exact[2]-vel[2])*(tar_vel_exact[2]-vel[2]));
//      /sqrt(tar_vel_exact[0]*tar_vel_exact[0]+tar_vel_exact[1]*tar_vel_exact[1]+tar_vel_exact[2]*tar_vel_exact[2]);
      if(rel_error_vel < tmp) rel_error_vel = tmp;

      tmp=fabs(tar_press[ti/3] - tar_press_ref[ti/3] - tar_press_exact[0]);
      if(max_press_exact < fabs(tar_press_exact[0])) max_press_exact = fabs(tar_press_exact[0]);
      if(rel_error_press < tmp) rel_error_press = tmp;
      ti+=3;
	  }
  }
  rel_error_press /= max_press_exact;
  std::cout<<"r_tar, max rel_error_vel rel_error_press : "<<r_tar<<"  "<<rel_error_vel<<" "<<rel_error_press<<'\n';
	std::ofstream myoutputfile;
	myoutputfile.open ("./rel_error.txt");
	myoutputfile<<std::scientific<<std::setprecision(2)<<r_tar<<" "<<std::scientific<<std::setprecision(15)<<rel_error_vel<<'\n';
  myoutputfile.close();
}

void GetRotationMatrix(T alpha, T beta, T gamma, T *RM)
{
  RM[0] = cos(gamma)*cos(beta);
  RM[1] = cos(gamma)*sin(beta)*sin(alpha)-sin(gamma)*cos(alpha);
  RM[2] = cos(gamma)*sin(beta)*cos(alpha)+sin(gamma)*sin(alpha);
  RM[3] = sin(gamma)*cos(beta);
  RM[4] = sin(gamma)*sin(beta)*sin(alpha)+cos(gamma)*cos(alpha);
  RM[5] = sin(gamma)*sin(beta)*cos(alpha)-cos(gamma)*sin(alpha);
  RM[6] =-sin(beta);
  RM[7] = cos(beta)*sin(alpha);
  RM[8] = cos(beta)*cos(alpha);
}

T crat = 1.0;
void SetGeometry(int shape, int M_gl, staticargs &sargs, MPI_Comm comm)
{
  int np,myrank,Mg=sargs.Mg,Ng=sargs.Ng;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);
  PVFMMVec theta,thetaunit,shc,grid,tmpv_;
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
		SphericalHarmonics<T>::SHC2Grid(shc, 2, sargs.sh_order, tmpv_);
		if(M_gl == 1)
		{
			sargs.rsc=1;
			T inx = 0.5,
        iny = 0.5,
        inz = 0.5,
        inr = 0.15;
			sargs.centers[0]=inx;
      sargs.centers[1]=iny;
      sargs.centers[2]=inz;
			for(long i=0;i<Ng;i++)
			{
				sargs.v_[0*Ng+i]=inx+inr*tmpv_[0*Ng+i];
				sargs.v_[1*Ng+i]=iny+inr*tmpv_[1*Ng+i];
				sargs.v_[2*Ng+i]=inz+inr*tmpv_[2*Ng+i];
        std::cout<<tmpv_[0*Ng+i]<<" "<<tmpv_[1*Ng+i]<<" "<<tmpv_[2*Ng+i]<<'\n';
			}
		}
		else if(M_gl == 2)
		{
      sargs.rsc=0.125;
      T px[2] = {0.375-(0.125/(2*crat)),0.625+(0.125/(2*crat))};
			T py[2] = {0.5,0.5};
			T pz[2] = {0.5,0.5};
      //T px[2] = {0.2,0.8};
      //T py[2] = {0.5,0.5};
      //T pz[2] = {0.5,0.5};
			T inx,iny,inz,inr;
      int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
      int end = start + M_gl/np + (M_gl%np>myrank?1:0);
			for(long j=start;j<end;j++)
			{
        inx=px[j];
        iny=py[j];
        inz=pz[j];
        inr=0.125;
        sargs.centers[3*(j-start)+0]=inx;
        sargs.centers[3*(j-start)+1]=iny;
        sargs.centers[3*(j-start)+2]=inz;
        std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
        for(long i=0;i<Ng;i++)
        {
          sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*tmpv_[0*Ng+i];
          sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*tmpv_[1*Ng+i];
          sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*tmpv_[2*Ng+i];
        }
			}
		}
		else if(M_gl == 8)
		{
			sargs.rsc=0.5;
			T px[8] = {1,1,1,1,-1,-1,-1,-1};
			T py[8] = {1,1,-1,-1,1,1,-1,-1};
			T pz[8] = {1,-1,1,-1,1,-1,1,-1};
			T inx,iny,inz,inr;
      int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
      int end = start + M_gl/np + (M_gl%np>myrank?1:0);
      for(long j=start;j<end;j++)
			{
        inx=px[j];
        iny=py[j];
        inz=pz[j];
        inr=0.5;
        sargs.centers[3*(j-start)+0]=inx;
        sargs.centers[3*(j-start)+1]=iny;
        sargs.centers[3*(j-start)+2]=inz;
        std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
        for(long i=0;i<Ng;i++)
        {
          sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*tmpv_[0*Ng+i];
          sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*tmpv_[1*Ng+i];
          sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*tmpv_[2*Ng+i];
        }
			}
		}
		else if(M_gl == 59)
		{
			sargs.rsc=0.1;
			T inx,iny,inz,inr;
			std::fstream myinputfile("./rand_geo.txt");
			if (myinputfile.is_open())
			{
        int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
        int end   = start + M_gl/np + (M_gl%np>myrank?1:0);
        for(long j=0;j<M_gl;j++)
				{
					myinputfile>>inx;
					myinputfile>>iny;
					myinputfile>>inz;
					myinputfile>>inr;
          if(j>=start && j<end)
          {
            sargs.centers[3*(j-start)+0]=inx;
            sargs.centers[3*(j-start)+1]=iny;
            sargs.centers[3*(j-start)+2]=inz;
					  std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
					  for(long i=0;i<Ng;i++)
					  {
              sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*tmpv_[0*Ng+i];
              sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*tmpv_[1*Ng+i];
              sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*tmpv_[2*Ng+i];
					  }
          }
				}
			}
			myinputfile.close();
    }
    else if(M_gl == 153)
		{
			sargs.rsc=1e-01;
			T inx,iny,inz,inr;
			std::fstream myinputfile("sphere_geometry_153.txt");
			if (myinputfile.is_open())
			{
        int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
        int end   = start + M_gl/np + (M_gl%np>myrank?1:0);
        for(long j=0;j<M_gl;j++)
				{
					myinputfile>>inx;
					myinputfile>>iny;
					myinputfile>>inz;
					myinputfile>>inr;
          if(j>=start && j<end)
          {
            sargs.centers[3*(j-start)+0]=inx;
            sargs.centers[3*(j-start)+1]=iny;
            sargs.centers[3*(j-start)+2]=inz;
					  std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<0.99*inr<<'\n';
					  for(long i=0;i<Ng;i++)
					  {
              sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+0.99*inr*tmpv_[0*Ng+i];
              sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+0.99*inr*tmpv_[1*Ng+i];
              sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+0.99*inr*tmpv_[2*Ng+i];
					  }
          }
				}
			}
			myinputfile.close();
    }
		else if(M_gl == 2048)
		{
			sargs.rsc=1;
			T inx,iny,inz,inr;
			std::fstream myinputfile("spheres_geometry_2048.txt");
			if (myinputfile.is_open())
			{
        int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
        int end = start + M_gl/np + (M_gl%np>myrank?1:0);
        for(long j=0;j<M_gl;j++)
				{
					myinputfile>>inx;
					myinputfile>>iny;
					myinputfile>>inz;
					myinputfile>>inr;
          if(j>=start && j<end)
          {
            sargs.centers[3*(j-start)+0]=inx;
            sargs.centers[3*(j-start)+1]=iny;
            sargs.centers[3*(j-start)+2]=inz;
					  std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
					  for(long i=0;i<Ng;i++)
					  {
              sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*tmpv_[0*Ng+i];
              sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*tmpv_[1*Ng+i];
              sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*tmpv_[2*Ng+i];
					  }
          }
				}
			}
			myinputfile.close();
    }
    else if(M_gl == 12800)
		{
			sargs.rsc=0.01;
			T inx,iny,inz,inr;
			std::fstream myinputfile("sphere_geometry_12800.txt");
			if (myinputfile.is_open())
			{
        int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
        int end = start + M_gl/np + (M_gl%np>myrank?1:0);
        for(long j=0;j<M_gl;j++)
				{
					myinputfile>>inx;
					myinputfile>>iny;
					myinputfile>>inz;
					myinputfile>>inr;
          if(j>=start && j<end)
          {
            sargs.centers[3*(j-start)+0]=inx;
            sargs.centers[3*(j-start)+1]=iny;
            sargs.centers[3*(j-start)+2]=inz;
					  std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
					  for(long i=0;i<Ng;i++)
					  {
              sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*tmpv_[0*Ng+i];
              sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*tmpv_[1*Ng+i];
              sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*tmpv_[2*Ng+i];
					  }
          }
				}
			}
			myinputfile.close();
    }
    else
		{
			sargs.rsc=1.0/30.0;
			SphericalHarmonics<T>::SHC2Grid(shc, 2, sargs.sh_order, tmpv_);
			T inx,iny,inz,inr;
      int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
      int end = start + M_gl/np + (M_gl%np>myrank?1:0);
			for(long j=start;j<end;j++)
			{
        inx = 3.0/60.0 + 6.0*(j%10)/60.0;
        iny = 3.0/60.0 + 6.0*((j/10)%10)/60.0;
        inz = 3.0/60.0 + 0.1*floor(j/100);//-3* (T) np/2.0 + 3/2.0 + 3*((j*np)/M_gl);
        inr = 1.0/30.0;
        sargs.centers[3*(j-start)+0]=inx;
        sargs.centers[3*(j-start)+1]=iny;
        sargs.centers[3*(j-start)+2]=inz;
        std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
        for(long i=0;i<Ng;i++)
        {
          sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*tmpv_[0*Ng+i];
          sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*tmpv_[1*Ng+i];
          sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*tmpv_[2*Ng+i];
        }
			}
		}
  }
  else if(shape==2)
  {
    int p0 = sargs.sh_order;

    std::vector<T> qx(p0+1), qw(p0+1);
    qx.resize(p0+1);
    cgqf(p0+1, 1, 0.0, 0.0, -1.0, 1.0, &qx[0], &qw[0]);

    grid.ReInit(3*Ng);
    for(long i=0;i<(p0+1);i++)
    {
      for(long j=0;j<2*p0;j++)
      {
        T th = acos(qx[i]);
        //T th = -M_PI/2.01 + i*2*M_PI/(2.01*(p0+1));
        T phi = j*M_PI/p0;
        //T xi = -1*cos(th);
        //T yi = (0.3+pow(sin(2*th),2))*sin(phi);
        //T zi = (0.3+pow(sin(2*th),2))*cos(phi);
        T xi = -2*cos(th);
        T yi = 1*sin(th)*sin(phi);
        T zi = 1*sin(th)*cos(phi);
        grid[0*Ng+2*i*p0+j]=xi;
        grid[1*Ng+2*i*p0+j]=yi;
        grid[2*Ng+2*i*p0+j]=zi;
      }
    }
		SphericalHarmonics<T>::Grid2SHC(grid, p0, p0, shc);
		SphericalHarmonics<T>::SHC2Grid(shc, p0, p0, tmpv_);
		if(M_gl == 1)
		{
			sargs.rsc=1;
			T inx = 0,
        iny = 0,
        inz = 0,
        inr = 1;
			sargs.centers[0]=inx; sargs.centers[1]=iny; sargs.centers[2]=inz;
			std::cout<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
			for(long i=0;i<Ng;i++)
			{
				sargs.v_[0*Ng+i]=inx+inr*tmpv_[0*Ng+i];
				sargs.v_[1*Ng+i]=iny+inr*tmpv_[1*Ng+i];
				sargs.v_[2*Ng+i]=inz+inr*tmpv_[2*Ng+i];
        std::cout<<tmpv_[0*Ng+i]<<" "<<tmpv_[1*Ng+i]<<" "<<tmpv_[2*Ng+i]<<'\n';
			}
		}
		else if(M_gl == 2)
		{
			sargs.rsc=1;
			T px[2]    = {0.3,0.7};
			T py[2]    = {0.5,0.5};
			T pz[2]    = {0.5,0.5};
      T alpha[2] = {0,0};
      T beta [2] = {0,M_PI/2};
      T gamma[2] = {0,0};
      T RM[9];
			T inx,iny,inz,inr;
      int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
      int end = start + M_gl/np + (M_gl%np>myrank?1:0);
			for(long j=start;j<end;j++)
			{
        inx=px[j];
        iny=py[j];
        inz=pz[j];
        inr=0.1;
        sargs.centers[3*(j-start)+0]=inx;
        sargs.centers[3*(j-start)+1]=iny;
        sargs.centers[3*(j-start)+2]=inz;
        std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
        theta[3*(j-start)+0]=alpha[j]-alpha[0];
        theta[3*(j-start)+1]=beta [j]-beta [0];
        theta[3*(j-start)+2]=gamma[j]-gamma[0];
        GetRotationMatrix(alpha[j],beta[j],gamma[j],&RM[0]);
        for(long i=0;i<Ng;i++)
        {
          sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*(RM[0]*tmpv_[0*Ng+i]+RM[1]*tmpv_[1*Ng+i]+RM[2]*tmpv_[2*Ng+i]);
          sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*(RM[3]*tmpv_[0*Ng+i]+RM[4]*tmpv_[1*Ng+i]+RM[5]*tmpv_[2*Ng+i]);
          sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*(RM[6]*tmpv_[0*Ng+i]+RM[7]*tmpv_[1*Ng+i]+RM[8]*tmpv_[2*Ng+i]);
        }
			}
		}
		else if(M_gl == 3)
		{
			sargs.rsc=1;
			T px[3]    = {0,-2, 2};
			T py[3]    = {0, 0, 0};
			T pz[3]    = {2,-1,-1};
      T alpha[3] = {0,0,0};
      T beta [3] = {0,M_PI/3,2*M_PI/3};
      T gamma[3] = {0,0,0};
      T RM[9];
			T inx,iny,inz,inr;
      int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
      int end = start + M_gl/np + (M_gl%np>myrank?1:0);
      for(long j=start;j<end;j++)
			{
        inx=px[j];
        iny=py[j];
        inz=pz[j];
        inr=1;
        sargs.centers[3*(j-start)+0]=inx;
        sargs.centers[3*(j-start)+1]=iny;
        sargs.centers[3*(j-start)+2]=inz;
        std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
        theta[3*(j-start)+0]=alpha[j]-alpha[0];
        theta[3*(j-start)+1]=beta [j]-beta [0];
        theta[3*(j-start)+2]=gamma[j]-gamma[0];
        GetRotationMatrix(alpha[j],beta[j],gamma[j],&RM[0]);
        for(long i=0;i<Ng;i++)
        {
          sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*(RM[0]*tmpv_[0*Ng+i]+RM[1]*tmpv_[1*Ng+i]+RM[2]*tmpv_[2*Ng+i]);
          sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*(RM[3]*tmpv_[0*Ng+i]+RM[4]*tmpv_[1*Ng+i]+RM[5]*tmpv_[2*Ng+i]);
          sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*(RM[6]*tmpv_[0*Ng+i]+RM[7]*tmpv_[1*Ng+i]+RM[8]*tmpv_[2*Ng+i]);
        }
			}
		}
    else if(M_gl == 4)
		{
			sargs.rsc=1;
			T px[4]    = {7,2,8,4};
			T py[4]    = {5,5,5,5};
			T pz[4]    = {3,3,8,8};
      T alpha[4] = {0,0,0,0};
      T beta [4] = {M_PI/3,M_PI/3,2*M_PI/3,2*M_PI/3};
      T gamma[4] = {0,0,0,0};
      T RM[9];
			T inx,iny,inz,inr;
      int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
      int end = start + M_gl/np + (M_gl%np>myrank?1:0);
      for(long j=start;j<end;j++)
			{
        inx=px[j];
        iny=py[j];
        inz=pz[j];
        inr=1;
        sargs.centers[3*(j-start)+0]=inx;
        sargs.centers[3*(j-start)+1]=iny;
        sargs.centers[3*(j-start)+2]=inz;
        std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
        theta[3*(j-start)+0]=alpha[j]-alpha[0];
        theta[3*(j-start)+1]=beta [j]-beta [0];
        theta[3*(j-start)+2]=gamma[j]-gamma[0];
        GetRotationMatrix(alpha[j],beta[j],gamma[j],&RM[0]);
        for(long i=0;i<Ng;i++)
        {
          sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*(RM[0]*tmpv_[0*Ng+i]+RM[1]*tmpv_[1*Ng+i]+RM[2]*tmpv_[2*Ng+i]);
          sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*(RM[3]*tmpv_[0*Ng+i]+RM[4]*tmpv_[1*Ng+i]+RM[5]*tmpv_[2*Ng+i]);
          sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*(RM[6]*tmpv_[0*Ng+i]+RM[7]*tmpv_[1*Ng+i]+RM[8]*tmpv_[2*Ng+i]);
        }
			}
		}
    else if(M_gl == 9261)
		{
			sargs.rsc=1;
      T alpha[4] = {0,0,0,0};
      T beta [4] = {M_PI/3,M_PI/3,2*M_PI/3,2*M_PI/3};
      T gamma[4] = {0,0,0,0};
      T RM[9];
			T inx,iny,inz,inr;
      int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
      int end = start + M_gl/np + (M_gl%np>myrank?1:0);
      for(long j=start;j<end;j++)
			{
        inx=0.05+0.045*(j%21);
        iny=0.05+0.045*((j/21)%21);
        inz=0.03+0.045*floor((j/441));
        inr=0.01;
        sargs.centers[3*(j-start)+0]=inx;
        sargs.centers[3*(j-start)+1]=iny;
        sargs.centers[3*(j-start)+2]=inz;
        std::cout<<"proc "<<myrank<<" "<<inx<<" "<<iny<<" "<<inz<<" "<<inr<<'\n';
        theta[3*(j-start)+0]=0;
        theta[3*(j-start)+1]=0;
        theta[3*(j-start)+2]=j/441%2?0:M_PI/2;
        GetRotationMatrix(theta[3*(j-start)+0],theta[3*(j-start)+1],theta[3*(j-start)+2],&RM[0]);
        for(long i=0;i<Ng;i++)
        {
          sargs.v_[0*Ng+i+3*(j-start)*Ng]=inx+inr*(RM[0]*tmpv_[0*Ng+i]+RM[1]*tmpv_[1*Ng+i]+RM[2]*tmpv_[2*Ng+i]);
          sargs.v_[1*Ng+i+3*(j-start)*Ng]=iny+inr*(RM[3]*tmpv_[0*Ng+i]+RM[4]*tmpv_[1*Ng+i]+RM[5]*tmpv_[2*Ng+i]);
          sargs.v_[2*Ng+i+3*(j-start)*Ng]=inz+inr*(RM[6]*tmpv_[0*Ng+i]+RM[7]*tmpv_[1*Ng+i]+RM[8]*tmpv_[2*Ng+i]);
        }
			}
		}

  }
  WriteVTK(sargs.v_,sargs.sh_order,sargs.sh_order,"geometry");
  sargs.stokesdl = new SVel(sargs.sh_order,sargs.sh_order_up,sargs.box_size, 1e-03, comm);
	sargs.unitdl   = new SVel(sargs.sh_order,sargs.sh_order_up,            -1, 1e-03, comm);


  sargs.unitv_ = tmpv_;
  sargs.x0.resize(Mg,sargs.sh_order);
	for(long i=0;i<sargs.v_.Dim();i++) sargs.x0.begin()[i]=sargs.v_[i];
	sargs.S_ = new Sur_t(sargs.sh_order, *sargs.mats, &sargs.x0);

  sargs.x0.resize(1,sargs.sh_order);
	for(long i=0;i<tmpv_.Dim();i++) sargs.x0.begin()[i]=tmpv_[i];
	sargs.unitS_ = new Sur_t(sargs.sh_order, *sargs.mats, &sargs.x0);


  sargs.stokesdl->SetSrcCoord(sargs.v_);
  sargs.unitdl->SetSrcCoord(tmpv_);
  sargs.stokesdl->SetThetaRot(&theta);
  sargs.unitdl->SetThetaRot(&thetaunit);
}

void GetVTK(Vec x, int *ngrid, int testcase, T grid_size, T *v_back, staticargs &sargs,const char *vtkfilename, MPI_Comm comm, PVFMMVec sstok, PVFMMVec srot, PVFMMVec flowcenters)
{
  int np,myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);

  int Mg       = sargs.Mg,
      Ng       = sargs.Ng;
  T gridsft    = 0.001;
  T box_size   = sargs.box_size,
    rsc        =      sargs.rsc,
    gridxi     =    0 + gridsft,
    gridyi     =    0 + gridsft,
    gridzi     =    0 + gridsft,
    gridxf     = grid_size - gridsft,
    gridyf     = grid_size - gridsft,
    gridzf     = grid_size - gridsft,
    griddelx   = (gridxf - gridxi)/(ngrid[0]-1),
    griddely   = (gridyf - gridyi)/(ngrid[1]-1),
    griddelz   = (gridzf - gridzi)/(ngrid[2]-1);

  PVFMMVec f,f1,Df,Df1,Kf,tar,tar1,tar_vel,tar_press,tar_vel_exact;

	PetscErrorCode ierr;
  PetscScalar *x_ptr;
  ierr = VecGetArray(x, &x_ptr);
	f .ReInit(3*Mg*Ng);
	f1.ReInit(3*Mg*Ng);
	for(long i=0;i<3*Mg*Ng;i++)
  {
    f [i] = x_ptr[i];
    f1[i] = 1;
  }

  //rectangular grid
  int ngridx = ngrid[0],
      ngridy = ngrid[1],
      ngridz = ngrid[2];
  if(!myrank){
	  tar          .ReInit(3*ngridx*ngridy*ngridz);
	  tar1         .ReInit(3*ngridx*ngridy*ngridz);
    tar_vel      .ReInit(3*ngridx*ngridy*ngridz);
	  tar_vel_exact.ReInit(3*ngridx*ngridy*ngridz);
	  tar_press    .ReInit(  ngridx*ngridy*ngridz);

    T pt[3];
    for(long i=0;i<ngridx*ngridy*ngridz;i++)
    {
      pt[0] = gridxi+((i/              1)%ngridx)*griddelx;
      pt[1] = gridyi+((i/         ngridx)%ngridy)*griddely;
      pt[2] = gridzi+((i/(ngridx*ngridy))%ngridz)*griddelz;
      tar[3*i+0] = pt[0];
      tar[3*i+1] = pt[1];
      tar[3*i+2] = pt[2];
      if(box_size>0)
      {
        pt[0] = fmod(pt[0],box_size);
        pt[1] = fmod(pt[1],box_size);
        pt[2] = fmod(pt[2],box_size);
        if(pt[0] < 0) pt[0] += box_size;
        if(pt[1] < 0) pt[1] += box_size;
        if(pt[2] < 0) pt[2] += box_size;
        //if(pt[0] > box_size/2.0) pt[0] -= box_size;
        //if(pt[1] > box_size/2.0) pt[1] -= box_size;
        //if(pt[2] > box_size/2.0) pt[2] -= box_size;
      }
      tar1[3*i+0] = pt[0];
      tar1[3*i+1] = pt[1];
      tar1[3*i+2] = pt[2];
      //vel_eval  (sargs,&pt[0],NULL,x_ptr,&tar_vel  [3*i+0],0);
      //press_eval(sargs,&pt[0],NULL,x_ptr,&tar_press[  i+0],0);
      //if(testcase == 1) stokrot_eval(sstok,srot,flowcenters,Mg,&pt[0],&tar_vel_exact[3*i+0]);
    }
  }
  else{
    tar          .ReInit(3);
    //tar[0]=0;tar[1]=0;tar[2]=0;
    //tar1[0]=0;tar1[1]=0;tar1[2]=0;
	  tar1         .ReInit(3);
    tar_vel      .ReInit(3);
	  tar_vel_exact.ReInit(3);
	  tar_press    .ReInit(3);
  }

  //PVFMMVec normal_(3*Mg*Ng);
  //for(long i=0;i<3*Mg*Ng;i++)  normal_[i] = -1*sargs.S_->getNormal().begin()[i];


  if(testcase == 1) stokrot_eval(sargs,sstok,srot,flowcenters,&tar[0],ngridx*ngridy*ngridz,&tar_vel_exact[0]);

  PVFMMVec src_pos(3*Mg),stokes_den(3*Mg),rot_den(3*Mg),trg_pot_stokes, trg_pot_rot;

  if(!myrank)
  {
    trg_pot_stokes.ReInit(3*ngridx*ngridy*ngridz);
    trg_pot_rot   .ReInit(3*ngridx*ngridy*ngridz);
  }
  else
  {
    trg_pot_stokes.ReInit(3);
    trg_pot_rot   .ReInit(3);
  }

  for(long i=0;i<3*Mg;i++)
  {
    src_pos   [i] = sargs.centers[i];
    stokes_den[i] = x_ptr[3*Mg*Ng+i];
    rot_den   [i] = x_ptr[3*Mg*Ng+3*Mg+i];
  }
  MPI_Barrier(comm);
  if(!myrank) std::cout<<"HERE"<<'\n';
  PVFMMEval(&src_pos[0],&stokes_den[0],(T*) NULL,src_pos.Dim()/3,&tar[0],&trg_pot_stokes[0],trg_pot_stokes.Dim()/3,&sargs.sctx,setupsr);
  MPI_Barrier(comm);
  PVFMMEval(&src_pos[0],&rot_den   [0],(T*) NULL,src_pos.Dim()/3,&tar[0],&trg_pot_rot   [0],trg_pot_rot   .Dim()/3,&sargs.rctx,setupsr);

  MPI_Barrier(comm);
  if(!myrank) std::cout<<"HERE"<<'\n';
  if(!myrank){
    for(long i=0;i<3*ngridx*ngridy*ngridz;i++) tar_vel[i] = rsc*trg_pot_stokes[i] + rsc*rsc*trg_pot_rot[i];
  }

  PVFMMVec theta_rot = sargs.stokesdl->GetThetaRot();
	sargs.stokesdl->SetDensityDL(&f);
	sargs.stokesdl->SetTrgCoord(&tar);
  MPI_Barrier(comm);
  if(!myrank) std::cout<<"HERE"<<'\n';
	Df=sargs.stokesdl->operator()();

	//sargs.stokesdl->SetDensityDL(&f1);
	//Df1=sargs.stokesdl->operator()();

  SVel *stokesgeometry = new SVel(sargs.sh_order,sargs.sh_order_up, -1, 1e-03, comm);
  stokesgeometry->SetSrcCoord(sargs.v_);
	stokesgeometry->SetTrgCoord(&tar1);
	stokesgeometry->SetDensityDL(&f1);
  stokesgeometry->SetThetaRot(&theta_rot);
  MPI_Barrier(comm);
  if(!myrank) std::cout<<"HERE"<<'\n';
	Df1=stokesgeometry->operator()();

  MPI_Barrier(comm);
  if(!myrank) std::cout<<"HERE"<<'\n';

  /*
  sargs.stokesdl->SetKernel(sargs.box_size,1000,10,MAX_DEPTH,&pressure_kernel(),sargs.comm);
  sargs.stokesdl->SetSrcCoord(sargs.v_);
	sargs.stokesdl->SetTrgCoord(&tar);
	sargs.stokesdl->SetDensityDL(&f);
	Kf=sargs.stokesdl->operator()();
  */
  if(!myrank) std::cout<<"HERE"<<'\n';
  if(!myrank){
    std::ofstream myoutputfile;
    myoutputfile.open(vtkfilename);

    myoutputfile<<"# vtk DataFile Version 1.0\n3D Structured Grid\nASCII\n\nDATASET RECTILINEAR_GRID\n";
    myoutputfile<<"DIMENSIONS "<<ngridx<<" "<<ngridy<<" "<<ngridz;
    myoutputfile<<"\nX_COORDINATES "<<ngridx<<" double\n";
    for(long i=0;i<ngridx;i++) myoutputfile<<std::setprecision(5)<<gridxi+i*griddelx<<" ";
    myoutputfile<<"\nY_COORDINATES "<<ngridy<<" double\n";
    for(long i=0;i<ngridy;i++) myoutputfile<<std::setprecision(5)<<gridyi+i*griddely<<" ";
    myoutputfile<<"\nZ_COORDINATES "<<ngridz<<" double\n";
    for(long i=0;i<ngridz;i++) myoutputfile<<std::setprecision(5)<<gridzi+i*griddelz<<" ";
    myoutputfile<<"\n\nPOINT_DATA "<<ngridx*ngridy*ngridz<<'\n';
    myoutputfile<<"VECTORS velocity double\n";

    T max_vel_exact=0.0, max_rel_error=0.0;
    PVFMMVec geometry(ngridx*ngridy*ngridz), abs_error(ngridx*ngridy*ngridz);
    for(long i=0;i<ngridx*ngridy*ngridz;i++)
    {
      if(Df1[3*i+0]<0.1 && Df1[3*i+0]>-0.1)
      {
        tar_vel[3*i+0] += Df[3*i+0] + v_back[0];
        tar_vel[3*i+1] += Df[3*i+1] + v_back[1];
        tar_vel[3*i+2] += Df[3*i+2] + v_back[2];

        geometry [i] = 0;

        if(testcase == 1)
        {
          abs_error[i] = sqrt((tar_vel_exact[3*i+0]-tar_vel[3*i+0])*(tar_vel_exact[3*i+0]-tar_vel[3*i+0])+(tar_vel_exact[3*i+1]-tar_vel[3*i+1])*(tar_vel_exact[3*i+1]-tar_vel[3*i+1])
              +(tar_vel_exact[3*i+2]-tar_vel[3*i+2])*(tar_vel_exact[3*i+2]-tar_vel[3*i+2]));

          if(max_rel_error < abs_error[i]) max_rel_error = abs_error[i];

          T tmp=sqrt(tar_vel_exact[3*i+0]*tar_vel_exact[3*i+0]+tar_vel_exact[3*i+1]*tar_vel_exact[3*i+1]+tar_vel_exact[3*i+2]*tar_vel_exact[3*i+2]);
          if(max_vel_exact < tmp) max_vel_exact = tmp;
        }
      //tar_press[1*i+0] += Kf[1*i+0];

      }
      else
      {
        tar_vel[3*i+0] = 0;
        tar_vel[3*i+1] = 0;
        tar_vel[3*i+2] = 0;
        geometry [i] = 1;
        if(testcase == 1) abs_error[i] = 0;
      }

      myoutputfile<<std::scientific<<std::setprecision(15)<<tar_vel[3*i+0]<<" "<<tar_vel[3*i+1]<<" "<<tar_vel[3*i+2]<<'\n';
    }
    if(testcase == 1) max_rel_error /= max_vel_exact;

    myoutputfile<<"\nSCALARS geometry double\nLOOKUP_TABLE default\n";
    for(long i=0;i<ngridx*ngridy*ngridz;i++) myoutputfile<<std::scientific<<std::setprecision(15)<<geometry[i]<<'\n';

    if(testcase == 1)
    {
      myoutputfile<<"\nSCALARS rel_error double\nLOOKUP_TABLE default\n";
      for(long i=0;i<ngridx*ngridy*ngridz;i++) myoutputfile<<std::scientific<<std::setprecision(15)<<abs_error[i]/max_vel_exact<<'\n';
    }

    myoutputfile.close();

    std::cout<<"max rel error "<<max_rel_error<<'\n';
  }

}

void TestPeriodic()
{
  int N = 1;
  T box_size = 1;
  PVFMMVec src_pos(3*N), src_den(3*N), trg_pos(3*N), trg_pot_stokes_periodic(3*N), trg_pot_stokes_free(3*N);

  for(long i=0;i<3*N;i++)
  {
    src_pos[i] = drand48();
    trg_pos[i] = drand48();
    src_den[i] = drand48()-0.5;
  }

  void *periodic_ctx,*freespace_ctx;
  periodic_ctx  = PVFMMCreateContext<T>(box_size);
  freespace_ctx = PVFMMCreateContext<T>();
  PVFMMEval(&src_pos[0],&src_den[0],(T*) NULL,src_pos.Dim()/3,&trg_pos[0],&trg_pot_stokes_periodic[0],trg_pos.Dim()/3,&periodic_ctx,1);

  for(long img=0;img<100;img++)
  {
    long q = 2*img+1;
    PVFMMVec src_pos_free(3*N*pow(q,3)), src_den_free(3*N*pow(q,3));
    src_pos_free.SetZero();
    src_den_free.SetZero();
    for(long ix=-img;ix<=img;ix++)
    {
      for(long iy=-img;iy<=img;iy++)
      {
        for(long iz=-img;iz<=img;iz++)
        {
          for(long i=0;i<N;i++)
          {
            src_pos_free[3*N*(q*q*(ix+img)+q*(iy+img)+(iz+img))+3*i+0] = src_pos[3*i+0]+ix*box_size;
            src_pos_free[3*N*(q*q*(ix+img)+q*(iy+img)+(iz+img))+3*i+1] = src_pos[3*i+1]+iy*box_size;
            src_pos_free[3*N*(q*q*(ix+img)+q*(iy+img)+(iz+img))+3*i+2] = src_pos[3*i+2]+iz*box_size;

            src_den_free[3*N*(q*q*(ix+img)+q*(iy+img)+(iz+img))+3*i+0] = src_den[3*i+0];
            src_den_free[3*N*(q*q*(ix+img)+q*(iy+img)+(iz+img))+3*i+1] = src_den[3*i+1];
            src_den_free[3*N*(q*q*(ix+img)+q*(iy+img)+(iz+img))+3*i+2] = src_den[3*i+2];
          }
        }
      }
    }
    //std::cout<<"src_pos"<<'\n';
    //PrintVec(src_pos);
    //std::cout<<"src_pos_free"<<'\n';
    //PrintVec(src_pos_free);
    PVFMMEval(&src_pos_free[0],&src_den_free[0],(T*) NULL,src_pos_free.Dim()/3,&trg_pos[0],&trg_pot_stokes_free[0],trg_pos.Dim()/3,&freespace_ctx,1);
    RelativeError(trg_pot_stokes_periodic,trg_pot_stokes_free);
    std::cout<<"pot_free"<<'\n';
    PrintVec(trg_pot_stokes_free);
    std::cout<<"pot_periodic"<<'\n';
    PrintVec(trg_pot_stokes_periodic);

  }




/*

  T vel[3]={0,0,0}, box_size=sargs.box_size;

  T cx = src_pos[0];
  T cy = src_pos[1];
  T cz = src_pos[2];
  T rx = trg_pos[0]-cx;
  T ry = trg_pos[1]-cy;
  T rz = trg_pos[2]-cz;
  T ri = sqrt(rx*rx+ry*ry+rz*rz);
  vel[0]+=(1.0/(8*M_PI))*(src_den[0]/ri+(rx*rx*src_den[0]+rx*ry*src_den[1]+rx*rz*src_den[2])/(ri*ri*ri));
  vel[1]+=(1.0/(8*M_PI))*(src_den[1]/ri+(ry*rx*src_den[0]+ry*ry*src_den[1]+ry*rz*src_den[2])/(ri*ri*ri));
  vel[2]+=(1.0/(8*M_PI))*(src_den[2]/ri+(rz*rx*src_den[0]+rz*ry*src_den[1]+rz*rz*src_den[2])/(ri*ri*ri));

  T rel_error = ((vel[0]-trg_pot_stokes[0])*(vel[0]-trg_pot_stokes[0])+(vel[1]-trg_pot_stokes[1])*(vel[1]-trg_pot_stokes[1])+(vel[2]-trg_pot_stokes[2])*(vel[0]-trg_pot_stokes[2]))
    /(trg_pot_stokes[0]*trg_pot_stokes[0]+trg_pot_stokes[1]*trg_pot_stokes[1]+trg_pot_stokes[2]*trg_pot_stokes[2]);
  std::cout<<"At images = 0,Relative Error = "<<rel_error<<'\n';


  for(long stp=1;stp<=100;stp++)
  {
    for(long lx=-stp;lx<=stp;lx+=2*stp)
    {
      for(long ly=-stp;ly<=stp;ly+=2*stp)
      {
        for(long lz=-stp;lz<=stp;lz+=2*stp)
        {
          cx = src_pos[0]+lx*box_size;
          cy = src_pos[1]+ly*box_size;
          cz = src_pos[2]+lz*box_size;
          rx = trg_pos[0]-cx;
          ry = trg_pos[1]-cy;
          rz = trg_pos[2]-cz;
          ri = sqrt(rx*rx+ry*ry+rz*rz);
          vel[0]+=(1.0/(8*M_PI))*(src_den[0]/ri+(rx*rx*src_den[0]+rx*ry*src_den[1]+rx*rz*src_den[2])/(ri*ri*ri));
          vel[1]+=(1.0/(8*M_PI))*(src_den[1]/ri+(ry*rx*src_den[0]+ry*ry*src_den[1]+ry*rz*src_den[2])/(ri*ri*ri));
          vel[2]+=(1.0/(8*M_PI))*(src_den[2]/ri+(rz*rx*src_den[0]+rz*ry*src_den[1]+rz*rz*src_den[2])/(ri*ri*ri));
        }
      }
    }
    rel_error = ((vel[0]-trg_pot_stokes[0])*(vel[0]-trg_pot_stokes[0])+(vel[1]-trg_pot_stokes[1])*(vel[1]-trg_pot_stokes[1])+(vel[2]-trg_pot_stokes[2])*(vel[0]-trg_pot_stokes[2]))
      /(trg_pot_stokes[0]*trg_pot_stokes[0]+trg_pot_stokes[1]*trg_pot_stokes[1]+trg_pot_stokes[2]*trg_pot_stokes[2]);
    std::cout<<"At images = "<<stp<<",Relative Error = "<<rel_error<<'\n';
  }
  */
}

int main(int argc, char **argv)
{

  /* 1: Mg, 2: sh_order, 3: upsampling, 4: tolerance, 5: preconditioning, 6: box_size, 7: grid_size
   *
   *
   */

  MPI_Init(NULL, NULL);
	PetscErrorCode ierr;
	PetscInitialize(&argc,&argv,0,help);
	MPI_Comm comm_self, comm = MPI_COMM_WORLD;

  int np,myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_split(comm, myrank, myrank, &comm_self);

	staticargs sargs;
  sargs.            comm = comm;
  sargs.            M_gl = atoi(argv[1]);
	sargs.        sh_order = atoi(argv[2]);
	sargs.     sh_order_up = atoi(argv[3]);
  sargs.        box_size = atof(argv[6]);
  sargs.params_.sh_order = sargs.sh_order;
	sargs.              Mg = sargs.M_gl/np + (myrank < sargs.M_gl % np ? 1 : 0);
  sargs.              Ng = 2*sargs.sh_order*(sargs.sh_order+1);
  sargs.            sctx = PVFMMCreateContext<T>(sargs.box_size,1000,10,MAX_DEPTH,&StokesKernel<T>::Kernel(),comm);
  sargs.            rctx = PVFMMCreateContext<T>(sargs.box_size,1000,10,MAX_DEPTH,   &rotlet_kernel(),comm);
	sargs.centers.Resize(3*sargs.Mg);
  sargs.v_.Resize(3*sargs.Mg*sargs.Ng);
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

  SetGeometry(shape,M_gl,sargs,comm);
  MPI_Barrier(comm);
  if(!myrank) std::cout<<"Set Geometry"<<'\n';
  T t0,t1;
  t0 = MPI_Wtime();
  //PVFMMVec DLM;
  //DLM.ReInit(0);
  //std::cout<<"size v: "<<sargs.v_.Dim()<<'\n';
  //SphericalHarmonics<T>::StokesSingularInteg(sargs.v_, sargs.sh_order, sargs.sh_order_up, NULL, &DLM);
	sargs.stokesdl->MonitorError();
  t1 = MPI_Wtime();
  if(!myrank) std::cout<<"Time taken by MonitorError = "<<t1-t0<<'\n';
	//sargs.unitdl->MonitorError();


  Mat A;
  // Create Matrix. A
	MatCreateShell(comm,m,n,PETSC_DETERMINE,PETSC_DETERMINE,NULL,&A);
	MatShellSetOperation(A,MATOP_MULT,(void(*)(void))mult);
	MatShellSetContext(A, &sargs);

  Vec f,x,b;
	// Create vectors
	VecCreateMPI(comm,n,PETSC_DETERMINE,&f);
	VecCreateMPI(comm,n,PETSC_DETERMINE,&b);
	VecCreateMPI(comm,n,PETSC_DETERMINE,&x); // Ax=b

	// Create Input Vector. f
	PetscInt f_size;
	ierr = VecGetLocalSize(f, &f_size);

  PetscScalar *f_ptr;
	ierr = VecGetArray(f, &f_ptr);
	for(long i=0;i<3*Mg*Ng+6*Mg;i++) f_ptr[i]=0.0;//drand48();
	f_ptr[Ng+4]=1.0;
	ierr = VecRestoreArray(f, &f_ptr);

  // Create Input Vector.
	PetscInt b_size;
	ierr = VecGetLocalSize(b, &b_size);

  PetscScalar *b_ptr;
   PetscScalar *x_ptr;
	ierr = VecGetArray(b, &b_ptr);

  PVFMMVec sstok(3*Mg), srot(3*Mg), flowcenters(3*Mg);
  T v_back[3];

	if(testcase==1)
  {
    T pt[3],tmpb[3];
		for(long i=0;i<3*Mg;i++)
		{
			sstok[i] = 1;//((T) rand()/(RAND_MAX));
			srot [i] = 1;//((T) rand()/(RAND_MAX));
			flowcenters[i] = sargs.centers[i]; + 0.1;//*((T) rand()/(RAND_MAX));
		}

    PVFMMVec tar(3*Ng*Mg),trg_pot_stokes(3*Ng*Mg), trg_pot_rot(3*Ng*Mg);

    for(long j=0;j<Mg;j++)
		{
			for(long i=0;i<Ng;i++)
			{
        tar[3*Ng*j+3*i+0] = sargs.v_[0*Ng+3*Ng*j+i];
        tar[3*Ng*j+3*i+1] = sargs.v_[1*Ng+3*Ng*j+i];
        tar[3*Ng*j+3*i+2] = sargs.v_[2*Ng+3*Ng*j+i];
      }
    }

    PVFMMEval(&flowcenters[0],&sstok[0],(T*) NULL,Mg,&tar[0],&trg_pot_stokes[0],trg_pot_stokes.Dim()/3,&sargs.sctx,1);
    PVFMMEval(&flowcenters[0], &srot[0],(T*) NULL,Mg,&tar[0],   &trg_pot_rot[0],   trg_pot_rot.Dim()/3,&sargs.rctx,1);

		for(long j=0;j<Mg;j++)
		{
			for(long i=0;i<Ng;i++)
			{
        b_ptr[0*Ng+3*Ng*j+i] = trg_pot_stokes[3*Ng*j+3*i+0] + trg_pot_rot[3*Ng*j+3*i+0];
        b_ptr[1*Ng+3*Ng*j+i] = trg_pot_stokes[3*Ng*j+3*i+1] + trg_pot_rot[3*Ng*j+3*i+1];
        b_ptr[2*Ng+3*Ng*j+i] = trg_pot_stokes[3*Ng*j+3*i+2] + trg_pot_rot[3*Ng*j+3*i+2];
      }
    }

    /*
		for(long j=0;j<Mg;j++)
		{
			for(long i=0;i<Ng;i++)
			{
        pt[0]=sargs.v_[0*Ng+3*Ng*j+i];
        pt[1]=sargs.v_[1*Ng+3*Ng*j+i];
        pt[2]=sargs.v_[2*Ng+3*Ng*j+i];
				stokrot_eval(sstok,srot,flowcenters,Mg,&pt[0],&tmpb[0]);
        b_ptr[0*Ng+3*Ng*j+i]=tmpb[0];
        b_ptr[1*Ng+3*Ng*j+i]=tmpb[1];
        b_ptr[2*Ng+3*Ng*j+i]=tmpb[2];
			}
		}
    */
    v_back[0] = 0; v_back[1] = 0; v_back[2] = 0;
	}
  else if(testcase==2)
  {
		for(long j=0;j<Mg;j++)
		{
			for(long i=0;i<Ng;i++)
			{
				b_ptr[0*Ng+3*Ng*j+i] =  0;
				b_ptr[1*Ng+3*Ng*j+i] =  0;
				b_ptr[2*Ng+3*Ng*j+i] = -1;
			}
		}
    v_back[0] = 0; v_back[1] = 0; v_back[2] = 1;
	}

	for(long i=0;i<6*Mg;i++) b_ptr[3*Mg*Ng+i]=0.0;
  ierr = VecRestoreArray(b, &b_ptr);

	// Create solution vector
	ierr = VecDuplicate(f,&x); CHKERRQ(ierr);
	PetscInt x_size;
	ierr = VecGetLocalSize(x, &x_size);

	// Create linear solver context
    
	KSP ksp; ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);

  if(precondition == 1)
  {
    PC pc;
    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCSHELL);
		PCShellSetContext(pc, &sargs);
    setDiagonalPreconditionerMatrix(&sargs,comm_self);
    PCShellSetApply(pc,diagonalPreconditioner);
  }
  //KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

	// Set operators. Here the matrix that defines the linear system
	// also serves as the preconditioning matrix.
	ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

	// Set runtime options
	KSPSetType(ksp, KSPGMRES);
	KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);

	int tole=atoi(argv[4]);//pow(10,-8);
	KSPSetTolerances(ksp, pow(10,-1*tole), PETSC_DEFAULT,PETSC_DEFAULT, 500);
	KSPGMRESSetRestart(ksp, 500);

	ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

	// -------------------------------------------------------------------
	// Solve the linear system: Ax=b
	// -------------------------------------------------------------------
  // saveMat(A,x,f);
  // linearitycheck(A,x,f);
  // std::cout<<"Matrix saved"<<'\n';

	std::string vtkfile,solfile;
	std::ostringstream ss;
	ss << "proc_" <<np<< "_n_" <<sargs.M_gl << "_sh_" << sargs.sh_order << "_" << sargs.sh_order_up << "_tol_" << tole<< "_precond_" << precondition<<"_gridsize_"<<grid_size<<"_testcase_"<<testcase;
  if(box_size>0) ss << "_periodic";
  else           ss << "_free";
  if(M_gl==2 && shape==1) ss << "_crat_" << crat;

  vtkfile = "./vels_" + ss.str() + ".vtk";
	solfile = "./solution_" + ss.str() + ".txt";
  std::ifstream finsol(solfile.c_str());

  //if(!finsol)
  //{
  t0 = MPI_Wtime();
	ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  t1 = MPI_Wtime();
  if(!myrank) std::cout<<"Time taken by gmres = "<<t1-t0<<'\n';
  // std::cout<<"x = "; print(x);

  savesolution(x,&sargs,ss,comm);
  //}
  /*else
  {
    ierr = VecGetArray(x, &x_ptr);
    T inp;
    int start = myrank*(M_gl/np) + (M_gl%np>myrank?myrank:M_gl%np);
    for(long j=0;j<M_gl;)
    {
      if(j==start)
      {
        for(long i=0;i<3*Ng*Mg+6*Mg;i++) finsol>>x_ptr[i];
        j+=Mg;
      }
      else
      {
        for(long i=0;i<3*Ng+6;i++) finsol>>inp;
        j++;
      }
    }
    finsol.close();
		ierr = VecRestoreArray(x, &x_ptr);
  }*/

  GetVTK(x,ngrid,testcase,grid_size,v_back,sargs,vtkfile.c_str(),comm,sstok,srot,flowcenters);

/*
  T t_fin, del_t;
  int t_steps, n_tracers;
  t_fin = 20;
  del_t = 0.1;
  t_steps = t_fin/del_t + 1;
  n_tracers = 0;
  PVFMMVec pos(3*n_tracers*t_steps);
  pos.SetZero();
  std::cout<<pos.Dim()<<'\n';
  for(long i=0;i<n_tracers;i++)
  {
    pos[3*i+0] = 0.7-0.13*i; pos[3*i+2] = -1.4;
  }
  TimeStepper(t_fin, del_t, n_tracers, sargs, pos, v_back,x_ptr, 0);


	s = "/scratch/romir/vels/tracer_free_euler_" + oss.str() + ".txt";
	myoutputfile.open (s.c_str());

	for(long i=0;i<3*n_tracers*t_steps;i++)
	{
		myoutputfile<<std::scientific<<std::setprecision(15)<<pos[i]<<'\n';
  }
  myoutputfile.close();

*/
    
  //if(M_gl==1 && box_size==-1 && testcase==2 && shape==1) for(T r_tar=1.1;r_tar<2.1;r_tar+=0.1) single_sphere_uniform_flow(sargs,x,v_back,r_tar);

  ierr = VecRestoreArray(x, &x_ptr);

	//View info about the solver
	//KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	// Iterations

	PetscInt its;
	ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Iterations %D\n",its); CHKERRQ(ierr);
	// Free work space.  All PETSc objects should be destroyed when they
	// are no longer needed.
	ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&b);CHKERRQ(ierr);
	ierr = MatDestroy(&A);CHKERRQ(ierr);



	ierr = PetscFinalize();
  MPI_Finalize();
	return 0;
}
