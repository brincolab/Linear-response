/* Example of linear response in BMS model
   B. Cessac, I. Ampuero, 12-2019
*/
#include <pranasCore/BMSPotential.h>
#include <pranasCore/ParametricPotential.h>
#include <pranasCore/ParametricPotentialMonomial.h>
#include <pranasCore/PlotProbability.h>
#include <pranasCore/PlotRaster.h>
#include <pranasCore/PlotWordsProbabilities.h>
#include <pranasCore/sys.h>
#include <vector>
#include <cstdio>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include "stimuli.cpp"
//#include "utilities.cpp"

using namespace pranas;
double spikeRate(RasterBlock* raster,int i, int interval, int t);

class rate : public RasterBlockObservable {//Observable omega_{i_0}(0)
public:
    
  std::string asString() const {//print the observable form
    std::string str=sys::echo("w_%d(0)",i0);
    return str;
  }
  double phi(const RasterBlock& word) const{//Returns the value of the observable
    return word.getEvent(i0,word.getNumberOfTimeSteps()-1);//If word.getNumberOfTimeSteps()=1 one takes omega(0)
  }
  void reset(int i){//Initialize i0
    // printf("Reseting rate\n");
    i0=i;
    // printf("i0=%d\n",i0);
  }
  int getIndex(){//Return the value of i0
    return i0;
  }
private:
  int i0;
};

class pairwise : public RasterBlockObservable {//Observable \omega_{i_1}(n_1)\omega_{i_2}(n_2)
public:
    
  std::string asString() const {//print the observable form
    std::string str=sys::echo("w_%d(%d)w_%d(%d)",i1,n1,i2,n2);
    return str;
  }
  double phi(const RasterBlock& word) const{//Returns the value of the observable
    return word.getEvent(i1,n1)*word.getEvent(i2,n2);
  }
  void reset(int ii1,int nn1,int ii2,int nn2){//Initialize i1,n1,i2,n2
    // printf("Reseting rate\n");
    i1=ii1;n1=nn1;i2=ii2;n2=nn2;
    // printf("i0=%d\n",i0);
  }
  int getDepth(){//Memory depth of the observable
    return abs(n2-n1)+1;
  }
private:
  int i1,n1,i2,n2;
};

int main(int argc, char *argv[]){

    bool plot=0;
    int N = 30; //Number of neurons
    int  transient=1000;//Transient a time in bin units;
    int  T =2000;//Simulation time in bin units;
    int   OF=0;//Offset time in bin units;
    double ew = 0.2, iw = -2, leak = 0.6, sigma = 0.2,unit_threshold=1;
    //double ew = 0, iw = 0, leak = 0.6, sigma = 0.2,unit_threshold=1;
    double Ie=0.3; //Constant current to control the level of spontaneous activity
    //double Ie=0.45; //Constant current to control the level of spontaneous activity
    double A=0.03125;//atof(argv[0]);
    int M=50000;//atoi(argv[1]);
    int ic=N/2; //Center neuron
    int i1=ic-3,n1=0,i2=ic,n2=2;//Neuron and time indices for the pairwise observable
    rate f0; f0.reset(ic);
    pairwise f1;f1.reset(i1,n1,i2,n2);
    RasterBlockObservable *f;
    int depth=0;
    //printf("A=%lg\tM=%d\n",A,M);   exit(0);
    double epsilon=1E-3;
    int tau_gamma=0;
    
    //Initializations
    sys::echo("Initializing the model\n");
    sys::setSeed(-1);
    
    BMSPotential bms; bms.reset(N, leak, sigma);
    for (int i=0;i<N;i++) {
        bms.setUnitCurrent(i,Ie);
    }
    
    //Set w_i,i+1 = Exitatory || w_i,i+2 Inhibitory
    for (int i=0;i<N;i++){
        for (int j=0; j<N;j++){
            if(abs(i-j)==1) bms.setWeight(i,j,ew);
            if(abs(i-j)==2) bms.setWeight(i,j,iw);
        }
    }
    
    std::vector <double> pow_gamma=bms.power_gamma(&tau_gamma,epsilon);
    int D=(ceil)(-1.0/log(leak));D=10;
    int R=D+1;
    tau_gamma=18;
    char chaine_rate_spont[1000],chaine_plot_spont[1000],chaine_plot_stim[1000],chaine_test[1000],chaine_rate[1000],chaine_cor_FDT[1000],chaine_cor_app[1000],chaine_cor_ex[1000],chaine_lin_rep[1000],chaine_dist[1000];
    char messg1[1000],messg2[1000],chaine_av[1000];

    
    printf("Example of linear response in BMS model where the observable is ");
    
    int choice=1;//0 => rate; 1 => Pairwise
    switch(choice){
        case 0:
            printf("the rate of neuron %d\n",ic);
            sprintf(chaine_dist,"N%d/Error_Rate_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_Ie_%lg_M%d",N,N,ew,iw,leak,sigma,Ie,M);
            break;
        case 1:
            printf(" pairwise (%d,%d), (%d,%d) \n",i1,n1,i2,n2);
            sprintf(chaine_dist,"N%d/Error_Pairwise_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_Ie_%lg_i1_%d_n1_%d_i2_%d_n2_%d_M%d",N,N,ew,iw,leak,sigma,Ie,i1,n1,i2,n2,M);
            break;
        default:
            printf("%d is not an acceptable choice\n",choice);
            exit(0);
    }
    
    FILE* f_dist;//=fopen(chaine_dist,"w");fclose(f_dist);
    
    //A=0.2;
    while (A<=4.0){//A loop
        
        switch(choice){
            case 0:
                sprintf(messg1,"Computing the spontaneous rate of neuron %d ",ic);
                sprintf(messg2,"Computing the rate response ic=%d to the stimulus\n",ic);
                sprintf(chaine_rate,"N%d/RateStim_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_Ie_%lg_A%lg_M%d",N,N,ew,iw,leak,sigma,Ie,A,M);
                sprintf(chaine_cor_FDT,"N%d/Correlation_Pairwise_rate_temp_ic_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_i1_%d_n1_%d_i2_%d_n2_%d_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,i1,n1,i2,n2,M);
                sprintf(chaine_cor_app,"N%d/Correlation_temp_ic_xi_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,M);
               // sprintf(chaine_cor_ex,"N%d/Correlation_temp_ic_ex_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,M);
                //sprintf(chaine_av,"N%d/Averages_ex_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,M);
                sprintf(chaine_lin_rep,"N%d/LinRepRateStim_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,M);
                f=&f0;
                depth=1;
                printf("A=%lg\n",A);
                break;
            case 1:
                sprintf(messg1,"Computing the spontaneous average of the pairwise function f(i1=%d,n1=%d,i2=%d,n2=%d) ",i1,n1,i2,n2);
                sprintf(messg2,"Computing the  pairwise response i1=%d,n1=%d,i2=%d,n2=%d to the stimulus \n",i1,n1,i2,n2);
                sprintf(chaine_rate,"N%d/PairwiseStim_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_Ie_%lg_A%lg_i1_%d_n1_%d_i2_%d_n2_%d_M%d",N,N,ew,iw,leak,sigma,Ie,A,i1,n1,i2,n2,M);
                sprintf(chaine_cor_FDT,"N%d/Correlation_temp_ic_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,M);
                sprintf(chaine_cor_app,"N%d/Correlation_Pairwise_temp_ic_xi_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_i1_%d_n1_%d_i2_%d_n2_%d_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,i1,n1,i2,n2,M);
              //  sprintf(chaine_cor_ex,"N%d/Correlation_Pairwise_temp_ic_ex_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_i1_%d_n1_%d_i2_%d_n2_%d_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,i1,n1,i2,n2,M);
                //sprintf(chaine_av,"N%d/Averages_Pairwise_ex_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_i1_%d_n1_%d_i2_%d_n2_%d_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,i1,n1,i2,n2,M);
                sprintf(chaine_lin_rep,"N%d/LinRepPairwiseStim_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_tg%d_D%d_Ie%lg_A%lg_i1_%d_n1_%d_i2_%d_n2_%d_M%d",N,N,ew,iw,leak,sigma,tau_gamma,D,Ie,A,i1,n1,i2,n2,M);
                depth=f1.getDepth();
                f=&f1;
                printf("A=%lg\tdepth=%d \n",A,depth);
                break;
            default:
                printf("%d is not an acceptable choice\n",choice);
                exit(0);
        }
        
    
        //Spontaneous activity
        sys::echo("Generating a raster in spontaneous activity\n");
        RasterBlock *raster_sp = NULL;
        
        if (plot){
            raster_sp = bms.getRasterBlock(transient,T);
            sprintf(chaine_plot_spont,"N%d/SpontaneousN%dew%lgiw%lggamma%lgsigma%lgIe%lg",N,N,ew,iw,leak,sigma,Ie);
            PlotRaster fig_stim;
            PlotCurves cfig_stim;
            fig_stim.reset(chaine_plot_spont, 0, T+OF, "time", 0, N, "units");
            fig_stim.draw(*raster_sp).show();
            fig_stim.save();
        }
        
        //Computing spontaneous rates
        int Mr=M/T;//M is used for linear response, Mr for spontaneous rates (using ergodic theorem M must be of the same order at Mr*T
        std::vector <double> rates(N,0);
        double SpontPair=0;
        RasterBlock wEmp;
        
        printf("%s \n",messg1);
        
        for (int m=0;m<Mr;m++){//Sample loop
            if (!(m*T%1000)) printf("m=%d\n",m*T);
            raster_sp = bms.getRasterBlock(transient,T);
            for (int t=0;t<T-depth;t++){//Time loop
                wEmp.resetSubSequence(*raster_sp,t,depth);//omega_{0}^{depth}
                if (f->phi(wEmp)==1) {
                   // printf("%d,%d\n",wEmp.getEvent(i1,n1),wEmp.getEvent(i2,n2));
                    SpontPair++;
                }
            }//End of time loop
            //    exit(0);
        }//End of sample loop
        printf("=> done\t");
        
        SpontPair/=(double)(M);
        printf("Value= %lg\n Done \n",SpontPair);
        
        // exit(0);
        
        //printf("tau_gamma=%d, D=%d\n",tau_gamma, D);//exit(0);
        //exit(0);
        
        double mean_field=Ie;
        double thetaL=(unit_threshold-Ie/(1-leak))/(sigma/sqrt(2*(1-leak)));
        /*for (int j=0;j<N;j++){
         mean_field+=bms.getWeight(ic,j)*rates[j];
         }
         mean_field*=(1-pow(leak,D+1))/(1-leak);*/
        // double sigma_g=(1-pow(leak,D+1))/(1-leak)*sigma;
        double gammakl=-2*sqrt(1-leak)*(bms.pi_prime_divby_pi(thetaL)- bms.pi_prime_divby_one_minus_pi(thetaL))/sigma;
        //printf("thetaL=%lg\tMean field=%lg\tsigma=%lg\tPosition in the sigmoid=%lg\tgammakl=%lg\n",thetaL,mean_field,sigma,(unit_threshold-mean_field)/sigma_g,gammakl);
        //  printf("thetaL=%lg\tsigma=%lg\tgammakl=%lg\n",thetaL,sigma,gammakl);exit(0);
        
        //Stimulated activity
        sys::echo("Generating a raster when the stimulus is a moving bar \n");
        double L=30;//Network width (say in millimeters)
        double delta=L/N;//Lattice spacing (say in millimeters)
        double width=delta;//Bar width (say in millimeters)
        double Amp=A*sqrt(2*M_PI); //Bar contrast
        double v=2;//Bar speed (say in mm/s)
        double t0=300; //Time when the bar starts in s
        double dx=delta/10; //Space integration step
        double dt=dx/v; //Time integration step
        double bin=0.01; //Binning length in s
        double window=tau_gamma;//Sliding window width to compute firing rates
        //int samp=100; //Time delay between two computations of the linear response
        
        std::vector <double> params(6);
        params[0]=v;
        params[1]=Amp;
        params[2]=width;
        params[3]=delta;
        params[4]=bin;
        params[5]=t0;
        
        if (plot){//Plot
            sprintf(chaine_test,"N%d/Test_bar_Gaussian_N%d_ew%lg_iw%lg_gamma%lg_sigma_%lg_A%lg",N,N,ew,iw,leak,sigma,A);
            FILE *fp=fopen(chaine_test,"w");
            coord X; X.x=X.y=X.z=0;
            
            /* for (double t=0;t<T*bin;t+=dt){//Time loop in real time
             for (double x=0;x<L;x+=dx){//space loop
             X.x=x;
             fprintf(fp,"%lg\t%lg\t%lg\n",x,t,MovingBarUniform1D(X,v,t,Amp,width));
             //fprintf(fp,"%lg\t%lg\t%lg\n",t,x,MovingBarUniform1D((int)(x/delta),t,params));
             }
             }
             fclose(fp);
             }*/
            
            for (int t=0;t<T-depth;t++){//Time loop in bin time
                //printf("t=%d\n",t);
                for (int i=0;i<N;i++){//Neuron loop
                    fprintf(fp,"%d\t%d\t%lg\n",t,i,MovingGaussianUniform1D(i,t,params));
                }
            }
            fclose(fp);
        }
        
        RasterBlock *raster_stim = bms.getRasterBlock(0,T,MovingGaussianUniform1D,params);
        
        if (plot){
            sprintf(chaine_plot_stim,"N%d/MovingBarN%dew%lgiw%lggamma%lgsigmaIe%lg%lgA%lg",N,N,ew,iw,leak,sigma,Ie,A);
            PlotRaster fig_stim;
            PlotCurves cfig_stim;
            fig_stim.reset(chaine_plot_stim, 0, T+OF, "time", 0, N, "units");
            fig_stim.draw(*raster_stim).show();
            fig_stim.save();
            //    exit(0);
        }
        
        
        printf("=> done\n");
        std::vector <double> Resp(T,0);//Matrix of rates
        
        //Computing the firing rate of neurons by averaging over rasters
        printf("%s",messg2);
        printf("Data will be stored in %s\n",chaine_rate);//exit(0);
        
        for (int m=0;m<M;m++){//Rasters loop
            if (!(m%1000)) printf("\tm=%d\n",m);
            raster_stim = bms.getRasterBlock(0,T,MovingGaussianUniform1D,params);
            for (int t=0;t<T-depth;t++){//Time loop
                wEmp.resetSubSequence(*raster_stim,t,depth);//omega_{t}^{t}
                if (f->phi(wEmp)==1) Resp[t]++;
            }//End of time loop
            delete raster_stim;
        }//End of rasters loop
        
        //Normalisation
        for (int t=0;t<T-depth;t++){//Time loop
            Resp[t]/=M;
        }
        printf("=> done\n");
        
        FILE *fr=fopen(chaine_rate,"w");
        for (int t=0;t<T-depth;t++){//Time loop in bin time
            fprintf(fr,"%d\t%lg\n",t,Resp[t]-SpontPair);
        }
        fclose(fr);
        
        //Computing the linear response from fluctuation-dissipation theorem
        
        printf("Computing the linear response from fluctuation-dissipation theorem\n");
        printf("Correlations will be stored in %s\n",chaine_cor_FDT);//exit(0);
        printf("Linear response will be stored in %s\n",chaine_lin_rep);//exit(0);
        
        //RasterBlock w; w.reset(N,1);
        
        RasterBlockObservable** f2=new RasterBlockObservable*[N];//One computes the correlation of f with omega(n) leading to the first order in the Hammersley-Clifford development
        rate*ff=new rate[N];
        
        for (int k=0;k<N;k++){
            // f2[k]=new rate[1];
            //w.setEvent(k,0,1);
            ff[k].reset(k);
            f2[k]=const_cast<rate*>(&ff[k]);
            //printf("&ff[%d]=%x, phi=%lg, index=%d\n",k,&ff[k],ff[k].phi(w),ff[k].getIndex());
            //printf("&f2[%d]=%x, phi=%lg\n",k,f2[k],f2[k]->phi(w));
        }
        
        Matrix Csp;
        printf("\tComputing the pairwise correlations f f2\n");
        Csp=bms.Csp(M,f,f2,R);
        FILE *fcor=fopen(chaine_cor_FDT,"w");
        for (int r=0;r<=D;r++)
            for (int k=0;k<N;k++)
                fprintf(fcor,"%d\t%d\t%lg\n",k,r,Csp[k][r]);
        fclose(fcor);
        printf("\t=> done\n");
        
        printf("\tComputing the linear response from fluctuation-dissipation theorem\n");
        std::vector <double> dmu_fdt=bms.delta_mu1_app(Csp,pow_gamma,MovingGaussianUniform1D,params,R,T);
        printf("=> done\n");
        
        std::vector <double> dmu_app(T,0),dmu_ex(T,0);
        
        //Computing linear response to first order with a first order expansion of the potential
        
        printf("\nComputing the linear response to first order with a first order expansion of the potential\n");
        
        printf("\tComputing the pairwise correlations f xi\n");
        Csp=bms.Csp_xi(M,f,R);
        // printf("%s\n",chaine_cor); exit(0);
        printf("Data will be stored in %s\n",chaine_cor_app);//exit(0);
        fcor=fopen(chaine_cor_app,"w");
        for (int r=0;r<=D;r++)
            for (int k=0;k<N;k++)
                fprintf(fcor,"%d\t%d\t%lg\n",k,r,Csp[k][r]);
        fclose(fcor);
        printf("\t=> done\n");
        
        //First order approximation
        printf("\tComputing the linear response to first order with a first order expansion of the potential\n");
        
        dmu_app=bms.delta_mu1_app(Csp,pow_gamma,MovingGaussianUniform1D,params,R,T);
        printf("=> done\n");
        
        
        //Computing the theoretical linear response of Cessac & Cofre
        /*  printf("\nComputing the linear response exact to first order\n");
         std::vector <double> musp_fn(T,0);
         std::vector <double> musp_dphir(T,0);
         
         //dmu_ex=bms.delta_mu1_ex(M,Csp,musp_fn,musp_dphir,f,*raster_sp,MovingGaussianUniform1D,params,pow_gamma,R,T);
         printf("=> done\n");
         
         printf("\n\tStoring correlation Csp(f(n),deltaphi(r))\n");
         
         fcor=fopen(chaine_cor_ex,"w");
         for (int n=D;n<T-depth;n++)
         for (int r=n-D;r<=n;r++)
         fprintf(fcor,"%d\t%d\t%lg\n",n,r,Csp[n][r]);
         fclose(fcor);
         printf("\t=> done\n");
         
         printf("\n\tStoring averages\n");
         FILE* f_av=fopen(chaine_av,"w");
         for (int n=0;n<T-depth;n++)
         fprintf(f_av,"%d\t%lg\t%lg\n",n,musp_fn[n],musp_dphir[n]);
         fclose(f_av);
         printf("\t=> done\n");*/
        
        
        printf("Storing Linear responses\n");
        FILE *frep=fopen(chaine_lin_rep,"w");
        
        for (int n=0;n<T-depth;n++){//Time loop in bin time
            fprintf(frep,"%d\t%lg\t%lg\t%lg\n",n,gammakl*dmu_fdt[n],dmu_app[n],dmu_ex[n]);
            
        }
        fclose(frep);
        printf("\t=> done\n");
        
        printf("Computing the distance between experiments and theory\n");
        double dFDT=0,dapp=0,sigmaFDT=0,sigmaapp=0,normeFDT=0,normeapp=0;
        double dFDTn=0,dappn=0;
        
        for (int n=0;n<T-depth;n++){
            normeFDT+=pow(gammakl*dmu_fdt[n],2);
            dFDTn=pow(gammakl*dmu_fdt[n]-Resp[n],2);
            dFDT+=dFDTn;
            sigmaFDT+=pow(dFDTn,2);
            normeapp+=pow(dmu_app[n],2);
            dappn=pow(dmu_app[n]-Resp[n],2);
            dapp+=dappn;
            sigmaapp+=pow(dapp,2);
        }
        normeFDT/=(T-depth);
        normeFDT=sqrt(normeFDT);
        dFDT/=((T-depth));
        sigmaFDT/=(T-depth);
        sigmaFDT-=pow(dFDT,2);
        dFDT=sqrt(dFDT);
        sigmaFDT=sqrt(sqrt(sigmaFDT));
        
        normeapp/=(T-depth);
        normeapp=sqrt(normeapp);
        dapp/=((T-depth));
        sigmaapp/=(T-depth);
        sigmaapp-=pow(dapp,2);
        dapp=sqrt(dapp);
        sigmaapp=sqrt(sqrt(sigmaapp));
      
       // printf("dFDT=%lg +- %lg, dapp=%lg +- %lg\n",dFDT,sigmaFDT,normeFDT,dapp,sigmaapp,normeapp);
        f_dist=fopen(chaine_dist,"a");
        fprintf(f_dist,"%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",A,dFDT,sigmaFDT,normeFDT,dapp,sigmaapp,normeapp);
        fclose(f_dist);
        
        A*=sqrt(2.0);
       // exit(0);
    }//End of A loop
    
    printf("Exiting\n");
  //Exiting
  //  delete  raster_sp,raster_stim;
  return EXIT_SUCCESS;
}


//Fonctions

// Computes the spike rate of a given neuron in the raster of length T using an interval.
double spikeRate(RasterBlock* raster,int i, int interval, int t){
  double sum=0;
  // for (int j = std::max(0,t-interval/2);j<std::min(T,t+interval/2);j++){
  for (int j = t-interval/2;j<t+interval/2;j++){
    sum+=raster->getEvent(i,j);
  }
  sum = sum/interval;
    
  return sum;
}


//Observables whose linear response is computed
/*RasterBlockObservable rate(int k,RasterBlock word){
  RasterBlockObservable f;
 
  return f;
  // return word.getEvent(k, 0);
  }*/

