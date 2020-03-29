/* Space time functions describing propagating stimuli
 Such fontions have the form f(X-vt) where:
 X is a 3-dimensional object coord with the 3-dimensional spatial coordinates
 t is the time
 v is the speed
 B. Cessac, I. Ampuero, 12-2019
 */

typedef struct Coord{
    double x;
    double y;
    double z;
} coord;

/** Null stimulus  used for tests***/
double ZeroStim(Coord X,double v,double t,double Amp,double width){
     
    return 0.0;
}

/** Bar moving at constant speed v on the x axis. The bar is a pulse with height Amp and width width. **/
double MovingBarUniform1D(Coord X,double v,double t,double Amp,double width){
    double u=X.x-v*t;
    
    return (((u>-width)&&(u<width)) ? Amp: 0.0);
}

/** Bar stimulus for BMS potential. Space is expressed in delta units. Time is expressed in bins **/
double MovingBarUniform1D(int i,double t,std::vector <double> params){
    double v=params[0];
    double Amp=params[1];
    double width=params[2];
    double delta=params[3];
    double bin=params[4];
    double t0=params[5];//Time when the bar starts
   // Coord X; X.x=i*delta;X.y=0;X.z=0;
    
    double u=i*delta-v*(t-t0)*bin;
    
    return (((u>-width)&&(u<width)) ? Amp: 0.0);
}

#define one_over_sqrt2pi 0.398942280401433

/** Gaussian stimulus for BMS potential. Space is expressed in delta units. Time is expressed in bins **/
double MovingGaussianUniform1D(int i,double t,std::vector <double> params){
    double v=params[0];
    double Amp=params[1];
    double width=params[2];
    double delta=params[3];
    double bin=params[4];
    double t0=params[5];//Time when the bar starts
   // Coord X; X.x=i*delta;X.y=0;X.z=0;
    
    double u=(i*delta-v*(t-t0)*bin)/width;
    
    return Amp*one_over_sqrt2pi*exp(-u*u/2.0)/width;
}
