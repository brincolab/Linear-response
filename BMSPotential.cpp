/** Library for the discrete time Integrate and Fire model (called BMS in https://arxiv.org/abs/0706.0077 https://arxiv.org/pdf/1002.3275.pdf **/

#include "pranasCore/BMSPotential.h"

#include "pranasCore/sys.h"
#include "pranasCore/ParametricPotential.h"
#include "pranasCore/ParametricPotentialMonomial.h"
#include "pranasCore/RasterBlockGrammarComparator.h"
#include "pranasCore/RasterBlockEventGrammar.h"
#include "pranasCore/RasterBlock.h"
#include "pranasCore/PlotWordsProbabilities.h"
#include "pranasCore/math_pranas.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <hdf5.h>
#include <hdf5_hl.h>

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

#define one_over_sqrt2pi 0.398942280401433 //1/sqrt(2 pi)

#define tol_pi 7 //If |x| > tol_pi Taylor expansions of pi and derivative are used.

using namespace std;

namespace pranas {
    
  // Classes
    
  // Members
    
  // Methods
  BMSPotential::BMSPotential() : generatedRaster(NULL) {}
  BMSPotential::~BMSPotential()
  {
    if (generatedRaster)
      delete generatedRaster;
    generatedRaster = NULL;
  }
    
  // Resets the potential for a given network parameters.
  BMSPotential& BMSPotential::reset(unsigned int N, double leak, double sigmaB, double C, double threshold)
  {
    sys::check(0 <= leak && leak < 1, "in BMSPotential::reset the leak term is not in the interval [0,1[ -> %f", leak);
    sys::check(0<= sigmaB, " in BMSPotential::reset the noise standard deviation sigmaB is negative -> %f", sigmaB);
    number_of_units = N;
    int R=100; //Infinite range in pratice. This can be set to small values like 8 or 10
    resetEmpty(N,R);
        
    unit_leak = leak;
    unit_capacitance = C;
    unit_currents.resize(N, 0);
    unit_noise = sigmaB;
    unit_thresholds.resize(N, threshold);
    W.resize(N * N, 0.0);
        
    generatedRaster = NULL;
    potentialV.assign(number_of_units,0);
    for(unsigned int i = 0; i < number_of_units; i++)
      potentialV[i] = 2 * sys::random()-1;
#ifdef _OPENMP
    number_of_threads = omp_get_max_threads();
#else
    number_of_threads = 1;
#endif
    unsigned int r = number_of_units%number_of_threads;
    unsigned int myNeuronItStart,myNeuronItEnd;
    myNeuronItStartVector.clear();
    myNeuronItEndVector.clear();
    for (unsigned int id = 0;id<r;id++){
      myNeuronItStart = id*(floor(number_of_units/number_of_threads)+1);
      myNeuronItEnd = (id+1)*(floor(number_of_units/number_of_threads)+1);
      myNeuronItStartVector.push_back(myNeuronItStart);
      myNeuronItEndVector.push_back(myNeuronItEnd);
    }
        
    for (unsigned int id = r;id<number_of_threads;id++){
      myNeuronItStart = id*(floor(number_of_units/number_of_threads))+r;
      myNeuronItEnd = (id+1)*(floor(number_of_units/number_of_threads))+r;
      myNeuronItStartVector.push_back(myNeuronItStart);
      myNeuronItEndVector.push_back(myNeuronItEnd);
    }
    return *this;
  }
    
  // Gets the number of units.
  unsigned int BMSPotential::getNumberOfUnits() const
  {
    return number_of_units;
  }
    
  // Sets the current of unit i
  void BMSPotential::setUnitCurrent(unsigned int i, double val)
  {
    unit_currents[i] = val;
  }
    
  // Sets the threshold of unit i
  void BMSPotential::setUnitThreshold(size_t i, double val)
  {
    unit_thresholds[i] = val;
  }
  // Sets the threshold of all units
  void BMSPotential::setUnitThresholds(double val)
  {
    unit_thresholds.clear();
    unit_thresholds.resize(number_of_units,val);
  }
    
  // Sets the threshold of unit i
  double BMSPotential::getUnitThreshold(size_t i) const
  {
    return unit_thresholds[i];
  }
    
    
  void BMSPotential::setUnitThresholds(const std::vector<double> thresholds)
  {
    unit_thresholds = thresholds;
  }
    
    
  // Sets the capacitance of units
  void BMSPotential::setCapacitance(double val){unit_capacitance = val;}
    
  // Sets the current of units
  void BMSPotential::setUnitCurrents(const std::vector<double> currents)
  {
    unit_currents = currents;
  }
    
  // Gets the synaptic weight.
  double BMSPotential::getWeight(unsigned int i, unsigned int j) const
  {
    sys::check(i < number_of_units, "in BMSPotential::getWeight, bad post-synaptic neuron index %d not in {0, %d{", i, number_of_units);
    sys::check(j < number_of_units, "in BMSPotential::getWeight, bad pre-synaptic neuron index %d not in {0, %d{", j, number_of_units);
    return W[i * number_of_units + j];
  }
    
  std::vector<double> BMSPotential::getWeights() const
  {
    return W;
  }
    
  // Gets current.
  double BMSPotential::getCurrent(unsigned int i) const
  {
    sys::check(i < number_of_units, "in BMSPotential::getWeight, bad post-synaptic neuron index %d not in {0, %d{", i, number_of_units);
        
    return unit_currents[i];
  }
    
  // Sets the synaptic weight.
  void BMSPotential::setWeight(unsigned int i, unsigned int j, double value)
  {
    sys::check(i < number_of_units, "in BMSPotential::setWeight, bad post-synaptic neuron index %d not in {0, %d{", i, number_of_units);
    sys::check(j < number_of_units, "in BMSPotential::setWeight, bad  pre-synaptic neuron index %d not in {0, %d{", j, number_of_units);
    W[i * number_of_units + j] = value;
  }
    
    
  // Sets currents values.
  void BMSPotential::setCurrents(const std::vector<double>& ie)
  {
    unsigned int size = ie.size();
        
    sys::check(size == number_of_units, "in BMSPotential::setCurrents, bad currents size (%d), not equal to number of units (%d)", size, number_of_units);
        
    unit_currents = ie;
  }
    
  // Read weights values.
  void BMSPotential::readWeights(const string& fileName)
  {
    // Initialization - field
        
    hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        
    sys::check(file_id >= 0, "in BMSPotential::readWeights, unable to open %s",fileName.c_str());
        
    //////////////////////////////////////////////////////////////////////////////////////////////
        
    bool vrExists = H5Lexists(file_id, "/VirtualRetina", 0);
        
    sys::check(vrExists, "in BMSPotential::readWeights, weights not found");
        
    bool connectivityExists = H5Lexists(file_id, "/VirtualRetina/LateralConnectivity", 0);
        
    sys::check(connectivityExists, "in BMSPotential::readWeights, weights not found");
        
    bool weightsExists = H5Lexists(file_id, "/VirtualRetina/LateralConnectivity/Weights", 0);
        
    sys::check(weightsExists, "in BMSPotential::readWeights, weights not found");
        
    /////////////////////////////////////////////////////////////////////////////////////////////
        
      hid_t dataset_id = H5Dopen(file_id, "/VirtualRetina/LateralConnectivity/Weights", H5P_DEFAULT);
        
      hsize_t weightsSize[1] = {1};
      hid_t weightsSpace = H5Dget_space(dataset_id);
      H5Sget_simple_extent_dims(weightsSpace, weightsSize, NULL);
        
      vector<double> weights(weightsSize[0]);
        
      H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, weights.data());
        
      H5Dclose(dataset_id);
        
      /////////////////////////////////////////////////////////////////////////////////////////////
        
        unsigned int size = sqrt(weightsSize[0]);
        unsigned int sizeMin = std::min(number_of_units, size);
        
        unsigned int oi = 0;
        unsigned int index = 0;
        for(unsigned int i = 0; i < sizeMin; i++) {
	  unsigned int j = 0;
	  for(; j < sizeMin; j++, oi++, index++) {
	    W[oi] = weights[index];
	  }
	  for(; j < size; j++) {
	    index++;
	  }
	  for(; j < number_of_units; j++) {
	    oi++;
	  }
        }
        
        // Free
        
        H5Sclose(weightsSpace);
        
        // Close file
        
        H5Fclose(file_id);
  }
    
  // Write weights values.
  void BMSPotential::writeWeights(const string& fileName)
  {
    unsigned int Wsize = number_of_units*number_of_units;
        
    // create file
        
    hid_t dataset_id;
        
    hid_t file_id = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        
    if (file_id < 0)
      return;
        
    //////////////////////////////////////////////////////////////////////////////////////////////
        
    bool virtualRetinaExists = H5Lexists(file_id, "/VirtualRetina", 0);
        
    hid_t virtualRetina_id = 0;
    if (!virtualRetinaExists) {
      virtualRetina_id = H5Gcreate(file_id, "/VirtualRetina", 0, H5P_DEFAULT, H5P_DEFAULT);
    }
        
    //////////////////////////////////////////////////////////////////////////////////////////////
        
    // Create VirtualRetinaConnectivity
        
    bool virtualRetinaConnectivityExists = H5Lexists(file_id, "/VirtualRetina/LateralConnectivity", 0);
        
    hid_t virtualRetinaConnectivity_id = 0;
    if (!virtualRetinaConnectivityExists) {
      virtualRetinaConnectivity_id = H5Gcreate(file_id, "/VirtualRetina/LateralConnectivity", 0, H5P_DEFAULT, H5P_DEFAULT);
    }
        
    //////////////////////////////////////////////////////////////////////////////////////////////
        
    // Weights
        
    hsize_t weightsSize[] = {(hsize_t)Wsize};
    hid_t weightsSpace = H5Screate_simple(1, weightsSize, NULL);
        
    dataset_id = H5Dcreate(file_id, "/VirtualRetina/LateralConnectivity/Weights", H5T_NATIVE_DOUBLE, weightsSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, W.data());
        
    H5Dclose(dataset_id);
        
    //////////////////////////////////////////////////////////////////////////////////////////////
        
    // Free
        
    if (!virtualRetinaExists) {
      H5Gclose(virtualRetina_id);
    }
        
    if (!virtualRetinaConnectivityExists) {
      H5Gclose(virtualRetinaConnectivity_id);
    }
        
    H5Sclose(weightsSpace);
        
    // close file
        
    H5Fclose(file_id);
  }
    
  // Sets weights values.
  void BMSPotential::setWeights(const std::vector<double>& weights)
  {
    unsigned int size = weights.size();
    unsigned int Wsize = number_of_units*number_of_units;
    sys::check(size == Wsize, "in BMSPotential::setWeights, bad weights size (%d), not equal to number of units * number of units (%d)", size, Wsize);
        
    W = weights;
  }
    
    
  // Sets random weights values in a fully connected net.
  void BMSPotential::setWeights(double sigma, double bias)
  {
    double mean = sys::getGaussianBias(bias, sigma);
    for(unsigned int oi = 0; oi < number_of_units * number_of_units; oi++) {
      W[oi] = sys::gaussian(mean, sigma);
    }
  }
    
  // Sets random weights values in a sparse connected net.
  void BMSPotential::setSparseWeights(int K,double sigma, double bias)
  {
    if (bias > 1) bias/=100;
    double mean = sys::getGaussianBias(bias, sigma);
    std::vector<int> vois(K);
    for(unsigned int oi = 0; oi < number_of_units;oi++) {
      // sys::echo("oi=%d",oi);
      int nbvois = 0;
      while (nbvois<K){
	int idx = sys::random(0,number_of_units);
	// sys::echo("idx=%d",idx);
	int j = 0;
	while ((j<nbvois)&&(idx!=vois[j]))   j++;
	if (j == nbvois){
	  vois[nbvois]=idx;
	  //sys::echo("Adding a new neighbour=%d",idx);
	  nbvois++;
	}
      }
      //sys::echo("List of neighbours");
      for (int j=0;j<K;j++){
	W[oi*number_of_units+vois[j]] = sys::gaussian(mean, sigma);
	//sys::echo("%d",vois[j]);
      }
    }
  }
    
  // Initializes the generated raster-block.
  void BMSPotential::resetRasterBlock(unsigned int length)
  {
    if (generatedRaster)
      delete generatedRaster;
    generatedRaster = new RasterBlock();
    generatedRaster->reset(number_of_units, length);
  }
    
  // Generates a raster-plot from this BMS network.
  RasterBlock *BMSPotential::getRasterBlockEvent(unsigned int length)
  {   resetRasterBlock(length);
    for (unsigned int t = 0;t<length;t++)
      generateEvent(t);
    return this->generatedRaster;
  }
    
  // Generates a raster-plot from this BMS network.
  RasterBlock *BMSPotential::getRasterBlock(unsigned int transients,unsigned int length) const
  {
    RasterBlock *raster = new RasterBlock();
    raster->reset(number_of_units, length);
    // Initial potential
    std::vector<double> Vvp(number_of_units);
    std::vector<double> Vv(number_of_units);
    std::vector<double> V(number_of_units);
    for(unsigned int i = 0; i < number_of_units; i++)
      Vv[i] = 2 * sys::random()-1;
        
    // Iterates on time and units: transients
    for(unsigned int t = 0; t < transients; t++) {
      for(unsigned int i = 0; i < number_of_units; i++) {
	Vvp[i] = unit_currents[i] + unit_noise *sys::gaussian();
	for(unsigned int j = 0; j < number_of_units; j++)
	  if (Vv[j] >= unit_thresholds[j])
	    Vvp[i] += getWeight(i,j);
	if(Vv[i] < unit_thresholds[i])
	  Vvp[i] += unit_leak * Vv[i];
      }
      // Pingpong the buffers
      V = Vv;
      Vv = Vvp;
      Vvp = V;
    }
        
    // Iterates on time and units: stores the raster
    for(unsigned int t = 0; t < length; t++) {
      for(unsigned int i = 0; i < number_of_units; i++) {
	Vvp[i] = unit_currents[i] + unit_noise *sys::gaussian();
	for(unsigned int j = 0; j < number_of_units; j++)
	  if (Vv[j] >= unit_thresholds[j])
	    Vvp[i] += getWeight(i,j);
	if(Vv[i] >= unit_thresholds[i])
	  raster->setEvent(i, t, true);
	else
	  Vvp[i] += unit_leak * Vv[i];
      }
      // Pingpong the buffers
      V = Vv;
      Vv = Vvp;
      Vvp = V;
    }
        
    return raster;
  }
    
  // Generates a raster-plot from this BMS network under stimulation with a time dependent stimulus
  RasterBlock *BMSPotential::getRasterBlock(unsigned int transients,unsigned int length,double (*S)(int i,double t,std::vector <double> params),std::vector <double> params) const
  {
    RasterBlock *raster = new RasterBlock();
    raster->reset(number_of_units, length);
    // Initial potential
    std::vector<double> Vvp(number_of_units);
    std::vector<double> Vv(number_of_units);
    std::vector<double> V(number_of_units);
    for(unsigned int i = 0; i < number_of_units; i++)
      Vv[i] = 2 * sys::random()-1;
        
    // Iterates on time and units: transients. Stimulus is not applied during transients
    for(unsigned int t = 0; t < transients; t++) {
      for(unsigned int i = 0; i < number_of_units; i++) {
	Vvp[i] =  unit_currents[i] + unit_noise *sys::gaussian();
	for(unsigned int j = 0; j < number_of_units; j++)
	  if (Vv[j] >= unit_thresholds[j])
	    Vvp[i] += getWeight(i,j);
	if(Vv[i] < unit_thresholds[i])
	  Vvp[i] += unit_leak * Vv[i];
      }
      // Pingpong the buffers
      V = Vv;
      Vv = Vvp;
      Vvp = V;
    }
        
    // Iterates on time and units: stores the raster while the stimulus is applied
    for(unsigned int t = 0; t < length; t++) {
      for(unsigned int i = 0; i < number_of_units; i++) {
	Vvp[i] = unit_currents[i] + S(i,t,params)  + unit_noise *sys::gaussian();
	//printf("S(%d,%d)=%lg\n",i,t,S(i,t,params));
	for(unsigned int j = 0; j < number_of_units; j++)
	  if (Vv[j] >= unit_thresholds[j])
	    Vvp[i] += getWeight(i,j);
	if(Vv[i] >= unit_thresholds[i])
	  raster->setEvent(i, t, true);
	else
	  Vvp[i] += unit_leak * Vv[i];
      }
      // Pingpong the buffers
      V = Vv;
      Vv = Vvp;
      Vvp = V;
    }
        
    return raster;
  }
    
    
  void BMSPotential::generateEvent(unsigned int t)
  {
    std::vector<std::vector<unsigned int> > Z(number_of_threads);
        
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int n= 0; n<number_of_threads;n++){
      for(int  i= myNeuronItStartVector[n]; i<myNeuronItEndVector[n];i++) {
	if (potentialV[i]>=unit_thresholds[i]) {//meaning: should the unit i spike?
	  potentialV[i] = 0;
	  Z[n].push_back(i);
	}
	else potentialV[i] *= unit_leak;
	potentialV[i] += ((unit_currents[i] + unit_noise *sys::gaussian())/unit_capacitance);//all the units receive input current divided by the capacitance and some noise. Note that here the dt is assumed to be 1
      }
    }
        
    for(unsigned int n= 0; n<number_of_threads;n++) {
      for (std::vector<unsigned int>::iterator j = Z[n].begin() ; j != Z[n].end(); ++j){//if the unit j spike, then the potentials of all units i connected to j should be updated
	generatedRaster->setEvent(*j, t, true);
                
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int i = 0; i<number_of_units;i++) potentialV[i] += (getWeight(i,*j)/unit_capacitance);
      }
    }
  }
    
  RasterBlock* BMSPotential::getRasterBlock() const
  {
    return this->generatedRaster;
  }
    
  /** Implementation of the BMS conditional probability of the spike event \f$\omega_k(D)=0,1\f$ given the past \f$\omega_0^{D-1} \f$.
      k neuron index
      omega_k_D the event  \f$\omega_k(D)\f$
      word the block \f$word=\omega_0^{D-1}\f$
  **/
  double BMSPotential::getEventConditionalProbability(unsigned int k, bool omegak,const RasterBlock& word) const
  {
    sys::check(word.getNumberOfUnits() == number_of_units, "in BMSPotential::getEventConditionalProbability uncoherent block size %d != %d the units number", word.getNumberOfUnits(), number_of_units);
    sys::check(word.getNumberOfTimeSteps() > 0, "in BMSPotential::getEventConditionalProbability word range = %d must be >= ", word.getNumberOfTimeSteps());
        
    // sys::echo("in getEventConditionalProbability\nBlock\t"+sys::echo("","w",word.asEventDataString()))+"\tProba=%lg\n", pi(getX(k,word)));
    if (omegak==1) return pi(getX(k,word));
    else return (1-pi(getX(k,word)));
  }
    
    
  /** Implementation of the BMS conditional probability of a spike pattern. If \f$word = \omega(D) \omega_0^{D-1}\f$ then
      the routine returns \f$P\left[\omega(D) \, | \,\omega_0^{D-1} \right]\f$
  **/
  double BMSPotential::getConditionalProbability(const RasterBlock& event,const RasterBlock& condition) const
  {
    sys::check(condition.getNumberOfUnits() == number_of_units, "in BMSPotential::getConditionalProbability uncoherent block size %d != %d the units number", condition.getNumberOfUnits(), number_of_units);
    sys::check(condition.getNumberOfTimeSteps() > 1, "in BMSPotential::getConditionalProbability word range = %d must be >= 2", condition.getNumberOfTimeSteps());
        
    double P=1;
    for (unsigned int k=0;k<number_of_units;k++)
      if (event.getEvent(k,0)) P*=pi(getX(k,condition));
      else P*=(1-pi(getX(k,condition)));
    return P;
  }
    
  /** Implementation of the BMS conditional probability of the spike event \f$\omega_k(D)=0,1\f$ given the past \f$\omega_0^{D-1} \f$.
      @param w =  \f$\omega_0^{D} \f$
      @return the conditionalprobability
  **/
  double BMSPotential::getConditionalProbability(const RasterBlock& w) const
  {
    unsigned int R=w.getNumberOfTimeSteps(),D=R-1;
    sys::check(w.getNumberOfUnits() == number_of_units, "in BMSPotential::getEventConditionalProbability uncoherent block size %d != %d the units number", w.getNumberOfUnits(), number_of_units);
    sys::check(R > 0, "in BMSPotential::getEventConditionalProbability word range = %d must be >= 0", R);
        
    double P=1;
    RasterBlock event;event.resetSubSequence(w,D,1);
    RasterBlock condition;condition.resetSubSequence(w,0,D);
    //sys::echo("event"+event.asEventDataString());
    //sys::echo("condition"+condition.asEventDataString());
        
    for (unsigned int k=0;k<number_of_units;k++)
      if (event.getEvent(k,0)) P*=pi(getX(k,condition));
      else P*=(1-pi(getX(k,condition)));
    //sys::echo("P=%lg",P);
    return P;
  }
    
  /** Implementation of the BMS potential in spontaneous activity or with a constant current Ie.
      If \f$word = \omega(D) \omega_0^{D-1}\f$ then
      the routine returns \f$log P\left[\omega(D) \, | \,\omega_0^{D-1} \right]\f$.
  **/
  double BMSPotential::phi(const RasterBlock& w) const
  {
    unsigned int R=w.getNumberOfTimeSteps(),D=R-1;
    // sys::check(w.getNumberOfUnits() == number_of_units, "in BMSPotential::phi uncoherent block size %d != %d the units number", w.getNumberOfUnits(), number_of_units);
    //sys::check(R > 0, "in BMSPotential::phi word range = %d must be >= 0", R);
        
    RasterBlock event;event.resetSubSequence(w,D,1);
    RasterBlock condition;condition.resetSubSequence(w,0,D);
    /*sys::echo("block"+w.asEventDataString());
      sys::echo("event"+event.asEventDataString());
      sys::echo("condition"+condition.asEventDataString());*/
    double fi=0;
    for (unsigned int k=0;k<number_of_units;k++)
      {
	/*     sys::echo("tau_%d=%d",k,getTau(k,condition));
	       sys::echo("Sigma_%d=%lg",k,getSigma(k,condition));
	       sys::echo("Ie_%d=%lg",k,getIe(k,condition));
	       for (unsigned int j=0;j<number_of_units;j++)  sys::echo("tau_%d=%d,Eta(%d,%d)=%lg",k,getTau(k,condition),j,k,getEta(j,k,condition));
	       sys::echo("X_%d=%lg",k,getX(k,condition));
	       sys::echo("log(pi(X_%d)=%lg",k,lpi(getX(k,condition)));
	       sys::echo("log(1-pi(X_%d)=%lg",k,ilpi(getX(k,condition)));*/
	if (event.getEvent(k,0)) fi+=lpi(getX(k,condition));
	else fi+=ilpi(getX(k,condition));
	//	sys::echo("fi=%lg",fi);
      }
    //  sys::echo("\n");
    return fi;
  }
   
  /** Implementation of the BMS potential in the presence of a stimulus (S).
      If \f$word = \omega(D) \omega_0^{D-1}\f$ then
      the routine returns \f$log P_t\left[\omega(D) \, | \,\omega_0^{D-1} \right]\f$.
  **/
  double BMSPotential::phi(unsigned int t, const RasterBlock& w,double (*S)(int i,double t,std::vector <double> params),std::vector <double> params,std::vector <double> pow_gamma,int tau_gamma) const
  {
    unsigned int R=w.getNumberOfTimeSteps(),D=R-1;
    // sys::check(w.getNumberOfUnits() == number_of_units, "in BMSPotential::phi uncoherent block size %d != %d the units number", w.getNumberOfUnits(), number_of_units);
    //sys::check(R > 0, "in BMSPotential::phi word range = %d must be >= 0", R);
        
    RasterBlock event;event.resetSubSequence(w,D,1);
    RasterBlock condition;condition.resetSubSequence(w,0,D);
    /*sys::echo("block"+w.asEventDataString());
      sys::echo("event"+event.asEventDataString());
      sys::echo("condition"+condition.asEventDataString());*/
    double fi=0;
    for (unsigned int k=0;k<number_of_units;k++)
      {
	/* sys::echo("tau_%d=%d",k,getTau(k,condition));
	   sys::echo("Sigma_%d=%lg",k,getSigma(k,condition));
	   sys::echo("Ie_%d=%lg",k,getIe(k,condition));
	   for (unsigned int j=0;j<number_of_units;j++)  sys::echo("tau_%d=%d,Eta(%d,%d)=%lg",k,getTau(k,condition),j,k,getEta(j,k,condition));
	   sys::echo("X_%d=%lg",k,getX(k,condition));
	   sys::echo("log(pi(X_%d)=%lg",k,lpi(getX(k,condition)));
	   sys::echo("log(1-pi(X_%d)=%lg",k,ilpi(getX(k,condition)));*/
	if (event.getEvent(k,0)) fi+=lpi(getX(k,t,condition,S,params,pow_gamma,tau_gamma));
	else fi+=ilpi(getX(k,t,condition,S,params,pow_gamma,tau_gamma));
	//	sys::echo("fi=%lg",fi);
      }
    // sys::echo("\t\tfi=%lg\n",fi);
    return fi;
  }

  /** Computing the firing rate of N neurons by averaging over M rasters of length T. The average can be time dependent **/
  Matrix BMSPotential::spikeRate(int M, int T, double (*S)(int k,double t,std::vector <double> params),std::vector <double> param){

    int N=number_of_units;
    Matrix r;
    resetMatrix(r,N,T);
        
    for (int m=0;m<M;m++){//Rasters loop
      if (!(m%1000)) printf("\tm=%d\n",m);
      RasterBlock *raster_stim = getRasterBlock(0,T,S,param);
      for (int i=0;i<N;i++){//Neurons loop
	for (int t=0;t<T;t++){//Time loop
	  r[i][t]+=raster_stim->getEvent(i,t);
	}
      }//End of neurons loop
      delete raster_stim;
    }//End of rasters loop
      
    //Normalisation
    for (int i=0;i<N;i++){//Neurons loop
      for (int t=0;t<T;t++){//Time loop
	r[i][t]/=M;
      }
    }//End of neurons loop
        
    return r;
  }

  /** Computing the average of observables f   by averaging over M rasters of length T. The average can be time dependent **/
  Matrix BMSPotential::spikeRate(const RasterBlockObservable **f,int depth,int dim,int M, int T, double (*S)(int k,double t,std::vector <double> params),std::vector <double> param){

      int N=number_of_units;
      Matrix r;
      resetMatrix(r,N,T);
      RasterBlock wEmp;
      
      for (int m=0;m<M;m++){//Rasters loop
          if (!(m%1000)) printf("\tm=%d\n",m);
          RasterBlock *raster_stim = getRasterBlock(0,T,S,param);
          for (int t=0;t<T-depth;t++){//Time loop
              wEmp.resetSubSequence(*raster_stim,t,t+depth);//omega_{0}^{depth}
              for (int i=0;i<dim;i++){//Observable loop
                  r[i][t]+=f[i]->phi(wEmp);
              }//End of observable loop
          }//End of neurons loop
          delete raster_stim;
      }//End of rasters loop
      
      //Normalisation
      for (int i=0;i<dim;i++){//Neurons loop
          for (int t=0;t<T-depth;t++){//Time loop
              r[i][t]/=M;
          }
      }//End of neurons loop
      
      return r;
  }

    
  /** Implementation of the first order term xi in the linear response for BMS in spontaneous activity
      If \f$word = \omega(D) \omega_0^{D-1}\f$ then
      the routine returns the vector of \f$\xi_k \left[\omega(D) \, | \,\omega_0^{D-1} \right] = \frac{H^{(1)}_k(0,\omega)}{\sigma_k(-1,\omega)}\f$ \f$k=1 \dots N $\f, defined in https://hal.inria.fr/hal-01895095/document eq (57) (where it is called \f$\zeta(n,\omega) \f$.
  **/
  std::vector <double> BMSPotential::xi(const RasterBlock& w) const
  {
    unsigned int R=w.getNumberOfTimeSteps(),D=R-1;//Blocks have depth R
    //sys::check(w.getNumberOfUnits() == number_of_units, "in BMSPotential::phi uncoherent block size %d != %d the units number", w.getNumberOfUnits(), number_of_units);
    int N=number_of_units;
    //sys::check(R > 0, "in BMSPotential::phi word range = %d must be >= 0", R);
        
    RasterBlock event;event.resetSubSequence(w,D,1);//omega(D)
    RasterBlock condition;condition.resetSubSequence(w,0,D);//omega_0^{D-1}
    /*sys::echo("block"+w.asEventDataString());
      sys::echo("event"+event.asEventDataString());
      sys::echo("condition"+condition.asEventDataString());*/
    std::vector <double> dfi(N,0);
        
    for (int k=0;k<N;k++){
      /*     sys::echo("tau_%d=%d",k,getTau(k,condition));
             sys::echo("Sigma_%d=%lg",k,getSigma(k,condition));
             sys::echo("Ie_%d=%lg",k,getIe(k,condition));
             for (unsigned int j=0;j<number_of_units;j++)  sys::echo("tau_%d=%d,Eta(%d,%d)=%lg",k,getTau(k,condition),j,k,getEta(j,k,condition));
             sys::echo("X_%d=%lg",k,getX(k,condition));
             sys::echo("log(pi(X_%d)=%lg",k,lpi(getX(k,condition)));
             sys::echo("log(1-pi(X_%d)=%lg",k,ilpi(getX(k,condition)));*/
      if (event.getEvent(k,0)) dfi[k]=pi_prime_divby_pi(getX(k,condition))/getSigma(k,condition);
      else dfi[k]=pi_prime_divby_one_minus_pi(getX(k,condition))/getSigma(k,condition);
      //	sys::echo("dfi=%lg",dfi);
      //  sys::echo("\n");
    }
        
    return dfi;
  }
    
    
  /** Computes the  first order variation of the average of the observable f when submitted to a time dependent stimulus, using the linear response theory  defined in https://arxiv.org/abs/1704.05344 eq. (30). One computes the sum of time correlations between f and delta_phi computed as the difference between phi with stimulus and phi without stimulus. Correlations are computed from the empirical measure of a raster.
      tau_gamma is the time horizon.
      S is the stimulus.
  **/
  std::vector <double> BMSPotential::delta_mu1_ex(int M,Matrix& Csp,std::vector <double> &musp_fn,std::vector <double>& musp_dphir,const RasterBlockObservable *f,const RasterBlock& EmpiricalRaster,double (*S)(int k,double t,std::vector <double> params),std::vector <double> params,std::vector <double> pow_gamma,int R,int T) const{

    int D=R-1;
    //        int N=number_of_units;
    int length=T-R;
    int tau_gamma=pow_gamma.size();
    //Computing the correlation matrix Csp(f(n,.),delta phi(r,.))
    printf("\tComputing the correlation matrix Csp(f(n,.),delta phi(r,.))\n");
        
    resetMatrix(Csp,T,T);//Correlation matrix Csp(f(n,.),delta phi(r,.))
    //std::vector <double> musp_fn(T,0),fn(T,0);
    //std::vector <double> musp_dphir(T,0),dphir(T,0);
    std::vector <double> fn(T,0);
    std::vector <double> dphir(T,0);
    RasterBlock w_n;
      
    //Storing averages
    printf("\tStoring averages\n");
    for (int m=0;m<M;m++){//Rasters loop
      if (!(m%100))printf("\tm=%d\n",m);
      RasterBlock *raster_sp = getRasterBlock(1000,T);//Averages are taken with respect to spontaneous dynamics
      for (int n=D;n<T;n++){//Time n loop
          //printf("\t\tn=%d\n",n);
          w_n.resetSubSequence(*raster_sp,n-D,R);//omega_{n-D}^{n}
          // printf("OK \n");
          fn[n]=f->phi(w_n);//Potential phi of the observable f. Later phi will depend on n. Need to solve the C++ issue first
          double phisp=phi(w_n);//phi BMS potential
          dphir[n]=phi(n,w_n,S,params,pow_gamma,tau_gamma)-phisp;//Potential variation at time n
          musp_fn[n]+=fn[n];
          musp_dphir[n]+=dphir[n];
      }
      for (int n=D;n<T;n++){//Time n loop
          for (int r=D;r<T;r++){//Time  r loop
              Csp[n][r]+=fn[n]*dphir[r];
          }//End of time r loop
      }//End of time  n loop
      delete raster_sp;
    }//End of rasters loop
      
    //Normalisation
    for (int n=D;n<T;n++){//Time n loop
        musp_fn[n]/=(double)M;
        musp_dphir[n]/=(double)M;
    }
      
    for (int n=D;n<T;n++){//Time n loop
      for (int r=D;r<T;r++){//Time  r loop
          Csp[n][r]/=(double)M;
          Csp[n][r]-=musp_fn[n]*musp_dphir[r];
          //printf("Csp_ex[%d][%d]=%lg\n",n,r,Csp[n][r]);
      }//End of time r loop
    }//End of time  n loop
      
    printf("done\n");
      
    printf("Computing the linear response\n");
        
    std::vector <double> dmu(T,0);
    for (int n=D;n<T;n++){//Time n loop
      for (int r=n-D;r<=n;r++){//Time  r loop
          dmu[n]+=Csp[n][r];
      }//End of time r loop
    }//End of time  n loop
        
    return dmu;
  }
       
  
  /** Computes the spontaneous correlation matrix  \f$C(f(r),xi_k(0))\f$, with two indices \f$k=0 \dots N-1\f$ the neuron index, and \f$r=0 \dots D\f$, the time index.
   * @param M The number of samples
   * @param f the observable.
   * @param R the memory depth of xi
   * @param T the time length where the linear response is computed
   * @return the correlation matrix with entries \f$C_{f,xi_k}(r)\f$.
   */
  Matrix BMSPotential::Csp_xi(int M,const RasterBlockObservable* f,int R) const{
      int D=R-1;
      int N=number_of_units;
      unsigned int RasterLength=2*D+2;
      
      //Computing the correlation matrix Csp(f(n,.),delta phi(r,.))
      printf("\tComputing the correlation matrix Csp(f(n,.),xi(r,.))\n");
      
      RasterBlock wf,wEmp;
      Matrix Csp; resetMatrix(Csp,N,R);//Correlation matrix Csp(f(n,.),xi(r,.))=Csp(f(r,.),xi(0,.)) where n=0..D, 0 <= r <= D.
      std::vector <double> musp_fn(R,0); // musp(f(n,.))
      std::vector <double> Xi(N,0);
      std::vector <double> musp_xi(N,0);
      double fnw=0;
      
      for (int m=0;m<M;m++){//Rasters loop
          if (!(m%1000))printf("\tm=%d\n",m);
          RasterBlock* EmpiricalRaster = getRasterBlock(1000,RasterLength);
          wEmp.resetSubSequence(*EmpiricalRaster,0,R);//omega_{0}^{D}
          Xi=xi(wEmp);
          for (int k=0;k<N;k++){//Neuron loop
              musp_xi[k]+=Xi[k];
          }//End of neuron loop
          
          for (int r=0;r<=D;r++){//Time  r loop. r is the time of the last spike pattern in the block.
              wf.resetSubSequence(*EmpiricalRaster,r,R);//omega_{r}_{r+D}
              fnw=f->phi(wf);//Later phi will depend on n. Need to solve the C++ issue first
              if (fnw!=0){//Non vanishing condition
                  //   printf("Sample %d, r=%d, fn=1\n",m,r);
                  musp_fn[r]+=fnw;
                  for (int k=0;k<N;k++){//Neuron loop
                      Csp[k][r]+=fnw*Xi[k];
                  }//End of neuron loop
              }//End of non vanishing condition
          }//End of time r loop
          
          delete EmpiricalRaster;
      }//End of rasters loop
      
      for (int k=0;k<N;k++) musp_xi[k]/=(double)M;
      
      //Normalisation
      for (int r=0;r<=D;r++){//Time  r loop
          musp_fn[r]/=(double)M;
          for (int k=0;k<N;k++){//Neuron loop	  
              Csp[k][r]/=(double)M;
              Csp[k][r]-=musp_fn[r]*musp_xi[k];
              Csp[k][r]= -Csp[k][r];
          }//End of neuron loop
      }//End of time r loop
      
      printf("\tdone\n");
      
      return Csp;
  }

  /** Computes the spontaneous correlation matrix  \f$C(f1(r),f2(0))\f$ between two observables, where f1 is a scalar observable and f2 a vector observable with dimension the number of neurons. Thus the matrix has two indices \f$k=0 \dots N-1\f$ the neuron index, and \f$r=0 \dots D\f$, the time index.
   * @param M The number of samples to compute the correlations
   * @param f1 the first observable.
   * @param f2 the second observable.
   * @param R the memory depth 
   * @return the correlation matrix with entries \f$C_{f,xi_k}(r)\f$.
   */
  Matrix BMSPotential::Csp(int M,const RasterBlockObservable* f1,RasterBlockObservable** f2,int R) const{
      int D=R-1;
      int N=number_of_units;
      unsigned int RasterLength=2*D+2;
      
      //Computing the correlation matrix Csp(f(n,.),delta phi(r,.))
      printf("\tComputing the correlation matrix Csp(f1(r,.),f2(0,.))\n");
      /* RasterBlock w; w.reset(N,1);
       for (int k=0;k<N;k++){
       w.setEvent(k,0,1);
       printf("&f2[%d]=%x, phi=%lg\n",k,f2[k],f2[k]->phi(w));
       }*/
      
      RasterBlock wf,wEmp;
      Matrix Csp; resetMatrix(Csp,N,R);//Correlation matrix Csp(f(n,.),f2(r,.))=Csp(f(r,.),f2(0,.)) where n=0..D, 0 <= r <= D.
      std::vector <double> musp_f1(R,0); // musp(f(n,.))
      std::vector <double> F2(N,0);
      std::vector <double> musp_f2(N,0);
      double f1w=0;
      
      for (int m=0;m<M;m++){//Rasters loop
          if (!(m%1000))printf("\tm=%d\n",m);
          RasterBlock* EmpiricalRaster = getRasterBlock(1000,RasterLength);//1000 steps of transients
          wEmp.resetSubSequence(*EmpiricalRaster,0,R);//omega_{0}^{D}
          for (int k=0;k<N;k++){//Neuron loop
              F2[k]=f2[k]->phi(wEmp);
              musp_f2[k]+=F2[k];
          }//End of neuron loop
          
          for (int r=0;r<=D;r++){//Time  r loop. r is the time of the last spike pattern in the block.
              wf.resetSubSequence(*EmpiricalRaster,r,R);//omega_{r}_{r+D}
              f1w=f1->phi(wf);//Later phi will depend on n. Need to solve the C++ issue first
              if (f1w!=0){
                  musp_f1[r]+=f1w;
                  for (int k=0;k<N;k++){//Neuron loop
                      Csp[k][r]+=f1w*F2[k];
                  }//End of neuron loop
              }//End of if condition
          }//End of time r loop
          delete EmpiricalRaster;
          
      }//End of rasters loop
      
      for (int k=0;k<N;k++) musp_f2[k]/=(double)M;
      
      //Normalisation
      for (int r=0;r<=D;r++){//Time  r loop
          musp_f1[r]/=(double)M;
          for (int k=0;k<N;k++){//Neuron loop
              Csp[k][r]/=(double)M;
              Csp[k][r]-=musp_f1[r]*musp_f2[k];
          }//End of neuron loop
      }//End of time r loop
      
      printf("\tdone\n");
      
      return Csp;
  }
  
    
  /** Computes the approached first order variation of the average of the observable f when submitted to a time dependent stimulus, using linear response theory  defined in https://arxiv.org/abs/1704.05344 (eq (59) adapted to BMS model) . Here a first order expansion of the potential phi is used (eq. (59)) .
      The matrix Csp and the vector pow_gamma are precomputed. Csp might correspond to different types of approximations
  **/
    std::vector <double> BMSPotential::delta_mu1_app(Matrix Csp,std::vector <double> pow_gamma,double (*S)(int k,double t,std::vector <double> params),std::vector <double> params,int R,int T) const{
        printf("\tBMS potential delta_mu1_app \n");
        int N=number_of_units;
        int D=R-1;
        int tau_gamma=pow_gamma.size();
        
        std::vector <double> dmu(T,0);
        for (int n=0;n<T;n++){//Time loop
            if (!(n%100)) printf("\tn=%d\n",n);
            for (int k=0;k<N;k++){//Neuron loop
                for (int m=0;m<=D+1;m++){//Memory loop
                    //  printf("\tk=%d, r=%d\n",k,r);
                    double convol=0;
                    for (int l=0;l<tau_gamma;l++){//Convolution loop \sum_m gamma^m S(k,n-m-l)
                        convol+=pow_gamma[l]*S(k,n-m-l,params);
                    }//End of convolution loop
                    //  printf("\tconvol at k=%d, r=%d =%lg\n",k,r,convol);
                    //printf("\tC[%d][%d]=%lg\n",k,r,Csp[k][r]);
                    dmu[n]+=Csp[k][m]*convol;
                    //   printf("\tdmu[%d]=%lg\n",n,dmu[n]);
                }//End of memory loop
            }//End of neuron loop
            //  printf("\tdmu[%d]=%lg\n",n,dmu[n]);
        }//End of time loop
        printf("\tdone\n");
        
        return dmu;
    }
    
    /** Computes the empirical correlation between an observable f1 and f2 at time r>=0.
     //To be revisited. It works only for observables of range 1
     double BMSPotential::getCorrelation_f1_f2(const RasterBlockObservable *f1,const RasterBlockObservable *f2,const RasterBlock& EmpiricalRaster,int r) const{
     
     double C=0,moyf1=0,moyf2=0;
     double vf1=0,vf2=0;
     int T=EmpiricalRaster.getNumberOfTimeSteps ();
     double length=T-r;
     RasterBlock wf,wEmp;
     
     for (int t=0;t<T-r;t++){
     wf.resetSubSequence(EmpiricalRaster,t,1);//omega(t)
     wEmp.resetSubSequence(EmpiricalRaster,t+r,1);//omega(t+r)
     vf1=f1->phi(wf);
     vf2=f2->phi(wEmp);
     moyf1+=vf1;
     moyf2+=vf2;
     C+=vf1*vf2;
     }
     moyf1/=length;
     moyf2/=length;
     C/=length;
     
     //printf("Csp[%d]=%lg\n",r,C);
     
     return C-moyf1*moyf2;
     }
     **/
    
    
    /** Computes the empirical correlation between an observable  and the vector xi, with entries xi_k, at time r < R.
     std::vector <double> BMSPotential::getCorrelation_f_xi(const RasterBlockObservable *f,const RasterBlock& EmpiricalRaster,int R,int r) const{
     int N=number_of_units;
     std::vector <double> C(N,0),moyxi(N,0),vxi(N,0);
     double vf=0,moyf=0;
     int T=EmpiricalRaster.getNumberOfTimeSteps ();
     double length=T-R;
     RasterBlock wf,wEmp;
     //sys::check(T > R, "in BMSPotential::getCorrelation: Empirical raster length=%d must be larger than the range R=%d",T,R);
     //sys::check(R >= r, "in BMSPotential::getCorrelation: memory index r=%d must be smaller than the range R=%d",r,R);
     for (int t=R;t<T-R;t++){
     wf.resetSubSequence(EmpiricalRaster,t+r,1);//omega(t+r)
     wEmp.resetSubSequence(EmpiricalRaster,t-R,R);//omega_{t-R}^{t}
     vf=f->phi(wf);
     vxi=xi(wEmp);
     moyf+=vf;
     for (int k=0;k<N;k++){
     moyxi[k ]+=vxi[k];
     if (vf!=0)
     C[k]+=vf*vxi[k];
     }
     }
     moyf/=length;
     
     for (int k=0;k<N;k++){
     moyxi[k]/=length;
     C[k]/=length;
     C[k]-=moyf*moyxi[k];
     }
     
     return C;
     }
     **/
    
    
    /** Computes the spontaneous correlation matrix  \f$C(f1(r),f2(0))\f$ between observables f1 and f2. Csp has  one indice \f$r=0 \dots D\f$, the time index where r is constrained between \f$[0,D]\f$.
     
     std::vector <double> BMSPotential::Csp(const RasterBlockObservable *f1,const RasterBlockObservable *f2,const RasterBlock& EmpiricalRaster,int R) const{
     //sys::check("R>=0","in BMSPotential::getEta. The word range %d must be larger than 0",R);
     std::vector <double> C(R);
     
     for (int r=0;r<R;r++)
     C[r]=getCorrelation_f1_f2(f1,f2,EmpiricalRaster,r);
     
     return C;
     }
     **/
    
  //Computes the vector with entries \f$\gamma^m\f$ until the power is lower than epsilon.
  std::vector <double> BMSPotential::power_gamma(int *tau_gamma,double epsilon){
    double puis=1.0;
    std::vector <double> pow_gamma;
    while (puis>epsilon){
      pow_gamma.push_back(puis);
      puis*=unit_leak;
      //   printf("puis=%lg\n",puis);
    }
 
    *tau_gamma=pow_gamma.size();

    return pow_gamma;
  }
    
    
  /** Computes the mean square deviation of the membrane potential in BMS model for the block \f$\omega_0^{D-1} \f$.
      Eq (8) of http://lanl.arxiv.org/pdf/1002.3275.
  **/
  double BMSPotential::getSigma(unsigned int i, const RasterBlock& word) const {
    //  sys::echo("call getSigma");
    int D=word.getNumberOfTimeSteps();
    double sigma=unit_noise*sqrt((1-pow(unit_leak,int(2*(D-word.getTau(i)))))/(1-pow(unit_leak,2)));
    //sys::echo("sigma=%lg",sigma);
    return  sigma;
  }
    
  /** Computes the integrated constant current  in BMS model for the block \f$\omega_0^{D-1} \f$.
      Eq (5) of http://lanl.arxiv.org/pdf/1002.3275.
  **/
  double BMSPotential::getIe(unsigned int i, const RasterBlock& word) const {
    //  sys::echo("call getIe");
    int D=word.getNumberOfTimeSteps();
    double Ie=unit_currents[i]*(1-pow(unit_leak,int(D-word.getTau(i))))/(1-unit_leak);
    //sys::echo("Ie=%lg",Ie);
    return  Ie;
  }
    
    
  /** Computes the integrated current  in BMS model for the block \f$\omega_0^{D-1} \f$.
      Eq (5) of http://lanl.arxiv.org/pdf/1002.3275.
  **/
  double BMSPotential::getIe(unsigned int i, const RasterBlock& word, vector<double> stimulus) const {
    int D=word.getNumberOfTimeSteps();
    double Ie = 0;//TODO:Daniela Put the currect equation
    // double Ie=unit_currents[i]*(1-pow(unit_leak,int(D-word.getTau(i))))/(1-unit_leak);
    return  Ie;
  }
    
  /** Computes the integrated pre-synaptic current for neuron j to neuron i in BMS model  for the block \f$\omega_0^{D-1} \f$.
      Eq (5) of http://lanl.arxiv.org/pdf/1002.3275. (We note here eta instead of X)
  **/
  double BMSPotential::getEta(unsigned int i,unsigned int j, const RasterBlock& word) const {
    //  sys::echo("call getEta");
    double eta=0;
    int D=word.getNumberOfTimeSteps(); sys::check("D>=1","in BMSPotential::getEta. The word range %d must be larger than 1",D);
    unsigned int Dmoins1=D-1;
    for (unsigned int l=word.getTau(i);l<=Dmoins1;l++)
      if (word.getEvent(j,l))
	eta+=pow(unit_leak,int(Dmoins1-l));
    //  sys::echo("eta=%lg",eta);
    return eta;
  }
    
  /**Computes the total, normalized voltage  in BMS model  for the block \f$\omega_0^{D-1}. Here the external current is constant. \f$.
   **/
  double BMSPotential::getX(unsigned int i, const RasterBlock& word) const {
    //  sys::echo("call getX");
    double X=0;
    double sigma=getSigma(i,word);
    //sys::check(sigma>0, "in BMSPotential::getX : the cumulative noise variance = %lg must be >0", sigma);
    for (unsigned int j=0;j<number_of_units;j++) X+=  getWeight(i,j)*getEta(i,j,word);
    double Y=(unit_capacitance*unit_thresholds[i]-X-getIe(i,word))/sigma;
    //sys::echo("X=%lg",X);
    return Y;
  }
    
  /**Computes the total, normalized voltage  in BMS model  for the block \f$\omega_0^{D-1}, for neuron i, at time t. Here the external current (stimulus) is a time dependent function. \f$.
   **/
  double BMSPotential::getX(unsigned int i,unsigned int t, const RasterBlock& word,double (*S)(int i,double t,std::vector <double> params),std::vector <double> params,std::vector <double> pow_gamma,int tau_gamma) const {
    //sys::echo("call getX");
    // sys::check(pow_gamma.size()>0,"Error: the vector pow_gamma must be initialized");
    //sys::check(params.size()>0,"Error: the vector params must be initialized");
        
    double X=0;
    double sigma=getSigma(i,word);
    for (unsigned int j=0;j<number_of_units;j++) X+=  getWeight(i,j)*getEta(i,j,word);
    double sum=0;
    for (int l=0;l<=min(tau_gamma,(int)(t-word.getTau(i)));l++){//Convolution loop
      sum+=pow_gamma[l]*S(i,t-l,params);
    }
    double Y=(unit_capacitance*unit_thresholds[i]-X-getIe(i,word)-sum)/sigma;
    // sys::echo("\t\tY=%lg\n",Y);
    return Y;
  }
    
    
  /**Computes the total, normalized voltage  in BMS model  for the block \f$\omega_0^{D-1} \f$ with a constant current.
   **/
  double BMSPotential::getX(unsigned int i, const RasterBlock& word, vector<double> stimulus) const {
    double X=0;
    double sigma=getSigma(i,word);
    sys::check(sigma>0, "in BMSPotential::getX : the cumulative noise variance = %lg must be >0", sigma);
    for (unsigned int j=0;j<number_of_units;j++) X+=  getWeight(i,j)*getEta(i,j,word);
    double Y=(unit_capacitance*unit_thresholds[i]-X-getIe(i,word,stimulus))/sigma;
    return Y;
  }
    
  // Gaussian integral pi(x)=1/sqrt(2 pi) int_x^{+infty} exp(-x^2/2)dx as defined in section 2.1.4 of http://lanl.arxiv.org/pdf/1002.3275
  // Note: \f$erf(x) = \frac{2}{\sqrt(pi)}  \int_0^x e^{-t^2} dt\f$
  double BMSPotential::pi(double x) const
  {
    /* if (x<-10)
       return (1.000000000+.3989422802/x-3989422802/pow(x,3)+1.196826841/pow(x,5)-5.984134204/pow(x,7)+41.88893943/pow(x,9)-377.0004548/pow(x,11))*exp(-pow(x,2)/2);
       if (x>10)
       return (.3989422802/x-.3989422802/pow(x,3)+1.196826841/pow(x,5)-5.984134204/pow(x,7)+41.88893943/pow(x,9)-377.0004548/pow(x,11))*exp(-pow(x,2)/2);*/
    return 0.5 * (1 - erf(x / sqrt(2.0)));
  }
    
  //Approximation of log pi, taking into account overflows problems.
  double BMSPotential::lpi(double x) const
  {
    if (x<-tol_pi) //Taylor expansion near -\infty
      return exp(-pow(x,2)/2)*(3989422802/x-0.3989422802/pow(x,3)+1.196826841/pow(x,5)-.984134204/pow(x,7) +41.88893942/pow(x,9)-377.0004548/pow(x,11)+4147.005003/pow(x,13));
    if (x>tol_pi) //Taylor expansion near +\infty
      return -0.5*pow(x,2)-0.9189385335-log(x)-1/pow(x,2)+2.5/pow(x,4)-12.33333333/pow(x,6)+88.25/pow(x,8)-816.2000000/pow(x,10)+9200.833335/pow(x,12);
    return log(pi(x));
  }
    
  // Returns log(1 - pi(x))
  double BMSPotential::ilpi(double x) const
  {
    if (x<-tol_pi)
      return -0.5*pow(x,2)-0.9189385335-log(-x)-1/pow(x,2)+2.5/pow(x,4)-12.33333333/pow(x,6)+88.25/pow(x,8)-816.2000000/pow(x,10)+9200.833335/pow(x,12);
    if (x>tol_pi)
      return exp(-pow(x,2)/2)*(-.3989422802/x+.3989422802/pow(x,3)-1.196826841/pow(x,5)+5.984134204/pow(x,7)-41.88893942/pow(x,9)+377.0004548/pow(x,11)-4147.005003/pow(x,13));
    return log(1-pi(x));
  }
    
  /*Returns pi'(x)/pi=-exp(-x^2/2)/ int_x^{-infty} exp(-x^2/2)dx.
    A Taylor expansion is used when |x| is too large.*/
    
  double BMSPotential::pi_prime_divby_pi(double x) const{
    if(x>tol_pi){
      return (-0.9999999992*x-0.9999999992/x+1.999999999/pow(x,3)-9.999999992/pow(x,5)+73.99999996/pow(x,7)-705.9999997/pow(x,9)+8161.999994/pow(x,11)-1.104099999*pow(double(10),int(5))/pow(x,13)+1.708393999*pow(double(10),int(6))/pow(x,15));
    }
    else
      return -one_over_sqrt2pi*exp(-pow(x,2)/2)/pi(x);
  }
    
  /*return pi'(x)/(1-pi)=-exp(-x^2/2)/ (1-int_x^{-infty} exp(-x^2/2)dx).
    A Taylor expansion is used when |x| is too large*/
  double BMSPotential::pi_prime_divby_one_minus_pi(double x) const{
    if(x<=-tol_pi){
      return (x+1/x-2.000000001/pow(x,3)+10.00000001/pow(x,5)-73.99999996/pow(x,7)+706.0000002/pow(x,9)-8162.000006/pow(x,11)+1.104100000*pow(double(10),int(5))/pow(x,13)-1.708394001*pow(double(10),int(6))/pow(x,15));
    }
    else
      return -one_over_sqrt2pi*exp(-pow(x,2)/2)/(1-pi(x));
        
  }
    
  /* Infers the synaptic weights by the learning rule:
     dW/dt=-gradWeights(dKL)=gradPressureWeights -grad(kulbLiebDivergence)=grad(empiricalAverage(Phi))-GradPressure */
  /* grammarSought is generated by the raster we'd like to approximate*/
  /* grammarBMS is the raster that change every time we iterate the method*/
    
  std::vector<double> BMSPotential::getGradientDivergenceWeights(RasterBlockGrammar *grammarSought, RasterBlockGrammar* grammarBMS, unsigned int R) {
    unsigned int N=number_of_units;
    std::vector<double> Gdkl(N*N, 0);
    /*
      for(unsigned i=0;i<N;i++)
      for(unsigned j=0;j<N;j++)
      Gdkl[i*N+j]=0;
    */
    /** Computing gradient of the KL divergence **/
    unsigned int D=R-1;
        
    for(RasterBlockIterator& it = grammarSought->reset(); it.hasNext();)
      {
	const RasterBlock& word = it.getItem();
	RasterBlock event;event.resetSubSequence(word,D,1);
	RasterBlock condition;condition.resetSubSequence(word,0,D);
	double Pjoin=grammarSought->getJoinProbability(word);
            
	for(unsigned i=0;i<N;i++){
	  // std::string str=sys::echo("","w","\ni=%d",i);
	  double ui=0, xi=getX(i,condition);
	  if (event.getEvent(i,0)) {
	    //if (event.getEvent(i,0)==1) {
	    ui=-pi_prime_divby_pi(xi);
	  }
	  else {
	    ui=pi_prime_divby_one_minus_pi(xi);
	  }
	  //str+=sys::echo("","w"," \tpi=%lg",ui);
	  ui=ui*Pjoin*unit_capacitance/getSigma(i,condition);
	  //str+=sys::echo("","w","\tpi P/sigma=%lg",ui);
                
	  for(unsigned j=0;j<N;j++){
	    Gdkl[i*N+j]+=ui*getEta(i,j,condition);
	    //str+=sys::echo("","w","\n \tu_i eta(%d) =%lg",j,ui*getEta(i,j,condition));
	    // str+=sys::echo("","w","\n\tG(%d,%d)=%lg", i,j,Gdkl[i*N+j]);
	  }
	  //sys::echo(str);//sys::echo("\n");
	}
      }
        
    /* Computing the gradient of the pressure. We use /f$ \nabla_W P(\psi) = \mu()\nabla \psi/f$ where \f$\mu\f$ is replaced by the empirical measure of the current BMS raster (grammarBMS).*/
    for(RasterBlockIterator& k = grammarBMS->reset(); k.hasNext();)
      {
	const RasterBlock& wordBMS = k.getItem();
	RasterBlock eventBMS;
	eventBMS.resetSubSequence(wordBMS,D,1);
	RasterBlock conditionBMS;
	conditionBMS.resetSubSequence(wordBMS,0,D);
	double PjoinBMS=grammarBMS->getJoinProbability(wordBMS);
            
	for(unsigned i=0;i<N;i++){
	  //std::string str=sys::echo("","w","\ni=%d",i);
	  double uiBMS=0,xiBMS=getX(i,conditionBMS);
	  if (eventBMS.getEvent(i,0)==1) {
	    uiBMS=-pi_prime_divby_pi(xiBMS);
	  }
	  else {
	    uiBMS=pi_prime_divby_one_minus_pi(xiBMS);
	  }
	  //str+=sys::echo("","w"," \tpi=%lg",ui);
	  uiBMS=uiBMS*PjoinBMS*unit_capacitance/getSigma(i,conditionBMS);
	  //str+=sys::echo("","w","\tpi P/sigma=%lg",ui);
                
                
	  for(unsigned j=0;j<N;j++){
	    Gdkl[i*N+j]-=uiBMS*getEta(i,j,conditionBMS);
	    //str+= sys::echo("","w","\n\tG(%d,%d)=%lg", i,j,Gdkl[i*N+j]);
	  }
                
	}
      }
    //  Normalizig gradient
    double norm=0;
    for(unsigned int i = 0; i < N; i++){
      for(unsigned int j= 0; j < N; j++){
	norm+=pow(Gdkl[i*N+j],int(2));
      }
    }
    norm=sqrt(norm);
    //sys::echo("In gradDiv Weights: norm=%lg",norm);
    /*
    // sys::echo("Gradient");
    if (norm > 1)
    for(unsigned int i = 0; i < N; i++){
    for(unsigned int j= 0; j < N; j++){
    Gdkl[i*N+j]/=norm;
    }
    }
    */
    /*for(unsigned int i = 0; i < N; i++){
      for(unsigned int j= 0; j < N; j++){
      printf(" %5.3lg ", Gdkl[i*N+j]);
      }
      printf("\n");
      }*/
        
        
    return Gdkl;
  }
    
    
  /**  Infers the synaptic weights by the learning rule:
   * dW/dt=-gradWeights(dKL)=gradPressureWeights -grad(kulbLiebDivergence)=grad(empiricalAverage(Phi))-GradPressure considering a time-varying stimulus
   * @param bookSought is generated by the raster we'd like to approximate
   * @param bookBMS is the raster that change every time we iterate the method*/
    
  std::vector<double> BMSPotential::getGradientDivergenceWeights(RasterBlockBook *bookSought, RasterBlockBook* bookBMS, unsigned int R) {
    unsigned int N=number_of_units;
    std::vector<double> Gdkl(N*N, 0);
    /** Computing gradient of the KL divergence **/
    unsigned int D=R-1;
        
    for(RasterBlockIterator& it = bookSought->reset(); it.hasNext();){
      const RasterBlock& word = it.getItem();
      RasterBlock event;event.resetSubSequence(word,D,1);
      RasterBlock condition;condition.resetSubSequence(word,0,D);
      double Pjoin=bookSought->getJoinProbability(word);
      vector <double> stimulus = bookSought->getAverageStimulus (word, N, R);
      for(unsigned i=0;i<N;i++){
	double ui=0, xi=getX(i,condition,stimulus);
	if (event.getEvent(i,0)) ui=-pi_prime_divby_pi(xi);
	else ui=pi_prime_divby_one_minus_pi(xi);
	ui=ui*Pjoin*unit_capacitance/getSigma(i,condition);
                
	for(unsigned j=0;j<N;j++)
	  Gdkl[i*N+j]+=ui*getEta(i,j,condition);
                
      }
    }
        
    /* Computing the gradient of the pressure. We use /f$ \nabla_W P(\psi) = \mu()\nabla \psi/f$ where \f$\mu\f$ is replaced by the empirical measure of the current BMS raster (bookBMS).*/
    for(RasterBlockIterator& k = bookBMS->reset(); k.hasNext();){
      const RasterBlock& wordBMS = k.getItem();
      RasterBlock eventBMS;
      eventBMS.resetSubSequence(wordBMS,D,1);
      RasterBlock conditionBMS;
      conditionBMS.resetSubSequence(wordBMS,0,D);
      double PjoinBMS=bookBMS->getJoinProbability(wordBMS);
      vector <double> stimulus = bookBMS->getAverageStimulus (wordBMS, N, R);
            
      for(unsigned i=0;i<N;i++){
	double uiBMS=0,xiBMS=getX(i,conditionBMS,stimulus);
	if (eventBMS.getEvent(i,0)==1) uiBMS=-pi_prime_divby_pi(xiBMS);
	else uiBMS=pi_prime_divby_one_minus_pi(xiBMS);
	uiBMS=uiBMS*PjoinBMS*unit_capacitance/getSigma(i,conditionBMS);
	for(unsigned j=0;j<N;j++)
	  Gdkl[i*N+j]-=uiBMS*getEta(i,j,conditionBMS);
      }
    }
    return Gdkl;
  }
    
  /* implement dI/dt= grad_I(empiricalAverage(Phi))-Grad_IPressure*/
  std::vector<double> BMSPotential::getGradientDivergenceCurrent(RasterBlockGrammar *grammarSought, RasterBlockGrammar* grammarBMS,unsigned int R) {
        
    unsigned int N=number_of_units;
    std::vector<double> GIdkl(N);
    for(unsigned i=0;i<N;i++) GIdkl[i]=0;
        
    /** Computing gradient **/
    unsigned int D=R-1;
    for(RasterBlockIterator& it = grammarSought->reset(); it.hasNext();)
      {
	const RasterBlock& word = it.getItem();
	//	sys::echo("\n\t"+word.asEventDataString());
            
	RasterBlock event;event.resetSubSequence(word,D,1);
	RasterBlock condition;condition.resetSubSequence(word,0,D);
	double Pjoin=grammarSought->getJoinProbability(word);
            
	for(unsigned i=0;i<N;i++){
	  // std::string str=sys::echo("","w","\ni=%d",i);
	  double ui=0,xi=getX(i,condition);
	  if (event.getEvent(i,0)==1) {
	    ui=-pi_prime_divby_pi(xi);
	  }
	  else {
	    ui=pi_prime_divby_one_minus_pi(xi);
	  }
	  //str+=sys::echo("","w"," \tpi=%lg",ui);
	  ui=ui*Pjoin*((1-pow(unit_leak,int(D-condition.getTau(i))))/((1-unit_leak)*getSigma(i,condition)));
	  GIdkl[i]+=ui;
	  //str+=sys::echo("","w","\n \tu_i eta(%d) =%lg",j,ui*getEta(i,j,condition));
	  //str+=sys::echo("","w","\n\tG(%d,%d)=%lg", i,j,Gdkl[i*N+j]);
	}
      }
        
    /* computing the gradient of the pressure.
       We use /f$ \nabla_W P(\psi) = \mu()\nabla \psi/f$ where \f$\mu\f$ is replaced by the empirical measure of the current BMS raster (grammarBMS). */
    for(RasterBlockIterator& k = grammarBMS->reset(); k.hasNext();)
      {
	const RasterBlock& wordBMS = k.getItem();
            
	RasterBlock eventBMS;eventBMS.resetSubSequence(wordBMS,D,1);
	RasterBlock conditionBMS;conditionBMS.resetSubSequence(wordBMS,0,D);
	double PjoinBMS=grammarBMS->getJoinProbability(wordBMS);
            
	for(unsigned i=0;i<N;i++){
	  //std::string str=sys::echo("","w","\ni=%d",i);
	  double uiBMS=0,xiBMS=getX(i,conditionBMS);
	  if (eventBMS.getEvent(i,0)) {
	    //if (eventBMS.getEvent(i,0)==1) {
	    uiBMS=-pi_prime_divby_pi(xiBMS);
	  }
	  else {
	    uiBMS=pi_prime_divby_one_minus_pi(xiBMS);
	  }
	  uiBMS=uiBMS*PjoinBMS*((1-pow(unit_leak,int(D-conditionBMS.getTau(i))))/((1-unit_leak)*getSigma(i,conditionBMS)));
	  GIdkl[i]-=uiBMS;
	}
      }
        
    /** Normalizing gradient **/
    double norm=0;
    for(unsigned int i = 0; i < N; i++){
      norm+=pow(GIdkl[i],2);
    }
        
    norm=sqrt(norm);
    //sys::echo("In gradDivCurrent: norm=%lg",norm);
        
    //sys::echo("\n Gradient \n");
        
    /*    for(unsigned int i = 0; i < N; i++){
	  if (norm > 1)
	  GIdkl[i]/=norm;
	  printf("  %lg  ", GIdkl[i]);
	  }*/
        
        
    return GIdkl;
  }
    
    
  // implement the Weights based on Wij(t+1)=Wij(t)-d(KulbLiebDivergence)/dWij
  // grammarSought is generated by the rasterd we'd like to approx
  // grammarBMS is the raster that change every time we iterate the method
    
  void BMSPotential::setWeightsDivergence(RasterBlockGrammar* grammar,RasterBlockGrammar* grammarBMS, unsigned int R){
        
    std::vector<double> G = getGradientDivergenceWeights(grammar,grammarBMS,R);
    unsigned int N=number_of_units;
    double epsilon = 1;
    //double epsilon = 0.05;
    for (unsigned int i=0;i<N;i++) {
      for (unsigned int j=0;j<N;j++) {
	setWeight(i,j,W[i*N+j]+epsilon*G[i*N+j]);
      }
    }
  }
    
  /**
   * sets Weights based on Wij(t+1)=Wij(t)-d(KulbLiebDivergence)/dWij
   * @param bookSought is generated by the rasterd we'd like to approx
   * @param bookBMS is the raster that change every time we iterate the method
   **/
  void BMSPotential::setWeightsDivergence(RasterBlockBook* bookSought,RasterBlockBook* bookBMS, unsigned int R){
        
    std::vector<double> G = getGradientDivergenceWeights(bookSought,bookBMS,R);
    unsigned int N=number_of_units;
    double epsilon = 1;
    //double epsilon = 0.05;
    for (unsigned int i=0;i<N;i++) {
      for (unsigned int j=0;j<N;j++) {
	setWeight(i,j,W[i*N+j]+epsilon*G[i*N+j]);
      }
    }
        
  }
    
  /*implement the current based on Ii(t+1)=Ii(t)-d(KulbLiebDivergence)/dIi*/
  void BMSPotential::setCurrentDivergence(RasterBlockGrammar* grammar,RasterBlockGrammar* grammarBMS,unsigned int R){
    double epsilon = 0.05;
    std::vector<double> G = getGradientDivergenceCurrent(grammar,grammarBMS,R);
    unsigned int N = number_of_units;
    for (unsigned int i=0;i<N;i++) unit_currents[i]+=epsilon*G[i];//We take epsilon =0.05
  }
    
  /**
     Estimation of the mean square deviation in the plot  empirical versus theoretical probability
  **/
  double  BMSPotential::delta(double x,int T)
  {
    return sqrt(x*(1-x))/sqrt(T);
  }
    
  /** Gets the KL divergence between the empirical measure \f$\pi \f$ of a Raster and the exact measure \f$\mu\f$ of a finite range R BMS potential approximation. The algorithm uses the equation
      /f$ dKL(\mu,\pi) = P(\phi) - \pi(\phi) - h(\pi)/f$, where \f$P(\phi) \f$ the pressure of \f$\phi \f$ is zero; \f$\pi(\phi) \f$ is the empirical average of \f$\phi \f$ and \f$h(\pi) \f$ is the entropy of \f$\pi \f$ (estimated with Strong et al method).
      @param the Raster
      @return the divergence
  **/
  double BMSPotential::getDivergence(const RasterBlock *Raster,unsigned int R)
  {
    double dist=0;
    RasterBlockGrammar grammar;
    grammar.reset(*Raster, R);
    for(RasterBlockIterator& i = grammar.reset(); i.hasNext();)
      {
	const RasterBlock& word = i.getItem();
	dist-=phi(word)*grammar.getJoinProbability(word);
      }
    return dist-grammar.getStrongEtAlEntropy();
  }
    
  std::vector<BMSPotential::ConditionalProbabilitiesData> BMSPotential::getConditionalProbabilities(RasterBlockGrammar *grammar,unsigned int T)
  {
    std::vector<BMSPotential::ConditionalProbabilitiesData> conditionalProbabilities;
        
    unsigned int D = grammar->getRangeOfBlocks()-1;
        
    for(RasterBlockIterator& i = grammar->reset(); i.hasNext();)
      {
	const RasterBlock& word = i.getItem();
	// RasterBlock event;event.resetSubSequence(word,D,1);
	// RasterBlock conditioning_event;conditioning_event.resetSubSequence(word,0,D);
	//	  sys::echo(event.asEventDataString()+conditioning_event.asEventDataString());
	double pi_join=grammar->getJoinProbability(word), pi_marg=grammar->getMarginalProbability(word);
	double pi_cond=grammar->getConditionalProbability(word),P_cond=getConditionalProbability(word);
	/**
	   Computes an estimate of the error on the conditional empirical probability.
	   delta P_cond/P_cond=sqrt((delta Pjoin/Pjoin)^2+(delta Pmarg/Pmarg)^2)
	**/
	double delta_pi_join = delta(pi_join,T-D-1)/pi_join;//This is a rough estimate of the error since pi_join ought be replaced by P_join, the exact probability, which is not known.
	double delta_pi_marg = delta(pi_marg,T-D)/pi_marg;//This is a rough estimate of the error since pi_marg ought be replaced by P_marg, the exact probability, which is not known.
            
	double err = sqrt(pow(delta_pi_join,2)+pow(delta_pi_marg,2));//Sum of errors
            
	BMSPotential::ConditionalProbabilitiesData data;
            
	data.empiricalJoinProbabilities = pi_join; // the empirical join probabilities of blocks from a BMS raster;
	data.empiricalMarginalProbabilities = pi_marg; // the empirical marginal probabilities of blocks from a BMS raster;
	data.empiricalConditionalProbabilities = pi_cond; // the empirical conditional probabilities of blocks from a BMS raster;
	data.theoreticalConditionalProbabilities = P_cond; // the theoretical conditional probabilities of blocks from a BMS potential;
	data.error = err; // the error on empirical conditional probabilities of blocks from a BMS raster;
            
	conditionalProbabilities.push_back(data);
      }
        
    return conditionalProbabilities;
  }
    
  /** Generates a  file containing:
   *- Each line corresponds to a raster block with
   * - On  the first row, the empirical join probabilities of blocks from a BMS raster;
   * - On  the second row, the empirical marginal probabilities of blocks from a BMS raster;
   * - On  the third row, the empirical conditional probabilities of blocks from a BMS raster;
   * - On  the fourth row, the theoretical conditional probabilities of blocks from a BMS potential;
   * - On  the fifth row, the error on empirical conditional probabilities of blocks from a BMS raster;
   */
  void BMSPotential::saveConditionalProbabilities(RasterBlockGrammar *grammar,unsigned int T,const char* chaine)
  {
    FILE *fp=fopen(chaine,"w");
        
    const std::vector<BMSPotential::ConditionalProbabilitiesData>& conditionalProbabilities = getConditionalProbabilities(grammar, T);
        
    int size = conditionalProbabilities.size();
        
    for (int i = 0; i<size; i++) {
      double pi_join = conditionalProbabilities[i].empiricalJoinProbabilities;
      double pi_marg = conditionalProbabilities[i].empiricalMarginalProbabilities;
      double pi_cond = conditionalProbabilities[i].empiricalConditionalProbabilities;
      double P_cond = conditionalProbabilities[i].theoreticalConditionalProbabilities;
      double err = conditionalProbabilities[i].error;
      /** plot file with
	  -first column: pi_join
	  -second column: pi_marg
	  -third column: pi_cond
	  -fourth column: P_cond
	  -fifth column: err
      **/
      fprintf(fp,"%lg\t%lg\t%lg\t%lg\t%lg\n",pi_join,pi_marg,pi_cond,P_cond,err);
    }
    fclose(fp);
  }
    
  /** Generates a gnuplot file of the conditional probabilities with on abscissa the empirical conditional probabilities of blocks, and, on ordinate, the  theoretical conditional probabilities of blocks. Additionnally, the colours of points is proportional to -log join empirical probability opf blocks. The DataFileName is generated by BMSPotential::SaveConditionalProbabilities
   */
  void BMSPotential::plotConditionalProbabilities(String DataFileName,String GnuplotFileName)
  {
    // Builds and returns the plot command
    std::string s = (std::string)
      "set autoscale xy\n"+
      "set logscale x 10\n"+
      "set logscale y 10\n"+
      "set notitle \n"+
      //"set title \""+ DataFileName +"\"\n"+
      "set xtics 0,10,1 \n"+
      "set ytics 0,0.1,1 \n"+
      "set format x \"10^{\%TL}\" \n"+
      "set format y \"10^{\%TL}\" \n"+ // The "T" is just to put something, if it is taken out the L desapears and the gnuplot does not work. This is an ugly patch
      "set yrange [1E-10:1] \n"+
      "set xlabel \""+ "Observed Probability"+ "\"\n"+
      "set ylabel \""+"Theoretical Probability"+"\"\n"+
      "set cblabel \""+ "Block occurrence empirical probability relative error"+"\"\n"+
      "set colorbox vertical origin screen 0.9, 0.2, 0 size screen 0.05, 0.6, 0 front bdefault \n"+
      "xps = 1\n"+
      //  "plot \""+DataFileName+"\"u 3:4:(-log($2)/log(10))  pt 7 ps xps lc palette tit '', x lc -1 not \n";
      //   "plot \""+DataFileName+"\"u 3:4:(($5)>0 ? log($5)/log(10) :30)   pt 7 ps xps lc palette tit '', x lc -1 not \n";
      "plot \""+DataFileName+"\"u 3:4:5   pt 7 ps xps lc palette tit '', x lc -1 not \n"+
      "set term png enhanced \n "+
      "set output \""+GnuplotFileName + ".png\" \n "+
      "replot";
    sys::echo(GnuplotFileName,"w", s);  
  }
    
  /** Generates a gnuplot file of the conditional probabilities with on abscissa the conditional probabilities of blocks of the BMS potential, and, on ordinate, the  conditional probabilities of blocks for a ParametricPotential. 
   * @param H. The parametric potential.
   * @param String Title. The string appearing in the title of the plot.
   * @param String GnuplotFileName. The name of output gnuplot file.
   */
  void BMSPotential::plotConditionalProbabilities(ParametricPotential *H,String Title,String GnuplotFileName)
  {
    unsigned int N=H->getNumberOfUnits();
    unsigned int R=H->getPotentialRange();
    FILE *fp=fopen(Title.c_str(),"w");
        
    sys::check(N*R<20,"In PlotConditionalProbabilities. Too large NR=%d",N*R);
    for(RasterBlockIterator i(N,R) ; i.hasNext();) {//loop on all possible blocks. Complexity 2^NR
      const RasterBlock& w = i.getItem();
      double Pbms=getConditionalProbability(w);
      double PH=H->getConditionalProbability(w);
      if ((Pbms>1E-10)&&(PH>1E-10))
	fprintf(fp,"%lg\t%lg\n",Pbms,PH);
    }
    fclose(fp);
    // Builds and returns the plot command
    std::string s = (std::string)
      "set autoscale xy\n"+
      "set logscale x 10\n"+
      "set logscale y 10\n"+
      "unset title \n"+
      //"set title \""+ Title +"\"\n"+
      "set xlabel \""+ " Exact conditional Probability"+ "\"\n"
      "set ylabel \""+"Reconstructed conditional Probability"+"\"\n"
      //"set cblabel \""+ "Block occurrence empirical probability relative error"+"\"\n"
      //"set colorbox vertical origin screen 0.9, 0.2, 0 size screen 0.05, 0.6, 0 front bdefault \n"+
      "xps = 1\n"+
      "plot \""+Title.c_str()+"\"u 1:2 not, x lc -1 not \n";
    // "plot \""+Title.c_str()+"\"u 1:2:(-log($2)/log(10))  pt 7 ps xps lc palette tit '', x lc -1 not \n";
    //   "plot \""+DataFileName+"\"u 3:4:(($5)>0 ? log($5)/log(10) :30)   pt 7 ps xps lc palette tit '', x lc -1 not \n";
    //"plot \""+Title+"\"u 3:4:5   pt 7 ps xps lc palette tit '', x lc -1 not \n";
        
    sys::echo(GnuplotFileName,"w", s);  
  }
    
  /** Given a raster, computes an estimation of
      the derivative of the topological pressure with respect to the synaptic weight Wkl **/ 
  //@todo: Utiliser grad_W(P)=\mu(\grad_W phi) \sim \pi(\grad_W phi)
  double BMSPotential::getPressureWeightsGradient(RasterBlock* raster,int k, int l){
        
    double dPkl=0;
    sys::check(0,"getPressureWeightsGradient: To be implemented");
    return dPkl;
  }
    
  // Write graph
  void BMSPotential::writeGraph() const
  {
    unsigned int N = getNumberOfUnits();
    char chaineG[100];
    sprintf(chaineG,"BMS_Graph_N%d_leak%lg_sigmaB%lg",N,unit_leak,unit_noise);
    FILE *fp=fopen(chaineG,"w");
    double nu = 2*M_PI/(double)(N);
    for (unsigned  int i=0;i<N;i++) {
      for (unsigned int j=0;j<N;j++)  {
	double xi = cos(nu*(double)i);
	double yi = sin(nu*(double)i);
	double xj = cos(nu*(double)j);
	double yj = sin(nu*(double)j);
	fprintf(fp,"%lg\t%lg\t%lg\t%lg\t",xj,yj,xi-xj,yi-yj);
	double weight = getWeight(i,j);
	if (weight!=0) {
	  fprintf(fp,"%lg\n",weight);
	  //sys::echo("%d->%d : %lg",j,i,weight);
	}
	else 
	  fprintf(fp,"%lg\n",0.0);
      }
    }
    fclose(fp);
  }
    
  double BMSPotential::estimateUnit_Threshold(double leak, double capacitance, double leak_characteristic_time,double spikeRate,double I,vector<double> W, size_t i){
    spikeRate /= 1000;
    double leak_conductance = capacitance/leak_characteristic_time;
    double meanFieldInputCurrent = I;
    for (size_t j = 0; j<number_of_units; j++) meanFieldInputCurrent +=getWeight(i,j)*spikeRate;
    return (1-pow(leak,1/spikeRate+1))*meanFieldInputCurrent/leak_conductance;
  }
    
    
  /**Computes the normalized voltage without external current in BMS model  for the block \f$\omega_0^{D-1} \f$.
     double BMSPotential::getXsp(unsigned int i, const RasterBlock& word) const {
     //  sys::echo("call getX");
     double X=0;
     double sigma=getSigma(i,word);
     sys::check(sigma>0, "in BMSPotential::getX : the cumulative noise variance = %lg must be >0", sigma);
     for (unsigned int j=0;j<number_of_units;j++) X+=  getWeight(i,j)*getEta(i,j,word);
     double Y=(unit_capacitance*unit_thresholds[i]-X)/sigma;
     return Y;
     }
  **/
    
  /** Computes the integrated current  in BMS model for the block \f$\omega_0^{D-1} \f$. 
      Eq (5) of http://lanl.arxiv.org/pdf/1002.3275.
      @param i the neuron index
      @param word the raster block \f$\omega_0^{D-1} \f$. 
      @param stimulus. The stimulus in the form of a matrix time x neuron index
  **/
  double BMSPotential::getIe(unsigned int i, const RasterBlock& word, vector<vector<double>> stimulus) const {
    int D=word.getNumberOfTimeSteps();
    //printf("STI AT %d %d  IS %f \n",i,D,stimulus[i][D]);
    double Ie=stimulus[D][i]*(1-pow(unit_leak,int(D-word.getTau(i))))/(1-unit_leak);
    return  Ie;
  }
    
  // Generates a raster-plot from this BMS network.
  RasterBlock *BMSPotential::getRasterBlockTCurrent(unsigned int transients,unsigned int length, std::vector<std::vector<double>> tUnitCurrents) const
  {
    RasterBlock *raster = new RasterBlock();
    raster->reset(number_of_units, length+tUnitCurrents.size());
    // Initial potential
    std::vector<double> Vvp(number_of_units);
    std::vector<double> Vv(number_of_units);
    std::vector<double> V(number_of_units);
    for(unsigned int i = 0; i < number_of_units; i++)
      //Vv[i] = 2 * sys::random()-1;
      Vv[i] = 0;
        
    // Iterates on time and units: transients
    for(unsigned int t = 0; t < transients; t++) {
      for(unsigned int i = 0; i < number_of_units; i++) {
	Vvp[i] = unit_currents[i] + unit_noise *sys::gaussian();
	for(unsigned int j = 0; j < number_of_units; j++)
	  if (Vv[j] >= unit_thresholds[j])
	    Vvp[i] += getWeight(i,j);
	if(Vv[i] < unit_thresholds[i])
	  Vvp[i] += unit_leak * Vv[i];
      }
      // Pingpong the buffers
      V = Vv;
      Vv = Vvp;
      Vvp = V;
    }
        
    // Iterates on time and units: stores the raster
    for(unsigned int t = 0; t < length; t++) {
      for(unsigned int i = 0; i < number_of_units; i++) {
	Vvp[i] = unit_currents[i] + unit_noise *sys::gaussian();
	for(unsigned int j = 0; j < number_of_units; j++)
	  if (Vv[j] >= unit_thresholds[j])
	    Vvp[i] += getWeight(i,j);
	if(Vv[i] >= unit_thresholds[i])
	  raster->setEvent(i, t, true);
	else
	  Vvp[i] += unit_leak * Vv[i];
      }
      // Pingpong the buffers
      V = Vv;
      Vv = Vvp;
      Vvp = V;
    }
        
    // Iterates on time and units: stores the raster and uses tUnitCurrents
    for(unsigned int t = 0; t < tUnitCurrents.size(); t++) {
      for(unsigned int i = 0; i < number_of_units; i++) {
	Vvp[i] = tUnitCurrents[t][i] + unit_noise *sys::gaussian();
	if(t<-5){
	  //printf("vv[%d] at %d = %f and %f and %d\n",i,t,Vv[i], unit_thresholds[i],length);
	  //printf("%d %.16f\n",Vv[i]>=unit_thresholds[i],unit_thresholds[i]-Vv[i]);
	}
	for(unsigned int j = 0; j < number_of_units; j++){
	  if (Vv[j] >= unit_thresholds[j]){
	    Vvp[i] += getWeight(i,j);
	  }
	}   
	if(Vv[i] >= unit_thresholds[i]){
	  raster->setEvent(i, length+t, true);
	}
	else Vvp[i] += unit_leak * Vv[i];
      }
      // Pingpong the buffers
      V = Vv;
      Vv = Vvp;
      Vvp = V;
    }
        
    return raster;
  }
    
} // Namespace

