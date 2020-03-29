/** Library for the discrete time Integrate and Fire model (called BMS in https://arxiv.org/abs/0706.0077 https://arxiv.org/pdf/1002.3275.pdf **/

#ifndef pranas_BMSPotential_h
#define pranas_BMSPotential_h

#include "pranasCore/GibbsPotential.h"
#include "pranasCore/RasterBlockBookComparator.h"
#include "pranasCore/bits.h"
#include "pranasCore/math_pranas.h"

#include <list>

namespace pranas {
  typedef std::map<RasterBlock*,double, RasterBlockComparator> WeightedBlock;
    
  class ParametricPotential;
  template<class T> class RasterBlockGrammarComparator;
  class RasterBlockEventGrammar;
  class RasterBlock;
  class PlotWordsProbabilities;
    
  /** Defines a Gibbs BMS network Gibbs potential.
   * - The BMS model dynamics has the form: \f$V_i(t) = \gamma  \, V_i(t-1) \, (1- Z_i(t-1)) + \sum_{j=1}^N W_{ij} \, Z_j(t-1) + I_e + \sigma_B {\cal N}(0, 1)\f$
   * - - Here \f$t\f$ is time, \f$V_i\f$ is the membrane potential of neuron \f$i\f$;
   * - - Neurons have a spiking threshold equal to <tt>1</tt>;
     
   * - - \f$Z_i(t)\f$ is a binary function which is <tt>1</tt> if neuron <tt>i</tt> fires and is <tt>0</tt> otherwise;
   * - - unit_leak, corresponds to \f$\gamma\f$ in Cessac's papers, is a constant leak;
   * - - \f$I_e\f$ is an constant external current;
   * - - \f$W_{ij}\f$ is the synaptic weight from pre-synaptic neuron <tt>j</tt> to post-synaptic neuron <tt>i</tt>,
   * - - \f${\cal N}(0,1)\f$ stands for an additive Gaussian noise, and \f$\sigma_B\f$ is the noise amplitude.
   *
   * - Since dynamics is stochastic one is seeking the conditional probability that a neuron fires at time 0 given the past spiking events of the whole network, \f$Prob\left[\omega_i(0) | \omega(-1), \omega(-2), .. \omega(-R)...\right]\f$. This probability can be exactly computed ad well as the related Gibbs potential (One can indeed show that spike train statistics in BMS model is a Gibbs distribution).
   * - This potential has mathematically an infinite range (i.e. the probability that a neuron fires at time 0 depends on the infinite past) but  Markov approximations exist, with a memory depth  <tt>R</tt>. Then the transition probabiblity is a function of block omega of range <tt>R</tt>. The log of this transition probability is what we compute here.
   * - Ref:
   * - -[soula-beslon-etal:06], H. Soula and G. Beslon and O. Mazet, Spontaneous dynamics of asymmetric random recurrent spiking neural networks,Neural Computation,18,1,(2006);
   * - - [cessac:10] <a href="http://lanl.arxiv.org/abs/1002.3275">B. Cessac, A discrete time neural network model with spiking neurons. II. Dynamics with noise. J.Math. Biol., accepted, 2010</a>
   * - See <a href="BMSPotential_8cpp.html">usage examples</a>.
   **/
  class PRANASCORE_EXPORT BMSPotential : public GibbsPotential {
  public:
    typedef struct {
      double empiricalJoinProbabilities; // the empirical join probabilities of blocks from a BMS raster;
      double empiricalMarginalProbabilities; // the empirical marginal probabilities of blocks from a BMS raster;
      double empiricalConditionalProbabilities; // the empirical conditional probabilities of blocks from a BMS raster;
      double theoreticalConditionalProbabilities; // the theoretical conditional probabilities of blocks from a BMS potential;
      double error; // the error on empirical conditional probabilities of blocks from a BMS raster;
    } ConditionalProbabilitiesData;
        
    // Classes
        
    // Methods
    BMSPotential();
    virtual ~BMSPotential();
        
    /** Resets the potential for a given network parameters.
     * @param N The number of units in the network.
     * @param leak Unit leak in <tt>[0,1[</tt> (also called \f$\gamma\f$).
     * @param sigmaB Additive Gaussian noise standard deviation.
     * @param capacitance Membrane capacitance
     * @param threshold Firing Threshold
     * @return this.
     */
    BMSPotential& reset(unsigned int N, double leak, double sigmaB, double capacitance = 1, double threshold = 1);
        
    /** Gets the number of units.
     * @return The number of units.
     */
    unsigned int getNumberOfUnits() const;
        
    /** Gets the unit threshold.
     * @param i unit
     * @return The unit threshold.
     */
    double getUnitThreshold(size_t i) const;
        
    /** Sets the unit current
     * @param i unit
     * @param val new value
     **/
    void setUnitCurrent(unsigned int i, double val);
        
    /** Sets the unit threshold
     * @param i unit
     * @param val new value
     **/
    void setUnitThreshold(size_t i, double val);
    /** Sets the unit thresholds
     **/
    void setUnitThresholds(const std::vector<double> currents);
        
    /** Sets the unit thresholds
     **/
    void setUnitThresholds(double val);
        
    /** Sets the unit capacitance
     * @param val new value
     **/
    void setCapacitance(double val);
        
        
    /** Sets the unit currents
     **/
    void setUnitCurrents(const std::vector<double> currents);
        
    /** Gets the synaptic weight matrix.
     * @return The vector of doubles correspondig to the synaptic weight matrix.
     */
    std::vector<double> getWeights() const;
        
        
    /** Gets the synaptic weight \f$W_{ij}\f$.
     * @param i Postsynaptic unit index, in <tt>{0, N{</tt>.
     * @param j Presynaptic unit index, in <tt>{0, N{</tt>.
     * @return The weight value.
     */
    double getWeight(unsigned int i, unsigned int j) const;
        
    /** Sets the synaptic weight \f$W_{ij}\f$.
     * @param i Postsynaptic unit index, in <tt>{0, N{</tt>.
     * @param j Presynaptic unit index, in <tt>{0, N{</tt>.
     * @param value The weight value <tt>W_ojd</tt>.
     */
    void setWeight(unsigned int i, unsigned int j, double value);
        
    /** Gets the current \f$I_i\f$.
     * @param i Neuron index
     * @return the current I_i*/
    double getCurrent(unsigned int i) const;
        
    /** Sets currents values.
     * @param ie The array of currents values.
     */
    void setCurrents(const std::vector<double>& ie);
        
    /** Read weights values.
     * @param fileName The file of weights values.
     */
    void readWeights(const std::string& fileName);
        
    /** Write weights values.
     * @param fileName The file of weights values.
     */
    void writeWeights(const std::string& fileName);
        
    /** Sets weights values.
     * @param weights The array of weights values.
     */
    void setWeights(const std::vector<double>& weights);
        
    /** Sets random weights values.
     * - Weights are drawn from a zero mean random Gaussian distribution with a standard-deviation \f$\sigma\f$.
     * @param sigma The  standard-deviation.
     * @param bias Pourcentage of positive weight, between 0 .. 1. Taken arbitrary here  as 0.8.
     */
    void setWeights(double sigma = 0, double bias = 0.8);
        
    /** Sets random weights values. in a sparse net. Each neuron receives K synapses.
     * - Weights are drawn from a zero mean random Gaussian distribution with a standard-deviation \f$\sigma\f$.
     * @param K The number of incoming weights per neuron.
     * @param sigma The  standard-deviation.
     * @param bias Pourcentage of positive weight, between 0 .. 1. Taken arbitrary here  as 0.8.
     */
    void setSparseWeights(int K,double sigma = 0, double bias = 0.8);
        
    /** Gets the conditional probability  \f$Prob(\omega(R) | \omega(R-1), \omega(R-2), .. \omega(1))\f$.
     * @param w The range <tt>R</tt> raster-word.
     * @return The conditional probability \f$Prob(\omega(R) | \omega(R-1), \omega(R-2), .. \omega(1))\f$, and 0 if undefined.
     * @see isRPFWord
     */
    virtual double getConditionalProbability(const RasterBlock& w) const;
        
    /** Generates a raster-plot from this BMS network.
     * - The raster-plot is obtained drawing random initial conditions <tt>V_i(0)</tt> in the <tt>[0, 2]</tt> interval.
     * - This routine is used as follows: <pre>
     * RasterBlock *raster = bmsPotential.getRaster(length);
     * ../..
     * delete raster;</pre>
     * @param transients The number of time iterations.
     * @param length The raster-plot length.
     * @return The generated raster-plot. To be deleted after use.
     */
    RasterBlock *getRasterBlock(unsigned int transients,unsigned int length) const;
        
    /** Generates a raster-plot from this BMS network.
     * - The raster-plot is obtained drawing random initial conditions <tt>V_i(0)</tt> in the <tt>[0, 2]</tt> interval.
     * - This routine is used as follows: <pre>
     * RasterBlock *raster = bmsPotential.getRaster(length);
     * ../..
     * delete raster;</pre>
     * @param transients The number of time iterations.
     * @param length The raster-plot length.
     * @param S The simulus as a function of i, neuron index, t current time and params vector containing the parameters defining the motion (e.g. for a moving bar with uniform translation these parameters are the speed, amplitude and width of the bar).
     *@param params the parameters constraining the stimulus (e.g. amplitude, width and speed of a moving bar)
     * @return The generated raster-plot. To be deleted after use.
     */
    RasterBlock *getRasterBlock(unsigned int transients,unsigned int length,double (*S)(int i,double t,std::vector <double> params),std::vector <double> params) const;
        
    /** Generates a raster-plot from this BMS network.
     * - The raster-plot is obtained drawing random initial conditions <tt>V_i(0)</tt> in the <tt>[0, 2]</tt> interval.
     * - This routine is used as follows: <pre>
     * RasterBlock *raster = bmsPotential.getRaster(length);
     * ../..
     * delete raster;</pre>
     * @param length The raster length.
     * @return The generated raster-plot. To be deleted after use.
     */
    RasterBlock *getRasterBlockEvent(unsigned int length);
        
        
        
    /** Initializes the generated raster-block.
     * @param length The raster-plot length.
     */
    void resetRasterBlock(unsigned int length);
        
    /*Generate spikes (events) from this BMS network at this current time stamp
     * in the end updates the time stamp, the currents and the "fired" label
     * @param t The time to generate event.
     */
    void generateEvent(unsigned int t);
        
    /** Gets a the generated raster-block from this BMS network.
     * @return The generated raster-block.
     */
    RasterBlock *getRasterBlock() const;
        
    /** Implementation of the BMS conditional probability of a spike event.
     * Notice: Redundant with getPotential().
     * If \f$word \equiv \omega_{0}^D \f$ this routine computes \f$P\left[\omega_k(D) \,|\, \omega_{0}^{D-1} \right]\f$ in BMS model.
     * @param k The neuron index.
     * @param omegak The event omegak\f$\omega_k(D) \in (0,1) \f$.
     * @param word The RasterBlock \f$\omega_{0}^{D-1}\f$ .
     * @return The conditional probability
     */
    double getEventConditionalProbability(unsigned int k, bool omegak,const RasterBlock& word) const;
        
    /** Implementation of the BMS conditional probability of event=\f$\omega(D)\f$ given the block condition \f$\omega_0^{D-1} \f$, \f$P\left[event \,|\, condition \right]\f$ in BMS model Eq (22) of cessac:11.
     * @param event The event=\f$\omega(D)\f$.
     * @param condition The block condition \f$\omega_0^{D-1} \f$.
     * @return The conditional probability
     */
    double getConditionalProbability(const RasterBlock& event,const RasterBlock& condition) const;//tested in BMSPotential.cpp
        
    /** Implementation of the BMS potential  in spontaneous activity or with a constant current Ie,  for the block \f$\omega_0^{D-1} \f$. Eq (32) of cessac:10.
     *  If \f$w \equiv \omega_{0}^R \f$ this routine computes \f$log P\left[\omega(R) \,|\, \omega_{0}^{R-1} \right]\f$ in BMS model.
     * @param word The RasterBlock.
     * @return The potential
     */
    virtual double phi(const RasterBlock& word) const;
  
        
    /** Implementation of the BMS potential in the presence of a stimulus (S).
	If \f$word = \omega(D) \omega_0^{D-1}\f$ then
	the routine returns \f$log P_t\left[\omega(D) \, | \,\omega_0^{D-1} \right]\f$.
	* @param t, the time index
	* @param word The RasterBlock (of range D).
	* @param S the time dependent stimulus.
	* @param params the parameters constraining the stimulus (e.g. amplitude, width and speed of a moving bar)
	* @param pow_gamma auxilary vector containing powers of gamma, the leak rate.
	* @param tau_gamma The time of summation in the discrete convolution (typically tau_gamma=1/abs(log(gamma))
	* @return The potential
	**/
    double phi(unsigned int t, const RasterBlock& word,double (*S)(int i,double t,std::vector <double> params),std::vector <double> params,std::vector <double> pow_gamma,int tau_gamma) const;
    
    /** Computing the firing rate of N neurons by averaging over M spontaneous rasters of length T
     * @param M The number of samples
     * @param T The time length of the raster
     * @param S the time dependent stimulus.
     * @param params the parameters constraining the stimulus (e.g. amplitude, width and speed of a moving bar)
     * @return The matrix of rate r[i][t] where i is the neuron index and t the time
     **/
    Matrix spikeRate(int M, int T, double (*S)(int k,double t,std::vector <double> params),std::vector <double> param);

    /** Computing the average of observables f   by averaging over M rasters of length T. The average can be time dependent
     * @param f the vector of observables
     * @param depth, the depth of the observable: for an observable \f$\prod_{k=1}^r \omega_{i_k}(n_k) \f$ the depth is the maximal diffference between the times \f$ n_{i_k}\f$
     * @param dim the dimension of the vector f
     * @param M The number of samples
     * @param T The time length of the raster
     * @param S the time dependent stimulus.
     * @param params the parameters constraining the stimulus (e.g. amplitude, width and speed of a moving bar)
     * @return The matrix of rate r[i][t] where i is the observable index and t the time 
**/
    Matrix spikeRate(const RasterBlockObservable **f,int depth,int dim,int M, int T, double (*S)(int k,double t,std::vector <double> params),std::vector <double> param);
    
        
    /** Implementation of the first order term xi in the linear response for BMS.
	If \f$word = \omega(D) \omega_0^{D-1}\f$ then
	the routine returns \f$\xi_k \left[\omega(D) \, | \,\omega_0^{D-1} \right] = \frac{H^{(1)}_k(0,\omega)}{\sigma_k(-1,\omega)}\f$ defined in https://hal.inria.fr/hal-01895095/document eq (57).
	* @param word The RasterBlock.
	* @return xi
	*/
    std::vector <double>  xi(const RasterBlock& word) const;
        
    /** Computes the exact first order variation of the average of the observable f when submitted to a time dependent stimulus, using the linear response theory  defined in https://arxiv.org/abs/1704.05344 eq. (30). One computes the sum of time correlations between f and delta_phi computed as the difference between phi with stimulus and phi without stimulus. The averages with respect to the spontaneous measure musp are obtained by averaging over M spontaneous rasters of length T.
     * @param M The number of samples
     * @param Csp_ex the correlation matrix Csp(f(n),deltaphi(r)) computed to obtain the linear response.
     * @param f, the observable.
     * @param EmpiricalRaster, the empirical raster.
     * @param S the stimulus, a neuron dependent and  time dependent function
     * @param params the parameters constraining the stimulus (e.g. amplitude, width and speed of a moving bar)
     * @param pow_gamma auxilary vector containing powers of gamma, the leak rate.
     * @param R The range (R=D+1 where D is the Markov memory depth)
     * @param T the time length where the linear response is computed
     * @return delta_mu
     **/
    std::vector <double> delta_mu1_ex(int M,Matrix& Csp_ex,std::vector <double> &musp_fn,std::vector <double>& musp_dphir,const RasterBlockObservable *f,const RasterBlock& EmpiricalRaster,double (*S)(int k,double t,std::vector <double> params),std::vector <double> params,std::vector <double> pow_gamma,int R,int T) const;

    /** Computes the spontaneous correlation matrix  \f$C(f(r),xi_k(0))\f$, with two indices \f$k=0 \dots N-1\f$ the neuron index, and \f$r=0 \dots D\f$, the time index.
     * @param M The number of samples
     * @param f the observable.
     * @param R the memory depth of xi
     * @return the correlation matrix with entries \f$C_{f,xi_k}(r)\f$.
     */
    Matrix Csp_xi(int M,const RasterBlockObservable* f,int R) const;

    /** Computes the spontaneous correlation matrix  \f$C(f1(r),f2(0))\f$ between two observables, where f1 is a scalar observable and f2 a vector observable with dimension the number of neurons. Thus the matrix has two indices \f$k=0 \dots N-1\f$ the neuron index, and \f$r=0 \dots D\f$, the time index.
     * @param M The number of samples to compute the correlations
     * @param f1 the first observable.
     * @param f2 the second observable.
     * @param R the memory depth 
     * @return the correlation matrix with entries \f$C_{f,xi_k}(r)\f$.
     */
    Matrix Csp(int M,const RasterBlockObservable* f1,RasterBlockObservable** f2,int R) const;  
    
    /** Computes the approached first order variation of the average of the observable f when submitted to a time dependent stimulus, using linear response theory  defined in https://arxiv.org/abs/1704.05344 (eq (59) adapted to BMS model) . Here a first order expansion of the potential phi is used (eq. ()) .
     *@param Csp, the correlation matrix with entries  \f$C_{f,xi_k}(r)\f$.
     *@param pow_gamm, the vector with entries \f$\gamma^m\f$.
     *@param S the stimulus, a neuron dependent and  time dependent function
     *@param params the parameters constraining the stimulus (e.g. amplitude, width and speed of a moving bar)
     *@param R The range (R=D+1 where D is the Markov memory depth)
     *@param tau_gamma The time of summation in the discrete convolution (typically tau_gamma=1/abs(log(gamma))
     *@param T The time length where the response is computed
     * @return delta_mu
     */
    std::vector <double>  delta_mu1_app(Matrix Csp,std::vector <double> pow_gamma,double (*S)(int k,double t,std::vector <double> params),std::vector <double> params,int R,int T) const;

        
    /** Computes the correlation between an observable  f and xi at time r < R.
     * @param f the observable.
     * @param EmpiricalRaster the raster used to compute the empirical correlation.
     *@param R the memory depth of xi
     *@param r the time of correlation (positive)
     * @return the correlation \f$C_{f,xi_k}(r)\f$
     */
    std::vector <double> getCorrelation_f_xi(const RasterBlockObservable* f,const RasterBlock& EmpiricalRaster,int R,int r) const;

        
    /** Computes the empirical correlation between a observable f1 and f2 at time r >=0.
     * @param f1 the first observable.
     * @param f2 the second observable.
     * @param EmpiricalRaster the raster used to compute the empirical correlation.
     *@param r the time of correlation (positive)
     * @return the correlation \f$C_{f,xi_k}(r)\f$
     */
    double getCorrelation_f1_f2(const RasterBlockObservable *f1,const RasterBlockObservable *f2,const RasterBlock& EmpiricalRaster,int r) const;
        
        
    /** Computes the spontaneous correlation matrix  \f$C(f1(r),f2(0))\f$ between observables f1 and f2. Csp has  one indice \f$r=0 \dots D\f$, the time index where r is constrained between \f$[0,D]\f$.
     **/
    std::vector <double> Csp(const RasterBlockObservable *f1,const RasterBlockObservable *f2,const RasterBlock& EmpiricalRaster,int R) const;
        
        
    /** Computes the vector with entries \f$\gamma^m\f$ until the power is lower than epsilon.
     * @param tau_gamma The maximal power, the dimension of the returned vector is tau_gamma+1
     * @param epsilon. Fixes the maximal power of gamma.
     *@return The vector.
     */
    std::vector <double> power_gamma(int *tau_gamma,double epsilon);
        
        
#ifndef SWIGJAVA
    // Methods
    // Avoids the implicit copy of the object
    BMSPotential(const BMSPotential& f) {};
#endif
        
    /** Computes the last firing time  \f$\tau_k(\omega_0^{D-1})\f$ of neuron $k$, in the RasterBlock word \f$\omega_0^{D-1}\f$.
     * Here, \f$\tau_k(\omega_0^{D-1})\f$ is defined as \f$min_{0\leq l \leq D-1} \lbrace \omega_k(l)=1\rbrace\f$. It is set to $0$ is neuron $k$ does not fire between times m and n
     * @param i The neuron index.
     * @param word The RasterBlock (of range D).
     * @return The last firing time.
     */
    //    unsigned int getTau(unsigned int i, const RasterBlock& word) const;
        
    //    unsigned int getTau2(unsigned int i, const RasterBlock& raster) const;
    //    unsigned int getTau3(unsigned int i, const RasterBlock& raster) const;
    //    unsigned int getTau4(unsigned int i, const RasterBlock& raster) const;
        
    /** Computes the mean square deviation of the membrane potential in BMS model for the block \f$\omega_0^{D-1} \f$. Eq (8) of http://lanl.arxiv.org/pdf/1002.3275.
     * @param i The neuron index.
     * @param word The RasterBlock (of range D).
     * @return  The mean-square deviation.
     */
    double getSigma(unsigned int i, const RasterBlock& word) const;
        
    /** Computes the integrated external current  in BMS model for the block \f$\omega_0^{D-1} \f$. Eq (5) of http://lanl.arxiv.org/pdf/1002.3275.
     * @param i The neuron index.
     * @param word The RasterBlock (of range D).
     * @return  The integrated external current.
     */
    double getIe(unsigned int i, const RasterBlock& word) const;
        
        
    /** Computes the integrated external current  in BMS model for the block \f$\omega_0^{D-1} \f$ considering a time varying stimulus
     * @param i The neuron index.
     * @param word The RasterBlock (of range D).
     * @param stimulus the average stimulus during this word
     * @return  The integrated external current.
     */
    double getIe(unsigned int i, const RasterBlock& word, std::vector<double> stimulus) const;
        
        
        
    /** Computes the integrated pre-synaptic current from neuron j to neuron i in BMS model  for the block \f$\omega_0^{D-1} \f$. Eq (5) of http://lanl.arxiv.org/pdf/1002.3275. (We note here eta instead of x).
     * @param j The pre synaptic neuron index.
     * @param i The post synaptic neuron index.
     * @param word The RasterBlock (of range D).
     * @return  The integrated pre-synaptic current from neuron j to neuron i.
     */
    double getEta(unsigned int j,unsigned int i, const RasterBlock& word) const;
        
    /**Computes the total, normalized voltage  in BMS model  for the block \f$\omega_0^{D-1}\f$.
     * @param i The neuron index.
     * @param word The RasterBlock (of range D).
     * @return  The  total, normalized voltage .
     */
    double getX(unsigned int i, const RasterBlock& word) const;
        
        
    /**Computes the total, normalized voltage  in BMS model  for the block \f$\omega_0^{D-1}\f$ considering a time varying stimulus
     * @param i The neuron index.
     * @param word The RasterBlock (of range D).
     * @param stimulus the average stimulus that occurred during this word.
     * @return  The  total, normalized voltage .
     */
    double getX(unsigned int i, const RasterBlock& word, std::vector<double> stimulus) const;
        
    /**Computes the total, normalized voltage  in BMS model  for the block \f$\omega_0^{D-1}. Here the external current is a time dependent function. \f$.
     * @param i The neuron index.
     * @param t, the time index
     * @param word The RasterBlock (of range D).
     * @param S the time dependent stimulus.
     * @param params the parameters constraining the stimulus (e.g. amplitude, width and speed of a moving bar)
     * @param pow_gamma auxilary vector containing powers of gamma, the leak rate.
     *@param tau_gamma The time of summation in the discrete convolution (typically tau_gamma=1/abs(log(gamma))
     * @return  The  total, normalized voltage .
     **/
    double getX(unsigned int i,unsigned int t, const RasterBlock& word,double (*S)(int i,double t,std::vector <double> params),std::vector <double> params,std::vector <double> pow_gamma,int tau_gamma) const;
        
    /** Log of Gaussian integral as defined in section 2.1.4 of cessac:10
     *  Note: erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t * t) dt
     */
    double pi(double x) const;
    double lpi(double x) const; // log(pi(x));
    double ilpi(double x) const; // log(1 - pi(x));
        
    // compute pi'(x)/pi(x)
    double pi_prime_divby_pi(double x) const;
        
    // compute pi'(x)/(1-pi(x))
    double pi_prime_divby_one_minus_pi(double x) const;
        
    // gradient matrix of EmpiricalAverage(Phi) with respect to weights
    std::vector<double> getGradientDivergenceWeights(RasterBlockGrammar* grammarSought,RasterBlockGrammar* grammarBMS,unsigned int R);
        
    // gradient matrix of EmpiricalAverage(Phi) with respect to weights considering a time-varying stimulus
    std::vector<double> getGradientDivergenceWeights(RasterBlockBook* bookSought,RasterBlockBook* bookBMS,unsigned int R);
        
    // Weights based on gradient of Kulback Liebler divergence
    void setWeightsDivergence(RasterBlockGrammar* grammar,RasterBlockGrammar* grammarBMS, unsigned int R);
        
    // Weights based on gradient of Kulback Liebler divergence considering a time-varying stimulus
    void setWeightsDivergence(RasterBlockBook* bookSought,RasterBlockBook* bookBMS, unsigned int R);
        
    // gradient vector of EmpiricalAverage(Phi) with respect to current
    std::vector<double> getGradientDivergenceCurrent(RasterBlockGrammar* grammarSought,RasterBlockGrammar* grammarBMS,unsigned int R);
        
    // Current based on gradient of Kulback Liebler divergence
    void setCurrentDivergence(RasterBlockGrammar* grammar,RasterBlockGrammar* grammarBMS,unsigned int R);
        
    // Estimation of the mean square deviation in the plot empirical versus theoretical probability
    double delta(double x,int T);
        
    /** Gets the KL divergence between the empirical measure \f$\pi \f$ of a Raster and the exact measure \f$\mu\f$ of a finite range R BMS potential approximation. The algorithm uses the equation
     * \f$ dKL(\mu,\pi) = P(\phi) - \pi(\phi) - h(\pi)\f$, where \f$P(\phi)\f$ the pressure of \f$\phi\f$ is zero; \f$\pi(\phi) \f$ is the empirical average of \f$\phi \f$ and \f$h(\pi) \f$ is
     * the entropy of \f$\pi \f$ (estimated with Strong et al method).
     @param Raster the Raster
     @param R the range
     @return the divergence
    */
    double getDivergence(const RasterBlock* Raster,unsigned int R);
        
    // Computes a vector of ConditionalProbabilitiesData from a raster grammar where
    //- Each index corresponds to a raster block with
    // - the empirical join probabilities of blocks from a BMS raster;
    // - the empirical marginal probabilities of blocks from a BMS raster;
    // - the empirical conditional probabilities of blocks from a BMS raster;
    // - the theoretical conditional probabilities of blocks from a BMS potential;
    // - the error on empirical conditional probabilities of blocks from a BMS raster;
    ///
      std::vector<ConditionalProbabilitiesData> getConditionalProbabilities(RasterBlockGrammar *grammar,unsigned int T);
        
      /** Generates a file from a raster grammar where
       *- Each line corresponds to a raster block with
       * - On  the first raw, the empirical join probabilities of blocks from a BMS raster;
       * - On  the second raw, the empirical marginal probabilities of blocks from a BMS raster;
       * - On  the third raw, the empirical conditional probabilities of blocks from a BMS raster;
       * - On  the fourth raw, the theoretical conditional probabilities of blocks from a BMS potential;
       * - On  the fifth raw, the error on empirical conditional probabilities of blocks from a BMS raster;
       * @param grammar the grammar.
       * @param T the raster length.
       * @param chaine the file name where data are saved.
       */
      void saveConditionalProbabilities(RasterBlockGrammar *grammar,unsigned int T,const char* chaine);
        
      /** Generates a gnuplot file of the conditional probabilities with on abscissa the empirical conditional probabilities of blocks, and, on ordinate, the  theoretical conditional probabilities of blocks. Addionnally, the colours of points is proportional to -log join empirical probability opf blocks.
       * @param DataFileName The file name.
       * @param GnuplotFileName The name of output gnuplot file.
       */
      void plotConditionalProbabilities(String DataFileName,String GnuplotFileName);
        
      /** Generates a gnuplot file of the conditional probabilities with on abscissa the conditional probabilities of blocks of the BMS potential, and, on ordinate, the  conditional probabilities of blocks for a ParametricPotential. Additionnally, the colours of points is proportional to -log join empirical probability of blocks.
       * @param H The parametric potential.
       * @param Title The string appearing in the title of the plot.
       * @param GnuplotFileName The name of output gnuplot file.
       */
      void plotConditionalProbabilities(ParametricPotential *H,String Title,String GnuplotFileName);
        
      /** Given a raster raster, computes an estimation of
       * the derivative of the topological pressure with respect to the synaptic weight \f$W_{kl}\f$
       * @param raster the raster.
       * @param k the index of \f$W_{kl}\f$.
       * @param l the index of \f$W_{kl}\f$.
       * @return The gradient.
       **/
      double getPressureWeightsGradient(RasterBlock* raster,int k, int l);
        
      // Write graph
      void writeGraph() const;
        
      /** Estimate the unit_threshold given a average spikeRate (assuming the current is constant)
       * @param leak in ms
       * @param capacitance in miuFarrad
       * @param leak_characteristic_time usually denoted by Tau_L in milisecs
       * @param spikeRate Spike Rate in Hertz
       * @param I current value in pAmperes
       * @param W TODO
       * @param i TODO
       * @return the unit-threshold
       **/
      double estimateUnit_Threshold(double leak, double capacitance, double leak_characteristic_time, double spikeRate,double I,std::vector<double> W, size_t i);
      /** Generates a raster-plot from this BMS network with variable current.
       * - The raster-plot is obtained drawing random initial conditions <tt>V_i(0)</tt> in the <tt>[0, 2]</tt> interval.
       * - The currents during lenght is the constant BMS given current
       * - The current after length is the matrix of currents given as input
       * - The raster dimention is N x (lenght+tUnitCurrents.size())
       * - This routine is used as follows: <pre>
       * RasterBlock *raster = bmsPotential.getRasterBlockTcurrent(transients,length,tCurrents);
       * ../..
       * delete raster;</pre>
       * @param transients The number of time iterations.
       * @param length The raster-plot length.
       * @param tUnitCurrents The time dependant currents after the lenght.
       * @param noise Integer to add noise (1/0)
       * @return The generated raster-plot.
       */
      RasterBlock *getRasterBlockTCurrent(unsigned int transients,unsigned int length, std::vector<std::vector<double>> tUnitCurrents) const;
        
      /** Computes the integrated external current  in BMS model for the block \f$\omega_0^{D-1} \f$ considering a time varying stimulus 
       * @param i The neuron index.
       * @param word The RasterBlock (of range D).
       * @param stimulus the average stimulus during this word
       * @return  The integrated external current.
       */
      double getIe(unsigned int i, const RasterBlock& word, std::vector<std::vector<double>> stimulus) const;
        
      /**Computes the total, normalized voltage  in BMS model  for the block \f$\omega_0^{D-1}\f$. 
       * @param i The neuron index.
       * @param word The RasterBlock (of range D).
       * @return  The  total, normalized voltage .
       */
      double getXsp(unsigned int i, const RasterBlock& word) const;
        
  private:
      // Members
      // BMS network parameters
      unsigned int number_of_units;
      double unit_leak, unit_noise, unit_capacitance;
      std::vector<double> unit_currents, unit_thresholds, W;
      RasterBlock *generatedRaster;
      //Introduced to integrate with Virtual Retina
      std::vector<double> potentialV; // membrane potential
      unsigned int number_of_threads;
      std::vector<unsigned int> myNeuronItStartVector;
      std::vector<unsigned int> myNeuronItEndVector;
  };
}

#endif // pranas_BMSPotential_h

