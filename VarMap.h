#ifndef VARMAP_H
#define VARMAP_H

#include "TrainingPoint.h"
#include "Global.h"
#include "NeuralNetwork.h"
//#include "../Algorithms/PSO/Swarm.h"
//#include "../Algorithms/PSO/Particle.h"
//#include "../Algorithms/Chromosome.h"
#include <iostream>
using namespace std;

int domination(const vector<double> &a, const vector<double> &b);
double calDistance(const vector<double> a, const vector<double> b, int startIndex = 0);
void OutputVector(const vector<double> &x);
vector<vector<double> > generate_weight(int nobj, int p);
vector<double> define_refer_point(const vector<CSolution> & archive);
double VectorNorm2(const vector <double> &vec1);
double innerProduct(const vector <double> &vec1, const vector <double> &vec2);
double cal_pbi(const vector<double> &y_obj, const vector<double> &namda, const vector<double> &referencepoint, double &d1, double &d2);
vector<vector<int> > determine_sol_weight(const vector<CSolution> &archive, const vector<vector<double> > &weights, vector<double> &pbi_value);
vector<vector<int> > pair_sol_detected_change(const vector<CSolution> &a, const vector<CSolution> &b, const vector<vector<double> > &weights, bool use_ep=true, bool pbipair = false);
vector<double> train_network(const vector<solpair> &set, const vector<vector<int> > &related_dims, int maxEpochs, int envi_num);
vector<vector<double> > predict_sol_by_network(int cur_envi, int envi_num, const vector<vector<int> > &relateddim, bool main_indep_var=false);
vector<double> predict_a_sol_by_NN(int cur_envi, int envi_num, const vector<vector<int> > &relateddim, const vector<double> &x, bool sameIndep = false);
void initialNNInfo(int i);

vector<double> cal_x_change_bet_envi(const vector<solpair> &set, int num_dim);
double cal_MIC(double *x, double *y, int n);
vector<vector<double> > cal_mic_bet_envi(const vector<solpair> &set, int num_dim);
vector<vector<double> > cal_mic_in_last_envi(int i, int num_dim);
vector<vector<double> > cal_mic_inside_all_envi(int cur_envi, int num_dim);
void find_connect_dim_in_envi(const vector<vector<int> >  &cor_matrix, vector<vector<int> > &connect_block, vector<int> &block_index, int num_dim);
vector<vector<int> > define_corrleration(const vector<vector<double> >& mic_last_envi, int num_dim, double threthold);
vector<int> define_dimblock_ind_var(const vector<vector<int> > &connect_block, const vector<double> &x_change);

vector<vector<double> > predict_pop_by_network(const vector<CSolution>& pop, int i, int envi_num, const vector<vector<int> >& relateddim, const vector<int> &self_variable, bool main_indep_var);





class Solution{
public:
	vector<double> x;
	double f;
public:
	void setSolution(vector<double> sx, double fit){
		//	x.resize(sx.size());
		x = sx;
		f = fit;
	}
	bool operator<(const Solution &p){
		if (DynPara::optType == MIN)
			return f < p.f;
		else return f> p.f;
	}
};

void updateYMaxMin();
void normilizeDataXWithGivenBound(vector<TrainingPoint> &dataSet, vector<double> xmax, vector<double> xmin, int dim);
void normilizeDataYWithGivenBound(vector<TrainingPoint> &dataSet, vector<double> ymax, vector<double> ymin, int dim);
void normilizeDataY(vector<TrainingPoint> &dataSet, const int ydim, vector<double> &maxyvalue, vector<double> &minyvalue);
void outputConOptima(int curenvi, int num_run);
void recordConOptima(int curenvi);
void clusterSol(const vector<TrainingPoint> &dataSet, vector<int> &index, vector<int> &ceterIndex, const int clstnum);
void clstSolByOptima(vector<TrainingPoint> &dataSet, vector<int> &belongCluster, vector<vector<int> > &dataEachClst, vector<TrainingPoint> &center, const int clstnum, vector<vector<int> > &clstSize, bool clstSizeLimit);
//i: the current environment
//g: the current generation in SPSO
//num: the number of solutions map pairs
void outputMapSolutionsToFile(int i, int g, const vector<int> mapIndex, const vector<int> mapCIndex, const int num, const vector<int> lastClusterId,const vector<int> curClusterId, const vector<double> lminfit, const vector<double> lmaxfit, const vector<double> ecminfit, const vector<double> ecmaxfit);
void outputMapSolutionEachEnviToFile(int i, const int num);
void outputConOptMapToFile(int i, int num, const vector<double> lastXFit, const vector<double> curXFit, const vector<int> lastpeak, const vector<int> curpeak);

//vector<vector<int> > &cIndexInEachClst: the cluster result of the solutions in the current environment
//vector<vector<int> > indexInEachClst: the cluster result of the solutions in the last environment
void trainCurVarNN(int i, bool useVarNN, int maxEpochs, vector<int> &cbelongCluster, vector<vector<int> > &cIndexInEachClst, vector<vector<int> > indexInEachClst, int clstnum);
void initConOptNNStruct(const int i, const int samplenum, bool useConOptVarNN);
void trainConOptVarNN(int i, bool useConOptVarNN, bool accumulateOptMap, bool lastToNewAllDim, int  maxEpochs);
void addNewPointToCradleByVarNN(int i, int &fes, bool useConOptVarNN, bool useVarNN, bool lastToNewAllDim, vector<vector<double> > &newpointset, const int newpointnum);
//void addNewPointToCradleByVarNN(int i, int &fes, bool useConOptVarNN, bool lastToNewAllDim, vector<vector<double> > &newpointset, const int newpointnum);
void addNewPointToMpSingleByVarNN(int i, bool useConOptVarNN, bool lastToNewAllDim, vector<vector<double> > &mp_single, const int newpointnum);
void trainCurFunNN(int i, bool useFunNN, int maxEpochs);
void trainInverseFToXNN(int i, bool useInverseFToXModel, int maxEpochs);

void initialNNInfo(int i, bool useFunNN, bool useVarNN, bool useConOptVarNN, bool useInverseFToXModel);
void recordCurNNToPast(int i);
double evaluateByFunNN(int i, bool useFunNN, double *x);

void updateFunAndVarNN(int i, int &fes, int ite, bool useFunNN, bool useVarNN, vector<vector<double> > &cradle, const int selectnum, const int maxEpochs, vector<int> &cbelongCluster, vector<vector<int>> &cIndexInEachClst, vector<vector<int>> indexInEachClst, const int clstnum);

double eval_movpeaks(double x[], bool flag);
bool fitBetter(double a, double b);
#endif