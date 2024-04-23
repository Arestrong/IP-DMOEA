#ifndef __GLOBAL_H_
#define __GLOBAL_H_

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include <vector>
#include <windows.h>
#include "NeuralNetwork.h"
#include "TrainingPoint.h"

using namespace std;

//#define TEST

static string strTestInstance;
static string strFunctionType;
static string preOutRoute;
static bool correct; //correct solutions using gaussian distribution on the solutions
static bool use_random; // randomly generate a part of population
static bool use_pop_predict = true;
static bool keep_self_variable = false;
static bool use_pro = true; //probability to detemine the predition and correction
static bool use_relevant_variables = true; //only use the relevant variables to train preiction models
static bool use_time_condition = true;

enum OPT_TYPE { MIN, MAX };

// demensionality of variables and objectives
//int     numVariables;
//int     numObjectives;

// distribution indexes in SBX and polynomial mutation
static int     id_cx = 20;    // crossover
static int     id_mu = 20;    // for mutation

// ideal point used in decomposition methods
static double  *idealpoint;
double cal_length(vector<double> x);

// parameters for random number generation
//int     seed = 237;
//long    rnd_uni_init;

class CSolution {
public:
	vector<double> x;
	vector<double> f;
	vector<double> nf;
	int assoReferLine; //the reference line that the solution is associated to
	double d1; //the perpendicular distance between the normilzed solution and the reference line
	double d2; //the distance between the origin and the projection point of the norimized solution on the reference line
	int label;

	CSolution();
	CSolution& operator=(const CSolution &s);
	void reMemory();
};

class DynPara {
public:
	static string proName;
	static int dimNum;
	static int objNum;
	static int changeFre;
	static int firstEnviFes;
	static int enviNum;
	static int totalFes;
	static int severity;
	static int runTime;
	static vector<double> lowBound;
	static vector<double> upperBound;
	static const string PFDataPre;

	static int initDimNum;  //the dimensional number in the first environment
	static int initObjNum;  //the objective number in the first environment

	static vector<int> dimNumAllEnvi;  //the dim number in each environment
	static vector<int> objNumAllEnvi; //the objective number of in each environment

	static const OPT_TYPE optType;

	static int freRate;

	static int taut;
	static int t0;
	static int nt;
	static bool test_DF;
	static bool test_SDP;
	
	static bool change_nt_ct; //changing nt ct
	static vector<int> change_nt; // [31] ; // = { 10, 7, 15, 10, 7, 15, 10, 10, 7, 7, 15, 10, 7, 7, 10, 15, 15, 10, 10, 10, 10, 15, 15, 15, 7, 10, 10, 10, 7, 7, 15 };   //10, 15, 7
	static vector<int> change_ct;// [31] ; // = { 10, 10, 5, 15, 10, 15, 5, 15, 15, 10, 10, 10, 15, 10, 5, 5, 10, 10, 15, 5, 10, 10, 15, 10, 10, 5, 15, 10, 15, 10, 5 };  //10, 5, 15

};

void defineBoundary(string proName);

//***********evaluate the performance of the resutls*******************
static string PFDataPre;
static vector<vector<double> > igdPoints;
static vector<double> hvPoint;



class solpair {
public:
	vector<double> x;
	vector<double> y;
	int t;
	solpair(vector<double> x, vector<double> y, int t) {
		this->x = x;
		this->y = y;
		this->t = t;
	}
};

class InfoMemory {
public:
	//static vector<TrainingPoint> solInLastEnvi;
	//static vector<TrainingPoint> solInCurEnvi; //the soluitons that have been found in the current environment
	//static vector<TrainingPoint> solMapBetEnvi; //the solutions map from the last environment to the current environment

	static vector<NeuralNetwork> NNEachDim; //the var x to x (all to one dimension)
	static vector<TrainingPoint> train_set;
	//static vector<vector<vector<int> > > pairEachEnvi; //the pair between each neighboring environment
	static vector<vector<solpair> > solPairEachEnvi; //the solution pairs in each environment
	//static vector<vector<vector<int> > > pairIndexEachEnvi; //the index pairs in each environment

	static vector<vector<CSolution> > noCorrectEnviSol; //the prediction solution before correction 
	static vector<vector<CSolution> > detectedEnviSol; //the solution set stored before the algorithm detects that the environment has changed
	static vector<vector<CSolution> > predictEnviSol; //the predcit set in each environment using 
	static vector<int> predictEnviIndex; // the predict solutions in which environment index

	static vector<vector<CSolution> > initPop; //the init population after environmental changes
	static vector<vector<CSolution> > algSolAllEnvi;

	static vector<vector<double> > predict_time; // the time used for whole prediction part in each environment in each run
	static vector<vector<double> > train_time; // the time used in each environment in each run
	static vector<vector<double> > mic_time; // the time used in each environment in each run
	static vector<vector<double> > nnpredict_time; //the time used for the NN prediction in each environmetn in each run
	static vector<vector<double> > correct_time; //the time used for the correct procedure in each environment in each run
	static vector<vector<double> > update_time;// the time used for update population using predicted solutions in each environment in each run
	static vector<vector<vector<vector<int> > > > relevance_set; //the relevance set of each prediction target in each environment in each run
	static int cur_run; //the index of the current run
	static vector<vector<int> > datasize; //the number of samples in the data set in each enviroment in each run
	static vector<vector<double> > train_error;  //the average training error of all networks
	static vector<vector<double> > predict_error; //the average eduliance distance of the predicted solutions to the optimal solutions

	static vector<double> xmax;
	static vector<double> xmin;
	static vector<double> ymax;
	static vector<double> ymin; //the min y value for normilization
	static bool useNN;

	static bool storeTime; // 

	static vector<CSolution> detector;
	static int numFixDetector;
	static int time_step;
	static bool random_init; // randomly initialize the population after the environment changes

	static vector<vector<vector<double> > > center_gen_envi;
	static vector<vector<vector<double> > > f_center_gen_envi;
	static bool useCusteringSVM; //the SVM classier
	static bool useKneeSVM; //the class to use knee point for tranfer learning 
	static bool useSVR;
	static bool useITSVM; //individual transfer 
	static bool useAutoEncoding;

	static vector<vector<double> > igd_process; //the igd procedure
	static vector<vector<double> > hv_process; //the hv procedure
	static vector<vector<int> > fes_process; //the fes record the igd and hv values
	static int sampleFre;


	//parameter for the SVR method
	static int q; //=4
	static int C; //=1000
	static double eplison; //=0.05
	static double gamma; //=1/num_dim

	static vector<vector<CSolution> > orignalPop; //the orignal population at the begining of the envirommnet, directly inhert from the last environment
	static vector<vector<double> > hist_move; //the movement of optimal solutions in the historical environments   
	
	static vector<int> change_nt; // [31] ; // = { 10, 7, 15, 10, 7, 15, 10, 10, 7, 7, 15, 10, 7, 7, 10, 15, 15, 10, 10, 10, 10, 15, 15, 15, 7, 10, 10, 10, 7, 7, 15 };   //10, 15, 7
	static vector<int> change_ct;// [31] ; // = { 10, 10, 5, 15, 10, 15, 5, 15, 15, 10, 10, 10, 15, 10, 5, 5, 10, 10, 15, 5, 10, 10, 15, 10, 10, 5, 15, 10, 15, 10, 5 };  //10, 5, 15
};

#include "random.h"

#endif