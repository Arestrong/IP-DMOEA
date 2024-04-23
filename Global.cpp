#include "Global.h"
using namespace std;

string DynPara::proName;
int DynPara::dimNum;
int DynPara::objNum;
int DynPara::changeFre;
int DynPara::firstEnviFes;
int DynPara::enviNum;
int DynPara::totalFes;
int DynPara::severity;
int DynPara::runTime;
int DynPara::initObjNum;
int DynPara::initDimNum;
const string DynPara::PFDataPre = "pf_data/";
vector<double> DynPara::lowBound;
vector<double> DynPara::upperBound;

vector<int> DynPara::dimNumAllEnvi;
vector<int> DynPara::objNumAllEnvi;

const OPT_TYPE DynPara::optType = MIN;
int DynPara::freRate = 200;

int DynPara::taut;
int DynPara::t0;
int DynPara::nt;
bool DynPara::test_DF;
bool DynPara::test_SDP;
bool DynPara::change_nt_ct; //changing nt ct

vector<int> DynPara::change_nt; // [31] ; // = { 10, 7, 15, 10, 7, 15, 10, 10, 7, 7, 15, 10, 7, 7, 10, 15, 15, 10, 10, 10, 10, 15, 15, 15, 7, 10, 10, 10, 7, 7, 15 };   //10, 15, 7
vector<int> DynPara::change_ct;// [31] ; // = { 10, 10, 5, 15, 10, 15, 5, 15, 15, 10, 10, 10, 15, 10, 5, 5, 10, 10, 15, 5, 10, 10, 15, 10, 10, 5, 15, 10, 15, 10, 5 };  //10, 5, 15


double cal_length(vector<double> x) {
	double l = 0;
	for (int j = 0; j < x.size(); ++j)
		l += x[j] * x[j];
	return sqrt(l);
}

////////////////////////////////////////////////////////////////////////////////////////////
CSolution::CSolution() {
	x.resize(DynPara::dimNum);
	f.resize(DynPara::objNum);
	assoReferLine = -1;
	d1 = 0;
	d2 = 0;
}
CSolution& CSolution::operator=(const CSolution &s) {
	x = s.x;
	f = s.f;
	assoReferLine = s.assoReferLine;
	d1 = s.d1;
	d2 = s.d2;
	label = s.label;
	return *this;
}
void CSolution::reMemory() {
	vector<double> ox = x;
	vector<double> of = f;
	x.resize(DynPara::dimNum);
	f.resize(DynPara::objNum);
	for (int j = 0; j < x.size(); ++j) {
		if (j < ox.size()) x[j] = ox[j];
		else x[j] = (DynPara::lowBound[j] + DynPara::upperBound[j]) / 2;
		//else x[j] = random(GlobalPara::lowBound[j], GlobalPara::upperBound[j]);
	}
	for (int j = 0; j < f.size(); ++j) {
		if (j < of.size()) f[j] = of[j];
		else f[j] = 1e6;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////

//define problem boundary of the search space
void defineBoundary(string proName) {
	//define the boundary of the search space
	if (DynPara::lowBound.size() != DynPara::dimNum) {
		DynPara::lowBound.resize(DynPara::dimNum);
	}
	if (DynPara::upperBound.size() != DynPara::dimNum) {
		DynPara::upperBound.resize(DynPara::dimNum);
	}

	if (proName == "SDP1") {
		for (int j = 0; j < DynPara::objNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = DynPara::objNum; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "SDP2") {
		//the x is extended to [1,4] in the objective function, so the bound is [0,1]
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = DynPara::objNum - 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "SDP3" || proName == "SDP4" || proName == "SDP10" || proName == "SDP12") {
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = DynPara::objNum - 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "SDP5" || proName == "SDP6" || proName == "SDP7" || proName == "SDP9" || proName == "SDP11"
		|| proName == "SDP13" || proName == "SDP14" || proName == "SDP15") {
		for (int j = 0; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "SDP8") {
		for (int j = 0; j < DynPara::objNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = DynPara::objNum; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "DF1" || proName == "DF2") {
		for (int j = 0; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "DF3") {
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = -1;
			DynPara::upperBound[j] = 2;
		}
	}
	if (proName == "DF4") {
		for (int j = 0; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = -2;
			DynPara::upperBound[j] = 2;
		}
	}
	if (proName == "DF5" || proName == "DF6") {
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = -1;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "DF7") {
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 1;
			DynPara::upperBound[j] = 4;
		}
		for (int j = 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "DF8" || proName == "DF9") {
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = -1;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "DF10") {
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = DynPara::objNum - 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = -1;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "DF11") {
		for (int j = 0; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "DF12") {
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = DynPara::objNum - 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = -1;
			DynPara::upperBound[j] = 1;
		}
	}
	if (proName == "DF13" || proName == "DF14") {
		for (int j = 0; j < DynPara::objNum - 1; ++j) {
			DynPara::lowBound[j] = 0;
			DynPara::upperBound[j] = 1;
		}
		for (int j = DynPara::objNum - 1; j < DynPara::dimNum; ++j) {
			DynPara::lowBound[j] = -1;
			DynPara::upperBound[j] = 1;
		}
	}
}

vector<NeuralNetwork> InfoMemory::NNEachDim; //the var x to x (all to one dimension)
vector<TrainingPoint> InfoMemory::train_set;
//static vector<vector<vector<int> > > pairEachEnvi; //the pair between each neighboring environment
vector<vector<solpair> > InfoMemory::solPairEachEnvi; //the solution pairs in each environment
//static vector<vector<vector<int> > > pairIndexEachEnvi; //the index pairs in each environment

vector<vector<CSolution> > InfoMemory::noCorrectEnviSol;
vector<vector<CSolution> > InfoMemory::detectedEnviSol; //the solution set stored before the algorithm detects that the environment has changed
vector<vector<CSolution> > InfoMemory::predictEnviSol; //the predcit set in each environment using 
vector<int> InfoMemory::predictEnviIndex;
vector<vector<CSolution> > InfoMemory::algSolAllEnvi;

vector<vector<double> > InfoMemory::predict_time; // the time used for whole prediction part in each environment in each run
vector<vector<double> > InfoMemory::train_time; // the time used in each environment in each run
vector<vector<double> > InfoMemory::mic_time; // the time used in each environment in each run
vector<vector<double> > InfoMemory::nnpredict_time; //the time used for the NN prediction in each environmetn in each run
vector<vector<double> > InfoMemory::correct_time; //the time used for the correct procedure in each environment in each run
vector<vector<double> > InfoMemory::update_time;// the time used for update population using predicted solutions in each environment in each run
vector<vector<vector<vector<int> > > > InfoMemory::relevance_set; //the relevance set of each prediction target in each environment in each run
int InfoMemory::cur_run; //the index of the current run
vector<vector<int> > InfoMemory::datasize; //the number of samples in the data set in each enviroment in each run
vector<vector<double> > InfoMemory::train_error;  //the average training error of all networks
vector<vector<double> > InfoMemory::predict_error; //the average eduliance distance of the predicted solutions to the optimal solutions

vector<double> InfoMemory::xmax;
vector<double> InfoMemory::xmin;
vector<double> InfoMemory::ymax;
vector<double> InfoMemory::ymin; //the min y value for normilization
bool InfoMemory::useNN = false;
bool InfoMemory::storeTime = false; // 
vector<CSolution> InfoMemory::detector;
int InfoMemory::numFixDetector;
int InfoMemory::time_step;
vector<vector<vector<double> > > InfoMemory::center_gen_envi;
vector<vector<vector<double> > > InfoMemory::f_center_gen_envi;
bool InfoMemory::useCusteringSVM;
bool InfoMemory::useKneeSVM;
bool InfoMemory::useSVR;
bool InfoMemory::useITSVM; //individual transfer 
bool InfoMemory::useAutoEncoding;

vector<vector<double> > InfoMemory::igd_process; //the igd procedure
vector<vector<double> > InfoMemory::hv_process; //the hv procedure
vector<vector<int> > InfoMemory::fes_process; //the fes record the igd and hv values
int InfoMemory::sampleFre;

int InfoMemory::q = 4;
int InfoMemory::C = 1000;
double InfoMemory::eplison = 0.05;
double InfoMemory::gamma = 0.1;// 1 / num_dim

vector<vector<CSolution> > InfoMemory::orignalPop; //the orignal population at the begining of the envirommnet, directly inhert from the last environment
vector<vector<double> > InfoMemory::hist_move; //the movement of optimal solutions in the historical environments  

vector<int> InfoMemory::change_nt; // [31] ; // = { 10, 7, 15, 10, 7, 15, 10, 10, 7, 7, 15, 10, 7, 7, 10, 15, 15, 10, 10, 10, 10, 15, 15, 15, 7, 10, 10, 10, 7, 7, 15 };   //10, 15, 7
vector<int> InfoMemory::change_ct;// [31] ; // = { 10, 10, 5, 15, 10, 15, 5, 15, 15, 10, 10, 10, 15, 10, 5, 5, 10, 10, 15, 5, 10, 10, 15, 10, 10, 5, 15, 10, 15, 10, 5 };  //10, 5, 15
