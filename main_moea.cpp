/*==========================================================================
//  Implementation of X.-F. Liu, X.-X. Xu, Z.-H. Zhan, Y. Fang, and J. Zhang, 
//  "Interaction- based prediction for dynamic multiobjective optimization," IEEE Trans. Evol. Comput., early access, 2023, doi:10.1109/TEVC.2023.3234113.
//  
//  Based on Multiobjective Evolutionary Algorithm Decomposition (MOEA/D) For Continuous Multiobjective Optimization Problems (2006)
//
//  See the details of MOEA/D in the following paper
//  Q. Zhang and H. Li, MOEA/D: A Multi-objective Evolutionary Algorithm Based on Decomposition, 
//  IEEE Trans. on Evolutionary Computation, in press, 2007
//
//  The source code of MOEA/D was implemented by Hui Li and Qingfu Zhang  
//
===========================================================================*/


#include "global.h"
#include "dmoea.h"
#include "nsga2.h"
#include <sstream>
#include <iomanip>
#include <io.h>
#include <direct.h>
#include <assert.h>
using namespace std;

void initialProPara(string proName, int numOfObj, int numOfDim, int enviseverity) {
	//
	DynPara::proName = proName;
	DynPara::objNum = numOfObj;
	DynPara::dimNum = numOfDim; //10
	DynPara::changeFre = DynPara::freRate * 10;// numOfDim; //100*10
	DynPara::firstEnviFes = 500 * numOfDim;//500 * numOfDim;
	DynPara::enviNum = 31;
/*
#ifndef TEST
	if(numOfDim == 10)
		DynPara::runTime = 20;
	else
		DynPara::runTime = 20;
#else
	DynPara::runTime = 1;
#endif
*/
	DynPara::totalFes = DynPara::changeFre*(DynPara::enviNum - 1) + DynPara::firstEnviFes;
	if (DynPara::change_nt_ct) {
		DynPara::totalFes = DynPara::firstEnviFes;
		for (int j = 1; j < DynPara::enviNum; ++j) {
			DynPara::totalFes += DynPara::changeFre * DynPara::change_ct[j];
		}
	}

	DynPara::severity = enviseverity;

	DynPara::lowBound.resize(DynPara::dimNum);
	DynPara::upperBound.resize(DynPara::dimNum);
	defineBoundary(proName);

	DynPara::initDimNum = DynPara::dimNum;
	DynPara::initObjNum = DynPara::objNum;

}

void init_Alg_Para() {
	if (InfoMemory::useSVR) {
		InfoMemory::q = 4;
		InfoMemory::C = 1000;
		InfoMemory::eplison = 0.05;
		InfoMemory::gamma = 1.0 / DynPara::dimNum; // num_dim
	}
}

void storeOneRunBestResult(string proName, int numOfObj, int i, const vector<double> &igd, const vector<double> &hv) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	stringstream strRun;
	strRun << i;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_igd.out";
	fstream ft;

	ft.open(fileName, ios::out);
	ft << "Envi\tIGD\n";
	for (int j = 0; j < igd.size(); ++j) {
		ft << j << "\t" << igd[j] << "\n";
	}
	ft.close();


	//stringstream strRun;
	//strRun << i;
	fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_hv.out";

	ft.open(fileName, ios::out);
	ft << "Envi\tHV\n";
	for (int j = 0; j < hv.size(); ++j) {
		ft << j << "\t" << hv[j] << "\n";
	}
	ft.close();
}

void storeOneRunProcessResult(string proName, int numOfObj, int i) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	stringstream strRun;
	strRun << i;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_igd.pro";
	fstream ft;

	ft.open(fileName, ios::out);
	ft << "Fes\tIGD\n";
	for (int j = 0; j < InfoMemory::igd_process[i].size(); ++j) {
		ft << InfoMemory::fes_process[i][j] << "\t" << InfoMemory::igd_process[i][j] << "\n";
	}
	ft.close();


	//stringstream strRun;
	//strRun << i;
	fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_hv.pro";

	ft.open(fileName, ios::out);
	ft << "Fes\tHV\n";
	for (int j = 0; j < InfoMemory::hv_process[i].size(); ++j) {
		ft << InfoMemory::fes_process[i][j] << "\t" << InfoMemory::hv_process[i][j] << "\n";
	}
	ft.close();
}

void output_noCorrect_sols(string proName, int numOfObj, int numOfDim, int i) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	stringstream strRun;
	strRun << i;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_noCorrect.sol";
	fstream ft;

	ft.open(fileName, ios::out);
	ft << "Envi\tId\t";
	for (int j = 0; j < numOfObj; ++j) {
		ft << "obj" << j << "\t";
	}
	for (int j = 0; j < numOfDim; ++j) {
		ft << "x" << j << "\t";
	}
	ft << "\n";
	for (int j = 0; j < InfoMemory::noCorrectEnviSol.size(); ++j) {
		for (int k = 0; k < InfoMemory::noCorrectEnviSol[j].size(); ++k) {
			ft << InfoMemory::predictEnviIndex[j+1] << "\t" << k << "\t";
			for (int m = 0; m < numOfObj; ++m) {
				ft << InfoMemory::noCorrectEnviSol[j][k].f[m] << "\t";
			}
			for (int m = 0; m < numOfDim; ++m) {
				ft << InfoMemory::noCorrectEnviSol[j][k].x[m] << "\t";
			}
			ft << "\n";
		}
	}
	ft.close();
}

void output_predict_sols(string proName, int numOfObj, int numOfDim, int i) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	stringstream strRun;
	strRun << i;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_prediction.sol";
	fstream ft;

	ft.open(fileName, ios::out);
	ft << "Envi\tId\t";
	for (int j = 0; j < numOfObj; ++j) {
		ft << "obj" << j << "\t";
	}
	for (int j = 0; j < numOfDim; ++j) {
		ft << "x" << j << "\t";
	}
	ft << "\n";
	for (int j = 0; j < InfoMemory::predictEnviSol.size(); ++j) {
		for (int k = 0; k < InfoMemory::predictEnviSol[j].size(); ++k) {
			ft << InfoMemory::predictEnviIndex[j] << "\t" << k << "\t";
			for (int m = 0; m < numOfObj; ++m) {
				ft << InfoMemory::predictEnviSol[j][k].f[m] << "\t";
			}
			for (int m = 0; m < numOfDim; ++m) {
				ft << InfoMemory::predictEnviSol[j][k].x[m] << "\t";
			}
			ft << "\n";
		}
	}
	ft.close();
}

void storeOneRunSol(string proName, int numOfObj, int i) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	stringstream strRun;
	strRun << i;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + ".sol";
	fstream ft;

	ft.open(fileName, ios::out);
	ft << "Envi\tId\t";
	for (int j = 0; j < numOfObj; ++j) {
		ft << "obj" << j << "\t";
	}
	for (int j = 0; j < DynPara::dimNum; ++j) {
		ft << "x" << j << "\t";
	}
	ft << "\n";
	for (int j = 0; j < InfoMemory::algSolAllEnvi.size(); ++j) {
		for (int k = 0; k < InfoMemory::algSolAllEnvi[j].size(); ++k) {
			ft << j << "\t" << k << "\t";
			for (int m = 0; m < InfoMemory::algSolAllEnvi[j][k].f.size(); ++m) {
				ft << InfoMemory::algSolAllEnvi[j][k].f[m] << "\t";
			}
			for (int m = 0; m < InfoMemory::algSolAllEnvi[j][k].x.size(); ++m) {
				ft << InfoMemory::algSolAllEnvi[j][k].x[m] << "\t";
			}
			ft << "\n";
		}
	}
	ft.close();
}

void storeOneRunIGD_HV(string proName, int numOfObj, int i, const vector<double> &igdValue, const vector<double> &hvValue) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	stringstream strRun;
	strRun << i;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_igd.out";
	fstream ft;

	ft.open(fileName, ios::out);
	ft << "Envi\tId\n";
	for (int j = 0; j < igdValue.size(); ++j) {
		ft << j << "\t" << igdValue[j] << "\n";
	}
	ft.close();

	fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_hv.out";
	ft.open(fileName, ios::out);
	ft << "Envi\tId\n";
	for (int j = 0; j < hvValue.size(); ++j) {
		ft << j << "\t" << hvValue[j] << "\n";
	}
	ft.close();
}

void storeOneRunRelevanceSet(string proName, int numOfObj, int i) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	stringstream strRun;
	strRun << i;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_relevanceSet.out";
	fstream ft;

	ft.open(fileName, ios::out);
	ft << "Envi\tId\t";
	for (int j = 0; j < DynPara::dimNum; ++j)
		ft << "x" << j << "\t";
	ft << "\n";
	for (int j = 0; j < InfoMemory::relevance_set[i].size(); ++j) {
		for (int k = 0; k < InfoMemory::relevance_set[i][j].size(); ++k) {
			ft << j + 2 << "\t" << k << "\t";
			for (int m = 0; m < InfoMemory::relevance_set[i][j][k].size(); ++m) {
				ft << InfoMemory::relevance_set[i][j][k][m] << "\t";
			}
			for (int m = InfoMemory::relevance_set[i][j][k].size(); m < DynPara::dimNum; ++m) {
				ft << "-1\t";
			}
			ft << "\n";
		}
	}
	
	ft.close();
}
void storeOneRunTime(string proName, int numOfObj, int i) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	stringstream strRun;
	strRun << i;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_cost.out";
	fstream ft;
	ft.open(fileName, ios::out);
	ft << "Envi\tdataSize\ttotalTime\tmicTime\ttrainTime\tnnpredictTime\tcorrectTime\tupdateTime\ttrainError\tpredictError\n";
	int start_id = InfoMemory::predict_time[i].size() - InfoMemory::mic_time[i].size();
	if (!InfoMemory::useNN) {
		for (int j = 0; j < InfoMemory::predict_time[i].size(); ++j) {
			ft << j + 1 << "\t" << "0" << "\t" << InfoMemory::predict_time[i][j];// << "\t";
			for (int k = 0; k < 7; ++k) {
				ft << "\t0";
			}
			ft << "\n";
		}
	}
	else {
		for (int j = 0; j < InfoMemory::mic_time[i].size(); ++j) {
			ft << j + 2 << "\t" << InfoMemory::datasize[i][j] << "\t" << InfoMemory::predict_time[i][start_id + j] << "\t" << InfoMemory::mic_time[i][j] << "\t" << InfoMemory::train_time[i][j] << "\t"
				<< InfoMemory::nnpredict_time[i][j] << "\t" << InfoMemory::correct_time[i][j] << "\t" << InfoMemory::update_time[i][j] << "\t"
				<< InfoMemory::train_error[i][j] << "\t" << InfoMemory::predict_error[i][j] << "\n";
		}
	}
	ft.close();
}

void storeProcessResultToFile(string proName, int numOfObj, int numOfDim) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;
	
	
	for (int i = 0; i < DynPara::runTime; ++i) {
		stringstream strRun;
		strRun << i;
		string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_igd.pro";
		fstream ft;
		
		ft.open(fileName, ios::out);
		ft << "Fes\tIGD\n";
		for (int j = 0; j < InfoMemory::igd_process[i].size(); ++j) {
			ft << InfoMemory::fes_process[i][j] << "\t" << InfoMemory::igd_process[i][j] << "\n";
		}
		ft.close();
	}
	
	
	for (int i = 0; i < DynPara::runTime; ++i) {
		stringstream strRun;
		strRun << i;
		string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_R" + strRun.str() + "_hv.pro";
		fstream ft;

		ft.open(fileName, ios::out);
		ft << "Fes\tHV\n";
		for (int j = 0; j < InfoMemory::hv_process[i].size(); ++j) {
			ft << InfoMemory::fes_process[i][j] << "\t" << InfoMemory::hv_process[i][j] << "\n";
		}
		ft.close();
	}
}

void output_mean_in_pro_run(vector<double> value, string proName, int numOfObj, string poststr) {
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;

	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + poststr;
	fstream ft;
	ft.open(fileName, ios::out);
	ft << "Run\tIGD\n";
	for (int j = 0; j < value.size(); ++j) {
		ft << j << "\t" << value[j] << "\n";
	}
	ft.close();
}

void cal_mean_value(const vector<double>& value, double &mean, double &std) {
	mean = 0;
	std = 0;
	for (int j = 0; j < value.size(); ++j) {
		mean += value[j];
	}
	mean /= value.size();
	for (int j = 0; j < value.size(); ++j) {
		std += (value[j] - mean) * (value[j] - mean);
	}
	if (value.size() > 1)
		std = sqrt(std / (value.size() - 1));
	else
		std = 0;
}


void storeResultToFile(string proName, int numOfObj, int numOfDim, vector<vector<double> > igdValue, vector<vector<double> > hvValue){
	//statistics on the data of the igd and hv
	vector<double> avgIGD(DynPara::runTime);
	vector<double> avgHV(DynPara::runTime); //the value in one run
	stringstream strObj; strObj << numOfObj;
	stringstream strEnvi; strEnvi << DynPara::enviNum;
	string fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_data.igd";
	fstream ft;

	ft.open(fileName, ios::out);
	ft << "EnviNum\t";
	for (int i = 0; i < DynPara::runTime; ++i) {
		ft << "Run" <<i << "\t";
	}
	ft << "\n";
	//ft << DynPara::runTime << "\t" << DynPara::enviNum << "\n";
	for (int j = 0; j < DynPara::enviNum; ++j) {
		ft << j << "\t";
		for (int i = 0; i < DynPara::runTime; ++i) {
			ft << igdValue[i][j] << "\t";
		}
		ft << "\n";
	}
	ft.close();
	fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + "_data.hv";
	ft.open(fileName, ios::out);
	//ft << "RunTime\t" << "EnviNum\n";
	//ft << DynPara::runTime << "\t" << DynPara::enviNum << "\n";
	ft << "EnviNum\t";
	for (int i = 0; i < DynPara::runTime; ++i) {
		ft << "Run" << i << "\t";
	}
	ft << "\n";
	for (int j = 0; j < DynPara::enviNum; ++j) {
		ft << j << "\t";
		for (int i = 0; i < DynPara::runTime; ++i) {
			ft << hvValue[i][j] << "\t";
		}
		ft << "\n";
	}
	ft.close();

	//=======================outout the process of the igd and hv during the whole evolutionary=======================================
	//storeProcessResultToFile()
	//===============================================================================================================================

	double totAvgIgd = 0;
	fileName = preOutRoute +  proName + "_M" + strObj.str() + "_E" + strEnvi.str() + ".igd";
	ft.open(fileName, ios::out);
	//ft << "RunTime\n";
	ft << "Run\tIGD\n";
	for (int i = 0; i < DynPara::runTime; ++i) {
		avgIGD[i] = 0;
		for (int j = 0; j < igdValue[i].size(); ++j) {
			avgIGD[i] += igdValue[i][j];
		}
		avgIGD[i] /= igdValue[i].size();
		totAvgIgd += avgIGD[i];
	}
	totAvgIgd /= DynPara::runTime;
	for (int i = 0; i < DynPara::runTime; ++i) {
		ft << i << "\t" << avgIGD[i] << endl;
	}
	ft.close();
	double totStdIgd = 0;
	for (int i = 0; i < DynPara::runTime; ++i) {
		totStdIgd += (totAvgIgd - avgIGD[i])*(totAvgIgd - avgIGD[i]);
	}
	if (DynPara::runTime > 1) totStdIgd = sqrt(totStdIgd / (DynPara::runTime - 1));
	else totStdIgd = 0;

	double totAvgHv = 0;
	double totStdHv = 0;
	fileName = preOutRoute + proName + "_M" + strObj.str() + "_E" + strEnvi.str() + ".hv";
	ft.open(fileName, ios::out);
	//ft << "RunTime\n";
	ft << "Run\tHV\n";
	for (int i = 0; i < DynPara::runTime; ++i) {
		avgHV[i] = 0;
		for (int j = 0; j < hvValue[i].size(); ++j) {
			avgHV[i] += hvValue[i][j];
		}
		avgHV[i] /= hvValue[i].size();
		totAvgHv += avgHV[i];
	}
	totAvgHv /= DynPara::runTime;
	for (int i = 0; i < DynPara::runTime; ++i) {
		ft << i << "\t" << avgHV[i] << endl;
	}
	ft.close();
	//	double totStdHv = 0;
	for (int i = 0; i < DynPara::runTime; ++i) {
		totStdHv += (totAvgHv - avgHV[i])*(totAvgHv - avgHV[i]);
	}
	if (DynPara::runTime > 1) totStdHv = sqrt(totStdHv / (DynPara::runTime - 1));
	else totStdHv = 0;

	double onlineigd = 0, onlineigd_std = 0;
	double onlinehv = 0, onlinehv_std = 0;
	vector<double> online_igd(DynPara::runTime);
	vector<double> online_hv(DynPara::runTime);
	for (int j = 0; j < DynPara::runTime; ++ j) {
		cal_mean_value(InfoMemory::igd_process[j], online_igd[j], onlineigd_std);
		cal_mean_value(InfoMemory::hv_process[j], online_hv[j], onlinehv_std);
	}
	cal_mean_value(online_igd, onlineigd, onlineigd_std);
	cal_mean_value(online_hv, onlinehv, onlinehv_std);     //vector<double> value, string proName, int numOfObj, string poststr
	output_mean_in_pro_run(online_igd, proName, numOfObj, "_online_igd.out");
	output_mean_in_pro_run(online_igd, proName, numOfObj, "_online_hv.out");

	string allFileName = "all_result.out";
	ft.open(allFileName, ios::out | ios::app);
	ft << preOutRoute << "\t" << proName << "\t" << numOfObj << "\t" << DynPara::enviNum << "\t" << numOfDim << "\t" << DynPara::severity << "\t"
		<< DynPara::taut << "\t" << DynPara::nt << "\t"
		<< totAvgIgd << "(" << totStdIgd << ")\t" << totAvgHv << "(" << totStdHv << ")\t"
		<< setiosflags(ios::fixed) << setprecision(4) << totAvgIgd << "(" << totStdIgd << ")\t"
		<< totAvgHv << "(" << totStdHv << ")\t"
		<< onlineigd << "(" << onlineigd_std << ")\t" << onlinehv << "(" << onlinehv_std << ")\n";
	ft.close();

	cout << preOutRoute << "\t" << proName << "\t" << numOfObj << "\t" << DynPara::enviNum << "\t" << numOfDim << "\t" << DynPara::severity << "\t"
		//<< totAvgIgd << "(" << totStdIgd << ")\t" << totAvgHv << "(" << totStdHv << ")\t"
		<< DynPara::taut << "\t" << DynPara::nt << "\t"
		<< setiosflags(ios::fixed) << setprecision(4) << totAvgIgd << "(" << totStdIgd << ")\t"
		<< totAvgHv << "(" << totStdHv << ")\t"
		<< onlineigd << "(" << onlineigd_std << ")\t" << onlinehv << "(" << onlinehv_std << ")\n";
}

double cal_mean(const vector<double> &v) {
	double value = 0;
	for (int i = 0; i < v.size(); ++i) {
		value += v[i];
	}
	value /= v.size();
	return value;
}

//output the optimal solutions in pf the problem
void output_problem_optimal(string proName, const int objNum, const int dimNum, int enviseverity) {
	int mEvas = 1;
	
	if (DynPara::test_SDP) initSDPSystem(proName, objNum, dimNum);
	initialProPara(proName, objNum, dimNum, enviseverity);
	TMOEAD  MOEAD;
	vector<vector<double> > set;
	vector<vector<double> > f;
	vector<double> x(dimNum);
	for (int i = 0; i < dimNum; ++i) {
		x[i] = DynPara::lowBound[i];
	}
	vector<double> optS;
	vector<double> optF(objNum);;
	getOptimalSolution(proName, x, optS, mEvas);
	objectives(optS, optF, mEvas);
	set.push_back(optS);
	f.push_back(optF);

	mEvas += DynPara::firstEnviFes;
	for (int i = 0; i < DynPara::enviNum; ++i) {
		int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
		MOEAD.introduceDynamic(mEvas);
		int cur_index = MOEAD.getEnviIndex(mEvas);
		vector<double> x(dimNum);
		for (int i = 0; i < dimNum; ++i) {
			x[i] = DynPara::lowBound[i];
		}
		vector<double> optS;
		vector<double> optF(objNum);;
		getOptimalSolution(proName, x, optS, mEvas);
		//getOptimalSolution(DynPara::proName, ox, real_y, mEvas);
		objectives(optS, optF, mEvas);
		set.push_back(optS);
		f.push_back(optF);
		mEvas += DynPara::changeFre;
		cout << i << "\t";
		OutputVector(optS);
		OutputVector(optF);
		cout << endl;
	}

	stringstream severitystr;
	severitystr << DynPara::nt;
	stringstream tautstr;
	tautstr << DynPara::taut;
	string casestr = "_taut" + tautstr.str() + "_nt" + severitystr.str();
	//st = "taut" + tautstr.str() + "_nt" + severitystr.str() + "/";
	string fileName = proName + casestr + "_opt.sol";

	fstream ft;
	ft.open(fileName, ios::out);
	ft << "Envi\t";
	for (int j = 0; j < DynPara::objNum; ++j) {
		ft << "obj" << j << "\t";
	}
	for (int j = 0; j < DynPara::dimNum; ++j) {
		ft << "x" << j << "\t";
	}
	ft << "\n";
	for (int i = 0; i < set.size(); ++i) {
		ft << i << "\t";
		for (int j = 0; j < f[i].size(); ++j) {
			ft << f[i][j] << "\t";
		}
		for (int j = 0; j < set[i].size(); ++j) {
			ft << set[i][j] << "\t";
		}
		ft << "\n";
	}
	ft.close();
	
}

void optimization(string proName, const int objNum, const int dimNum, int enviseverity, int startindex = 0, int end_run = 30, bool test_parameter=false, bool use_moead=true, int popsize = 200) {
	int runTime = DynPara::runTime;
	int enviNum = DynPara::enviNum;
	vector<vector<double> > igdValue(runTime);
	vector<vector<double> > hvValue(runTime);
	for (int j = 0; j < runTime; ++j) {
		igdValue[j].resize(enviNum);
		hvValue[j].resize(enviNum);
	}
	int niche = 20; // // neighborhood size
	InfoMemory::igd_process.clear();
	InfoMemory::hv_process.clear();
	InfoMemory::igd_process.resize(DynPara::runTime);
	InfoMemory::hv_process.resize(DynPara::runTime);
	InfoMemory::fes_process.clear();
	InfoMemory::fes_process.resize(DynPara::runTime);

	InfoMemory::predict_time.clear(); // the time used for whole prediction part in each environment in each run
	InfoMemory::train_time.clear(); // the time used in each environment in each run
	InfoMemory::mic_time.clear(); // the time used in each environment in each run
	InfoMemory::relevance_set.clear(); //the relevance set of each prediction target in each environment in each run
	InfoMemory::nnpredict_time.clear();
	InfoMemory::correct_time.clear();
	InfoMemory::update_time.clear();
	InfoMemory::train_error.clear();
	InfoMemory::predict_error.clear();

	InfoMemory::predict_time.resize(DynPara::runTime); // the time used for whole prediction part in each environment in each run
	InfoMemory::train_time.resize(DynPara::runTime); // the time used in each environment in each run
	InfoMemory::mic_time.resize(DynPara::runTime); // the time used in each environment in each run
	InfoMemory::relevance_set.resize(DynPara::runTime);

	InfoMemory::nnpredict_time.resize(DynPara::runTime);
	InfoMemory::correct_time.resize(DynPara::runTime);
	InfoMemory::update_time.resize(DynPara::runTime);
	InfoMemory::datasize.clear();
	InfoMemory::datasize.resize(DynPara::runTime);
	InfoMemory::train_error.resize(DynPara::runTime);
	InfoMemory::predict_error.resize(DynPara::runTime);

	/*
	for (int run = 0; run < DynPara::runTime; ++run) {
		InfoMemory::predict_time[run].resize(DynPara::enviNum);
		InfoMemory::train_time[run].resize(DynPara::enviNum);
		InfoMemory::mic_time[run].resize(DynPara::enviNum);
		InfoMemory::relevance_set[run].resize(DynPara::enviNum);
	}
	*/
	
	//InfoMemory::time_step = 3;
	//int popsize = 99;
	for (int run = 0; run < DynPara::runTime; ++ run) {
		if (run < startindex) continue;
		if (run > end_run) continue;
		InfoMemory::detectedEnviSol.clear();
		InfoMemory::solPairEachEnvi.clear();
		InfoMemory::center_gen_envi.clear();
		InfoMemory::f_center_gen_envi.clear();
		InfoMemory::predictEnviIndex.clear();
		InfoMemory::predictEnviSol.clear();
		InfoMemory::orignalPop.clear();
		InfoMemory::noCorrectEnviSol.clear();
		InfoMemory::cur_run = run;

		if(DynPara::test_SDP) initSDPSystem(proName, objNum, dimNum);
		initialProPara(proName, objNum, dimNum,enviseverity);
		//cout << nvar << "\t" << nobj << endl;
		
		int pointNum = 23;
		//bool test_parameter = true;
		if (test_parameter) {
			if (objNum == 2) pointNum = 199;
			else if (objNum == 3) pointNum = 23;
			else assert(false);
		}
		else {
			if (objNum == 2) pointNum = 99;   //99(100)
			else if (objNum == 3) pointNum = 13; //12(91); 13(105)
			else if (objNum == 5) pointNum = 5; //4(70); 5(126)
			else if (objNum == 7) pointNum = 3; //4(210); 3(84)
			else if (objNum == 10) pointNum = 2; //2(55); 3(220)
			else { cout << "The number of objectives is " << objNum << endl; }
		}
		
		//if (objNum == 3)  MOEAD.run(23, niche, max_gen, run,igdValue[run],hvValue[run]);  //23 -3  popsize 300
		//if (objNum == 2)  MOEAD.run(99, niche, max_gen, run, igdValue[run], hvValue[run]);  //99 -2  popsize 100
		if (use_moead) {
			TMOEAD  MOEAD;
			MOEAD.run(pointNum, niche, DynPara::totalFes, run, igdValue[run], hvValue[run]);  //23 -3  popsize 300
		}
		else {
			TNSGA NSGA;
			NSGA.run(popsize, niche, DynPara::totalFes, run, igdValue[run], hvValue[run]);  //23 -3  popsize 300
		}

		storeOneRunProcessResult(DynPara::proName, DynPara::objNum, run);
		storeOneRunSol(DynPara::proName, DynPara::objNum, run);
		storeOneRunIGD_HV(DynPara::proName, DynPara::objNum, run, igdValue[run], hvValue[run]);
		if(InfoMemory::storeTime) storeOneRunTime(DynPara::proName, DynPara::objNum, run);
		if(false && InfoMemory::useNN)storeOneRunRelevanceSet(DynPara::proName, DynPara::objNum, run);
		
		if (false && InfoMemory::useNN) {
			output_predict_sols(DynPara::proName, DynPara::objNum, DynPara::dimNum, run);
			output_noCorrect_sols(DynPara::proName, DynPara::objNum, DynPara::dimNum, run);
		}
		//storeOneRunBestResult(DynPara::proName, DynPara::objNum, run, igdValue[run], hvValue[run]);
		cout << igdValue[run].size() << "\t" << hvValue[run].size() << endl;
		cout << DynPara::proName << "\tM" << objNum <<"\t" << "R" << run << "\t" << cal_mean(igdValue[run]) << "\t" << cal_mean(hvValue[run]) << endl;
	}
	storeResultToFile(proName, objNum, dimNum, igdValue, hvValue);
}





void run_DF_CEC2018() {

	strFunctionType = "_TCH1";
	int  total_run = 20;         // totoal number of runs
	int  max_gen = 250;       // maximal number of generations
	int  niche = 20;        // neighborhood size
	int numOfObj = 2;
	const int numOfDim = 10;// 50;    //50维度的关系
	DynPara::runTime = 5;// 20;
	correct = true;// false;// true;// false;
	use_random = false;
	use_pop_predict = true;
	keep_self_variable = true; // true; //保持自变量值不变,对每个组的独立变量保持值不变，不用网络预测值
	use_pro = false;
	use_relevant_variables = true;  //神经网络输入是否采用所有变量维度，还是只采用相关变量进行训练
	use_time_condition = false;// true;  //将时间作为网络输入的一个维度
	if (!use_relevant_variables) keep_self_variable = false;
	bool R1_test_time = true;  //test computational time in the R1 version revision ========================================

#ifdef TEST
	DynPara::runTime = 1;
#endif

	DynPara::test_DF = true;
	DynPara::test_SDP = false;

	InfoMemory::useNN = true; // true;// true; // true;// true;
	InfoMemory::time_step = 3;
	DynPara::freRate = 10; //the frequencey rate of the proposed method
	InfoMemory::sampleFre = 5;
	InfoMemory::useCusteringSVM = false; // true;
	InfoMemory::useKneeSVM = false;
	InfoMemory::useSVR = false;
	InfoMemory::useITSVM = false;
	InfoMemory::useAutoEncoding = false;
	
	int which_pro, which_obj, which_severity;
	cout << "intput which problem to run:\t";
	cin >> which_pro;
	cout << "intput which severity:\t";
	cin >> which_severity;
	int start_index = 0;
	cout << "intput which run to start:\t";
	cin >> start_index;
	int end_index = 30;
	cout << "input which run to end:\t";
	cin >> end_index;

	srand(time(NULL));
	//set the name of problem instances
	int pronum = 14;
	vector<string> proInsName(pronum + 1);
	for (int j = 1; j <= pronum; ++j) {
		string temp = "DF";
		stringstream sstr;
		sstr << j;
		proInsName[j] = temp + sstr.str();
	}

	
	string allFileName = "all_result.out";
	fstream ft;
	ft.open(allFileName, ios::out | ios::app);
	ft << "proName\tnumOfObj\tenviNum\tnumOfDim\tseverity\ttaut\tnt\tfre\tIGD(std)\tHV(std)\tIGD(std)\tHV(std)\n";
	ft.close();

	string prepfpath = "process_pf/";
	PFDataPre = "process_pf/";// D: / research / dynamic_multiobjective / NNIP / pf_process / DF_PF_1000 / ";

	//PFDataPre = "../../CPSO/CPSO/pf_data/";
	//initial problems
	bool test_nt = true;// false;   //test the nt value 
	
	int severitynum = 4;
	int nt_value[4] = { 10, 5 };// 15, 20, 5};
	int taut_value[2] = { 10, 5 }; //the change frequency
	for (int scase = 0; scase < severitynum; ++ scase) {
		//if (scase == 0) continue;
		//if (scase > 1) continue;

#ifndef TEST
		if (scase != which_severity) continue;
#endif
		if (scase == 0) { DynPara::taut = taut_value[0]; DynPara::nt = nt_value[0]; }
		else if (scase == 1) { DynPara::taut = taut_value[1]; DynPara::nt = nt_value[0]; }
		else if (scase == 2) { DynPara::taut = taut_value[0]; DynPara::nt = nt_value[1]; }
		else if (scase == 3) { DynPara::taut = taut_value[1]; DynPara::nt = nt_value[1]; }
		else assert(false);

		if (test_nt) {
			DynPara::taut = 10;
			if (scase == 0) DynPara::nt = 7;
			if (scase == 1) DynPara::nt = 15;
			if (scase == 2) DynPara::nt = 20;
			//DynPara::nt = nt_value[scase];
		}

		int changeseverity = DynPara::nt;
		DynPara::freRate = 10 * DynPara::taut;
		
		stringstream severitystr;
		severitystr << changeseverity;
		stringstream tautstr;
		tautstr << DynPara::taut;
		string casestr = "taut" + tautstr.str() + "_nt" + severitystr.str();
		preOutRoute = "taut" + tautstr.str() + "_nt" + severitystr.str() + "/";
		cout << preOutRoute << endl;
		//PFDataPre = prepfpath + casestr + "//";
#ifndef TEST
		//PFDataPre = "pf_data/" + preOutRoute;
#endif
		if (InfoMemory::useNN) {
			if(correct)
				preOutRoute = casestr + "_NN";
			else
				preOutRoute = casestr + "_NN_noCorrect";
			if (use_pro) preOutRoute = preOutRoute + "_proPredict";
			if(!use_relevant_variables) preOutRoute = preOutRoute + "_allDim";
			if(!use_time_condition) preOutRoute = preOutRoute + "_noTimeCon";
			if(!keep_self_variable) preOutRoute = preOutRoute + "_noKeepSelfVar";
			if(R1_test_time) preOutRoute = preOutRoute + "_cost/"; //"_relation10/";// "_relation1/";// 
			else preOutRoute = preOutRoute + "_relation0_rand/"; //"_relation10/";// "_relation1/";// 

		}
		
		if (InfoMemory::useCusteringSVM)preOutRoute = casestr + "_CTSVM";
		if (InfoMemory::useKneeSVM)preOutRoute = casestr + "_KTSVM";
		if (InfoMemory::useSVR) preOutRoute = casestr + "_SVR";
		if (InfoMemory::useITSVM) preOutRoute = casestr + "_ITSVM";
		if (InfoMemory::useAutoEncoding) preOutRoute = casestr + "_AE";
		if (numOfDim != 10) {
			stringstream dimstr;
			dimstr << numOfDim;
			preOutRoute = preOutRoute + "_" + dimstr.str() + "D/";
		}
		else
			preOutRoute = preOutRoute + "/";

		if (_access(preOutRoute.c_str(), 0) != 0)  _mkdir(preOutRoute.c_str());

		for (int proindex = 1; proindex <= pronum; ++proindex) {
			if (proindex <= 9) numOfObj = 2;
			else numOfObj = 3;
#ifndef TEST			
			if (proindex != which_pro) continue;
#endif
			
			strTestInstance = proInsName[proindex];
			initialProPara(proInsName[proindex], numOfObj, numOfDim, changeseverity);
			optimization(proInsName[proindex], numOfObj, numOfDim, changeseverity, start_index, end_index);
			//output_problem_optimal(proInsName[proindex], numOfObj, numOfDim, changeseverity);
		}
	}
#ifdef TEST
	system("PAUSE");
#endif
	
}

void run_SDP()
{
	DynPara::test_DF = false;
	DynPara::test_SDP = true;
	// set the type of decomposition method
	// "_TCH1": Tchebycheff, "_TCH2": normalized Tchebycheff, "_PBI": Penalty-based BI
    strFunctionType = "_TCH1";  

	int  total_run       = 20;         // totoal number of runs
	int  max_gen         = 250;       // maximal number of generations
	int  niche           = 20;        // neighborhood size
	
	InfoMemory::useNN = false;// true;
	InfoMemory::time_step = 3;
	
	InfoMemory::useCusteringSVM = false; // true;
	InfoMemory::useKneeSVM = false;
	InfoMemory::useSVR = false;
	InfoMemory::useITSVM = false;
	InfoMemory::useAutoEncoding = false;

	int nt = 10;
	DynPara::freRate = 10*nt;
	InfoMemory::sampleFre = 50;
	DynPara::runTime = 20;

	int which_alg = 0;
	cout << "input which algorithm to run:\t1(IP), 2(CT), 3(KT), 4(SVR), 5(IT), 6(AE):\n";
	cin >> which_alg;
	if (which_alg == 1) InfoMemory::useNN = true;
	else if (which_alg == 2) InfoMemory::useCusteringSVM = true;
	else if (which_alg == 3) InfoMemory::useKneeSVM = true;
	else if (which_alg == 4) InfoMemory::useSVR = true;
	else if (which_alg == 5) InfoMemory::useITSVM = true;
	else if (which_alg == 6) InfoMemory::useAutoEncoding = true;
	else {
		assert(false);
	}

	///char *instances[]  = {"ZDT1","ZDT2","ZDT3","ZDT4","ZDT6","DTLZ1","DTLZ2"}; // names of test instances
	//int  nvars[]       = {30, 30, 30, 10, 10, 10, 10};                         // number of variables
	//int  nobjs[]       = {2, 2, 2, 2, 2, 3, 3};                                // number of objectives
	int which_pro, which_obj, which_severity;
	cout << "intput which problem to run: (SDP1-15)\n";
	cin >> which_pro;
	cout << "intput which objective instance to run:\t0(2), 1(3), 2(5), 3(7), 4(10)\n";
	cin >> which_obj;
	cout << "intput which severity:\t0(10), 1(7), 2(5), 3(3)\n";
	cin >> which_severity;
	int which_run = 0;
	cout << "input which run to begin.....\t";
	cin >> which_run;
	int end_run = 20;
	cout << "intput which run to end .....\t";
	cin >> end_run;

	srand(time(NULL));
	//set the name of problem instances
	int pronum = 15;
	vector<string> proInsName(pronum + 1);
	for (int j = 1; j <= pronum; ++j) {
		string temp = "SDP";
		stringstream sstr;
		sstr << j;
		proInsName[j] = temp + sstr.str();
	}
	//set the number of objectives in each problem instance
	int objCaseNum = 5;
	vector<int> objNum(objCaseNum);
	objNum[0] = 2; objNum[1] = 3; objNum[2] = 5; objNum[3] = 7; objNum[4] = 10;
	const int numOfDim = 30;

	string allFileName = "all_result.out";
	fstream ft;
	ft.open(allFileName, ios::out | ios::app);
	ft << "proName\tnumOfObj\tenviNum\tnumOfDim\tseverity\tIGD(std)\tHV(std)\tIGD(std)\tHV(std)\n";
	ft.close();

	PFDataPre = "pf_data/";
	
	//PFDataPre = "../../CPSO/CPSO/pf_data/";
	//initial problems
	int severitynum = 4;
	int severityvalue[4] = { 10, 7, 5, 3 };
	for (int scase = 0; scase < severitynum; ++ scase) {
		if (scase != which_severity)continue;
		int changeseverity = severityvalue[scase];
		//	if (scase != 3) {
		stringstream severitystr;
		severitystr << changeseverity;
		preOutRoute = "severity" + severitystr.str() + "/";
#ifndef TEST
		preOutRoute = "severity" + severitystr.str() + "/";
#endif
		stringstream changefrestr;
		changefrestr << nt;
		PFDataPre = PFDataPre + preOutRoute;
		if(InfoMemory::useNN) preOutRoute = "severity" + severitystr.str() + "_nt" + changefrestr.str() + "_IP_DMOEA/";
		if(InfoMemory::useCusteringSVM)preOutRoute = "severity" + severitystr.str() + "_nt" + changefrestr.str() + "_CTSVM/";
		if(InfoMemory::useKneeSVM)preOutRoute = "severity" + severitystr.str() + "_nt" + changefrestr.str() + "_KTSVM/";
		if(InfoMemory::useSVR) preOutRoute = "severity" + severitystr.str() + "_nt" + changefrestr.str() + "_SVR/";
		if(InfoMemory::useITSVM) preOutRoute = "severity" + severitystr.str() + "_nt" + changefrestr.str() +"_ITSVM/";
		if(InfoMemory::useAutoEncoding) preOutRoute = "severity" + severitystr.str() + "_nt" + changefrestr.str() + "_AE/";

		if (_access(preOutRoute.c_str(), 0) != 0)  _mkdir(preOutRoute.c_str());

		for (int proindex = 1; proindex <= pronum; ++proindex) {
			if (proindex != which_pro) continue;
			//cout << "problem " << proindex << endl;
			for (int objIndex = 0; objIndex < objCaseNum; ++ objIndex) {
#ifdef TEST
				if (objIndex >= 1) continue;
#endif
				if (objIndex != which_obj) continue;
				//if (objIndex != which_obj) continue;
				if (changeseverity < 10 && objIndex != 0) continue;
				//if (objIndex >= 1) continue;
				if (proindex >= 13 && objIndex >= 1) continue;
				int numOfObj = objNum[objIndex];
				if (proindex == 13) {
					numOfObj = 2;  //change from 2 50 5
				}
				else if (proindex == 14 || proindex == 15) {
					numOfObj = 5;
				}
				//test_rand(proInsName[proindex], numOfObj, numOfDim, changeseverity);
				
				//init the problem parameters in SDP.h
				strTestInstance = proInsName[proindex];
				initSDPSystem(proInsName[proindex], numOfObj, numOfDim);
				initialProPara(proInsName[proindex], numOfObj, numOfDim, changeseverity);
				optimization(proInsName[proindex], numOfObj, numOfDim, changeseverity, which_run, end_run);
			}
		}
	}
#ifdef TEST
	system("PAUSE");
#endif
    
}

int main() {
	DynPara::change_nt_ct = false; //changing nt ct 
	InfoMemory::storeTime = false;
	//test_CF_NT_combination();
	//test_parameter_K();
	//test_parameter_popsize();
	//test_base_solver(); //use NSGA-II
	//cout << sin(3 * pi * 0.0001) << endl;
	//system("PAUSE");

	//test_SDP();
	//test_cost();
	//test_changing_frequency_severity();
	run_DF_CEC2018();
	//run_SDP();
	return 0;
}