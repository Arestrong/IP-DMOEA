#pragma once
#ifndef __NSGA_H_
#define __NSGA_H_

#include "global.h"
#include "common.h"
#include "individual.h"
#include "scalarfunc.h"
#include "recombination.h"
#include "hv.h"
#include "VarMap.h"
#include "NeuralNetwork.h"
#include "mine.h"
#include "cppmine.h"
#include "Clustering.h"
#include "svm_utils.h"
#include <algorithm>
#include <assert.h>
#include <sstream>
#include <vector>
#include <iomanip>

class TNSGA
{
public:

	TNSGA();
	virtual ~TNSGA();

	void init_uniformweight(int sd);    // initialize the weights for subproblems
	void init_neighbourhood();          // calculate the neighbourhood of each subproblem
	void init_population();             // initialize the population
	void update_EP(const TIndividual& child);
	int dominate_comp(const vector<double>& a, const vector<double>& b);
	void storePSForDetectChange_DE();
	void reevaluate_EP();
	void init_detector();
	void reevaluate_detector();
	void reevaluate_pop(vector<double>& igdValue, vector<double>& hvValue);
	void reinitialize_pop(vector<double>& igdValue, vector<double>& hvValue, double rate = 0.1);
	void gaussian_pop(vector<double>& igdValue, vector<double>& hvValue, double rate = 0.1);

	void selection(vector<int> &a1);   //selection for the proposed method
	int tournament(const TIndividual& ind1, const TIndividual& ind2); //routine for binary tournament
	int dominate_cmp(const TIndividual& ind1, const TIndividual& ind2);
	vector<int> crowd_distance_selection(const vector<TIndividual>& temp, const int snum, const vector<double> max_f, const vector<double> min_f);

	void update_reference(TIndividual& ind);           // update the approximation of ideal point
	void update_problem(TIndividual& child, int id);   // compare and update the neighboring solutions
	void evolution(vector<double>& igdValue, vector<double>& hvValue);                                  // mating restriction, recombination, mutation, update
	void run(int sd, int nc, int maxfes, int rn, vector<double>& igdValue, vector<double>& hvValue);          // execute MOEAD
	void save_front(char savefilename[1024]);          // save the pareto front into files

	vector<vector<double> > weights; //the weights used to generate
	vector <TSOP>  population;  // current population 
	vector<TIndividual> offsprings;
	vector<TIndividual> EP;
	TIndividual* indivpoint;    // reference point
	int  niche;                 // neighborhood size
	int  pops;       // population   size	
	int mEvas;  //the number of function evaluations
	bool use_ep; //use archive
	int runid;
	vector<double> min_pf_value;
	double pro_predict; // the probability to update the solution using prediction
	double pro_correct; //the probability to correct the solution using 

	double cal_distance(const vector<double>& a, const vector<double>& b);
	void cal_maxmin_value(vector<double>& fmin, vector<double>& fmax);
	void operator=(const TNSGA& emo);


	bool enviChangeNextFes(int curFes);
	int getEnviIndex(int curFes);
	void metric(vector<double>& igdValue, vector<double>& hvValue);
	void metric();
	double calculateIGD(const vector<CSolution>& set, const int enviIndex, vector<double>& igdValue, bool update = true);
	void readReferPointForIGD(int enviIndex);
	void readReferPointForHV(int enviIndex, vector<double>& v);
	double calculateHV(const vector<CSolution>& set, const int enviIndex, vector<double>& hvValue, bool update = true);
	void introduceDynamic(int curFes);
	void reactToDimChange(bool varDimChange, bool objDimChange);
	void reactToEnviChange(vector<double>& igdValue, vector<double>& hvValue, const vector<TIndividual> &offsprings, const int oindex);
	bool detectChange(int i, vector<double>& igdValue, vector<double>& hvValue);

	void storeFinalPop();
	void storeAllPop(int runId);
	void storeInitPop();
	vector<double> eval_center_move();

	void transfer_cluster_SVM(vector<double>& IGD, vector<double>& HV);
	void transfer_knee_SVM(vector<double>& IGD, vector<double>& HV);
	void predict_by_neuralnetwork(vector<double>& IGD, vector<double>& HV);
	void cal_change_past_envi(vector<solpair>& set, vector<vector<int> >& select_indepvar, int start_envi, vector<int>& self_variables);
	void print_set(const vector<CSolution>& set);
	double cal_pop_x_error();
	double cal_EP_x_error();
	double cal_pop_f_error();
	double cal_EP_f_error();
	void init_pop_correlation();
	vector<double> cal_pop_center();
	void resample_pop(const vector<vector<double> >& sample, vector<double>& igdValue, vector<double>& hvValue);
	void reinit_hist_center(vector<double>& IGD, vector<double>& HV);
	vector<double> estimate_error(vector<double>& IGD, vector<double>& HV);
	vector<double> eval_move_error(); //evaluate the error of each dimension that the initial population have moved to the best solution

	//transfer learning using clustering
	void obtain_previous_data(vector<CSolution>& data);
	void obtain_new_data(vector<CSolution>& data, vector<double>& IGD, vector<double>& HV);
	int select_pre_cluster(Cluster& new_clst, Cluster& old_clst, int j, const vector<int>& visit);
	void record_metric();

	//transfer learning using knee points
	vector<CSolution> define_knee_in_set(const vector<CSolution>& set, vector<int>& subspace_index, vector<int>& label);
	vector<vector<double> > estimate_new_knee_point(const vector<CSolution>& a, const vector<int>& aindex, const vector<CSolution>& b, const vector<int>& bindex);
	vector<CSolution> generate_source_knee_data(const vector<CSolution>& set, const vector<int>& label);
	vector<CSolution> generate_target_knee_data(const vector<vector<double> >& estimated_knee);
	void init_pop_use_sol(const vector<vector<double> >& predict_knee, vector<double>& IGD, vector<double>& HV);

	//predict init population using SVR regression
	void predict_by_SVR(vector<double>& IGD, vector<double>& HV);

	//transfer learning using individuals
	void transfer_ind_SVM(vector<double>& IGD, vector<double>& HV);
	void presearch(vector<double>& IGD, vector<double>& HV, vector<int>& label, vector<vector<double> >& set);

	//predict half of the init population using autoencoding
	void predict_by_AE(vector<double>& IGD, vector<double>& HV);
	vector<int> rank_sol_crowd_distance(const vector<CSolution>& set);
	void init_half_pop_by_predict(const vector<CSolution>& set, vector<double>& IGD, vector<double>& HV);
	//void cal_center(const vector<double> &x);
	void cal_center(); //calculate center of the proposed method
	double cal_pop_error_to_optima();//calculate the error of the current population to the optimal solutions
	vector<double> eval_prediction_error(); //看当前环境预测的初始解与该环境最后的PS，通过pbi配对后看每个维度的预测误差，用于作为修正误差
	vector<vector<double> > correct_sol_by_predict_error(const vector<vector<double> >& set, const vector<CSolution>& osol, const vector<double>& train_error, const vector<int>& self_variable, const vector<int>& use_correct);

	void output_pop_errors();
	void outputVector(const vector<double> x);
	void eval_hist_move();   //calculate the historical movement

	void update_pop(const vector<TIndividual>& offsprings);

};


TNSGA::TNSGA()
{

	idealpoint = new double[DynPara::objNum];
	indivpoint = new TIndividual[DynPara::objNum];
	// initialize ideal point	
	for (int n = 0; n < DynPara::objNum; n++)
	{
		idealpoint[n] = 1.0e+30;
		indivpoint[n].rnd_init();
		indivpoint[n].obj_eval(mEvas);
	}
	use_ep = true;
	if (InfoMemory::useSVR)use_ep = false;
	pro_predict = 1;
	pro_correct = 1;
}

TNSGA::~TNSGA()
{
	delete[] idealpoint;
	delete[] indivpoint;
}

int TNSGA::dominate_comp(const vector<double>& a, const vector<double>& b) {
	int smaller = 0, equal = 0, large = 0;
	if (a.size() != b.size()) {
		cout << a.size() << "\t" << b.size() << endl;
		assert(false);
	}
	for (int i = 0; i < a.size(); ++i) {
		if (a[i] < b[i] && fabs(a[i] - b[i]) > 1e-6) smaller++;
		else if (a[i] > b[i] && fabs(a[i] - b[i]) > 1e-6) large++;
		else equal++;
	}
	if (smaller > 0 && large == 0) return 1;  //a is better
	else if (large > 0 && smaller == 0) return -1;
	else return 0;
}
double TNSGA::cal_distance(const vector<double>& a, const vector<double>& b) {
	double d = 0;
	for (int i = 0; i < a.size(); ++i) {
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	d = sqrt(d);
	return d;
}
void TNSGA::cal_maxmin_value(vector<double>& fmin, vector<double>& fmax) {
	fmin.resize(DynPara::objNum);
	fmax.resize(DynPara::objNum);
	for (int j = 0; j < EP.size(); ++j) {
		for (int k = 0; k < DynPara::objNum; ++k) {
			if (j == 0 || fmin[k] > EP[j].y_obj[k]) {
				fmin[k] = EP[j].y_obj[k];
			}
			if (j == 0 || fmax[k] < EP[j].y_obj[k]) {
				fmax[k] = EP[j].y_obj[k];
			}
		}
	}
	for (int j = 0; j < pops; ++j) {
		for (int k = 0; k < DynPara::objNum; ++k) {
			if (j == 0 || fmin[k] > population[j].indiv.y_obj[k]) {
				fmin[k] = population[j].indiv.y_obj[k];
			}
			if (j == 0 || fmax[k] < population[j].indiv.y_obj[k]) {
				fmax[k] = population[j].indiv.y_obj[k];
			}
		}
	}
}
void TNSGA::update_EP(const TIndividual& child) {
	//select the nondiminate solutions from population into ep
	if (EP.size() == 0) {
		EP.push_back(child);
		return;
	}
	int num_obj = DynPara::objNum;
	//OutputVector(child.y_obj);

	//check whether is the same
	for (int j = 0; j < EP.size(); ++j) {
		bool equal = true;
		for (int k = 0; k < num_obj; ++k) {
			if (fabs(EP[j].y_obj[k] - child.y_obj[k]) > 1e-6) {
				equal = false;
				break;
			}
		}
		if (equal) return;
	}
	//cout << "+\n";
	vector<int> be_dominated;
	bool add = true;
	for (int j = 0; j < EP.size(); ++j) {
		int relation = dominate_comp(child.y_obj, EP[j].y_obj);
		if (relation == 1) {
			be_dominated.push_back(j);
		}
		else if (relation == -1) {
			add = false;
			return;
		}
	}
	if (!add) return;
	//OutputVector(child.y_obj);
	//cout << endl;
	if (false && be_dominated.size() > 0) {
		cout << "child:\t";
		OutputVector(child.y_obj);
		cout << endl;
		for (int j = 0; j < be_dominated.size(); ++j) {
			cout << j << "\t";
			OutputVector(EP[be_dominated[j]].y_obj);
			cout << endl;
		}
	}
	for (int j = 0; j < be_dominated.size(); ++j) {
		EP.erase(EP.begin() + (be_dominated[j] - j));
	}
	if (add) {
		EP.push_back(child);
	}
	//print_set(EP);
	//check EP size
	if (EP.size() > pops) {
		//theta dominane
		vector<int> visit(EP.size(), 0);
		vector<double> minf;
		vector<double> maxf;
		cal_maxmin_value(minf, maxf);
		vector<int> label(EP.size());
		vector<double> pb_value(EP.size());
		vector<int> w_scount(weights.size(), 0);
		//vector<vector<double> > weights = generate_weight()
		double d1, d2;
		for (int j = 0; j < EP.size(); ++j) {
			vector<double> norm_f = EP[j].y_obj;
			for (int k = 0; k < norm_f.size(); ++k) {
				norm_f[k] = (norm_f[k] - minf[k]) / (maxf[k] - minf[k]);
			}
			vector<double> pbi_value(weights.size());
			int min_index = -1;
			for (int k = 0; k < weights.size(); ++k) {
				pbi_value[k] = cal_pbi(norm_f, weights[k], minf, d1, d2);
				if (k == 0 || pbi_value[k] < pbi_value[min_index])
					min_index = k;
			}
			label[j] = min_index;
			pb_value[j] = pbi_value[min_index];
		}
		int count = 0;
		while (count < pops) {
			int sindex = -1;
			for (int j = 0; j < EP.size(); ++j) {
				if (visit[j]) continue;
				if (sindex == -1 || w_scount[label[j]] < w_scount[label[sindex]] || (w_scount[label[j]] == w_scount[label[sindex]] && pb_value[j] < pb_value[sindex])) {
					sindex = j;
				}
			}
			visit[sindex] = 1;
			w_scount[label[sindex]] = w_scount[label[sindex]] + 1;
			count = count + 1;
		}

		int dcount = 0;
		for (int j = 0; j < visit.size(); ++j) {
			if (!visit[j]) {
				EP.erase(EP.begin() + (j - dcount));
				dcount = dcount + 1;
			}
		}
	}
}


void TNSGA::init_population()
{
	for (int i = 0; i < pops; i++)
	{
		population[i].indiv.rnd_init();
		population[i].indiv.obj_eval(mEvas);
		update_reference(population[i].indiv);
		mEvas++;
		record_metric();
	}

}

// initialize a set of evely-distributed weight vectors
void TNSGA::init_uniformweight(int sd)
{
	for (int i = 0; i < pops; ++i) {
		TSOP sop;
		for (int k = 0; k < DynPara::objNum; ++k) {
			sop.array.push_back(1);
		}
		/*sop.array.push_back(i);
		sop.array.push_back(j);
		sop.array.push_back(sd - i - j);*/
		for (int k = 0; k < sop.array.size(); k++)
			sop.namda.push_back(1.0 * sop.array[k] / sd);
		population.push_back(sop);
		weights.push_back(sop.namda);
	}

	//weights.clear();

}

// initialize the neighborhood of subproblems based on the distances of weight vectors
void TNSGA::init_neighbourhood()
{
	//vector<double> x(pops);
	//vector<int> idx(pops);
	double* x = new double[pops];
	int* idx = new int[pops];
	for (int i = 0; i < pops; i++)
	{
		for (int j = 0; j < pops; j++)
		{
			x[j] = distanceVector(population[i].namda, population[j].namda);
			idx[j] = j;
		}
		minfastsort(x, idx, pops, niche);
		for (int k = 0; k < niche; k++)
			population[i].table.push_back(idx[k]);

	}
	delete[] x;
	delete[] idx;
}

// update the best solutions of neighboring subproblems
void TNSGA::update_problem(TIndividual& indiv, int id)
{
	int replace_num = 2;
	int count = 0;
	for (int i = 0; i < niche; i++)
	{
		int    k = population[id].table[i];
		double f1, f2;
		f1 = scalar_func(population[k].indiv.y_obj, population[k].namda, indivpoint);
		f2 = scalar_func(indiv.y_obj, population[k].namda, indivpoint);
		if (f2 < f1) {
			population[k].indiv = indiv;
			count++;
		}
		if (InfoMemory::useNN)
			if (count >= replace_num) break;
	}
}

// update the reference point
void TNSGA::update_reference(TIndividual& ind)
{
	for (int n = 0; n < DynPara::objNum; n++)
	{
		if (ind.y_obj[n] < idealpoint[n])
		{
			idealpoint[n] = ind.y_obj[n];
			indivpoint[n] = ind;
		}
	}
}
bool TNSGA::enviChangeNextFes(int curFes) {
	int g = (curFes + 1 - DynPara::firstEnviFes) > 0 ? (curFes + 1 - DynPara::firstEnviFes) : 0;
	if (g % DynPara::changeFre == 1) return true;
	else return false;
}
int TNSGA::getEnviIndex(int curFes) {
	int g = (curFes - DynPara::firstEnviFes) > 0 ? (curFes - DynPara::firstEnviFes) : 0;
	if (curFes <= DynPara::firstEnviFes) return 0;
	else {
		int enviIndex = floor(1.0 * (g - 1) / DynPara::changeFre) + 1;
		return enviIndex;
	}
}
void TNSGA::introduceDynamic(int curFes) {
	if (DynPara::test_SDP) {
		changeEnviInNextFes(DynPara::proName, curFes, DynPara::firstEnviFes, DynPara::severity, DynPara::changeFre);
		DynPara::dimNum = getProDimNum();
		DynPara::objNum = getProObjNum();
	}
}
bool TNSGA::detectChange(int i, vector<double>& igdValue, vector<double>& hvValue) {
	int detectNum = 1;// population.size() *0.1;
	vector<int> index(1);
	index[0] = i;
	//vector<int> index(population.size());
	//random_shuffle(index.begin(), index.end());
	bool hasChanged = false;
	for (int j = 0; j < detectNum; ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(igdValue, hvValue);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; 
			int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		int sindex = index[0];

		record_metric();
		vector<double> of = InfoMemory::detector[sindex].f;
		objectives(InfoMemory::detector[sindex].x, InfoMemory::detector[sindex].f, mEvas);
		mEvas++;
		for (int k = 0; k < DynPara::objNum; ++k) {
			if (fabs(InfoMemory::detector[sindex].f[k] - of[k]) > 1e-6) {
				return true;
				break;
			}
		}
		/*vector<double> of = population[sindex].indiv.y_obj;
		population[sindex].indiv.obj_eval();
		for (int k = 0; k < DynPara::objNum; ++k) {
			if (fabs(population[sindex].indiv.y_obj[k] - of[k]) > 1e-6) {
				hasChanged = true;
				break;
			}
		}
		*/
		if (hasChanged) break;
	}
	return false;
}
void TNSGA::reactToDimChange(bool varDimChange, bool objDimChange) {
	//parameter change 
	if (objDimChange) {
		/*int pointNum = 23;
		if (objNum == 2) pointNum = 99;   //99(100)
		else if (objNum == 3) pointNum = 13; //12(91); 13(105)
		else if (objNum == 5) pointNum = 5; //4(70); 5(126)
		else if (objNum == 7) pointNum = 3; //4(210); 3(84)
		else if (objNum == 10) pointNum = 2; //2(55); 3(220)
		else { cout << "The number of objectives is " << objNum << endl; }*/
		int sd = 0;
		if (DynPara::objNum == 2) sd = 99;
		else if (DynPara::objNum == 3) sd = 13; //12(91); 13(105)
		else if (DynPara::objNum == 4) sd = 6;  //5(56); 6(84); 7(120)
		else if (DynPara::objNum == 5) sd = 5; //4(70); 5(126)
		else if (DynPara::objNum == 6) sd = 4; //4(126); 3(56); 5(152)
		else if (DynPara::objNum == 7) sd = 3; //4(210); 3(84)
		else if (DynPara::objNum == 8) sd = 3; //3(120); 2(36)
		else if (DynPara::objNum == 9) sd = 2;  //2(45); 3(165)
		else if (DynPara::objNum == 10) sd = 2; //2(55); 3(220)
		else {
			cout << DynPara::proName << "\t" << DynPara::objNum << endl;
			assert(false);
		}
		//store the solutions in the origin population
		population.clear();
		//cout << DynPara::objNum << "\t";
		init_uniformweight(sd); //23
		//cout << "weight\t"; cout << pops << "\t";
		init_neighbourhood();
		//cout << "neigh\t";
		init_population();
		//cout << "pop\t";
	}
	else if (varDimChange) {
		//cout << DynPara::objNum << "\t" << DynPara::dimNum << endl;
		for (int j = 0; j < population.size(); ++j) {
			population[j].indiv.reMemory();
			//cout << j << "\t" << population[j].indiv.x_var.size() << "\t" << population[j].indiv.y_obj.size() << endl;
			population[j].indiv.obj_eval(mEvas);
			mEvas++;
			record_metric();
		}
	}
	//population update
}
void TNSGA::storePSForDetectChange_DE() {

	//bool use_ep = false;

	if (use_ep) {
		//cout << EP.size() << endl;
		if (EP.size() == 0) assert(false);
		vector<CSolution> set;// (EP.size());
		for (int j = 0; j < EP.size(); ++j) {
			CSolution s;
			s.x = EP[j].x_var;
			s.f = EP[j].y_obj;
			set.push_back(s);
		}
		InfoMemory::detectedEnviSol.push_back(set);
	}
	else {
		//vector<CSolution> set(pops);
		/*vector<int> be_dominated(pops,0);
		for (int j = 0; j < pops; ++j) {
			for (int k = 0; k < pops; ++k) {
				if (j == k) continue;
				if (dominate_comp(population[j].indiv.y_obj, population[k].indiv.y_obj) == -1) {
					be_dominated[j] = 1;
					break;
				}
			}
		}*/
		vector<CSolution> set;
		for (int j = 0; j < pops; ++j) {
			//if (!InfoMemory::useSVR) {
			//	if (be_dominated[j]) continue;
			//}
			CSolution s;
			s.x = population[j].indiv.x_var;
			s.f = population[j].indiv.y_obj;
			set.push_back(s);
		}


		InfoMemory::detectedEnviSol.push_back(set);
	}
}
void TNSGA::init_detector() {
	InfoMemory::detector.clear();
	int detector_num = 10;
	InfoMemory::numFixDetector = pops * 0.1;
	InfoMemory::detector.resize(InfoMemory::numFixDetector);

	for (int k = 0; k < DynPara::dimNum; ++k) {
		InfoMemory::detector[0].x[k] = DynPara::lowBound[k];
		objectives(InfoMemory::detector[0].x, InfoMemory::detector[0].f, mEvas, true);
		mEvas += 1;
		record_metric();
	}
	for (int k = 0; k < DynPara::dimNum; ++k) {
		InfoMemory::detector[1].x[k] = DynPara::upperBound[k];
		objectives(InfoMemory::detector[1].x, InfoMemory::detector[1].f, mEvas, true);
		mEvas += 1;
		record_metric();
	}
	for (int k = 0; k < DynPara::dimNum; ++k) {
		InfoMemory::detector[2].x[k] = (DynPara::lowBound[k] + DynPara::upperBound[k]) / 2;
		objectives(InfoMemory::detector[2].x, InfoMemory::detector[2].f, mEvas, true);
		mEvas += 1;
		record_metric();
	}
	for (int j = 3; j < InfoMemory::numFixDetector; ++j) {
		for (int k = 0; k < DynPara::dimNum; ++k) {
			InfoMemory::detector[j].x[k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
		}
		objectives(InfoMemory::detector[j].x, InfoMemory::detector[j].f, mEvas, true);
		mEvas += 1;
		record_metric();
	}
}
void TNSGA::reevaluate_EP() {
	for (int j = 0; j < EP.size(); ++j) {
		EP[j].obj_eval(mEvas);
	}
	vector<int> be_dominated;
	for (int j = 0; j < EP.size(); ++j) {
		for (int k = 0; k < EP.size(); ++k) {
			if (j == k) continue;
			int d = domination(EP[j].y_obj, EP[k].y_obj);
			if (d == -1) {
				be_dominated.push_back(j);
				break;
			}
		}
	}
	for (int j = 0; j < be_dominated.size(); ++j) {
		EP.erase(EP.begin() + (be_dominated[j] - j));
	}
}
void TNSGA::print_set(const vector<CSolution>& set) {
	for (int j = 0; j < set.size(); ++j) {
		cout << j << " x:\t";
		OutputVector(set[j].x);
		cout << "\tf";
		OutputVector(set[j].f);
		cout << endl;
	}
}
void TNSGA::cal_change_past_envi(vector<solpair>& set, vector<vector<int> >& select_indepvar, int start_envi, vector<int>& self_variables) {

	int num_obj = DynPara::objNum;
	int H = 99;
	if (num_obj == 2) H = 99;
	else if (num_obj == 3) H = 12; //91 //105(H=13)
	else if (num_obj == 4) H = 6; //82(H=6) ; 120(H=7)
	else if (num_obj == 5) H = 4; //70(H=4); 126(H=5)
	else if (num_obj == 6) H = 3; //56(H=3); 126(H=4)
	else if (num_obj == 7) H = 3; //84(H=3); 210(H=4)
	else if (num_obj == 10) H = 2; //55(H=2); 220(H=3)
	else {
		cout << "The number of objectives is : " << num_obj << endl;
		assert(false);
	}
	vector<vector<double> > weights = generate_weight(num_obj, H);
	int cur_envi = InfoMemory::detectedEnviSol.size();
	int num_dim = DynPara::dimNum;
	vector<double> x_change(num_dim, 0);

	//int start_envi = 1;// cur_envi - 5;
	if (start_envi < 0) start_envi = 0;
	if (InfoMemory::detectedEnviSol.size() >= 2) {
		int j = InfoMemory::detectedEnviSol.size() - 1;
		//cout << "envi " << j << ":\t";
		//vector<vector<int> > pairs = pair_sol_between_envi(InfoMemory::algSolEachEnvi[j], InfoMemory::algSolEachEnvi[j - 1], weights);
		vector<vector<int> > pairs = pair_sol_detected_change(InfoMemory::detectedEnviSol[j], InfoMemory::detectedEnviSol[j - 1], weights, use_ep,true);
		//cout << pairs.size() << "\t";
		if (pairs.size() == 0) assert(false);
		vector<solpair> temp_set;
		for (int k = 0; k < pairs.size(); ++k) {
			//solpair p(InfoMemory::algSolEachEnvi[j - 1][pairs[k][1]].x, InfoMemory::algSolEachEnvi[j][pairs[k][0]].x, j - 1);
			solpair p(InfoMemory::detectedEnviSol[j - 1][pairs[k][1]].x, InfoMemory::detectedEnviSol[j][pairs[k][0]].x, j);
			temp_set.push_back(p);
		}
		InfoMemory::solPairEachEnvi.push_back(temp_set);
		//InfoMemory::pairIndexEachEnvi.push_back(pairs);
		//if (InfoMemory::solPairEachEnvi.size() != cur_envi - 1) {
			//assert(false);
		//}
	}

	for (int j = start_envi; j < InfoMemory::solPairEachEnvi.size(); ++j) {
		set.insert(set.end(), InfoMemory::solPairEachEnvi[j].begin(), InfoMemory::solPairEachEnvi[j].end());
#ifdef TEST
		if (false) {
			for (int k = 0; k < InfoMemory::solPairEachEnvi[j].size(); ++k) {
				cout << k << " x:\t";
				OutputVector(InfoMemory::solPairEachEnvi[j][k].x);
				cout << endl;
				cout << "y:\t";
				OutputVector(InfoMemory::solPairEachEnvi[j][k].y);
				cout << endl;
			}
		}
#endif
	}
	//cout << endl;
	//降低噪声维度: 根据预测准备率自适应调整MIC阈值；初始值等于0.6（相关维度，两个噪声维度，选择多少维度呢？）
	//加入时间维度
	//construct data set
	if (set.size() > 0) {
		clock_t start_time = clock();
		//step 0: 计算环境间变量的变化误差
		x_change = cal_x_change_bet_envi(set, num_dim);

		//step 1: 计算环境间变量的相关性
		vector<vector<double> > mic_value = cal_mic_bet_envi(set, num_dim);
		vector<vector<double> > mic_last_envi = cal_mic_in_last_envi(cur_envi - 1, num_dim);//计算上一环境内部

		//step 2: calcualte the dimension correlation relationship between different dimensions 
		if (false) {
			vector <vector<double> > mic_inside_envi = cal_mic_inside_all_envi(cur_envi, num_dim);
			//取环境间和环境内PS间维度相关性做均值
			vector<vector<double> > mean_mic = mic_value;
			for (int j = 0; j < mean_mic.size(); ++j) {
				for (int k = 0; k < mean_mic[j].size(); ++k) {
					mean_mic[j][k] = (mean_mic[j][k] + mic_inside_envi[j][k]) / 2;
				}
			}
			//check the max related dimension 
			cout << "mean corrlerated dim:================================================\n";
			for (int j = 0; j < num_dim; ++j) {
				int max_index = -1;
				int min_index = -1;
				for (int k = 0; k < num_dim; ++k) {
					if (k == j) continue;
					if (max_index == -1 || mean_mic[j][k] > mean_mic[j][max_index]) {
						max_index = k;
					}
					if (min_index == -1 || mean_mic[j][k] < mean_mic[j][min_index]) {
						min_index = k;
					}
				}
				cout << "corrlerated dim:\t" << j << "\t" << max_index << "\t" << mean_mic[j][max_index] << "\t" << min_index << "\t" << mean_mic[j][min_index] << endl;
			}
			//环境内部维度间相关性
			cout << "dim corrlerated inside envi.........................................................\n";
			for (int j = 0; j < num_dim; ++j) {
				int max_index = -1;
				int min_index = -1;
				for (int k = 0; k < num_dim; ++k) {
					if (k == j) continue;
					if (max_index == -1 || mic_inside_envi[j][k] > mic_inside_envi[j][max_index]) {
						max_index = k;
					}
					if (min_index == -1 || mic_inside_envi[j][k] < mic_inside_envi[j][min_index]) {
						min_index = k;
					}
				}
				cout << "corrlerated dim:\t" << j << "\t" << max_index << "\t" << mic_inside_envi[j][max_index] << "\t" << min_index << "\t" << mic_inside_envi[j][min_index] << endl;
			}
		}

		vector<vector<double> > cur_mean_mic = mic_value;
		for (int j = 0; j < cur_mean_mic.size(); ++j) {
			for (int k = 0; k < cur_mean_mic[j].size(); ++k) {
				cur_mean_mic[j][k] = (cur_mean_mic[j][k] + mic_last_envi[j][k]) / 2;
			}
		}
		//step 3: 根据环境间不同维度相关性和环境内部维度间相关性构造相关性图，然后确定自变量和因变量，时间维度
		//step 3.1: 环境内部（上衣环境）维度间相关性
		double threthold = 0.5;
		vector<vector<int> > cor_matrix = define_corrleration(mic_last_envi, num_dim, threthold);
		vector<vector<int> > fun_matrix = define_corrleration(mic_value, num_dim, threthold);

		//step 3.2: 根据环境内维度间的关系构造连通图,确定相关的维度块，并选取块的一个维度作为自变量
		vector<vector<int> > connect_block(num_dim);
		vector<int> block_index(num_dim);
		find_connect_dim_in_envi(cor_matrix, connect_block, block_index, num_dim);
		/*
		for (int j = 0; j < connect_block.size(); ++j) {
			cout << "block " << j << ":\t(";
			for (int k = 0; k < connect_block[j].size(); ++k) {
				cout << connect_block[j][k] << ",";
			}
			cout << ")\n";
		}
		*/
		//选择块的自变量
		vector<int> indepen_var = define_dimblock_ind_var(connect_block, x_change);
		self_variables = indepen_var;

		//step 3.3: 根据环境间的相关性确定相关维度
		vector<vector<int> > related_dims(num_dim);
		//vector<vector<int> > select_indepvar(num_dim); //每个变量相关的自变量
		select_indepvar.clear();
		select_indepvar.resize(num_dim);
		for (int j = 0; j < num_dim; ++j) {
			if (use_relevant_variables) {
				for (int k = 0; k < num_dim; ++k) {
					if (j == k) continue;
					if (fun_matrix[k][j] == 1) {
						related_dims[j].push_back(k);
					}
				}
				//related_dims[j].push_back(j);
				//根据环境内部解的相关性确定解的
				vector<int> include_block;
				for (int k = 0; k < related_dims[j].size(); ++k) {
					int bindex = block_index[related_dims[j][k]];
					if (find(include_block.begin(), include_block.end(), bindex) == include_block.end()) {
						include_block.push_back(bindex);
					}
				}
				for (int k = 0; k < include_block.size(); ++k) {
					select_indepvar[j].push_back(indepen_var[include_block[k]]);
				}
				if (find(select_indepvar[j].begin(), select_indepvar[j].end(), j) == select_indepvar[j].end()) {
					select_indepvar[j].push_back(j);;
				}
			}
			else {
				for (int k = 0; k < num_dim; ++k) {
					select_indepvar[j].push_back(k);
				}
			}
		}
		clock_t end_time = clock();
		double duration = (end_time - start_time) / CLOCKS_PER_SEC;
		InfoMemory::mic_time[InfoMemory::cur_run].push_back(duration);

#ifdef TEST
		/*
		for (int j = 0; j < select_indepvar.size(); ++j) {
			cout << "dep var " << j << ":\t";
			for (int k = 0; k < select_indepvar[j].size(); ++k) {
				cout << select_indepvar[j][k] << ",";
			}
			cout << endl;
		}
		*/
#endif

		//evaluate_predict_sol(new_sol);

		if (false) {
			cout << "dim corrlerated between envi.........................................................\n";
			//check the max related dimension 
			for (int j = 0; j < num_dim; ++j) {
				int max_index = -1;
				int min_index = -1;
				for (int k = 0; k < num_dim; ++k) {
					if (k == j) continue;
					if (max_index == -1 || mic_value[j][k] > mic_value[j][max_index]) {
						max_index = k;
					}
					if (min_index == -1 || mic_value[j][k] < mic_value[j][min_index]) {
						min_index = k;
					}
				}
				cout << "corrlerated dim:\t" << j << "\t" << max_index << "\t" << mic_value[j][max_index] << "\t" << min_index << "\t" << mic_value[j][min_index] << endl;
			}
			cout << "dim corrlerated mean the last envi:================================================\n";
			for (int j = 0; j < num_dim; ++j) {
				int max_index = -1;
				int min_index = -1;
				for (int k = 0; k < num_dim; ++k) {
					if (k == j) continue;
					if (max_index == -1 || cur_mean_mic[j][k] > cur_mean_mic[j][max_index]) {
						max_index = k;
					}
					if (min_index == -1 || cur_mean_mic[j][k] < cur_mean_mic[j][min_index]) {
						min_index = k;
					}
				}
				cout << "corrlerated dim:\t" << j << "\t" << max_index << "\t" << cur_mean_mic[j][max_index] << "\t" << min_index << "\t" << cur_mean_mic[j][min_index] << endl;
			}
		}

	}
}

void TNSGA::init_pop_correlation() {
	int cur_envi = InfoMemory::detectedEnviSol.size();
	int num_dim = DynPara::dimNum;
	vector<vector<double> > mic_last_envi = cal_mic_in_last_envi(cur_envi - 1, num_dim);

	//step 3: 根据环境间不同维度相关性和环境内部维度间相关性构造相关性图，然后确定自变量和因变量，时间维度
		//step 3.1: 环境内部（上衣环境）维度间相关性
	double threthold = 0.5;
	vector<vector<int> > cor_matrix = define_corrleration(mic_last_envi, num_dim, threthold);

	//step 3.2: 根据环境内维度间的关系构造连通图,确定相关的维度块，并选取块的一个维度作为自变量
	vector<vector<int> > connect_block(num_dim);
	vector<int> block_index(num_dim);
	find_connect_dim_in_envi(cor_matrix, connect_block, block_index, num_dim);
	/*
	for (int j = 0; j < connect_block.size(); ++j) {
		cout << "block " << j << ":\t(";
		for (int k = 0; k < connect_block[j].size(); ++k) {
			cout << connect_block[j][k] << ",";
		}
		cout << ")\n";
	}
	*/
	//选择块的自变量
	vector<double> xmin(num_dim, 1);
	vector<double> xmax(num_dim, 0);
	vector<double> range(num_dim, 0);
	for (int j = 0; j < InfoMemory::detectedEnviSol.back().size(); ++j) {
		for (int k = 0; k < num_dim; ++k) {
			if (j == 0 || InfoMemory::detectedEnviSol.back()[j].x[k] < xmin[k])
				xmin[k] = InfoMemory::detectedEnviSol.back()[j].x[k];
			if (j == 0 || InfoMemory::detectedEnviSol.back()[j].x[k] > xmax[k])
				xmax[k] = InfoMemory::detectedEnviSol.back()[j].x[k];
		}
	}
	for (int k = 0; k < num_dim; ++k) {
		range[k] = (xmax[k] - xmin[k]) / (DynPara::upperBound[k] - DynPara::lowBound[k]);
		cout << k << "\t" << range[k] << "\t" << xmin[k] << "\t" << xmax[k] << endl;
	}
	//system("PAUSE");
	//vector<int> indepen_var = define_dimblock_ind_var(connect_block, x_change);


}
void TNSGA::obtain_previous_data(vector<CSolution>& data) {
	int num_dim = DynPara::dimNum;
	int num_obj = DynPara::objNum;

	vector<double> std_pop(num_dim, 0);
	vector<double> mean_pop(num_dim, 0);
	for (int j = 0; j < pops; ++j) {
		for (int k = 0; k < num_dim; ++k) {
			mean_pop[k] += InfoMemory::detectedEnviSol.back()[j].x[k];
		}
	}
	for (int k = 0; k < num_dim; ++k) {
		mean_pop[k] /= pops;
	}
	for (int k = 0; k < num_dim; ++k) {
		for (int j = 0; j < pops; ++j) {
			std_pop[k] += (InfoMemory::detectedEnviSol.back()[j].x[k] - mean_pop[k]) * (InfoMemory::detectedEnviSol.back()[j].x[k] - mean_pop[k]);
		}
	}
	for (int k = 0; k < num_dim; ++k) {
		std_pop[k] /= sqrt(std_pop[k] / (pops - 1));
	}

	vector<vector<double> > noise_sol(pops);
	vector<int> label(pops);
	for (int i = 0; i < pops; ++i) {
		noise_sol[i].resize(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			//double noise = std_pop[k] * gaussian(0, 1);// random()* (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
			noise_sol[i][k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];// InfoMemory::detectedEnviSol.back()[i].x[k] + noise;
			if (noise_sol[i][k] < DynPara::lowBound[k])
				noise_sol[i][k] = DynPara::lowBound[k];
			if (noise_sol[i][k] > DynPara::upperBound[k])
				noise_sol[i][k] = DynPara::upperBound[k];
		}
		label[i] = 0;
	}
	for (int i = 0; i < InfoMemory::detectedEnviSol.back().size(); ++i) {
		noise_sol.push_back(InfoMemory::detectedEnviSol.back()[i].x);
		label.push_back(1);
	}
	data.clear();
	for (int i = 0; i < noise_sol.size(); ++i) {
		CSolution s;
		s.x = noise_sol[i];
		s.label = label[i];
		s.f = s.x;
		data.push_back(s);
	}
	//check the number of solutios
}
void TNSGA::obtain_new_data(vector<CSolution>& data, vector<double>& IGD, vector<double>& HV) {
	int num_dim = DynPara::dimNum;
	int num_obj = DynPara::objNum;
	//reevaluate the population
	for (int j = 0; j < pops; ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(IGD, HV);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum) {
				//cout << population.size() << "#";
				if (DynPara::proName != "SDP12" && DynPara::proName != "SDP13") assert(false);
				reactToDimChange(varDimChange, objDimChange);
			}
			//cout << "+++\t" << population.size() << "\t" << DynPara::dimNum << "\t" << DynPara::objNum << "\t";
		}
		if (mEvas >= DynPara::totalFes) break;
		population[j].indiv.obj_eval(mEvas);
		mEvas++;
	}
	evolution(IGD, HV);
	vector<vector<double> > rand_pop(pops);
	vector<vector<double> > rand_fit;

	vector<double> std_pop(num_dim, 0);
	vector<double> mean_pop(num_dim, 0);
	for (int j = 0; j < pops; ++j) {
		for (int k = 0; k < num_dim; ++k) {
			mean_pop[k] += population[j].indiv.x_var[k];
		}
	}
	for (int k = 0; k < num_dim; ++k) {
		mean_pop[k] /= pops;
	}
	for (int k = 0; k < num_dim; ++k) {
		for (int j = 0; j < pops; ++j) {
			std_pop[k] += (population[j].indiv.x_var[k] - mean_pop[k]) * (population[j].indiv.x_var[k] - mean_pop[k]);
		}
	}
	for (int k = 0; k < num_dim; ++k) {
		std_pop[k] /= sqrt(std_pop[k] / (pops - 1));
	}

	for (int i = 0; i < pops; ++i) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(IGD, HV);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum) {
				//cout << population.size() << "#";
				if (DynPara::proName != "SDP12" && DynPara::proName != "SDP13") assert(false);
				reactToDimChange(varDimChange, objDimChange);
			}
			//cout << "+++\t" << population.size() << "\t" << DynPara::dimNum << "\t" << DynPara::objNum << "\t";
		}
		if (mEvas >= DynPara::totalFes) break;
		rand_pop[i].resize(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			double noise = std_pop[k] * gaussian(0, 1);// random()* (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
			rand_pop[i][k] = population[i].indiv.x_var[k] + noise;
			if (rand_pop[i][k] < DynPara::lowBound[k])
				rand_pop[i][k] = DynPara::lowBound[k];
			if (rand_pop[i][k] > DynPara::upperBound[k])
				rand_pop[i][k] = DynPara::upperBound[k];
		}
		vector<double> f(num_obj);
		objectives(rand_pop[i], f, mEvas);
		rand_fit.push_back(f);
		mEvas++;
		record_metric();
	}
	for (int i = 0; i < pops; ++i) {
		rand_pop.push_back(population[i].indiv.x_var);
		rand_fit.push_back(population[i].indiv.y_obj);
	}
	//适应值归一化
	vector<double> max_f(num_obj);
	vector<double> min_f(num_obj);
	for (int i = 0; i < rand_pop.size(); ++i) {
		for (int k = 0; k < num_obj; ++k) {
			if (i == 0 || rand_fit[i][k] < min_f[k])
				min_f[k] = rand_fit[i][k];
			if (i == 0 || rand_fit[i][k] > max_f[k])
				max_f[k] = rand_fit[i][k];
		}
	}
	for (int i = 0; i < rand_pop.size(); ++i) {
		for (int k = 0; k < num_obj; ++k) {
			rand_fit[i][k] = (rand_fit[i][k] - min_f[k]) / (max_f[k] - min_f[k]);
		}
	}

	//非占优排序;直到选择的解大于N（种群规模）
	vector<int> visit(rand_pop.size(), 0);
	while (true) {
		int count = 0;
		vector<int> be_dominated(rand_pop.size(), 1);
		for (int j = 0; j < rand_pop.size(); ++j) {
			if (visit[j]) continue;
			for (int k = 0; k < rand_pop.size(); ++k) {
				if (j == k) continue;
				if (visit[k]) continue;
				int relation = dominate_comp(rand_fit[j], rand_fit[k]);
				if (relation == 1) be_dominated[k] = 0;
				if (relation == -1) be_dominated[j] = 0;
			}
		}
		//
		for (int j = 0; j < be_dominated.size(); ++j) {
			if (be_dominated[j] == 1)
				visit[j] = 1;
		}
		for (int j = 0; j < visit.size(); ++j)
			if (visit[j]) count += 1;
		if (count > pops) break;
	}
	//采用层次聚类操作将解划分为N类
	vector<CSolution> temp_sol;
	for (int j = 0; j < visit.size(); ++j) {
		if (visit[j]) {
			CSolution s;
			s.x = rand_pop[j];
			s.f = rand_fit[j];
			temp_sol.push_back(s);
		}
	}
	Cluster clust;
	clust.Initial(temp_sol, temp_sol.size());
	clust.Rough_Clustering(pops);
	//从每个类中选择一个解

	vector<vector<double> > new_pop;
	vector<vector<double> > new_fit;
	vector<int> new_label;
	for (int j = 0; j < clust.numbers; ++j) {
		for (int k = 0; k < clust.group[j].numbers; ++k) {
			new_pop.push_back(clust.group[j].best.x);
			new_fit.push_back(clust.group[j].best.f);
		}
	}
	//
	vector<int> dominated(new_pop.size(), 1);
	for (int j = 0; j < new_pop.size(); ++j) {
		for (int k = 0; k < new_pop.size(); ++k) {
			if (j == k) continue;
			int r = dominate_comp(new_fit[j], new_fit[k]);
			if (r == 1) dominated[k] = 0;
			if (r == -1) dominated[j] = 0;
		}
	}
	new_label = dominated;
	data.clear();
	for (int i = 0; i < new_pop.size(); ++i) {
		CSolution s;
		s.x = new_pop[i];
		s.label = new_label[i];
		s.f = s.x;
		data.push_back(s);
	}
}
int TNSGA::select_pre_cluster(Cluster& new_clst, Cluster& old_clst, int j, const vector<int>& visit) {
	int num_obj = DynPara::objNum;
	vector<double> angle(num_obj);
	vector<double> pro(num_obj);
	double sum = 0;
	double lsum = 0;
	for (int k = 0; k < num_obj; ++k) {
		if (visit[k]) continue;
		angle[k] = old_clst.Group_dist(old_clst.group[k], new_clst.group[j]);
		sum += angle[k];
	}
	for (int k = 0; k < num_obj; ++k) {
		if (visit[k]) continue;
		pro[k] = pro[k] / sum;
		lsum += log(pro[k]);
	}
	vector<double> lpro(num_obj);
	for (int k = 0; k < num_obj; ++k) {
		if (visit[k]) continue;
		lpro[k] = log(pro[k]) / lsum;
	}
	double r = random();
	sum = 0;
	int sindex = -1;
	for (int k = 0; k < num_obj; ++k) {
		if (visit[k]) continue;
		sum += lpro[k];
		if (r <= sum) {
			sindex = k;
			break;
		}
	}
	if (sindex == -1) sindex = num_obj - 1;
	return sindex;
}

vector<CSolution> TNSGA::define_knee_in_set(const vector<CSolution>& set, vector<int>& subspace_index, vector<int>& label) {
	int subspace = 10;
	int num_dim = DynPara::objNum;
	double maxvalue;
	double minvalue;
	int min_index = -1;
	int max_index = -1;
	label.resize(set.size());
	for (int j = 0; j < set.size(); ++j) label[j] = 0;
	for (int j = 0; j < set.size(); ++j) {
		if (min_index == -1 || minvalue > set[j].f[0]) {
			min_index = j;
			minvalue = set[j].f[0];
		}
		if (max_index == -1 || maxvalue < set[j].f[0]) {
			max_index = j;
			maxvalue = set[j].f[0];
		}
	}
	//define the line
	//double a = 

	double l_size = (maxvalue - minvalue) / subspace;
	vector<int> point_index; //store the point index of the 
	subspace_index.clear();
	vector<double> line(num_dim);
	double l_length = 0;
	for (int j = 0; j < num_dim; ++j) {
		line[j] = set[max_index].f[j] - set[min_index].f[j];
		l_length += line[j] * line[j];
	}
	l_length = sqrt(l_length);

	for (int j = 0; j < subspace; ++j) {
		double lb = l_size * j + minvalue;
		double ub = l_size * (j + 1) + minvalue;
		int sindex = -1;
		double maxdist = 0;
		for (int k = 0; k < set.size(); ++k) {
			if (set[k].f[0] >= lb && set[k].f[0] <= ub) {
				//calculate the d2 distance 
				vector<double> x(num_dim);
				for (int i = 0; i < num_dim; ++i) x[i] = set[k].f[i] - set[min_index].f[i];
				double d1 = innerProduct(x, line) / l_length;
				vector<double> norm_line(num_dim);
				for (int i = 0; i < num_dim; ++i) norm_line[i] = line[i] / l_length * d1;
				vector<double> y(num_dim);
				double d2 = 0;
				for (int i = 0; i < num_dim; ++i) {
					y[i] = x[i] - norm_line[i];
					d2 = d2 + y[i] * y[i];
				}
				d2 = sqrt(d2);
				if (sindex == -1 || maxdist < d2) {
					sindex = k;
					maxdist = d2;
				}
			}
		}
		if (sindex != -1) {
			point_index.push_back(sindex);
			subspace_index.push_back(j);
			label[sindex] = 1;
		}
	}


	vector<CSolution> knee;
	for (int i = 0; i < point_index.size(); ++i) {
		knee.push_back(set[point_index[i]]);
	}
	return knee;
}

//estimate new knee points according to the historical solutions
vector<vector<double> > TNSGA::estimate_new_knee_point(const vector<CSolution>& a, const vector<int>& aindex, const vector<CSolution>& b, const vector<int>& bindex) {
	vector<vector<double> > vset;
	int num_dim = DynPara::dimNum;
	for (int j = 0; j < aindex.size(); ++j) {
		int cindex = -1;
		for (int k = 0; k < bindex.size(); ++k) {
			if (aindex[j] == bindex[k]) {
				cindex = k;
				break;
			}
		}
		if (cindex == -1)continue;
		//construct vector
		vector<double> temp(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			temp[k] = a[j].x[k] - b[cindex].x[k];
		}
		//转换成极坐标表示
		vector<double> plor(num_dim);
		for (int k = 0; k < num_dim - 1; ++k) {
			double sum = 0;
			for (int m = k + 1; m < num_dim; ++m) {
				sum += temp[m] * temp[m];
			}
			sum = sqrt(sum);
			sum /= temp[k];
			plor[k] = atan(sum);
		}
		plor[num_dim - 1] = cal_length(temp);

		vset.push_back(plor);
		//randomly generate multiple deflection angles
		//-pi,pi
		int sample_num = 10;
		//generate new point
		vector<double> e_theta(num_dim);
		vector<double> e_point(num_dim);

		for (int k = 0; k < num_dim; ++k) {
			double prob = 0;
			double stheta = -1;
			for (int k = 0; k < sample_num; ++k) {
				double theta = random() * 2 * pi - pi;
				int sign = 1;
				if (theta < 0) sign = -1;
				double pvalue = exp(-sign * theta / plor[num_dim - 1]);
				if (k == 0 || pvalue > prob) {
					prob = pvalue;
					stheta = theta;
				}
			}
			e_theta[k] = stheta;
			if (k == 0) {
				e_point[k] = plor[num_dim - 1] * cos(plor[k] + stheta);
			}
			else if (k < num_dim - 1) {
				double accumate_sum = 1;
				for (int m = 0; m < k - 1; ++m) {
					accumate_sum *= sin(e_theta[m] + plor[m]);
				}
				e_point[k] = accumate_sum * plor[num_dim - 1] * cos(e_theta[k - 1] + plor[k - 1]);
			}
			else {
				double accumate_sum = 1;
				for (int m = 0; m < k - 1; ++m) {
					accumate_sum *= sin(e_theta[m] + plor[m]);
				}
				e_point[k] = accumate_sum * plor[num_dim - 1];
			}
		}

		//estimated point
		vector<double> new_point(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			new_point[k] = b[cindex].x[k] + e_point[k];
		}
		vset.push_back(new_point);
	}
	return vset;
}

vector<CSolution> TNSGA::generate_source_knee_data(const vector<CSolution>& set, const vector<int>& label) {
	int num_dim = DynPara::dimNum;
	int num_obj = DynPara::objNum;
	vector<vector<double> > noise_sol(pops);
	vector<int> noise_label(pops);
	//vector<int> label(pops);
	for (int i = 0; i < pops; ++i) {
		noise_sol[i].resize(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			noise_sol[i][k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
		}
		noise_label[i] = 0;
	}
	for (int i = 0; i < set.size(); ++i) {
		noise_sol.push_back(set[i].x);
		noise_label.push_back(label[i]);
	}
	vector<CSolution> data;
	for (int i = 0; i < noise_sol.size(); ++i) {
		CSolution s;
		s.x = noise_sol[i];
		s.label = noise_label[i];
		s.f = s.x;
		data.push_back(s);
	}
	return data;
}

vector<CSolution> TNSGA::generate_target_knee_data(const vector<vector<double> >& estimated_knee) {
	int num_dim = DynPara::dimNum;
	int num_obj = DynPara::objNum;
	vector<vector<double> > noise_sol(pops);
	vector<int> noise_label(pops);
	//vector<int> label(pops);
	for (int i = 0; i < pops; ++i) {
		noise_sol[i].resize(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			noise_sol[i][k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
		}
		noise_label[i] = 0;
	}
	for (int i = 0; i < estimated_knee.size(); ++i) {
		noise_sol.push_back(estimated_knee[i]);
		noise_label.push_back(1);
	}
	vector<CSolution> data;
	for (int i = 0; i < noise_sol.size(); ++i) {
		CSolution s;
		s.x = noise_sol[i];
		s.label = noise_label[i];
		s.f = s.x;
		data.push_back(s);
	}
	return data;
}

void TNSGA::init_pop_use_sol(const vector<vector<double> >& predict_knee, vector<double>& IGD, vector<double>& HV) {
	for (int j = 0; j < pops; ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(IGD, HV);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		if (j < predict_knee.size()) {
			population[j].indiv.x_var = predict_knee[j];
			int num_dim = population[j].indiv.x_var.size();
			for (int k = 0; k < num_dim; ++k) {
				if (population[j].indiv.x_var[k] < DynPara::lowBound[k])
					population[j].indiv.x_var[k] = DynPara::lowBound[k];
				if (population[j].indiv.x_var[k] > DynPara::upperBound[k])
					population[j].indiv.x_var[k] = DynPara::upperBound[k];
			}
			population[j].indiv.obj_eval(mEvas);
		}
		else {
			population[j].indiv.rnd_init();
			population[j].indiv.obj_eval(mEvas);
		}
		if (use_ep) update_EP(population[j].indiv);
		update_reference(population[j].indiv);
		mEvas++;
		record_metric();
	}
}

void TNSGA::transfer_knee_SVM(vector<double>& IGD, vector<double>& HV) {
	//define the knee points in the previous environments
	int cur_envi = InfoMemory::detectedEnviSol.size();
	int num_dim = DynPara::dimNum;
	int subspace = 10;
	vector<int> pre_k_index; vector<int> last_k_index;
	vector<int> pre_is_knee;
	vector<int> last_is_knee; //record whether the solution is a knee point in the PS
	vector<CSolution> pre_knee = define_knee_in_set(InfoMemory::detectedEnviSol[cur_envi - 2], pre_k_index, pre_is_knee);
	vector<CSolution> last_knee = define_knee_in_set(InfoMemory::detectedEnviSol[cur_envi - 1], last_k_index, last_is_knee);

	vector<vector<double> > estimate_knee = estimate_new_knee_point(pre_knee, pre_k_index, last_knee, last_k_index);

	//the obtained POS
	vector<CSolution> olddata = generate_source_knee_data(InfoMemory::detectedEnviSol[cur_envi - 1], last_is_knee);
	vector<CSolution> newdata = generate_target_knee_data(estimate_knee);
	int new_num = newdata.size();

	newdata.insert(newdata.end(), olddata.begin(), olddata.end());
	//train the SVM
	int qmax = 6;
	double cnk = 1;
	double ck = ((olddata.size() - subspace) * cnk) / subspace;
	int n1 = olddata.size();
	vector<double> d_weight(newdata.size());
	for (int j = 0; j < newdata.size(); ++j) {
		if (j >= new_num && newdata[j].label == 1) d_weight[j] = 1.0 / subspace;
		else if (j >= new_num && newdata[j].label == 0) d_weight[j] = 1.0 / (olddata.size() - subspace);
		else d_weight[j] = 1.0 / new_num;
	}
	Eigen::MatrixXf X(newdata.size(), num_dim);
	vector<int> label;
	vector<int> d_index; //is from source or target domain?
	for (int k = 0; k < newdata.size(); ++k) {
		for (int l = 0; l < num_dim; ++l) {
			X(k, l) = newdata[k].x[l];
		}
		label.push_back(newdata[k].label);
		if (k < new_num) d_index.push_back(1);
		else d_index.push_back(0);
	}
	vector<esvm::SVMClassifier> weak_svm(qmax);
	vector<double> predict(newdata.size(), 1);
	double threshold = 1;
	vector<double> svm_betaq(qmax);
	for (int q = 0; q < qmax; ++q) {
		weak_svm[q].train(X, label, d_weight);
		vector<int> result(newdata.size());
		weak_svm[q].test(X, result);

		double errorq = 0;
		double total_w = 0;
		int mis_new_num = 0;
		int mis_old_num = 0;
		for (int k = 0; k < newdata.size(); ++k) {
			if (d_index[k] == 1) {
				errorq += d_weight[k] * fabs(result[k] - label[k]);
				total_w += d_weight[k];
				if (label[k] == 1 && result[k] != 1) mis_new_num++;
			}
			else {
				if (label[k] == 1 && result[k] != 1) mis_old_num++;
			}
		}
		errorq /= total_w;

		double lou = (ck * mis_old_num + 0.001) / (cnk * mis_new_num + 0.001);
		double betaq = errorq / (1 - errorq);
		double beta = 1 / (1 + sqrt(2 * log((double)(n1) / qmax)));

		for (int k = 0; k < d_weight.size(); ++k) {
			if (d_index[k] == 1) d_weight[k] = d_weight[k] * pow(betaq, -fabs(result[k] - label[k]));
			else {
				if (label[k] == 0) {
					d_weight[k] = d_weight[k] * pow(beta, -fabs(result[k] - label[k]));
				}
				else {
					d_weight[k] = d_weight[k] * lou * pow(beta, -fabs(result[k] - label[k]));
				}
			}
		}
		svm_betaq[q] = betaq;
	}
	//组成一个强分类器,测试解
	int sample_num = 100 * pops;
	vector<vector<double> > rand_sol(sample_num);
	vector<int> pre_label(rand_sol.size(), 0);
	for (int j = 0; j < rand_sol.size(); ++j) {
		rand_sol[j].resize(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			rand_sol[j][k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
		}

	}
	Eigen::MatrixXf randX(rand_sol.size(), num_dim);
	for (int k = 0; k < rand_sol.size(); ++k) {
		for (int l = 0; l < num_dim; ++l) {
			randX(k, l) = rand_sol[k][l];
		}
	}
	vector<int> randY(rand_sol.size(), 0);
	vector<double> pvalue(rand_sol.size(), 0);
	for (int q = 0; q < qmax; ++q) {
		weak_svm[q].test(randX, randY);
		for (int j = 0; j < rand_sol.size(); ++j) {
			pvalue[j] += log(1 / svm_betaq[q]) * randY[j];//    pow(svm_betaq[q], -randY[j]);
		}
	}
	vector<vector<double> > predict_knee;
	for (int j = 0; j < rand_sol.size(); ++j) {
		if (pvalue[j] >= 0) predict_knee.push_back(rand_sol[j]);
	}
	//

	//使用初始种群
	vector<CSolution> pre_sol(predict_knee.size());
	for (int j = 0; j < predict_knee.size(); ++j) {
		pre_sol[j].x = predict_knee[j];
		for (int k = 0; k < DynPara::dimNum; ++k) {
			if (pre_sol[j].x[k] < DynPara::lowBound[k]) pre_sol[j].x[k] = DynPara::lowBound[k];
			if (pre_sol[j].x[k] > DynPara::upperBound[k]) pre_sol[j].x[k] = DynPara::upperBound[k];
		}
		objectives(pre_sol[j].x, pre_sol[j].f, mEvas);
	}
	InfoMemory::predictEnviSol.push_back(pre_sol);
	init_pop_use_sol(predict_knee, IGD, HV);
}
void TNSGA::transfer_cluster_SVM(vector<double>& IGD, vector<double>& HV) {
	//generate noise solutions
	int num_dim = DynPara::dimNum;
	int num_obj = DynPara::objNum;
	vector<CSolution> olddata;
	obtain_previous_data(olddata);

	//
	//evolution(IGD, HV);
	vector<CSolution> newdata;
	obtain_new_data(newdata, IGD, HV);
	vector<CSolution> best_set; //新数据中的非占优解
	for (int j = 0; j < newdata.size(); ++j) {
		if (newdata[j].label == 1)
			best_set.push_back(newdata[j]);
	}
	//去除里面的非占优解

	vector<double> mean_best(num_dim, 0);
	vector<double> std_best(num_dim, 0);
	for (int j = 0; j < num_dim; ++j) {
		for (int i = 0; i < best_set.size(); ++i) {
			mean_best[j] = mean_best[j] + best_set[i].x[j];
		}
		mean_best[j] = mean_best[j] / best_set.size();
	}

	//基于聚类的迁移
	//cout << olddata.size() << "\t";
	Cluster clust;
	clust.Initial(olddata, olddata.size());
	clust.Rough_Clustering(num_obj);
	/*
	int neg_num = 0;
	//cout << olddata.size() << "\t" << clust.group.size() << endl;
	for (int j = 0; j < num_obj; ++j) {
		//cout << "group " << j << "\t" << clust.group[j].members.size() << "\t" << clust.group[j].numbers << endl;
		for (int k = 0; k < clust.group[j].numbers; ++k) {
			if (clust.group[j].members[k].label == 0) neg_num++;
		}
	}
	cout << neg_num << endl;
	*/

	Cluster new_clust;
	new_clust.Initial(newdata, newdata.size());
	new_clust.Rough_Clustering(num_obj);

	/*
	//cout << newdata.size() << "\t" << new_clust.numbers << endl;
	for (int j = 0; j < num_obj; ++j) {
		//cout << "group " << new_clust.group[j].members.size() << "\t" << new_clust.group[j].numbers << endl;
		for (int k = 0; k < new_clust.group[j].numbers; ++k) {
			if (new_clust.group[j].members[k].label == 0) neg_num++;
		}
	}
	*/

	int qmax = 6;
	vector<int> visit(num_obj, 0);
	vector<vector<double> > init_pop;

	for (int j = 0; j < num_obj; ++j) {
		int sindex = select_pre_cluster(new_clust, clust, j, visit);
		visit[sindex] = 1;
		vector<vector<double> > data;
		vector<int> label;
		vector<int> d_index;
		vector<double> d_weight;
		int neg_num = 0;
		for (int k = 0; k < new_clust.group[j].numbers; ++k) {
			data.push_back(new_clust.group[j].members[k].x);
			label.push_back(new_clust.group[j].members[k].label);
			if (new_clust.group[j].members[k].label == 0) neg_num++;
			d_weight.push_back(1);
			d_index.push_back(1);
		}
		for (int k = 0; k < clust.group[sindex].numbers; ++k) {
			data.push_back(clust.group[sindex].members[k].x);
			label.push_back(clust.group[sindex].members[k].label);
			if (clust.group[sindex].members[k].label == 0) neg_num++;
			d_weight.push_back(1);
			d_index.push_back(0);
		}
		//cout << "########" << neg_num << "\t";
		Eigen::MatrixXf X(data.size(), data[0].size());
		for (int k = 0; k < data.size(); ++k) {
			for (int l = 0; l < num_dim; ++l) {
				X(k, l) = data[k][l];
			}
		}
		vector<esvm::SVMClassifier> weak_svm(qmax);
		vector<double> predict(data.size(), 1);
		double threshold = 1;
		vector<double> svm_betaq(qmax);
		for (int q = 0; q < qmax; ++q) {
			weak_svm[q].train(X, label, d_weight);
			vector<int> result(data.size());
			weak_svm[q].test(X, result);

			double errorq = 0;
			double total_w = 0;
			for (int k = 0; k < data.size(); ++k) {
				if (d_index[k] == 1) {
					errorq += d_weight[k] * fabs(result[k] - label[k]);
					total_w += d_weight[k];
				}
			}
			errorq /= total_w;
			int n1 = clust.group[sindex].numbers;
			double betaq = errorq / (1 - errorq);
			double beta = 1 / (1 + sqrt(2 * log((double)n1 / qmax)));

			for (int k = 0; k < d_weight.size(); ++k) {
				if (d_index[k] == 1) d_weight[k] = d_weight[k] * pow(betaq, -fabs(result[k] - label[k]));
				else d_weight[k] = d_weight[k] * pow(beta, -fabs(result[k] - label[k]));
			}
			if (q >= (double)qmax / 2 - 1) {
				threshold *= pow(betaq, -0.5);
			}
			svm_betaq[q] = betaq;
		}
		//组成一个强分类器,测试解
		vector<vector<double> > rand_sol(90 * pops);
		vector<int> pre_label(rand_sol.size(), 0);
		for (int i = 0; i < rand_sol.size(); ++i) {
			rand_sol[i].resize(num_dim);
			for (int k = 0; k < num_dim; ++k) {
				rand_sol[i][k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
			}

		}
		//对随机解加高斯噪声;在已有的历史解上做
		//选择聚类中的最好解组成集合，用来进行高斯分布采样

		for (int j = 0; j < num_dim; ++j) {
			for (int i = 0; i < best_set.size(); ++i) {
				std_best[j] = std_best[j] + (mean_best[j] - best_set[i].x[j]) * (mean_best[j] - best_set[i].x[j]);
			}
			if (best_set.size() > 1) std_best[j] = sqrt(std_best[j] / best_set.size());
		}
		for (int l = 0; l < 10; ++l) {
			vector<int> index_list(best_set.size());
			for (int i = 0; i < best_set.size(); ++i) index_list[i] = i;
			random_shuffle(index_list.begin(), index_list.end());
			for (int i = 0; i < best_set.size(); ++i) {
				vector<double> temp(num_dim);
				for (int d = 0; d < num_dim; ++d) {
					double noise = std_best[d] * gaussian(0, 1);
					temp[d] = best_set[index_list[i]].x[d] + noise;
					if (temp[d] < DynPara::lowBound[d])
						temp[d] = DynPara::lowBound[d];
					if (temp[d] > DynPara::upperBound[d])
						temp[d] = DynPara::upperBound[d];
				}
				rand_sol.push_back(temp);
			}
		}

		//设计解的噪声
		Eigen::MatrixXf randX(rand_sol.size(), num_dim);
		for (int k = 0; k < rand_sol.size(); ++k) {
			for (int l = 0; l < num_dim; ++l) {
				randX(k, l) = rand_sol[k][l];
			}
		}
		vector<int> randY(rand_sol.size(), 0);
		vector<double> pvalue(rand_sol.size(), 1);
		for (int q = (double)qmax / 2 - 1; q < qmax; ++q) {
			weak_svm[q].test(randX, randY);
			for (int j = 0; j < rand_sol.size(); ++j) {
				pvalue[j] *= pow(svm_betaq[q], -randY[j]);
			}
		}
		//选择前N/m个解
		int count = 0;
		for (int j = 0; j < rand_sol.size(); ++j) {
			if (pvalue[j] >= threshold) {
				randY[j] = 1;
				init_pop.push_back(rand_sol[j]);
				count++;
				if (count >= pops / num_obj) break;
			}
			else randY[j] = 0;
		}
		if (count == 0) {
			int max_num = pops / num_obj;
			if (max_num > new_clust.group[j].numbers) max_num = new_clust.group[j].numbers;
			for (int k = 0; k < max_num; ++k) {
				init_pop.push_back(new_clust.group[j].members[k].x);
			}
		}
	}
	//使用初始种群
	vector<CSolution> pre_sol(init_pop.size());
	for (int j = 0; j < init_pop.size(); ++j) {
		pre_sol[j].x = init_pop[j];
		for (int k = 0; k < DynPara::dimNum; ++k) {
			if (pre_sol[j].x[k] < DynPara::lowBound[k]) pre_sol[j].x[k] = DynPara::lowBound[k];
			if (pre_sol[j].x[k] > DynPara::upperBound[k]) pre_sol[j].x[k] = DynPara::upperBound[k];
		}
		objectives(pre_sol[j].x, pre_sol[j].f, mEvas);
	}
	InfoMemory::predictEnviSol.push_back(pre_sol);
	init_pop_use_sol(init_pop, IGD, HV);
}

void TNSGA::presearch(vector<double>& IGD, vector<double>& HV, vector<int>& label, vector<vector<double> >& set) {
	//p= 24 for 2-objective; //p = 9 for 3-objective
	int p;
	int num_obj = DynPara::objNum;
	if (num_obj == 2) p = 24;
	else if (num_obj == 3) p = 9;
	else if (num_obj == 5) p = 5;
	double lou = 0.9;
	vector<vector<double> > pre_weights = generate_weight(num_obj, p);
	int num_weight = pre_weights.size();
	vector<TIndividual> rand_pop(num_weight * 2);
	vector<double> refer(num_obj);
	int num_dim = DynPara::dimNum;

	for (int j = 0; j < rand_pop.size(); ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(IGD, HV);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		for (int k = 0; k < num_dim; ++k) {
			rand_pop[j].x_var[k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
		}
		objectives(rand_pop[j].x_var, rand_pop[j].y_obj, mEvas);
		for (int k = 0; k < num_obj; ++k) {
			if (j == 0 || rand_pop[j].y_obj[k] < refer[k])  refer[k] = rand_pop[j].y_obj[k];
		}
		mEvas++;
		record_metric();
		//
	}
	//random population to obtain a presearch procedure
	double d1, d2;
	vector<int> visit(rand_pop.size(), 0);
	vector<vector<int> > R_ind(pre_weights.size()); //the individual assigned to each weight
	for (int j = 0; j < pre_weights.size(); ++j) {
		int min_index = -1;
		int sec_index = -1;
		double min_value = -1;
		double sec_value = -1;
		for (int i = 0; i < rand_pop.size(); ++i) {
			if (visit[i]) continue;
			double d1 = cal_pbi(rand_pop[i].y_obj, pre_weights[j], refer, d1, d2);
			if (min_index == -1 || d1 < min_value) {
				if (min_index == -1) {
					min_index = i;
					min_value = d1;
				}
				else {
					sec_index = min_index;
					sec_value = min_value;
					min_index = i;
					min_value = d1;
				}
			}
			else {
				if (sec_index == -1 || d1 < sec_value) {
					sec_index = i;
					sec_value = d1;
				}
			}
		}
		visit[min_index] = 1;
		visit[sec_index] = 1;
		R_ind[j].push_back(min_index);
		R_ind[j].push_back(sec_index);
	}
	//define the neighbor of each reference vector
	vector<int> neigh(pre_weights.size());
	for (int j = 0; j < pre_weights.size(); ++j) {
		int min_index = -1;
		double min_dist = 0;
		for (int k = 0; k < pre_weights.size(); ++k) {
			if (j == k) continue;
			double d = distanceVector(pre_weights[j], pre_weights[k]);
			if (min_index == -1 || d < min_dist) {
				min_index = k;
				min_dist = d;
			}
		}
		//init_neighbourhood();
		neigh[j] = min_index;
	}
	//generate new individuals
	while (true) {
		//pre_search the individuals
		vector<TIndividual> Q;
		for (int j = 0; j < pre_weights.size(); ++j) {
			if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
				metric(IGD, HV);
				storeFinalPop();
				int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
				introduceDynamic(mEvas);
				defineBoundary(DynPara::proName);
				bool varDimChange = false;
				bool objDimChange = false;
				if (preDimNum != DynPara::dimNum) varDimChange = true;
				if (preObjNum != DynPara::objNum) objDimChange = true;
				if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
					reactToDimChange(varDimChange, objDimChange);
			}
			if (mEvas >= DynPara::totalFes) break;
			int n = neigh[j];
			TIndividual child, child2;
			realbinarycrossover(rand_pop[R_ind[j][0]], rand_pop[R_ind[n][0]], child, child2);
			child.obj_eval(mEvas);
			Q.push_back(child);
			mEvas++;
			record_metric();

			for (int k = 0; k < 2; ++k) {
				if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
					metric(IGD, HV);
					storeFinalPop();
					int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
					introduceDynamic(mEvas);
					defineBoundary(DynPara::proName);
					bool varDimChange = false;
					bool objDimChange = false;
					if (preDimNum != DynPara::dimNum) varDimChange = true;
					if (preObjNum != DynPara::objNum) objDimChange = true;
					if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
						reactToDimChange(varDimChange, objDimChange);
				}
				if (mEvas >= DynPara::totalFes) break;
				realmutation(rand_pop[R_ind[j][k]], 1.0 / DynPara::dimNum);
				rand_pop[R_ind[j][k]].obj_eval(mEvas);
				Q.push_back(rand_pop[R_ind[j][k]]);
				mEvas++;
				record_metric();
			}
			//
		}
		vector<int> visit(Q.size(), 0);
		R_ind.clear();
		R_ind.resize(pre_weights.size());
		double alg_d, perp_d;
		for (int j = 0; j < pre_weights.size(); ++j) {
			int min_index = -1;
			int sec_index = -1;
			double min_value = -1;
			double sec_value = -1;
			for (int i = 0; i < Q.size(); ++i) {
				if (visit[i]) continue;
				double d1 = cal_pbi(Q[i].y_obj, pre_weights[j], refer, alg_d, perp_d);
				if (min_index == -1 || d1 < min_value) {
					if (min_index == -1) {
						min_index = i;
						min_value = d1;
					}
					else {
						sec_index = min_index;
						sec_value = min_value;
						min_index = i;
						min_value = d1;
					}
				}
				else {
					if (sec_index == -1 || d1 < sec_value) {
						sec_index = i;
						sec_value = d1;
					}
				}
			}
			visit[min_index] = 1;
			visit[sec_index] = 1;
			R_ind[j].push_back(min_index);
			R_ind[j].push_back(sec_index);

		}
		//
		int count = 0;
		for (int j = 0; j < Q.size(); ++j) {
			if (visit[j]) {
				bool flag = true;
				for (int i = 0; i < rand_pop.size(); ++i) {
					if (domination(rand_pop[i].y_obj, Q[j].y_obj) == 1) {
						flag = false;
						break;
					}
				}
				if (flag) count += 1;
			}
		}
		double coverage = (double)count / (pre_weights.size() * 2);
		for (int i = 0; i < pre_weights.size(); ++i) {
			rand_pop.push_back(Q[R_ind[i][0]]);
			rand_pop.push_back(Q[R_ind[i][1]]);
			R_ind[i][0] = i * 2;
			R_ind[i][1] = i * 2 + 1;
		}
		if (coverage < lou) break;
	}

	set.clear();
	for (int i = 0; i < rand_pop.size(); ++i) {
		set.push_back(rand_pop[i].x_var);
	}
	label.clear();
	label.resize(set.size(), 1);
	for (int i = 0; i < rand_pop.size(); ++i) {
		for (int j = 0; j < rand_pop.size(); ++j) {
			if (i == j) continue;
			int relation = domination(rand_pop[i].y_obj, rand_pop[j].y_obj);
			if (relation == 1) label[j] = 0;
			if (relation == -1) label[i] = 0;
		}
	}
}

void TNSGA::transfer_ind_SVM(vector<double>& IGD, vector<double>& HV) {
	//reinitialize_pop(IGD, HV, 1.0);
	vector<vector<double> > target_sol;
	vector<int> target_label;
	vector<int> d_index; //from source or target
	vector<vector<double> > source_sol;
	vector<int> source_label;
	presearch(IGD, HV, target_label, target_sol);
	for (int i = 0; i < target_label.size(); ++i)
		d_index.push_back(1);

	for (int i = 0; i < InfoMemory::detectedEnviSol.back().size(); ++i) {
		source_sol.push_back(InfoMemory::detectedEnviSol.back()[i].x);
		source_label.push_back(1);
		d_index.push_back(0);
	}
	int num_dim = DynPara::dimNum;
	for (int i = 0; i < pops; ++i) {
		vector<double> temp(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			temp[k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
		}
		source_sol.push_back(temp);
		source_label.push_back(0);
		d_index.push_back(0);
	}

	//find the proposed method
	Eigen::MatrixXf X(source_sol.size() + target_sol.size(), num_dim);
	int data_size = source_sol.size() + target_sol.size();
	vector<double> d_weight(data_size, 1);
	for (int k = 0; k < target_sol.size(); ++k) {
		for (int l = 0; l < num_dim; ++l) {
			X(k, l) = target_sol[k][l];
		}
	}
	for (int k = 0; k < source_sol.size(); ++k) {
		for (int l = 0; l < num_dim; ++l) {
			X(k + target_sol.size(), l) = source_sol[k][l];
		}
	}
	vector<int> label = target_label;
	label.insert(label.end(), source_label.begin(), source_label.end());
	int qmax = 5;
	vector<esvm::SVMClassifier> weak_svm(qmax);
	vector<double> predict(data_size, 1);
	double threshold = 1;
	vector<double> svm_betaq(qmax);
	for (int q = 0; q < qmax; ++q) {
		weak_svm[q].train(X, label, d_weight);
		vector<int> result(data_size);
		weak_svm[q].test(X, result);

		double errorq = 0;
		double total_w = 0;
		for (int k = 0; k < data_size; ++k) {
			if (d_index[k] == 1) {
				errorq += d_weight[k] * fabs(result[k] - label[k]);
				total_w += d_weight[k];
			}
		}
		errorq /= total_w;
		int n1 = source_sol.size();
		double betaq = log((1 - errorq) / errorq) / 2;
		double beta = log(1 / (1 + sqrt(2 * log(qmax)))) / 2;

		for (int k = 0; k < d_weight.size(); ++k) {
			if (d_index[k] == 1) d_weight[k] = d_weight[k] * exp(betaq * fabs(result[k] - label[k]));
			else d_weight[k] = d_weight[k] * exp(beta * fabs(result[k] - label[k]));
		}
		svm_betaq[q] = betaq;
	}
	//组成一个强分类器,测试解
	vector<vector<double> > rand_sol(100 * pops);
	vector<int> pre_label(rand_sol.size(), 0);
	for (int j = 0; j < rand_sol.size(); ++j) {
		rand_sol[j].resize(num_dim);
		for (int k = 0; k < num_dim; ++k) {
			rand_sol[j][k] = random() * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
		}

	}
	Eigen::MatrixXf randX(rand_sol.size(), num_dim);
	for (int k = 0; k < rand_sol.size(); ++k) {
		for (int l = 0; l < num_dim; ++l) {
			randX(k, l) = rand_sol[k][l];
		}
	}
	vector<int> randY(rand_sol.size(), 0);
	vector<double> pvalue(rand_sol.size(), 0);
	for (int q = (double)qmax / 2 - 1; q < qmax; ++q) {
		weak_svm[q].test(randX, randY);
		for (int j = 0; j < rand_sol.size(); ++j) {
			pvalue[j] += svm_betaq[q] * randY[j];
		}
	}
	//选择前N/m个解
	vector<vector<double> > init_pop;
	for (int j = 0; j < rand_sol.size(); ++j) {
		if (pvalue[j] >= 0) {
			randY[j] = 1;
			init_pop.push_back(rand_sol[j]);
		}
		else randY[j] = 0;
	}

	//使用初始种群
	vector<CSolution> pre_sol(init_pop.size());
	for (int j = 0; j < init_pop.size(); ++j) {
		pre_sol[j].x = init_pop[j];
		for (int k = 0; k < DynPara::dimNum; ++k) {
			if (pre_sol[j].x[k] < DynPara::lowBound[k]) pre_sol[j].x[k] = DynPara::lowBound[k];
			if (pre_sol[j].x[k] > DynPara::upperBound[k]) pre_sol[j].x[k] = DynPara::upperBound[k];
		}
		objectives(pre_sol[j].x, pre_sol[j].f, mEvas);
	}
	InfoMemory::predictEnviSol.push_back(pre_sol);
	init_pop_use_sol(init_pop, IGD, HV);
}
void TNSGA::predict_by_SVR(vector<double>& IGD, vector<double>& HV) {
	//update the parameter in the proposed method 
	//each subproblem; each dimension
	int num_dim = DynPara::dimNum;
	int q = InfoMemory::q;
	int s = InfoMemory::detectedEnviSol.size();
	vector<vector<double> > predict_sol;
	vector<CSolution> pre_sol(pops);
	for (int i = 0; i < pops; ++i) {
		vector<double> y(num_dim);
		for (int j = 0; j < num_dim; ++j) {
			vector<vector<double> > train_in;
			vector<double> train_out;
			vector<double> d_weight;
			for (int k = q; k < s; ++k) {
				vector<double> temp(q);
				for (int l = 0; l < q; ++l) {
					int m = k - q + l;
					temp[l] = InfoMemory::detectedEnviSol[m][i].x[j];
				}
				train_in.push_back(temp);
				train_out.push_back(InfoMemory::detectedEnviSol[k][i].x[j]);
				d_weight.push_back(1);
				//temp[q] = InfoMemory::detectedEnviSol[k][i].x[j];
			}
			if (train_in.size() <= 0) assert(false);
			//train SVR
			Eigen::MatrixXf X(train_in.size(), q);
			for (int k = 0; k < train_in.size(); ++k) {
				for (int l = 0; l < q; ++l) {
					X(k, l) = train_in[k][l];
				}
			}
			esvm::SVMRegression svr;
			svr.setPara(InfoMemory::C, InfoMemory::eplison, InfoMemory::gamma);
			svr.train(X, train_out, d_weight);

			Eigen::MatrixXf TX(1, q);
			//cout << TX.rows() << endl;
			for (int k = s - q; k < s; ++k) {
				TX(0, k - (s - q)) = InfoMemory::detectedEnviSol[k][i].x[j];
			}
			vector<double> py(1);
			svr.test(TX, py);
			y[j] = py[0];
			if (y[j] < DynPara::lowBound[j]) y[j] = DynPara::lowBound[j];
			if (y[j] > DynPara::upperBound[j]) y[j] = DynPara::upperBound[j];

		}
		predict_sol.push_back(y);
		pre_sol[i].x = y;
		objectives(y, pre_sol[i].f, mEvas);
	}
	InfoMemory::predictEnviSol.push_back(pre_sol);
	init_pop_use_sol(predict_sol, IGD, HV);
}

vector<int> TNSGA::rank_sol_crowd_distance(const vector<CSolution>& set) {
	vector<double> dist(set.size(), 0);
	int num_obj = DynPara::objNum;
	for (int k = 0; k < num_obj; ++k) {
		vector<int> rank(set.size());
		vector<int> visit(set.size(), 0);
		for (int j = 0; j < set.size(); ++j) {
			int min_index = -1;
			for (int i = 0; i < set.size(); ++i) {
				if (visit[i]) continue;
				if (min_index == -1 || set[i].f[k] < set[min_index].f[k]) {
					min_index = i;
				}
			}
			//选择最好的个体
			visit[min_index] = 1;
			rank[j] = min_index;
		}
		//设置最好的个体
		for (int j = 0; j < rank.size(); ++j) {
			if (j == 0 || j == rank.size() - 1) {
				dist[rank[j]] = 1e10;
			}
			else {
				dist[rank[j]] = dist[rank[j]] + (set[rank[j + 1]].f[k] - set[rank[j - 1]].f[k]);
			}
		}
	}
	//根据距离排序
	vector<int> c_rank(set.size());
	vector<int> visit(set.size(), 0);
	for (int j = 0; j < set.size(); ++j) {
		int min_index = -1;
		for (int k = 0; k < set.size(); ++k) {
			if (visit[k]) continue;
			if (min_index == -1 || dist[min_index] < dist[k]) {
				min_index = k;
			}
		}
		c_rank[j] = min_index;
		visit[min_index] = 1;
	}
	//
	return c_rank;
}

void TNSGA::init_half_pop_by_predict(const vector<CSolution>& predict_sol, vector<double>& IGD, vector<double>& HV) {
	//predict solutions acccording to crowding distance
	vector<int> predict_rank = rank_sol_crowd_distance(predict_sol);
	vector<int> replace_index(pops);
	for (int j = 0; j < pops; ++j) replace_index[j] = j;
	random_shuffle(replace_index.begin(), replace_index.end());
	for (int m = 0; m < pops; ++m) {
		int j = replace_index[m];
		if (m < pops / 2) {
			if (m < predict_sol.size()) {
				int rindex = replace_index[j];
				population[j].indiv.x_var = predict_sol[predict_rank[m]].x;
				population[j].indiv.y_obj = predict_sol[predict_rank[m]].f;

				update_reference(population[j].indiv);
			}
			else {
				if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
					metric(IGD, HV);
					storeFinalPop();
					int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
					introduceDynamic(mEvas);
					defineBoundary(DynPara::proName);
					bool varDimChange = false;
					bool objDimChange = false;
					if (preDimNum != DynPara::dimNum) varDimChange = true;
					if (preObjNum != DynPara::objNum) objDimChange = true;
					if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
						reactToDimChange(varDimChange, objDimChange);
				}
				if (mEvas >= DynPara::totalFes) break;
				population[j].indiv.rnd_init();
				population[j].indiv.obj_eval(mEvas);
				update_reference(population[j].indiv);
				record_metric();
			}
		}
		else {
			if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
				metric(IGD, HV);
				storeFinalPop();
				int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
				introduceDynamic(mEvas);
				defineBoundary(DynPara::proName);
				bool varDimChange = false;
				bool objDimChange = false;
				if (preDimNum != DynPara::dimNum) varDimChange = true;
				if (preObjNum != DynPara::objNum) objDimChange = true;
				if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
					reactToDimChange(varDimChange, objDimChange);
			}
			if (mEvas >= DynPara::totalFes) break;
			population[j].indiv.obj_eval(mEvas);
			update_reference(population[j].indiv);
			record_metric();
		}
		if (use_ep) update_EP(population[j].indiv);
	}
}

void TNSGA::output_pop_errors() {
	int num_dim = DynPara::dimNum;
	vector<double> dim_error(num_dim, 0);
	for (int i = 0; i < pops; ++i) {
		vector<double> ox = population[i].indiv.x_var;
		vector<double> of(DynPara::objNum);
		objectives(ox, of, mEvas);

		vector<double> real_y(num_dim);
		vector<double> f(DynPara::objNum);
		getOptimalSolution(DynPara::proName, ox, real_y, mEvas);
		objectives(real_y, f, mEvas);

		for (int k = 0; k < num_dim; ++k) {
			dim_error[k] += fabs(real_y[k] - ox[k]);
		}
	}
	cout << "populat:("; //predict
	for (int i = 0; i < dim_error.size(); ++i) cout << dim_error[i] / pops << ",";
	cout << ")\n";
}

//Solving Dynamic Multiobjective Problem via Autoencoding Evolutionary Search, tcyb
void TNSGA::predict_by_AE(vector<double>& IGD, vector<double>& HV) {
	//autoencoding the proposed method
#ifdef TEST
	metric();
#endif
	int cur_envi = InfoMemory::detectedEnviSol.size();
	int num_dim = DynPara::dimNum;
	vector<int> source_rank = rank_sol_crowd_distance(InfoMemory::detectedEnviSol[cur_envi - 2]);
	vector<int> target_rank = rank_sol_crowd_distance(InfoMemory::detectedEnviSol[cur_envi - 1]);
	int dsize = source_rank.size();
	if (target_rank.size() < dsize) dsize = target_rank.size();
	Eigen::MatrixXd X(num_dim, dsize);
	Eigen::MatrixXd Y(num_dim, dsize);
	for (int i = 0; i < dsize; ++i) {
		for (int j = 0; j < num_dim; ++j) {
			X(j, i) = InfoMemory::detectedEnviSol[cur_envi - 2][source_rank[i]].x[j];
			Y(j, i) = InfoMemory::detectedEnviSol[cur_envi - 1][target_rank[i]].x[j];
		}
	}
	Eigen::MatrixXd M(num_dim, num_dim);
	Eigen::MatrixXd temp(num_dim, num_dim);
	Eigen::MatrixXd P(target_rank.size(), num_dim);
	temp = Y * Y.transpose();
	double detvalue = temp.determinant();
	bool un_inver = false;
	if (detvalue <= 0) {
		un_inver = true;
	}
	else {
		M = X * Y.transpose() * (Y * Y.transpose()).inverse(); //dsize * dsize * 
		//generate new solutions
		Eigen::MatrixXd Z(target_rank.size(), num_dim);
		for (int i = 0; i < target_rank.size(); ++i) {
			for (int j = 0; j < num_dim; ++j) {
				Z(i, j) = InfoMemory::detectedEnviSol[cur_envi - 1][target_rank[i]].x[j];
			}
		}
		P = Z * M;
	}
	vector<CSolution> predict_sol(target_rank.size());
	vector<double> predict_error(num_dim);
	for (int i = 0; i < target_rank.size(); ++i) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(IGD, HV);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		for (int j = 0; j < num_dim; ++j) {
			if (un_inver) predict_sol[i].x[j] = random() * (DynPara::upperBound[j] - DynPara::lowBound[j]) + DynPara::lowBound[j];
			else predict_sol[i].x[j] = P(i, j);
			//cout << predict_sol[i].x[j] << ",";
			if (predict_sol[i].x[j] < DynPara::lowBound[j])
				predict_sol[i].x[j] = DynPara::lowBound[j];
			if (predict_sol[i].x[j] > DynPara::upperBound[j])
				predict_sol[i].x[j] = DynPara::upperBound[j];
		}
		//cout << "\t";
		objectives(predict_sol[i].x, predict_sol[i].f, mEvas);
		mEvas++;
		record_metric();

		//生成对应的最优解，估计误差
		vector<double> real_y(num_dim);
		vector<double> f(DynPara::objNum);
		getOptimalSolution(DynPara::proName, InfoMemory::detectedEnviSol[cur_envi - 1][target_rank[i]].x, real_y, mEvas);
		objectives(real_y, f, mEvas);
		for (int d = 0; d < num_dim; ++d) {
			predict_error[d] += fabs(real_y[d] - predict_sol[i].x[d]);
		}

	}
	InfoMemory::predictEnviSol.push_back(predict_sol);

#ifdef TEST
	cout << "predict:(";
	for (int j = 0; j < num_dim; ++j) {
		cout << predict_error[j] / target_rank.size() << ",";
	}
	cout << ")\n";
#endif
	//设置优化算法
	double error = 0;
	double old_error = 0;
	//int num_dim = DynPara::dimNum;
	vector<double> dim_error(num_dim, 0);
	//bool use_ep = false;  //the proposed method
	int curEnvi = InfoMemory::detectedEnviSol.size();
	int startenvi = curEnvi - InfoMemory::time_step;

#ifdef TEST
	output_pop_errors();
#endif

	init_half_pop_by_predict(predict_sol, IGD, HV);
	for (int i = 0; i < pops; ++i) {
		vector<double> ox = population[i].indiv.x_var;
		vector<double> of(DynPara::objNum);
		objectives(ox, of, mEvas);

		vector<double> real_y(num_dim);
		vector<double> f(DynPara::objNum);
		getOptimalSolution(DynPara::proName, ox, real_y, mEvas);
		objectives(real_y, f, mEvas);

		for (int k = 0; k < num_dim; ++k) {
			dim_error[k] += fabs(real_y[k] - ox[k]);
		}
	}
#ifdef TEST
	cout << "predict:(";
	for (int j = 0; j < num_dim; ++j) {
		cout << dim_error[j] / pops << ",";
	}
	cout << ")\n";
#endif
}

//使用神经网络预测解
void TNSGA::predict_by_neuralnetwork(vector<double>& IGD, vector<double>& HV) {
	//cout << "predict by NN" << endl;
#ifdef TEST
	//if(use_ep && EP.size() > 0) cout << "archive error:\t" << cal_EP_x_error() << "\t" << "arc fit error:\t" << cal_EP_f_error() << endl;
	//cout << "pop error:\t" << cal_pop_x_error() <<  "\t" << "pop fit error:\t" << cal_pop_f_error() << endl;
#endif
	bool same_indepvar = false;

	//bool use_ep = false;  //the proposed method
	int curEnvi = InfoMemory::detectedEnviSol.size();
	//int time_step = 3;
	int startenvi = curEnvi - InfoMemory::time_step;
	initialNNInfo(curEnvi);
	vector<solpair> set;                    //construct the solution set between environment
	vector<vector<int> > select_indepvar;
	vector<int> self_variable;
	cal_change_past_envi(set, select_indepvar, startenvi, self_variable); //generate solution pairs to represent environment changes

	InfoMemory::datasize[InfoMemory::cur_run].push_back(set.size());
	InfoMemory::relevance_set[InfoMemory::cur_run].push_back(select_indepvar);  //记录独立变量集合

	vector<double> past_predict_error = eval_prediction_error(); //评估上一环境中模型的预测误差
	vector<double> past_change_error = eval_move_error(); //评估上一环境中原始种群与最终解的变化
	eval_hist_move(); //
	vector<double> correct_error(DynPara::dimNum, 0);

	vector<int> rand_index(pops);
	for (int j = 0; j < rand_index.size(); ++j) {
		rand_index[j] = j;
	}
	random_shuffle(rand_index.begin(), rand_index.end());
	vector<int> is_select_rand(pops, 0);
	int rand_num = 0;
	if (use_random) rand_num = pops * 0.1;
	for (int j = 0; j < rand_num; ++j) {
		is_select_rand[rand_index[j]] = 1;
	}
	//step 2: 定义修正的个体下表
	//定义更新的个体下标
	vector<int> predicted(pops, 0);
	vector<int> corrected(pops, 0);
	if (use_pro) {
		vector<int> pindex(pops);
		random_shuffle(pindex.begin(), pindex.end());
		for (int j = 0; j < pops * pro_predict; ++j) {
			predicted[pindex[j]] = 1;
		}
		random_shuffle(pindex.begin(), pindex.begin() + (pops * pro_predict));
		for (int j = 0; j < pops * pro_predict * pro_correct; ++j) {
			corrected[pindex[j]] = 1;
		}
	}
	else {
		for (int j = 0; j < pops; ++j) { predicted[j] = 1; corrected[j] = 1; }
	}


	//cout << "calculate past changes......\n";
#ifdef TEST
	if (false) {
		int num_dim = DynPara::dimNum;
		vector<double> deta(num_dim, 0);
		for (int j = 0; j < set.size(); ++j) {
			for (int k = 0; k < set[j].x.size(); ++k) {
				deta[k] += fabs(set[j].x[k] - set[j].y[k]);
			}
		}
		cout << "past chang:\t";
		for (int k = 0; k < num_dim; ++k) cout << deta[k] / set.size() << "\t";
		cout << endl;
		vector<double> temp = InfoMemory::detectedEnviSol.back()[0].x;
		vector<double> opt_x;
		getOptimalSolution(DynPara::proName, temp, opt_x, mEvas);
		cout << "opt change:\t";
		for (int k = 0; k < num_dim; ++k) cout << fixed << setprecision(6) << opt_x[k] - temp[k] << "\t";
		cout << "\n";
	}
	double olderror = cal_pop_error_to_optima();

#endif
	//cout << "predict new solutions.......\n";
	if (set.size() > 0) {
		//step 4: 根据自变量和因变量构造数据集和神经网络
		int maxEpochs = 30;
		int envi_num = 30;

		vector<vector<int> > pop_list(pops);
		vector<double> train_error = train_network(set, select_indepvar, maxEpochs, envi_num);
		double avg_train_error = 0;
		for (int j = 0; j < train_error.size(); ++j)
			avg_train_error += train_error[j];
		avg_train_error /= train_error.size();
		InfoMemory::train_error[InfoMemory::cur_run].push_back(avg_train_error);
#ifdef TEST
		for (int j = 0; j < select_indepvar.size(); ++j) {
			cout << j << "(";
			for (int k = 0; k < select_indepvar[j].size(); ++k) {
				cout << select_indepvar[j][k] << ",";
			}
			cout << ")\t";
		}
		cout << endl;
		cout << "Train error:\t(";
		for (int j = 0; j < train_error.size(); ++j) {
			cout << train_error[j] << ",";
		}
		cout << ")";
		cout << endl;
#endif
		//step 4.2 预测上一个环境PS在新环境的解位置
		vector<vector<double> > sols;
		bool use_pop_predict = true;
		vector<CSolution> use_pop;
		if (use_pop_predict) {
			vector<CSolution> cur_pop(pops);
			for (int j = 0; j < pops; ++j) {
				cur_pop[j].x = population[j].indiv.x_var;
			}
			clock_t start_time = clock();
			sols = predict_pop_by_network(cur_pop, InfoMemory::detectedEnviSol.size() - 1, envi_num, select_indepvar, self_variable, same_indepvar);
			clock_t end_time = clock();
			double duration = (end_time - start_time) / CLOCKS_PER_SEC;
			InfoMemory::nnpredict_time[InfoMemory::cur_run].push_back(duration);
			for (int j = 0; j < sols.size(); ++j) {
				pop_list[j].push_back(j); //每个个体最近的预测解
			}
			use_pop = cur_pop;
#ifdef TEST
			if (false) {
				for (int j = 0; j < pops; ++j) {
					cout << j << "\t";
					outputVector(cur_pop[j].x);
					cout << endl;
				}
				for (int j = 0; j < pops; ++j) {
					cout << "p" << j << "\t";
					outputVector(sols[j]);
					cout << endl;
				}
			}
#endif
		}
		else {
			sols = predict_sol_by_network(curEnvi, envi_num, select_indepvar, same_indepvar);
			vector<int> indiv_label(sols.size()); //上一环境中与当前个体最近的个体
			int k = InfoMemory::detectedEnviSol.size() - 1;
			for (int j = 0; j < sols.size(); ++j) {
				double mind = -1;
				int sindex = -1; //优化各部分的记录
				for (int i = 0; i < pops; ++i) {
					double dist = cal_distance(InfoMemory::detectedEnviSol[k][j].x, population[i].indiv.x_var);
					if (i == 0 || mind > dist) {
						sindex = i;
						mind = dist;
					}
				}
				indiv_label[j] = sindex; //
				pop_list[sindex].push_back(j); //每个个体最近的预测解
			}
			use_pop = InfoMemory::detectedEnviSol[k];
		}
		int num_dim = DynPara::dimNum;
		vector<double> dim_error(num_dim, 0);
		for (int i = 0; i < pops; ++i) {
			//cout << i << "\t";
			vector<double> fy(DynPara::objNum);
			vector<double> y = sols[i]; // predict_a_sol_by_NN(curEnvi, envi_num, select_indepvar, ox, same_indepvar);
			//objectives(y, fy, mEvas);
			vector<double> real_y = y;
			getOptimalSolution(DynPara::proName, y, real_y, mEvas);

			for (int k = 0; k < num_dim; ++k) {
				dim_error[k] += fabs(real_y[k] - y[k]);
			}
		}
		double avg_predict_error = 0;
		for (int k = 0; k < num_dim; ++k) avg_predict_error += dim_error[k];
		avg_predict_error /= num_dim;
		InfoMemory::predict_error[InfoMemory::cur_run].push_back(avg_predict_error);


		//记录预测的解
		vector<CSolution> temp_pop(sols.size());
		for (int j = 0; j < temp_pop.size(); ++j) {
			temp_pop[j].x = sols[j];
			objectives(sols[j], temp_pop[j].f, mEvas);
		}
		InfoMemory::noCorrectEnviSol.push_back(temp_pop);

		//vector<vector<double> > last_sol = predict_sol_by_network(-1, envi_num, select_indepvar, same_indepvar);
		//step 4.3 对解进行修正，利用上一环境的预测误差
		if (correct) {
			clock_t start_time = clock();
			sols = correct_sol_by_predict_error(sols, use_pop, train_error, self_variable, corrected);
			clock_t end_time = clock();
			double duration = (end_time - start_time) / CLOCKS_PER_SEC;
			InfoMemory::correct_time[InfoMemory::cur_run].push_back(duration);
#ifdef TEST
			if (false) {
				for (int j = 0; j < pops; ++j) {
					cout << "c" << j << "\t";
					outputVector(sols[j]);
					cout << endl;
				}
			}
			for (int j = 0; j < pops; ++j) {
				vector<double> real_y(DynPara::dimNum);
				getOptimalSolution(DynPara::proName, sols[j], real_y, mEvas);
				for (int k = 0; k < DynPara::dimNum; ++k) {
					correct_error[k] += fabs(sols[j][k] - real_y[k]);
				}
			}
			for (int k = 0; k < DynPara::dimNum; ++k) {
				correct_error[k] /= pops;
			}
#endif
		}
		vector<CSolution> pre_sol(sols.size());

		for (int i = 0; i < pre_sol.size(); ++i) {
			//cout  << i << "\t";
			if (predicted[i]) {
				if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
					metric(IGD, HV);
					storeFinalPop();
					int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
					introduceDynamic(mEvas);
					defineBoundary(DynPara::proName);
					bool varDimChange = false;
					bool objDimChange = false;
					if (preDimNum != DynPara::dimNum) varDimChange = true;
					if (preObjNum != DynPara::objNum) objDimChange = true;
					if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
						reactToDimChange(varDimChange, objDimChange);
				}
				if (mEvas >= DynPara::totalFes) break;
				mEvas++;
			}
			pre_sol[i].x = sols[i];
			objectives(pre_sol[i].x, pre_sol[i].f, mEvas);

			if (predicted[i]) {
				TIndividual ind;
				ind.x_var = pre_sol[i].x;
				ind.y_obj = pre_sol[i].f;

				if (use_ep) update_EP(ind);
				update_reference(ind);
				record_metric();
			}
		}
		InfoMemory::predictEnviSol.push_back(pre_sol);
#ifdef TEST
		//metric();
#endif
		vector<int> is_better(pops, 0);
		//step 5: 预测archive中解新的位置
		int better = 0, worse = 0;
		if (false && use_ep) {
			int num_dim = DynPara::dimNum;
			//对解进行修正
			bool correct = false;
			if (correct) {
				vector<double> xerror = estimate_error(IGD, HV);
				cout << "estimate x change:\t";
				OutputVector(xerror);
				cout << endl;
				for (int j = 0; j < sols.size(); ++j) {
					for (int k = 0; k < num_dim; ++k) {
						if (fabs(sols[j][k] - InfoMemory::detectedEnviSol.back()[j].x[k]) < xerror[k]) {
							sols[j][k] = sols[j][k];// +gaussian(0, xerror[k] - fabs(sols[j][k] - InfoMemory::detectedEnviSol.back()[j].x[k]) / 3);
						}
					}
				}
				//根据历史误差
			}
#ifdef TEST
			if (false) {
				vector<vector<double> > optSol(sols.size());
				double error = 0;
				vector<double> predict_error(num_dim, 0);
				vector<double> predict_change(num_dim, 0);
				for (int j = 0; j < optSol.size(); ++j) {
					optSol[j].resize(num_dim);
					vector<double> temp = InfoMemory::detectedEnviSol.back()[j].x;
					getOptimalSolution(DynPara::proName, temp, optSol[j]);
					for (int k = 0; k < num_dim; ++k) {
						predict_error[k] += fabs(sols[j][k] - optSol[j][k]);
						predict_change[k] += fabs(sols[j][k] - temp[k]);
					}


					error = error + calDistance(sols[j], optSol[j]);

				}
				//cout << "predict error of EP:\t" << error / sols.size() << endl;
				cout << "dim error:\t";
				for (int k = 0; k < num_dim; ++k) {
					predict_error[k] /= sols.size();
					cout << predict_error[k] << "\t";

				}
				cout << endl;
				cout << "dim chang:\t";
				for (int k = 0; k < num_dim; ++k) {
					predict_change[k] /= sols.size();
					cout << predict_change[k] << "\t";
				}
				cout << endl;
			}
#endif
			//use the ep solutions to update populations

			vector<TIndividual> predict_set;
			for (int j = 0; j < sols.size(); ++j) {
				TIndividual d;
				d.x_var = pre_sol[j].x;
				d.y_obj = pre_sol[j].f;
				update_EP(d);
				update_reference(d);
				predict_set.push_back(d);
				record_metric();
			}
			//计算了归属的个体
			vector<vector<double> > scale_value(predict_set.size());
			vector<int> visit(pops, 0);
			vector<int> aindex(pops, -1);
			vector<int> belong(predict_set.size(), -1);
			for (int j = 0; j < predict_set.size(); ++j) {

				scale_value[j].resize(weights.size());
				int sindex = -1;
				for (int k = 0; k < pops; ++k) {
					scale_value[j][k] = scalar_func(predict_set[j].y_obj, population[k].namda, indivpoint);
					if (visit[k]) continue;
					if (sindex == -1 || scale_value[j][k] < scale_value[j][sindex]) {
						sindex = k;
					}
				}
				belong[j] = sindex;
				//
				vector<double> oy(DynPara::objNum);
				objectives(population[sindex].indiv.x_var, population[sindex].indiv.y_obj, mEvas);

				if (!visit[sindex]) {
					population[sindex].indiv = predict_set[j];
					visit[sindex] = 1;
					aindex[sindex] = j;
				}
				else if (scale_value[j][sindex] < scale_value[aindex[sindex]][sindex]) {
					population[sindex].indiv = predict_set[j];
					visit[sindex] = 1;
					aindex[sindex] = j;
				}
				int relation = domination(population[sindex].indiv.y_obj, oy);
				if (relation > 0) {
					better++;
				}
				else if (relation < 0) worse++;

				if (false) {
					int   n = sindex;
					int   s = population[n].table.size();
					TIndividual child, child2;
					realbinarycrossover(population[n].indiv, predict_set[j], child, child2);
					realmutation(child, 1.0 / DynPara::dimNum);
					child.obj_eval(mEvas);
					update_reference(child);
					int relation = dominate_comp(child.y_obj, population[n].indiv.y_obj);
					if (relation >= 0) {
						population[n].indiv = child;
						update_EP(child);
					}
				}
				else {
					update_reference(population[sindex].indiv);
					update_problem(population[sindex].indiv, sindex);
					update_EP(population[sindex].indiv);
				}
				//update_problem(child, n);
			}
			for (int j = 0; j < visit.size(); ++j) {
				if (visit[j]) continue;
				if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
					metric(IGD, HV);
					storeFinalPop();
					int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
					introduceDynamic(mEvas);
					defineBoundary(DynPara::proName);
					bool varDimChange = false;
					bool objDimChange = false;
					if (preDimNum != DynPara::dimNum) varDimChange = true;
					if (preObjNum != DynPara::objNum) objDimChange = true;
					if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
						reactToDimChange(varDimChange, objDimChange);
				}
				if (mEvas >= DynPara::totalFes) break;
				vector<double> oy(DynPara::objNum);
				objectives(population[j].indiv.x_var, population[j].indiv.y_obj, mEvas);
				if (true) {
					vector<double> px = predict_a_sol_by_NN(curEnvi, envi_num, select_indepvar, population[j].indiv.x_var);
					population[j].indiv.x_var = px;
					for (int k = 0; k < num_dim; ++k) {
						if (population[j].indiv.x_var[k] < DynPara::lowBound[k]) population[j].indiv.x_var[k] = DynPara::lowBound[k];
						if (population[j].indiv.x_var[k] > DynPara::upperBound[k]) population[j].indiv.x_var[k] = DynPara::upperBound[k];
					}
				}
				population[j].indiv.obj_eval(mEvas);
				update_reference(population[j].indiv);
				update_EP(population[j].indiv);
				mEvas++;
				record_metric();

				int relation = domination(population[j].indiv.y_obj, oy);
				if (relation > 0) {
					better++;
				}
				else if (relation < 0) worse++;
			}
		}
		else {
			//update the population using the movement of individuals and in the proposed 
		   //cout << "update population ....\n";
			double error = 0;
			double old_error = 0;
			int num_dim = DynPara::dimNum;

			int num = 0;
#ifdef TEST
			output_pop_errors();
#endif
			if (true) {
				clock_t start_time = clock();
				//reevaluate the population
				for (int i = 0; i < pops; ++i) {
					if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
						metric(IGD, HV);
						storeFinalPop();
						int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
						introduceDynamic(mEvas);
						defineBoundary(DynPara::proName);
						bool varDimChange = false;
						bool objDimChange = false;
						if (preDimNum != DynPara::dimNum) varDimChange = true;
						if (preObjNum != DynPara::objNum) objDimChange = true;
						if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
							reactToDimChange(varDimChange, objDimChange);
					}
					if (mEvas >= DynPara::totalFes) break;
					population[i].indiv.obj_eval(mEvas);
					if (use_ep) update_EP(population[i].indiv);
					update_reference(population[i].indiv);
					mEvas++;
					record_metric();
				}

				for (int i = 0; i < pops; ++i) {
					vector<double> ox = population[i].indiv.x_var;
					vector<double> of(DynPara::objNum);
					objectives(ox, of, mEvas);

					//cout << i << "\t";
					vector<double> fy(DynPara::objNum);
					vector<double> y = pre_sol[i].x; // predict_a_sol_by_NN(curEnvi, envi_num, select_indepvar, ox, same_indepvar);
					objectives(y, fy, mEvas);

					vector<double> real_y(num_dim);
					vector<double> f(DynPara::objNum);
					getOptimalSolution(DynPara::proName, ox, real_y, mEvas);
					objectives(real_y, f, mEvas);

					//for (int k = 0; k < num_dim; ++k) {
				//		dim_error[k] += fabs(real_y[k] - y[k]);
				//	}
					if (domination(fy, of) == 1) {
						better++;
					}
					else if (domination(fy, of) == -1) {
						worse++;
					}

					old_error = old_error + calDistance(ox, real_y);
					if (predicted[i]) {
						/**
						if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
							metric(IGD, HV);
							storeFinalPop();
							int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
							introduceDynamic(mEvas);
							defineBoundary(DynPara::proName);
							bool varDimChange = false;
							bool objDimChange = false;
							if (preDimNum != DynPara::dimNum) varDimChange = true;
							if (preObjNum != DynPara::objNum) objDimChange = true;
							if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
								reactToDimChange(varDimChange, objDimChange);
						}
						if (mEvas >= DynPara::totalFes) break;

						population[i].indiv.obj_eval(mEvas);
						if (use_ep) update_EP(population[i].indiv);
						update_reference(population[i].indiv);
						mEvas++;
						*/
						if (pop_list[i].size() > 0 && is_select_rand[i] == 0) {
							for (int j = 0; j < pop_list[i].size(); ++j) {
								if (domination(pre_sol[pop_list[i][j]].f, population[i].indiv.y_obj) == 1) {
									is_better[j] = 1;
#ifdef TEST
									num++;
#endif
								}
								int relation = domination(pre_sol[pop_list[i][j]].f, population[i].indiv.y_obj);
								//if (j == 0 && relation == 1) {
								if (j == 0 && (relation == 1 || (relation == 0 && random() <= 0.5))) {
									population[i].indiv.x_var = pre_sol[pop_list[i][j]].x;
									population[i].indiv.y_obj = pre_sol[pop_list[i][j]].f;
									if (use_ep) update_EP(population[i].indiv);
									update_reference(population[i].indiv);
								}
							}
						}
						/*
						else {
							if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
								metric(IGD, HV);
								storeFinalPop();
								int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
								introduceDynamic(mEvas);
								defineBoundary(DynPara::proName);
								bool varDimChange = false;
								bool objDimChange = false;
								if (preDimNum != DynPara::dimNum) varDimChange = true;
								if (preObjNum != DynPara::objNum) objDimChange = true;
								if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
									reactToDimChange(varDimChange, objDimChange);
							}
							if (mEvas >= DynPara::totalFes) break;

							//population[i].indiv.obj_eval(mEvas);
							if (is_select_rand[i] == 1) population[i].indiv.rnd_init();
							population[i].indiv.obj_eval(mEvas);
							if (use_ep) update_EP(population[i].indiv);
							update_reference(population[i].indiv);
							mEvas++;
							record_metric();
						}
						*/

					}

					error = error + calDistance(y, real_y);
				}
				clock_t end_time = clock();
				double duration = (end_time - start_time) / CLOCKS_PER_SEC;
				InfoMemory::update_time[InfoMemory::cur_run].push_back(duration);
			}
			else {

				init_half_pop_by_predict(pre_sol, IGD, HV);
				//设置优化算法
				for (int i = 0; i < pops; ++i) {
					vector<double> ox = population[i].indiv.x_var;
					vector<double> of(DynPara::objNum);
					objectives(ox, of, mEvas);

					//cout << i << "\t";
					vector<double> fy(DynPara::objNum);
					vector<double> y = predict_a_sol_by_NN(curEnvi, envi_num, select_indepvar, ox, same_indepvar);
					objectives(y, fy, mEvas);

					vector<double> real_y(num_dim);
					vector<double> f(DynPara::objNum);
					getOptimalSolution(DynPara::proName, ox, real_y, mEvas);
					objectives(real_y, f, mEvas);

					for (int k = 0; k < num_dim; ++k) {
						dim_error[k] += fabs(real_y[k] - ox[k]);
					}
				}
			}
			for (int k = 0; k < num_dim; ++k) dim_error[k] /= pops;
#ifdef TEST

			cout << "predict:(";
			for (int k = 0; k < num_dim; ++k) {
				cout << dim_error[k] << ",";
			}
			cout << ")\t";
			cout << "***" << num << "****\n";
			cout << "correct:";
			outputVector(correct_error);
			cout << "\n";
			//cout <<"new error:\t" << error / pops << "\t";// endl;
			//OutputVector(dim_error); 
			//cout << "+" << better << "\t-" << worse << "\t=" << pops - better - worse;
			//cout << endl;
#endif
			//update the probability of the proposed method
			if (use_pro) {
				int suc_predict = 0;
				int suc_correct = 0;
				int total_predict = 0;
				int  total_correct = 0;
				double r_predict = 0;
				double r_correct = 0;
				for (int j = 0; j < predicted.size(); ++j) {
					if (predicted[j] == 1 && is_better[j] == 1)
						suc_predict++;
					if (predicted[j] == 1) total_predict++;
				}
				for (int j = 0; j < predicted.size(); ++j) {
					if (predicted[j] == 1 && corrected[j] == 1 && is_better[j] == 1) {
						suc_correct++;
					}
					if (corrected[j] == 1 && predicted[j] == 1)
						total_correct++;
				}
				r_correct = (double)suc_correct / total_correct;
				r_predict = (double)suc_predict / total_predict;
				pro_predict = 0.95 * pro_predict + 0.05 * r_predict;
				pro_correct = 0.95 * pro_correct + 0.05 * r_correct;
				if (pro_predict < 0.5) pro_predict = 0.5;
				if (pro_correct < 0.5) pro_correct = 0.5;
			}
			else {
				pro_predict = 1;
				pro_correct = 1;
			}
		}
		//predict 
		//step 6: 预测种群中的解新位置
#ifdef TEST
		double newerror = cal_pop_error_to_optima();
		//cout << "======\t";
		cout << "olderror:\t" << olderror << "\t" << "newerror:\t" << newerror << "\t+" << better << "\t-" << worse << "\t=" << pops - better - worse << endl;
		metric();

#endif


#ifdef TEST
		//cout << DynPara::proName << "\t" << mEvas << "\t+(" << better << ")\t=(" << pops - better - worse << ")\t-(" << worse << ")\tpredict_error(" << error / pops << ")\t"
		//	<< "old error(" << old_error/pops << ")" <<  endl;
		//cout << "======\t";
		//metric();
#endif
	}
#ifdef TEST
	if (false) {
		//计算解中不同维度的关联性
		init_pop_correlation();
	}
#endif
}
void TNSGA::reevaluate_detector() {
	for (int j = 0; j < InfoMemory::numFixDetector; ++j) {
		objectives(InfoMemory::detector[j].x, InfoMemory::detector[j].f, mEvas);
		mEvas++;
		record_metric();
	}
}
void TNSGA::gaussian_pop(vector<double>& igdValue, vector<double>& hvValue, double rate) {
	//randomly select 10% inviduals to be reinitialized
	int num = pops * rate;
	int num_dim = DynPara::dimNum;
	vector<int> slist(pops);
	for (int j = 0; j < pops; ++j) {
		slist[j] = j;
	}
	random_shuffle(slist.begin(), slist.end());
	vector<double> dim_mean(num_dim, 0);
	for (int k = 0; k < num_dim; ++k) {
		dim_mean[k] = (DynPara::upperBound[k] - DynPara::lowBound[k]) / 10;
	}
	double std = 1;
	for (int j = 0; j < num; ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(igdValue, hvValue);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		int i = slist[j];
		for (int k = 0; k < num_dim; ++k) {
			population[i].indiv.x_var[k] = population[i].indiv.x_var[k] + gaussian(dim_mean[k], std);
			if (population[i].indiv.x_var[k] < DynPara::lowBound[k])
				population[i].indiv.x_var[k] = DynPara::lowBound[k];
			if (population[i].indiv.x_var[k] > DynPara::upperBound[k])
				population[i].indiv.x_var[k] = DynPara::upperBound[k];
		}
		population[i].indiv.obj_eval(mEvas);
		update_reference(population[i].indiv);
		mEvas++;
		record_metric();
	}
	for (int j = num; j < pops; ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(igdValue, hvValue);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		int i = slist[j];
		//population[i].indiv.rnd_init();
		population[i].indiv.obj_eval(mEvas);
		update_reference(population[i].indiv);
		//update_reference(population[i].indiv);
		mEvas++;
		record_metric();
	}

	vector<CSolution> pre_sol(pops);
	for (int j = 0; j < pops; ++j) {
		pre_sol[j].x = population[j].indiv.x_var;
		pre_sol[j].f = population[j].indiv.y_obj;
	}
	InfoMemory::predictEnviSol.push_back(pre_sol);
}
//初始化种群规模
void TNSGA::reinitialize_pop(vector<double>& igdValue, vector<double>& hvValue, double rate) {
	//randomly select 10% inviduals to be reinitialized
	int num = pops * rate;
	vector<int> slist(pops);
	for (int j = 0; j < pops; ++j) {
		slist[j] = j;
	}
	random_shuffle(slist.begin(), slist.end());
	for (int j = 0; j < num; ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(igdValue, hvValue);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		int i = slist[j];
		population[i].indiv.rnd_init();
		population[i].indiv.obj_eval(mEvas);
		update_reference(population[i].indiv);
		mEvas++;
		record_metric();
	}

	for (int j = num; j < pops; ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(igdValue, hvValue);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		int i = slist[j];
		//population[i].indiv.rnd_init();
		population[i].indiv.obj_eval(mEvas);
		update_reference(population[i].indiv);
		//update_reference(population[i].indiv);
		mEvas++;
		record_metric();
	}

	vector<CSolution> pre_sol(pops);
	for (int j = 0; j < pops; ++j) {
		pre_sol[j].x = population[j].indiv.x_var;
		pre_sol[j].f = population[j].indiv.y_obj;
	}
	InfoMemory::predictEnviSol.push_back(pre_sol);
}
void TNSGA::reevaluate_pop(vector<double>& igdValue, vector<double>& hvValue) {
	for (int j = 0; j < population.size(); ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(igdValue, hvValue);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		population[j].indiv.obj_eval(mEvas);
		update_reference(population[j].indiv);
		mEvas++;
		if (use_ep) update_EP(population[j].indiv);
		record_metric();
	}
}
//估计算法中的第一个环境中中心点的变化; 在新环境中评估这些点的变化，估计新环境适应值和解移动程度的变换
vector<double> TNSGA::estimate_error(vector<double>& IGD, vector<double>& HV) {
	int num_obj = DynPara::objNum;
	int num_dim = DynPara::dimNum;
	vector<double> min_first_envi(num_obj);
	vector<double> max_first_envi(num_obj);
	vector<double> norm_sum(InfoMemory::center_gen_envi[0].size(), 0);
	for (int i = 0; i < InfoMemory::center_gen_envi[0].size(); ++i) {
		for (int k = 0; k < num_obj; ++k) {
			if (i == 0 || min_first_envi[k] > InfoMemory::f_center_gen_envi[0][i][k])
				min_first_envi[k] = InfoMemory::f_center_gen_envi[0][i][k];
			if (i == 0 || max_first_envi[k] < InfoMemory::f_center_gen_envi[0][i][k])
				max_first_envi[k] = InfoMemory::f_center_gen_envi[0][i][k];
		}
	}
	int aindex = 0;
	int bindex = InfoMemory::center_gen_envi[0].size() / 2;
	int cindex = InfoMemory::center_gen_envi[0].size() - 1;

	//normalized the fitness; 第一阶段进化的适应值占比
	vector<double> rate_half_stage(num_obj);
	for (int i = 0; i < InfoMemory::center_gen_envi[0].size(); ++i) {
		for (int k = 0; k < num_obj; ++k) {
			norm_sum[i] += (InfoMemory::f_center_gen_envi[0][i][k] - min_first_envi[k]) / (max_first_envi[k] - min_first_envi[k]);
		}
	}
	//找离0.5最近的点
	vector<int> sindex(3);
	sindex[0] = 0;  sindex[2] = cindex;
	int closest_index = -1;
	double d = 0;
	for (int i = 0; i < norm_sum.size(); ++i) {
		if (closest_index == -1 || fabs(norm_sum[i] - 0.5 * num_obj) < d) {
			closest_index = i;
			d = fabs(norm_sum[i] - 0.5 * num_obj);
		}
	}
	sindex[1] = closest_index;
	bindex = closest_index;
	for (int k = 0; k < num_obj; ++k) {
		rate_half_stage[k] = fabs(InfoMemory::f_center_gen_envi[0][aindex][k] - InfoMemory::f_center_gen_envi[0][bindex][k]) / (max_first_envi[k] - min_first_envi[k]);
	}
	for (int i = 0; i < 3; ++i) {
		cout << "old center:\t" << i << "\t"; OutputVector(InfoMemory::center_gen_envi[0][sindex[i]]); cout << "\t";
		OutputVector(InfoMemory::f_center_gen_envi[0][sindex[i]]);
		cout << endl;
	}

	//预估新的变化
	vector<double> new_min_f(num_obj);
	vector<double> new_max_f(num_obj);
	vector<vector<double> > new_f(3);
	for (int i = 0; i < 3; ++i) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(IGD, HV);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum)
				reactToDimChange(varDimChange, objDimChange);
		}
		if (mEvas >= DynPara::totalFes) break;
		TIndividual child;
		child.x_var = InfoMemory::center_gen_envi[0][sindex[i]];
		child.obj_eval(mEvas);
		//population[j].indiv.obj_eval();
		//update_reference(population[j].indiv);
		mEvas++;
		if (use_ep) update_EP(child);
		record_metric();
		new_f[i] = child.y_obj;
		for (int k = 0; k < num_obj; ++k) {
			if (i == 0 || new_min_f[k] > new_f[i][k]) {
				new_min_f[k] = new_f[i][k];
			}
			if (i == 0 || new_max_f[k] < new_f[i][k]) {
				new_max_f[k] = new_f[i][k];
			}
		}
		cout << "new envi:\t" << i << "\t"; OutputVector(child.x_var); cout << "\t";
		OutputVector(new_f[i]); cout << endl;
	}
	cout << "random one:\t";
	TIndividual child;
	child.x_var = InfoMemory::detectedEnviSol[0][0].x;
	OutputVector(child.x_var);
	cout << "\t";
	child.obj_eval(mEvas);
	OutputVector(child.y_obj);
	cout << endl;
	objective(DynPara::proName.c_str(), child.x_var, child.y_obj);
	OutputVector(child.y_obj);
	cout << endl;


	cout << "rate half stage:\t";
	OutputVector(rate_half_stage); cout << endl;

	cout << "current min f:\t"; OutputVector(new_min_f); cout << endl;
	cout << "current max f:\t"; OutputVector(new_max_f); cout << endl;
	vector<double> found_min_f = new_min_f;
	for (int k = 0; k < num_obj; ++k) {
		double ed = new_max_f[k] - fabs(new_f[0][k] - new_f[1][k]) / rate_half_stage[k];
		if (ed < found_min_f[k])  found_min_f[k] = ed;
	}
	cout << "estimate min f:\t"; OutputVector(found_min_f); cout << endl;
	//根据范围计算最优解离目标解距离
	vector<double> dist(num_obj, 0);
	double sum = 0;
	for (int k = 0; k < num_obj; ++k) {
		dist[k] = fabs(new_f[2][k] - found_min_f[k]);
		sum += dist[k] / (new_max_f[k] - found_min_f[k]);
	}
	cout << "normalize sum:\t";
	OutputVector(norm_sum);
	cout << endl;
	cout << "sum:\t" << sum << "\t";
	//找到第一个环境中适应值最接近的中心点的代数
	int clostindex = -1;
	double error = 0;
	for (int i = 0; i < InfoMemory::center_gen_envi[0].size(); ++i) {
		if (clostindex == -1 || (fabs(norm_sum[i] - sum) < error)) {
			clostindex = i;
			error = fabs(norm_sum[i] - sum);
		}
	}
	//选择中心点
	vector<double> xerror(num_dim);
	for (int j = 0; j < num_dim; ++j) {
		xerror[j] = fabs(InfoMemory::center_gen_envi[0].back()[j] - InfoMemory::center_gen_envi[0][clostindex][j]);
	}
	return xerror;
}
void TNSGA::reinit_hist_center(vector<double>& IGD, vector<double>& HV) {
	//回退历史解
	int num_obj = DynPara::objNum;
	int num_gen = 10;
	int sample_fre = InfoMemory::center_gen_envi[0].size() / num_gen;
	vector<vector<double> > samples;
	vector<vector<double> > f;
	for (int j = 0; j < num_gen; ++j) {
		int k = sample_fre * j;
		if (k >= InfoMemory::center_gen_envi[0].size()) continue;
		samples.push_back(InfoMemory::center_gen_envi[0][k]);
		vector<double> fitness(num_obj);
		objectives(samples.back(), fitness, mEvas);
		f.push_back(fitness);
	}
	//选择非占优解进行高斯变异
	vector<int> be_dominated;
	for (int j = 0; j < samples.size(); ++j) {
		for (int k = 0; k < samples.size(); ++k) {
			if (j == k) continue;
			if (dominate_comp(f[j], f[k]) == -1) {
				be_dominated.push_back(j);
				break;
			}
		}
	}
	//删除占优解
	for (int j = 0; j < be_dominated.size(); ++j) {
		samples.erase(samples.begin() + (be_dominated[j] - j));
		f.erase(f.begin() + (be_dominated[j] - j));
	}
	//更新EP
	for (int j = 0; j < samples.size(); ++j) {
		TIndividual child;
		child.x_var = samples[j];
		child.y_obj = f[j];
		//update_EP(child);
		//mEvas++;
	}
	resample_pop(samples, IGD, HV);
}
//igdvalue and hvvalue to calculate the hyperplane and hypervolumne
void TNSGA::reactToEnviChange(vector<double>& igdValue, vector<double>& hvValue, const vector<TIndividual>& offsprings, const int oindex) {
	//store the archive solutions in the detected 
	//cout << "enter rect to envi change.....\n";
	//bool use_ep = false;
	update_pop(offsprings);
	cal_center();
	storePSForDetectChange_DE();
	storeInitPop();
	reevaluate_detector();
	int curenvi = getEnviIndex(mEvas);

	clock_t start_time = clock();

	//reevaluate the population
	if (InfoMemory::useNN) {
		if (use_ep) reevaluate_EP();
		//EP.clear();
#ifdef TEST
		metric();
#endif
		//cout << "react to enviromnt change using NN begin.....\n";
		if (InfoMemory::detectedEnviSol.size() < 2) {
			//reinit_hist_center(igdValue, hvValue);
			//reinitialize_pop(igdValue, hvValue, 0.5);
			//reinitialize_pop(igdValue, hvValue, 0.1);
			InfoMemory::predictEnviIndex.push_back(curenvi);
			//gaussian_pop(igdValue, hvValue, 0.2);
			reinitialize_pop(igdValue, hvValue, 0.1);
		}
		else {
			predict_by_neuralnetwork(igdValue, hvValue);
			InfoMemory::predictEnviIndex.push_back(curenvi);
		}
		//cout << "react to enviromnt change using NN end.....\n";
	}
	else if (InfoMemory::useCusteringSVM) {
		EP.clear();
		transfer_cluster_SVM(igdValue, hvValue);
		InfoMemory::predictEnviIndex.push_back(curenvi);
	}
	else if (InfoMemory::useKneeSVM) {
		EP.clear();
		if (InfoMemory::detectedEnviSol.size() < 2) {
			reinitialize_pop(igdValue, hvValue, 1.0);
		}
		else {
			transfer_knee_SVM(igdValue, hvValue);

		}
		InfoMemory::predictEnviIndex.push_back(curenvi);
	}
	else if (InfoMemory::useITSVM) {
		EP.clear();
		transfer_ind_SVM(igdValue, hvValue);
		InfoMemory::predictEnviIndex.push_back(curenvi);
	}
	else if (InfoMemory::useSVR) {
		if (InfoMemory::detectedEnviSol.size() < InfoMemory::q + 1) {
			reevaluate_pop(igdValue, hvValue);
		}
		else {
			predict_by_SVR(igdValue, hvValue);
			InfoMemory::predictEnviIndex.push_back(curenvi);
		}
	}
	else if (InfoMemory::useAutoEncoding) {
		EP.clear();
		if (InfoMemory::detectedEnviSol.size() < 2) {
			reinitialize_pop(igdValue, hvValue, 0.5);
		}
		else {
			predict_by_AE(igdValue, hvValue);
		}
		InfoMemory::predictEnviIndex.push_back(curenvi);
	}
	else {
		EP.clear();
		reinitialize_pop(igdValue, hvValue, 0.1);
		InfoMemory::predictEnviIndex.push_back(curenvi);
		//reevaluate_pop(igdValue, hvValue);
		//for(int j = 0; )
		//reevaluate EP
	}
	clock_t end_time = clock();
	//int cur_id = DynPara::
	double duration = (end_time - start_time) / CLOCKS_PER_SEC;
	InfoMemory::predict_time[InfoMemory::cur_run].push_back(duration);

	if (InfoMemory::predictEnviIndex.size() != InfoMemory::predictEnviSol.size()) {
		cout << InfoMemory::predictEnviIndex.size() << "\t" << InfoMemory::predictEnviSol.size() << endl;
		assert(false);
	}
}

double TNSGA::cal_pop_error_to_optima() {
	double error = 0;
	int num_dim = DynPara::dimNum;
	for (int i = 0; i < pops; ++i) {
		vector<double> optima(num_dim);
		getOptimalSolution(DynPara::proName, population[i].indiv.x_var, optima, mEvas);
		error += calDistance(population[i].indiv.x_var, optima);
	}
	return error / pops;
}

//correct the solutions using prediction errors in the past environment;
vector<vector<double> > TNSGA::correct_sol_by_predict_error(const vector<vector<double> >& set, const vector<CSolution>& osol, const vector<double>& train_error, const vector<int>& self_variable, const vector<int>& use_correct) {
	//correct solutions using prediction error in the last environment
	//using gaussian distribution
	int num_dim = DynPara::dimNum;
	vector<vector<double> > cor_sols = set;
	vector<double> predict_error = eval_prediction_error();
	vector<double> change_error = eval_move_error(); //不做预测的初始种群的误差
	vector<double> last_move = InfoMemory::hist_move.back();
	vector<double> center_move = eval_center_move();
	if (InfoMemory::predictEnviSol.size() <= 0) {
		predict_error = train_error;
	}

	//判断原始解是否到达边界了;如果到达了，看看历史移动方向是否远远越过边界;如果是，则采用反向运动高斯（移动方向的反向值）
	int cur_envi = getEnviIndex(mEvas);

#ifdef TEST
	cout << "last move:\t";
	OutputVector(last_move);
	cout << endl;
	cout << "predict error:\t";
	OutputVector(predict_error);
	cout << endl;
	cout << "initpop error:\t";
	OutputVector(change_error);
	cout << endl;
#endif

	//check the diversity of the self-variables dimensions
	vector<double> diversity(num_dim, 0); //check the maximum and minmum value
	vector<double> min_value(num_dim, 0);
	vector<double> max_value(num_dim, 0);
	for (int k = 0; k < num_dim; ++k) {
		for (int i = 0; i < set.size(); ++i) {
			if (i == 0 || set[i][k] < min_value[k])
				min_value[k] = set[i][k];
			if (i == 0 || set[i][k] > max_value[k])
				max_value[k] = set[i][k];
		}
	}
	for (int k = 0; k < num_dim; ++k) {
		for (int i = 0; i < set.size(); ++i) {
			diversity[k] += set[i][k];
		}
		diversity[k] /= set.size();
	}
	vector<int> is_correct(num_dim, 1);
	for (int k = 0; k < self_variable.size(); ++k) {
		int d = self_variable[k];
		if (max_value[d] - min_value[d] > fabs(DynPara::upperBound[d] - DynPara::lowBound[d]) / 2) {
			is_correct[d] = 0;
		}
	}
	vector<int> is_predict(num_dim, 1);
	if (false) {  //测试的例子是true的

		for (int k = 0; k < num_dim; ++k) {
			if ((predict_error[k] - last_move[k]) >= last_move[k] * 0.5) {
				is_predict[k] = 0;
				is_correct[k] = 0;
			}
		}
	}

	vector<double> dist = predict_error;
	int b_index = InfoMemory::detectedEnviSol.size() - 2;
	for (int j = 0; j < set.size(); ++j) {
		if (use_correct[j] == 0) continue;
		for (int k = 0; k < num_dim; ++k) {
			if (is_predict[k] == 0) {
				cor_sols[j][k] = osol[j].x[k];
			}
			if (is_correct[k] != 1) continue;

			double threold = predict_error[k];
			//if (fabs(predict_error[k]) > fabs(last_move[k])) {
		//		threold = fabs(predict_error[k]) - fabs(last_move[k]);
		//	}
			double r = 0;
			//case 1: 上一环境已经到达边界，这次应该反向运动
			if (fabs(osol[j].x[k] - DynPara::lowBound[k]) < fabs(last_move[k]) / 20 // && osol[j].x[k] + last_move[k] < DynPara::lowBound[k]
				|| fabs(osol[j].x[k] - DynPara::upperBound[k]) < fabs(last_move[k]) / 20) { // && osol[j].x[k] + last_move[k] > DynPara::upperBound[k]) {
				threold = last_move[k];
				if (osol[j].x[k] + threold < DynPara::lowBound[k] || osol[j].x[k] + threold > DynPara::upperBound[k]) {
					threold = -last_move[k];
				}
				r = gaussian(threold, threold / 15);
				//r = gaussian(threold, threold / 6); //random() * 2 * threold - threold;// gaussian(threold / 2, threold / 6);// gaussian(0, threold);  gaussian(threold/2, threold/6);//
				//if (threold < 0 && r < threold) {
				//	r = threold + fabs(r - threold);
				//}
				//if (threold > 0 && r > threold) {
			//		r = threold - fabs(r - threold);
			//	}
				if (cor_sols[j][k] + r < DynPara::lowBound[k])
					r = fabs(cor_sols[j][k] + r - DynPara::lowBound[k]);
				if (cor_sols[j][k] + r > DynPara::upperBound[k])
					r = -fabs(cor_sols[j][k] + r - DynPara::upperBound[k]);
			}
			//case 2: 第一次到达边界，不改变值
			//set the problem in the proposed method;如果是第一次预测到边界;不管公式如何计算
			else if (fabs(cor_sols[j][k] - DynPara::lowBound[k]) < fabs(last_move[k]) / 20 || fabs(cor_sols[j][k] - DynPara::upperBound[k]) < fabs(last_move[k]) / 20) {
				//如果预测方向与上一次移动方向相反，则修正
				if ((cor_sols[j][k] - osol[j].x[k]) * (InfoMemory::detectedEnviSol.back()[j].x[k] - InfoMemory::detectedEnviSol[b_index][j].x[k]) < 0) {
					r = (InfoMemory::detectedEnviSol.back()[j].x[k] - InfoMemory::detectedEnviSol[b_index][j].x[k]);
				}
				if (false) {
					threold = predict_error[k];
					r = gaussian(threold * 0.5, threold / 30);
					//确定变异的方向，根据历史解的移动方向;
					if (InfoMemory::detectedEnviSol.back()[j].x[k] < InfoMemory::detectedEnviSol[b_index][j].x[k]) {
						r = -r;
					}
					//对生成解进行边界反弹
					if (cor_sols[j][k] + r < DynPara::lowBound[k])
						r = fabs(cor_sols[j][k] + r - DynPara::lowBound[k]);
					if (cor_sols[j][k] + r > DynPara::upperBound[k])
						r = -fabs(cor_sols[j][k] + r - DynPara::upperBound[k]);
				}
			}
			//case 3: 当前和过去都没有到达边界点，那么根据历史预测误差（过去种群进化方向）对解再进行移动，
			else {
				if (false) {
					if (train_error[k] >= 0.5 * last_move[k]) {
						threold = last_move[k];
					}
					else if (last_move[k] < predict_error[k]) {
						threold = last_move[k];// fabs(predict_error[k] - last_move[k]);
					}
				}
				r = gaussian(threold * 0.8, threold / 15);
				//r = gaussian(threold * 0.9, threold / 30);
				//double r = gaussian(threold*0.5, threold / 6);
				//double r = gaussian(threold*0.8, threold / 15); 
				//double r = gaussian(threold, threold / 6); //random() * 2 * threold - threold;// gaussian(threold / 2, threold / 6);// gaussian(0, threold);  gaussian(threold/2, threold/6);//
				//if (threold < 0 && r < threold) {
			//		r = threold + fabs(r - threold);
			//	}
			//	if (threold > 0 && r > threold) {
			//		r = threold - fabs(r - threold);
			//	}

															//if (r > threold) r = threold;
				//确定变异的方向，根据历史解的移动方向;
				if (InfoMemory::detectedEnviSol.back()[j].x[k] < InfoMemory::detectedEnviSol[b_index][j].x[k]) {
					r = -r;
				}
				if (false) {
					//对生成解进行边界反弹
					if (cor_sols[j][k] + r < DynPara::lowBound[k])
						r = fabs(cor_sols[j][k] + r - DynPara::lowBound[k]);
					if (cor_sols[j][k] + r > DynPara::upperBound[k])
						r = -fabs(cor_sols[j][k] + r - DynPara::upperBound[k]);
				}
			}
#ifdef TEST
			if (false && (cur_envi == 21 || cur_envi == 11)) {
				cout << k << "(" << osol[j].x[k] << "," << threold << "," << last_move[k] << "," << r << ")\n";
			}
#endif
			//if (predict_error[k] <= 0) continue;    //threold /2


			//if (random() <= 0.5) r = -r;
			//if (r < 0) r = 0;// threold;
			//if(r > 2*threold) r = 2*threold;
			//if (center_move[k] < 0) {
		//		r = -r;
		//	}
		//	if (cor_sols[j][k] + r < DynPara::lowBound[k] && fabs(osol[j].x[k] - DynPara::lowBound[k]) < fabs(predict_error[k])/10) {
		//		r = -r;
		//	}
		//	if (cor_sols[j][k] + r > DynPara::upperBound[k] && fabs(osol[j].x[k] - DynPara::upperBound[k]) < fabs(predict_error[k]) / 10) {
		//		r = -r;
		//	}
			//if (random() <= 0.5) r = -r;
			cor_sols[j][k] = cor_sols[j][k] + r;
			if (false) {
				if (cor_sols[j][k] < DynPara::lowBound[k])
					cor_sols[j][k] = (DynPara::lowBound[k] + set[j][k]) / 2;
				if (cor_sols[j][k] > DynPara::upperBound[k])
					cor_sols[j][k] = (DynPara::upperBound[k] + set[j][k]) / 2;
			}
			else {
				if (cor_sols[j][k] < DynPara::lowBound[k])
					cor_sols[j][k] = DynPara::lowBound[k];
				if (cor_sols[j][k] > DynPara::upperBound[k])
					cor_sols[j][k] = DynPara::upperBound[k];
			}
			//if (predict_error[k] > change_error[k] && random() <= 0.5) {
			//	cor_sols[j][k] = InfoMemory::detectedEnviSol.back()[j].x[k];
		//	}
		}
	}
	return cor_sols;
}

vector<double> TNSGA::eval_prediction_error() {
	int num_dim = DynPara::dimNum;
	vector<double> error(num_dim, 0);

	if (InfoMemory::predictEnviSol.size() > 0) {
		int a = InfoMemory::detectedEnviSol.size() - 1;
		int b = InfoMemory::predictEnviSol.size() - 1;
		vector<vector<int> > pair = pair_sol_detected_change(InfoMemory::detectedEnviSol.back(), InfoMemory::predictEnviSol.back(), weights, use_ep);

		if (pair.size() > 0) {
			for (int j = 0; j < pair.size(); ++j) {
				int aindex = pair[j][0];
				int bindex = pair[j][1];
				for (int k = 0; k < num_dim; ++k) {
					error[k] += fabs(InfoMemory::detectedEnviSol.back()[aindex].x[k] - InfoMemory::predictEnviSol.back()[bindex].x[k]);
					//error[k] += (InfoMemory::detectedEnviSol.back()[aindex].x[k] - InfoMemory::predictEnviSol.back()[bindex].x[k]);
				}
			}
			for (int j = 0; j < num_dim; ++j) {
				error[j] /= pair.size();
			}
		}
	}
	return error;
}

vector<double> TNSGA::eval_center_move() {
	int num_dim = DynPara::dimNum;
	vector<double> move_range(num_dim, 0);

	if (InfoMemory::predictEnviSol.size() > 0) {
		int a = InfoMemory::detectedEnviSol.size() - 1;
		int b = InfoMemory::predictEnviSol.size() - 1;

		vector<double> target_center(num_dim, 0);
		vector<double> init_center(num_dim, 0);

		for (int j = 0; j < num_dim; ++j) {
			for (int k = 0; k < InfoMemory::detectedEnviSol.back().size(); ++k) {
				target_center[j] += InfoMemory::detectedEnviSol.back()[k].x[j];
			}
			for (int k = 0; k < InfoMemory::predictEnviSol.back().size(); ++k) {
				init_center[j] += InfoMemory::predictEnviSol.back()[k].x[j];
			}
			target_center[j] /= InfoMemory::detectedEnviSol.back().size();
			init_center[j] /= InfoMemory::predictEnviSol.back().size();
			move_range[j] = target_center[j] - init_center[j];
		}
	}

	return move_range;

}

vector<double> TNSGA::eval_move_error() {
	int num_dim = DynPara::dimNum;
	vector<double> error(num_dim, 0);
	int sindex = InfoMemory::orignalPop.size() - 2;

	if (InfoMemory::orignalPop.size() >= 2) {
		for (int j = 0; j < InfoMemory::orignalPop[sindex].size(); ++j) {
			for (int k = 0; k < num_dim; ++k) {
				error[k] += fabs(InfoMemory::detectedEnviSol.back()[j].x[k] - InfoMemory::orignalPop[sindex][j].x[k]);
			}
		}
		for (int j = 0; j < num_dim; ++j) {
			error[j] /= InfoMemory::orignalPop[sindex].size();
		}
	}

	return error;

}

void TNSGA::cal_center() {
	//calculate the center of the population and the fitness value
	int num_dim = DynPara::dimNum;
	int num_obj = DynPara::objNum;
	vector<double> center(num_dim, 0);
	vector<double> f(num_obj, 0);
	//calculate the fitness center
	for (int j = 0; j < num_dim; ++j) {
		for (int i = 0; i < pops; ++i) {
			center[j] += population[i].indiv.x_var[j];
		}
		center[j] /= pops;
	}
	for (int j = 0; j < num_obj; ++j) {
		for (int i = 0; i < pops; ++i) {
			f[j] += population[i].indiv.y_obj[j];
		}
		f[j] /= pops;
	}

	//
	int enviindex = getEnviIndex(mEvas);
	if (InfoMemory::center_gen_envi.size() < enviindex + 1) {
		vector<vector<double> > temp;
		InfoMemory::center_gen_envi.push_back(temp);
		InfoMemory::center_gen_envi[enviindex].push_back(center);  //center of the population
		vector<vector<double> > f_temp;
		InfoMemory::f_center_gen_envi.push_back(f_temp);
		InfoMemory::f_center_gen_envi[enviindex].push_back(f);     //center of the population
	}
	else {
		InfoMemory::center_gen_envi[enviindex].push_back(center);  //center of the population
		InfoMemory::f_center_gen_envi[enviindex].push_back(f);     //center of the population
	}
}

void TNSGA::selection(vector<int>& parentlist) {   //selection for the proposed method
	parentlist.resize(pops);
	vector<int> a(pops);
	vector<int> b(pops);
	int temp;
	int i;
	int rand;
	for (i = 0; i < pops; i++)
	{
		a[i] = i;
		b[i] = i;
	}
	random_shuffle(a.begin(), a.end());
	random_shuffle(b.begin(), b.end());
	int k = 0;
	for (i = 0; i < pops; i += 4) {
		int pa1 = tournament(population[a[i]].indiv, population[a[i + 1]].indiv);
		parentlist[k] = a[i + pa1];
		int pb1 = tournament(population[a[i + 2]].indiv, population[a[i + 3]].indiv);
		parentlist[k + 1] = a[i + 2 + pb1];
		k = k + 2;
	}
	for (i = 0; i < pops; i += 4) {
		int pa1 = tournament(population[b[i]].indiv, population[b[i + 1]].indiv);
		parentlist[k] = b[i + pa1];
		int pb1 = tournament(population[b[i + 2]].indiv, population[b[i + 3]].indiv);
		parentlist[k + 1] = b[i + 2 + pb1];
		k = k + 2;
	}
}
int TNSGA::tournament(const TIndividual& a, const TIndividual& b) { //routine for binary tournament
	int i = 0;
	int better = 0, worse = 0;
	for (i = 0; i < nobj; i++)
	{
		if (a.y_obj[i] < b.y_obj[i])
		{
			better ++;

		}
		else
		{
			if (a.y_obj[i] > b.y_obj[i])
			{
				worse ++;
			}
		}
	}
	int cmb = 0;
	if (better > 0 && worse == 0) {
		return 0;
	}
	else if (better == 0 && worse > 0) {
		return 1;
	}
	else{
		if (random() <= 0.5) return 0;
		else return 1;
	}

}
int TNSGA::dominate_cmp(const TIndividual& a, const TIndividual& b) {
	int i = 0;
	int better = 0, worse = 0;
	for (i = 0; i < nobj; i++)
	{
		if (a.y_obj[i] < b.y_obj[i])
		{
			better++;

		}
		else
		{
			if (a.y_obj[i] > b.y_obj[i])
			{
				worse++;
			}
		}
	}
	if (better > 0 && worse == 0) return 1;
	else if (worse > 0 && better == 0) return -1;
	else return 0;
}

// recombination, mutation, update in MOEA/D
void TNSGA::evolution(vector<double>& igdValue, vector<double>& hvValue)
{
	//bool use_ep = false;
	//每一代10个随机个体进行环境变化检测
	bool useDetection = true;
	if (DynPara::proName == "SDP12" || DynPara::proName == "SDP13") useDetection = false;

	vector<int> parents(pops);
	selection(parents);
	//cout << population.size() << "\t";
	if(offsprings.size() != pops)offsprings.resize(population.size());
	//generate new childs
	for (int j = 0; j < population.size()/2; ++ j) {
		int i = j * 2;
		//int   n = i;
		//int   s = population.size();
		int   p1 = parents[i]; // int(s * random());
		int   p2 = parents[i + 1];// int(s * random());
		//if (p2 == p1) p2 = (p1 + 1) % s;
		TIndividual child, child2;
		realbinarycrossover(population[p1].indiv, population[p2].indiv, child, child2);
		realmutation(child, 1.0 / DynPara::dimNum);
		offsprings[i] = child;
		offsprings[i + 1] = child2;
	}

	for (int i = 0; i < population.size(); i++)
	{
		if (useDetection && i % 20 == 0) {
			int index = i / 20;
			//cout << i << "\t" << index << "\t";
			bool enviHasChanged = detectChange(index, igdValue, hvValue);
			//if (enviHasChanged) cout << "envi has changed...\n";
			if (enviHasChanged) {
				int enviindex = getEnviIndex(mEvas);
				vector<TIndividual> temp(i + 1);
				for (int j = 0; j <= i; ++j) temp[j] = offsprings[j];
				update_pop(temp);
				//cout << mEvas << "\t" << "envi has changed.....\n";
				reactToEnviChange(igdValue, hvValue, temp, i-1);
				offsprings.clear();
				break;
			}
		}
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			//update the population using the evaluated offsprings in the past environment
			vector<TIndividual> temp(i +1);
			for (int j = 0; j <= i; ++j) temp[j] = offsprings[j];
			update_pop(temp);
			metric(igdValue, hvValue);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum) {
				//cout << population.size() << "#";
				if (DynPara::proName != "SDP12" && DynPara::proName != "SDP13") assert(false);
				reactToDimChange(varDimChange, objDimChange);
			}
			//cout << "+++\t" << population.size() << "\t" << DynPara::dimNum << "\t" << DynPara::objNum << "\t";
		}
		if (mEvas >= DynPara::totalFes) break;
		//if (i >= population.size()) break;
		
		if (i < 0 || i > pops) { cout << i << "\t" << endl; assert(false); }
		offsprings[i].obj_eval(mEvas);
		update_reference(offsprings[i]);
		//update_problem(child, n);
		if (use_ep) update_EP(offsprings[i]);
		mEvas++;
		record_metric();
	}
	update_pop(offsprings); // update the population using crowding distances
	//
#ifdef TEST
	metric();
#endif
	cal_center();
	//vector<double> center = cal_pop_center();
	//InfoMemory::center_gen_envi[0].push_back(center);
	//
}

vector<int> TNSGA::crowd_distance_selection(const vector<TIndividual> &temp, const int snum, const vector<double> max_f, const vector<double> min_f) {
	double inf_value = 10000000000000;
	int tsize = temp.size();
	vector<double> crowd_dist(tsize, 0);
	vector<vector<int> > f_rank(DynPara::objNum);
	vector<int> ranks(tsize);
	for (int i = 0; i < DynPara::objNum; ++i) {
		for (int j = 0; j < tsize; ++j) {
			ranks[j] = j;
		}
		//vector<int> visit(tsize, 0);
		for (int j = 0; j < tsize; ++j) {
			double min_value = temp[ranks[j]].y_obj[i];
			int sindex = j;
			for (int k = j + 1; k < tsize; ++k) {
				if (temp[ranks[k]].y_obj[i] < min_value) {
					min_value = temp[ranks[k]].y_obj[i];
					sindex = k;
				}
			}
			int pindex = ranks[sindex];
			ranks[sindex] = ranks[j];
			ranks[j] = pindex;
		}
		f_rank[i] = ranks;
		crowd_dist[ranks[0]] += inf_value;
		crowd_dist[ranks[tsize - 1]] += inf_value;
		for (int j = 1; j < tsize - 1; ++j) {
			if (max_f[i] == min_f[i]) crowd_dist[ranks[j]] += 0;
			else crowd_dist[ranks[j]] += fabs(temp[ranks[j + 1]].y_obj[i] - temp[ranks[j - 1]].y_obj[i])/fabs(max_f[i] - min_f[i]);
		}
	}
	//select pops individuals from the set
	vector<int> visit(tsize, 1);
	for (int j = 0; j < tsize - snum; ++j) {
		int sindex = -1;
		for (int k = 0; k < tsize; ++k) {
			if (visit[k] == 0) continue;
			if (sindex == -1 || crowd_dist[k] < crowd_dist[sindex])
				sindex = k;
		}
		visit[sindex] = 0;
	}
	return visit;
}

void TNSGA::update_pop(const vector<TIndividual> &offsprings) {
	if (offsprings.size() == 0) return;
	//step 1: offsprings 
	int totalsize = pops + offsprings.size();
	vector<int> ranks(totalsize);
	vector<TIndividual> temp; // (totalsize);
	for (int j = 0; j < pops; ++j) temp.push_back(population[j].indiv);
	for (int j = 0; j < offsprings.size(); ++j) temp.push_back(offsprings[j]);

	//
	vector<double> max_f(DynPara::objNum);
	vector<double> min_f(DynPara::objNum);
	for (int j = 0; j < temp.size(); ++j) {
		
		for (int k = 0; k < DynPara::objNum; ++k) {
			if (j == 0 || temp[j].y_obj[k] < min_f[k])
				min_f[k] = temp[j].y_obj[k];
			if (j == 0 || temp[j].y_obj[k] > max_f[k])
				max_f[k] = temp[j].y_obj[k];
		}
	}

	//step 2: delete dominated solutions; 0是选中，1是淘汰
	vector<int> select_index(totalsize, 1);
	int scount = 0;
	while (true) {
		vector<int> dominated(totalsize, 0);
		for (int j = 0; j < totalsize; ++j) {
			if (select_index[j] == 0) continue;
			for (int k = 0; k < totalsize; ++k) {
				if (j == k) continue;
				if (select_index[k] == 0) continue;
				int r = dominate_cmp(temp[j], temp[k]);
				if (r == 1) dominated[k] = 1;
				else if (r == -1) dominated[j] = 1;
			}
		}
		//
		int count = 0;
		for (int j = 0; j < totalsize; ++j) {
			if (dominated[j] == 0) count += 1;
		}
		if (count > pops) {
			//根据crowd_distance 比较
			vector<TIndividual> fast_level;
			vector<int> oindex;
			scount = 0;
			for (int k = 0; k < totalsize; ++ k) {
				if (select_index[k] == 1 && dominated[k] == 0) {
					fast_level.push_back(temp[k]);
					oindex.push_back(k);
				}
				if (select_index[k] == 0)scount += 1;
			} 
			if (fast_level.size() <= pops - scount) assert(false);
			vector<int> final_select = crowd_distance_selection(fast_level, pops - scount, max_f, min_f);
			for (int k = 0; k < final_select.size(); ++k) {
				if (final_select[k] == 1) {
					select_index[oindex[k]] = 0;
				}
			}
			/*
			vector<int> delet_index(count);
			for (int j = 0; j < delet_index.size(); ++j) delet_index[j] = j;
			random_shuffle(delet_index.begin(), delet_index.end());
			sort(delet_index.begin(), delet_index.begin() + (count - pops));
			int cindex = 0;
			for (int j = 0; j < dominated.size(); ++j) {
				if (dominated[j] == 0) {
					cindex++;
					bool flag = false;
					for (int k = 0; k < count - pops; ++k) {
						if (cindex == delet_index[k]) {
							flag = true;
							break;
						}
					}
					if (!flag) select_index[j] = 0;
				}
				
			}
			*/
			//check the number of selected
			count = 0;
			for (int j = 0; j < select_index.size(); ++j)
				if (select_index[j] == 0) count++;
			if (count != pops) assert(false);
		}
		else {
			for (int j = 0; j < totalsize; ++j) {
				if (dominated[j] == 0) select_index[j] = 0;
			}
		}
		if (count >= pops) {
			break;
		}
	}
	//select individuals from the proposed method
	/*
	scount = 0;
	for (int j = 0; j < select_index.size(); ++j) {
		if (select_index[j] == 1) {
			temp.erase(temp.begin() + (j - scount));
			scount += 1;
		}
	}
	if (temp.size() != pops) assert(false);
	*/

	//step 3: select
	//crowd_distance_selection();
	//
	int k = 0;
	for (int j = 0; j < totalsize; ++j) {
		if (select_index[j] == 0) {
			population[k].indiv = temp[j];
			k++;
			if (k > pops) assert(false);
		}
	}
	if (k < pops) assert(false);
}

void TNSGA::run(int sd, int nc, int maxfes, int rn, vector<double>& igdValue, vector<double>& hvValue)
{
	// sd: integer number for generating weight vectors
	// nc: size of neighborhood
	// mg: maximal number of generations 
	runid = rn;
	InfoMemory::algSolAllEnvi.clear();
	InfoMemory::center_gen_envi.clear();
	InfoMemory::f_center_gen_envi.clear();
	InfoMemory::predictEnviSol.clear();
	InfoMemory::hist_move.clear();
	InfoMemory::noCorrectEnviSol.clear();
	if (InfoMemory::useNN) {
		use_ep = false;// true;
	}
	else {
		use_ep = false;
	}
	//use_ep = false;
	pops = sd; //the population size
	niche = nc;
	init_uniformweight(sd); //23
	init_neighbourhood();
	init_population();
	init_detector();
	record_metric();
	while (mEvas < maxfes) {
		evolution(igdValue, hvValue);
		//cout << mEvas << "\n";
	}
	metric(igdValue, hvValue);
	int enviIndex = getEnviIndex(mEvas);
	if (InfoMemory::algSolAllEnvi.size() != enviIndex + 1) {
		if (InfoMemory::algSolAllEnvi.size() == enviIndex) {
			storeFinalPop();
		}
		else {
			assert(false);
		}
	}
	if (InfoMemory::algSolAllEnvi.size() != enviIndex + 1) assert(false);
	storeAllPop(rn);
	//char savefilename[1024];
   // sprintf(savefilename,"ParetoFront/DMOEA_%s_R%d.dat",strTestInstance,rn);
	//save_front(savefilename);
	population.clear();
}

void TNSGA::storeFinalPop() {
	int curenvi = getEnviIndex(mEvas);
	//cout << mEvas << "\t" << curenvi << "\t" << InfoMemory::algSolAllEnvi.size() << endl;
	int nobj = DynPara::objNum;
	vector<vector<double> > osolSet(population.size());
	for (int j = 0; j < osolSet.size(); ++j) osolSet[j].resize(nobj);
	for (int n = 0; n < population.size(); ++n) {
		for (int k = 0; k < nobj; ++k) {
			osolSet[n][k] = population[n].indiv.y_obj[k];
		}
	}

	for (int n = 0; n < population.size(); ++n) {
		population[n].indiv.obj_eval(mEvas);
	}

	vector<CSolution> solSet(population.size());
	//for (int j = 0; j < solSet.size(); ++j) solSet[j].resize(nobj);
	for (int n = 0; n < population.size(); ++n) {
		solSet[n].x = population[n].indiv.x_var;
		solSet[n].f = population[n].indiv.y_obj;
		//for (int k = 0; k < nobj; ++k) {
			//solSet[n].x = population[n].indiv.x_var[k];
		//}
	}
	InfoMemory::algSolAllEnvi.push_back(solSet);

	for (int n = 0; n < population.size(); ++n) {
		for (int k = 0; k < nobj; ++k) {
			population[n].indiv.y_obj[k] = osolSet[n][k];
		}
	}

	
	if (InfoMemory::algSolAllEnvi.size() != curenvi + 1) { cout << curenvi << "\t" << mEvas << "\t" << InfoMemory::algSolAllEnvi.size() << endl; assert(false); }
	//record_metric();
}
void TNSGA::storeInitPop() {
	vector<CSolution> set;
	for (int j = 0; j < pops; ++j) {
		CSolution s;
		s.x = population[j].indiv.x_var;
		s.f = population[j].indiv.y_obj;
		set.push_back(s);
	}
	InfoMemory::orignalPop.push_back(set);
}
void TNSGA::storeAllPop(int runId) {
	stringstream objstr;
	objstr << DynPara::initObjNum;
	stringstream runstr;
	runstr << runId;
	stringstream envistr;
	envistr << DynPara::enviNum;
	string proname(strTestInstance);
	string filename = preOutRoute + proname + "_M" + objstr.str() + "_E" + envistr.str() + "_R" + runstr.str() + ".sol";
	fstream ft;
	//	if(runId == 0) 
	ft.open(filename, ios::out);
	//else ft.open(filename, ios::out | ios::app);
	if (!ft.fail()) {
		//if(runId == 0) ft << max_run << endl;
		ft << runId << "\t" << DynPara::enviNum << "\t";
		for (int i = 0; i < DynPara::objNum; ++i) {
			ft << "obj" << i << "\t";
		}
		for (int i = 0; i < DynPara::dimNum; ++i) {
			ft << "x" << i << "\t";
		}
		ft << "\n";
		for (int i = 0; i < InfoMemory::algSolAllEnvi.size(); ++i) {  //each environment
			ft << i << "\t" << InfoMemory::algSolAllEnvi[i].size() << "\t" << InfoMemory::algSolAllEnvi[i][0].f.size() << "\t" << InfoMemory::algSolAllEnvi[i][0].x.size() << "\n"; //enviIndex  numberofSolution numOfObj
			for (int j = 0; j < InfoMemory::algSolAllEnvi[i].size(); ++j) {
				ft << j << "\t";
				for (int k = 0; k < InfoMemory::algSolAllEnvi[i][j].f.size(); ++k) {
					ft << InfoMemory::algSolAllEnvi[i][j].f[k] << "\t";
				}
				for (int k = 0; k < InfoMemory::algSolAllEnvi[i][j].x.size(); ++k) {
					ft << InfoMemory::algSolAllEnvi[i][j].x[k] << "\t";
				}
				ft << "\n";
			}
		}
	}
	else {
		cout << "cannot open file: " << filename << endl;
	}
	ft.close();
}

void TNSGA::save_front(char saveFilename[1024])
{
	std::fstream fout;
	fout.open(saveFilename, std::ios::out);
	for (int n = 0; n < population.size(); n++)
	{
		for (int k = 0; k < DynPara::objNum; k++)
			fout << population[n].indiv.y_obj[k] << "  ";
		fout << "\n";
	}
	fout.close();
}


void TNSGA::operator=(const TNSGA& emo)
{
	pops = emo.pops;
	population = emo.population;
	indivpoint = emo.indivpoint;
	niche = emo.niche;
}
void TNSGA::record_metric() {
	if (mEvas == 1 || mEvas % InfoMemory::sampleFre == 0) {
		vector<double> igdvalue;
		vector<double> hvvalue;
		int enviIndex = getEnviIndex(mEvas);
		vector<vector<double> > oldObjValue(pops);

		//bool use_ep = false;
		double arc_igd = 0, arc_hv = 0;
		vector<CSolution> set;
		if (use_ep) {
			set.resize(EP.size());
			for (int j = 0; j < set.size(); ++j) {
				set[j].x = EP[j].x_var;
				int odim = EP[j].x_var.size();
				if (set[j].x.size() != DynPara::dimNum) {
					for (int k = odim; k < DynPara::dimNum; ++k) {
						set[j].x.push_back(DynPara::lowBound[k] + (DynPara::upperBound[k] - DynPara::lowBound[k]) * random());
					}
				}
				if (set[j].f.size() != DynPara::objNum) set[j].f.resize(DynPara::objNum);
				objectives(set[j].x, set[j].f, mEvas);
			}
			arc_igd = calculateIGD(set, enviIndex, igdvalue, false);
			arc_hv = calculateHV(set, enviIndex, hvvalue, false);
		}
		else {
			set.resize(pops);
			for (int j = 0; j < set.size(); ++j) {
				set[j].x = population[j].indiv.x_var;
				int odim = set[j].x.size();
				if (set[j].x.size() != DynPara::dimNum) {
					for (int k = odim; k < DynPara::dimNum; ++k) {
						set[j].x.push_back(DynPara::lowBound[k] + (DynPara::upperBound[k] - DynPara::lowBound[k]) * random());
					}
				}
				if (set[j].f.size() != DynPara::objNum) set[j].f.resize(DynPara::objNum);
				objectives(set[j].x, set[j].f, mEvas);
			}
			arc_igd = calculateIGD(set, enviIndex, igdvalue, false);
			arc_hv = calculateHV(set, enviIndex, hvvalue, false);
		}
		if (InfoMemory::fes_process[runid].size() <= 0 || InfoMemory::fes_process[runid].back() != mEvas) {
			InfoMemory::fes_process[runid].push_back(mEvas);
			InfoMemory::igd_process[runid].push_back(arc_igd);
			InfoMemory::hv_process[runid].push_back(arc_hv);
		}
	}
}
void TNSGA::metric() {
	vector<double> igdvalue;
	vector<double> hvvalue;
	int enviIndex = getEnviIndex(mEvas);
	vector<vector<double> > oldObjValue(pops);

	//bool use_ep = false;
	double arc_igd = 0;
	double arc_hv = 0;
	vector<CSolution> set;
	if (use_ep) {
		set.resize(EP.size());
		for (int j = 0; j < set.size(); ++j) {
			set[j].x = EP[j].x_var;
			objectives(set[j].x, set[j].f, mEvas);
		}
		arc_igd = calculateIGD(set, enviIndex, igdvalue, false);
		arc_hv = calculateHV(set, enviIndex, igdvalue, false);
	}
	if (true) {
		set.resize(pops);
		for (int j = 0; j < set.size(); ++j) {
			set[j].x = population[j].indiv.x_var;
			objectives(set[j].x, set[j].f, mEvas);
		}
	}
	double ivalue = calculateIGD(set, enviIndex, igdvalue, false);
	double hvalue = calculateHV(set, enviIndex, hvvalue, false);
	/*
	int nobj = DynPara::objNum;
	for (int j = 0; j < population.size(); ++j) {
		oldObjValue[j].resize(nobj);
		for (int k = 0; k < nobj; ++k)
			oldObjValue[j][k] = population[j].indiv.y_obj[k];//.F(k);
		population[j].indiv.obj_eval();
	}

	for (int j = 0; j < population.size(); ++j) {
		for (int k = 0; k < nobj; ++k) {
			population[j].indiv.y_obj[k] = oldObjValue[j][k];
		}
	}*/
#ifdef TEST
	cout << DynPara::proName << "\t" << mEvas << "\t" << EP.size() << "\tE" << enviIndex << "\tarc " << arc_igd << "\t" << arc_hv << "\tpop\t" << ivalue << "\t" << hvalue << "\n";
#endif
}

void TNSGA::metric(vector<double>& igdValue, vector<double>& hvValue) {
	int enviIndex = getEnviIndex(mEvas);
	//vector<vector<double> > oldObjValue(pops);
	//int nobj = DynPara::objNum;
	/*
	for (int j = 0; j < population.size(); ++j) {
		oldObjValue[j].resize(nobj);
		for (int k = 0; k < nobj; ++k)
			oldObjValue[j][k] = population[j].indiv.y_obj[k];//.F(k);
		population[j].indiv.obj_eval();
	}
	*/
	//bool use_ep = false;

	vector<CSolution> set;
	if (use_ep) {
		set.resize(EP.size());
		for (int j = 0; j < set.size(); ++j) {
			set[j].x = EP[j].x_var;
			if (DynPara::test_SDP) objective(DynPara::proName.c_str(), set[j].x, set[j].f);
			else if (DynPara::test_DF) set[j].f = cec2018_DF_eval(DynPara::proName.c_str(), set[j].x, mEvas, DynPara::taut, DynPara::nt);
		}
	}
	else {
		set.resize(pops);
		for (int j = 0; j < set.size(); ++j) {
			set[j].x = population[j].indiv.x_var;
			//objectives(set[j].x, set[j].f);

			if (DynPara::test_SDP) objective(DynPara::proName.c_str(), set[j].x, set[j].f);
			else if (DynPara::test_DF) set[j].f = cec2018_DF_eval(DynPara::proName.c_str(), set[j].x, mEvas, DynPara::taut, DynPara::nt);
		}
	}

	calculateIGD(set, enviIndex, igdValue, true);
	calculateHV(set, enviIndex, hvValue, true);
	if (InfoMemory::fes_process[runid].back() != mEvas) {
		InfoMemory::fes_process[runid].push_back(mEvas);
		InfoMemory::igd_process[runid].push_back(igdValue[enviIndex]);
		InfoMemory::hv_process[runid].push_back(hvValue[enviIndex]);
#ifdef TEST
		//cout << mEvas << "\t" << igdValue[enviIndex] << "\t" << hvValue[enviIndex] << endl;
#endif
	}
	/*
	for (int j = 0; j < population.size(); ++j) {
		for (int k = 0; k < nobj; ++k) {
			population[j].indiv.y_obj[k] = oldObjValue[j][k];
		}
	}
	*/
#ifdef TEST
	cout << DynPara::proName << "\t" << mEvas << "\t" << EP.size() << "\t" << enviIndex << "\t" << igdValue[enviIndex] << "\t" << hvValue[enviIndex] << "\n";
#endif
}

double TNSGA::calculateIGD(const vector<CSolution>& set, const int enviIndex, vector<double>& igdValue, bool update) {
	if (population.size() <= 0) assert(false);
	readReferPointForIGD(enviIndex);
	double pointNum = igdPoints.size();
	int objNum = DynPara::objNum;
	double minValue = 0;
	double distance = 0;
	double sum = 0;
	bool pfisworse = false;
	for (int j = 0; j < pointNum; ++j) {
		for (int i = 0; i < set.size(); ++i) {
			distance = 0;
			bool wrongPoint = false;
			int smaller = 0;
			int larger = 0;
			for (int k = 0; k < objNum; ++k) {
				if (set[i].f[k] < igdPoints[j][k] && fabs(set[i].f[k] - igdPoints[j][k]) > 1e-6) { smaller++; }
				else if (set[i].f[k] > igdPoints[j][k]) larger++;
			}
			if (smaller > 0 && larger == 0) {
				distance = 0;
				if (false) {
					stringstream enviStr;
					enviStr << enviIndex;
					stringstream tautstr, ntstr;
					tautstr << DynPara::taut;
					ntstr << DynPara::nt;
					string tstr = "//taut" + tautstr.str() + "_nt" + ntstr.str() + "//";
					//string tstr = "//taut" + tautstr.str() + "_nt" + ntstr.str() + "//";
					string filename = PFDataPre + tstr + DynPara::proName + "_E" + enviStr.str() + ".pf";

					//string filename = PFDataPre + DynPara::proName + "_E" + enviStr.str() + ".pf";
					cout << filename << endl;
					cout << "The PF point is dominated by the archive solution" << endl;
					int enviIndex = getEnviIndex(mEvas);
					cout << "Envi" << enviIndex << ":\n";
					cout << mEvas << "\t" << "archive:" << i << "\t";
					for (int k = 0; k < objNum; ++k) cout << population[i].indiv.y_obj[k] << ",";
					cout << endl;
					for (int k = 0; k < objNum; ++k) cout << set[i].f[k] << ",";
					cout << endl;
					cout << "igdPoint:" << j << "\t";
					for (int k = 0; k < objNum; ++k) cout << igdPoints[j][k] << ","; cout << endl;

					vector<double> y(DynPara::objNum);
					objectives(population[i].indiv.x_var, y, mEvas);

					pfisworse = true;
					distance = 0;
					assert(false);
				}
			}
			else {
				//if(dominance(Archive[i].y_obj,igdPoints[j]))
				for (int k = 0; k < objNum; ++k) {
					distance += (set[i].f[k] - igdPoints[j][k]) * (set[i].f[k] - igdPoints[j][k]);
				}
				distance = sqrt(distance);
			}
			if (i == 0 || distance < minValue)
				minValue = distance;
		}
		sum += minValue;
	}
	double value = sum / pointNum;
	if (update) {
		igdValue[enviIndex] = value;
	}
	if (pfisworse) cout << "+";
	return value;
}
void TNSGA::readReferPointForIGD(int enviIndex) {
	//read the reference points from the file
	stringstream objStr;
	objStr << DynPara::initObjNum;
	stringstream enviStr;
	enviStr << DynPara::enviNum;
	//string proName = strTestInstance;
	if (DynPara::test_SDP) {

		string filename = PFDataPre + DynPara::proName + "_M" + objStr.str() + "_E" + enviStr.str() + ".pf";
		fstream fin(filename, ios::in);
		if (fin.fail()) {
			cout << "cannot open pf_data file:" << filename << endl;
			assert(false);
		}
		int nobj = DynPara::objNum;
		bool findIGDPoint = false;
		int curEnviNum;
		fin >> curEnviNum;
		if (curEnviNum != DynPara::enviNum) { cout << filename << "\t" << curEnviNum << "\t" << DynPara::enviNum << endl; assert(false); }
		int pointNum; int objNum; int curEnviIndex;
		for (int j = 0; j < curEnviNum; ++j) {
			fin >> curEnviIndex >> pointNum >> objNum;
			if (curEnviIndex == enviIndex) {
				if (objNum != nobj) assert(false);
				int pointIndex;
				igdPoints.resize(pointNum);
				for (int k = 0; k < pointNum; ++k) {
					igdPoints[k].resize(objNum);
					fin >> pointIndex;
					if (pointIndex != k) assert(false);
					for (int i = 0; i < objNum; ++i) {
						fin >> igdPoints[k][i];
					}
				}
				findIGDPoint = true;
				break;
			}
			else {
				string line;
				double lvalue;
				int pointIndex;
				for (int k = 0; k < pointNum; ++k) {
					//getline(fin, line);
					fin >> pointIndex;
					if (pointIndex != k) assert(false);
					for (int i = 0; i < objNum; ++i) {
						fin >> lvalue;
					}
				}
				//
			}
		}
		fin.close();
		if (!findIGDPoint) {
			cout << "cannot find reference points for the environment " << enviIndex << endl;
			assert(false);
		}
	}
	else if (DynPara::test_DF) {
		stringstream enviStr;
		enviStr << enviIndex;
		stringstream tautstr, ntstr;
		tautstr << DynPara::taut;
		ntstr << DynPara::nt;
		string tstr = "//taut" + tautstr.str() + "_nt" + ntstr.str() + "//";
		//string tstr = "//taut" + tautstr.str() + "_nt" + ntstr.str() + "//";
		string filename = PFDataPre + tstr + DynPara::proName + "_E" + enviStr.str() + ".pf";
		int num = 1000;
		fstream fin(filename, ios::in);
		if (fin.fail()) {
			cout << "cannot open pf_data file:" << filename << endl;
			assert(false);
		}
		int objNum = DynPara::objNum;
		bool findIGDPoint = false;
		string str;

		int pointNum = 1000;
		int pointIndex;
		igdPoints.clear();
		//igdPoints.resize(pointNum);
		string line;
		getline(fin, line);
		//cout << line << endl;
		//cout << pointNum << "\t" << igdPoints.size() << "\t" << objNum << endl;
		while (!fin.eof()) {
			vector<double> temp(objNum);
			//[k].resize(objNum);
			fin >> pointIndex;
			//if (pointIndex != k) { cout << pointIndex << "\t" << k << endl;  assert(false); }
			for (int i = 0; i < objNum; ++i) {
				fin >> temp[i];
				//cout << igdPoints[k][i] << "\t";
			}
			igdPoints.push_back(temp);
		}
		findIGDPoint = true;
		fin.close();
		if (!findIGDPoint) {
			cout << "cannot find reference points for the environment " << enviIndex << endl;
			assert(false);
		}
	}
	else {
		assert(false);
	}
}
void TNSGA::readReferPointForHV(int enviIndex, vector<double>& v) {
	stringstream objStr;
	objStr << DynPara::initObjNum;
	stringstream enviStr;

	//string proName = strTestInstance;
	if (DynPara::test_SDP) {
		enviStr << DynPara::enviNum;
		string filename = PFDataPre + DynPara::proName + "_M" + objStr.str() + "_E" + enviStr.str() + ".hv";
		fstream fin(filename, ios::in);
		if (fin.fail()) {
			cout << "cannot open pf_data file:" << filename << endl;
			assert(false);
		}
		int curEnviNum;
		fin >> curEnviNum;
		if (curEnviNum != DynPara::enviNum) assert(false);
		int objNum; int curEnviIndex;
		vector<double> temp;
		for (int j = 0; j < DynPara::enviNum; ++j) {
			fin >> curEnviIndex >> objNum;
			if (curEnviIndex != j) assert(false);
			temp.resize(objNum);
			for (int k = 0; k < objNum; ++k) {
				fin >> temp[k];
			}
			if (curEnviIndex == enviIndex) {
				if (v.size() != objNum) v.resize(objNum);
				for (int k = 0; k < objNum; ++k)
					v[k] = temp[k];
				break;
			}
		}
	}
	else if (DynPara::test_DF) {
		enviStr << enviIndex;
		stringstream tautstr, ntstr;
		tautstr << DynPara::taut;
		ntstr << DynPara::nt;
		string tstr = "//taut" + tautstr.str() + "_nt" + ntstr.str() + "//";
		//string tstr = "//taut" + tautstr.str() + "_nt" + ntstr.str() + "//";
		string filename = PFDataPre + tstr + DynPara::proName + "_E" + enviStr.str() + ".hv";

		//string filename = PFDataPre + DynPara::proName + "_E" + enviStr.str() + ".hv";
		fstream fin(filename, ios::in);
		if (fin.fail()) {
			cout << "cannot open pf_data file:" << filename << endl;
			assert(false);
		}
		string line;
		getline(fin, line);
		vector<double> temp;
		int objNum = DynPara::objNum;

		if (v.size() != objNum) v.resize(objNum);
		min_pf_value.resize(objNum);
		for (int k = 0; k < objNum; ++k) {
			fin >> v[k];
		}
		for (int k = 0; k < objNum; ++k) {
			fin >> min_pf_value[k];
		}
	}
}
double TNSGA::calculateHV(const vector<CSolution>& set, const int enviIndex, vector<double>& hvValue, bool update) {
	if (population.size() == 0) assert(false);
	//vector<double> hvPoint(DynPara::objNum);
	if (hvPoint.size() != DynPara::objNum) hvPoint.resize(DynPara::objNum);
	int nobj = DynPara::objNum;
	readReferPointForHV(enviIndex, hvPoint);


	vector<vector<double> > solSet;
	for (int i = 0; i < set.size(); ++i) {
		vector<double> x(nobj);
		//solSet[i].resize(DynPara::objNum);
		bool smallerOne = true;
		for (int k = 0; k < nobj; ++k) {
			x[k] = set[i].f[k];
			if (x[k] > hvPoint[k]) {
				smallerOne = false;
			}
		}
		if (smallerOne) { solSet.push_back(x); }
	}
	if (DynPara::test_DF) {
		for (int k = 0; k < nobj; ++k) {
			if (hvPoint[k] == min_pf_value[k]) {
				//cout << "(" << hvPoint[k] << "," << min_pf_value[k] << ")\t";
				hvPoint[k] = min_pf_value[k] + 1;

			}
		}

		for (int j = 0; j < solSet.size(); ++j) {
			for (int k = 0; k < nobj; ++k) {

				solSet[j][k] = (solSet[j][k] - min_pf_value[k]) / (hvPoint[k] - min_pf_value[k]);
			}
		}
		for (int k = 0; k < hvPoint.size(); ++k) {
			hvPoint[k] = 1.1;
		}

	}
	double value = hypervolume(solSet, nobj, hvPoint);
	if (update) hvValue[enviIndex] = value;
	return value;
}

double TNSGA::cal_pop_x_error() {
	double error = 0;
	for (int i = 0; i < pops; ++i) {
		vector<double> optx;
		getOptimalSolution(DynPara::proName, population[i].indiv.x_var, optx);
		error += calDistance(optx, population[i].indiv.x_var);
	}
	return error / pops;
}

double TNSGA::cal_EP_x_error() {
	double error = 0;
	for (int i = 0; i < EP.size(); ++i) {
		vector<double> optx;
		getOptimalSolution(DynPara::proName, EP[i].x_var, optx);
		error += calDistance(optx, EP[i].x_var);
	}
	return error / EP.size();
}

double TNSGA::cal_pop_f_error() {
	double error = 0;
	for (int i = 0; i < pops; ++i) {
		vector<double> optx;
		getOptimalSolution(DynPara::proName, population[i].indiv.x_var, optx);
		vector<double> optf(DynPara::objNum);
		objectives(optx, optf, mEvas);
		error += calDistance(optf, population[i].indiv.y_obj);
	}
	return error / pops;
}

double TNSGA::cal_EP_f_error() {
	double error = 0;

	for (int i = 0; i < EP.size(); ++i) {
		vector<double> optx;
		getOptimalSolution(DynPara::proName, EP[i].x_var, optx);
		vector<double> optf(DynPara::objNum);
		objectives(optx, optf, mEvas);
		vector<double> tempf(DynPara::objNum);
		objectives(EP[i].x_var, tempf, mEvas);
		error += calDistance(optf, EP[i].y_obj);
		if (false && calDistance(tempf, EP[i].y_obj) >= 0.0001) {
			OutputVector(tempf);
			cout << "\t";
			OutputVector(EP[i].y_obj);
			cout << "\t";
			vector<double> tt(DynPara::objNum);
			objective(DynPara::proName.c_str(), EP[i].x_var, tt);
			OutputVector(tt);
			cout << endl;
			assert(false);
		}
	}
	return error / EP.size();
}

vector<double> TNSGA::cal_pop_center() {
	int num_dim = DynPara::dimNum;
	vector<double> center(num_dim, 0);
	for (int j = 0; j < pops; ++j) {
		for (int k = 0; k < num_dim; ++k) {
			center[k] = center[k] + population[j].indiv.x_var[k];
		}
	}
	for (int k = 0; k < num_dim; ++k) {
		center[k] = center[k] / pops;
	}
	return center;
}

void TNSGA::resample_pop(const vector<vector<double> >& sample, vector<double>& igdValue, vector<double>& hvValue) {
	for (int j = 0; j < pops; ++j) {
		if (enviChangeNextFes(mEvas) && mEvas < DynPara::totalFes) {
			metric(igdValue, hvValue);
			storeFinalPop();
			int preDimNum = DynPara::dimNum; int preObjNum = DynPara::objNum;
			introduceDynamic(mEvas);
			defineBoundary(DynPara::proName);
			bool varDimChange = false;
			bool objDimChange = false;
			if (preDimNum != DynPara::dimNum) varDimChange = true;
			if (preObjNum != DynPara::objNum) objDimChange = true;
			if (preDimNum != DynPara::dimNum || preObjNum != DynPara::objNum) {
				//cout << population.size() << "#";
				if (DynPara::proName != "SDP12" && DynPara::proName != "SDP13") assert(false);
				reactToDimChange(varDimChange, objDimChange);
			}
			//cout << "+++\t" << population.size() << "\t" << DynPara::dimNum << "\t" << DynPara::objNum << "\t";
		}
		if (mEvas >= DynPara::totalFes) break;
		int k = random() * sample.size();
		if (k >= sample.size()) k = 0;
		TIndividual p;
		p.x_var = sample[k];
		TIndividual child, child2;
		realbinarycrossover(population[j].indiv, p, child, child2);
		//cout << j << "\t" << calDistance(child.x_var, population[j].indiv.x_var) << "," << calDistance(child.x_var, p.x_var) << "\t";
		realmutation(child, 1.0 / DynPara::dimNum);
		//cout << calDistance(child.x_var, population[j].indiv.x_var) <<  "," << calDistance(child.x_var, p.x_var) <<"\t";
		child.obj_eval(mEvas);
		population[j].indiv = child;
		update_reference(child);
		//update_problem(child, n);
		if (use_ep) update_EP(child);
		mEvas++;
		record_metric();
	}
}

void TNSGA::outputVector(const vector<double> x) {
	cout << "(";
	for (int j = 0; j < x.size(); ++j) {
		cout << x[j] << ",";
	}
	cout << ")";
}

void TNSGA::eval_hist_move() {
	int b = InfoMemory::detectedEnviSol.size() - 1;
	int a = b - 1;
	vector<double> x(DynPara::dimNum, 0);
	for (int k = 0; k < DynPara::dimNum; ++k) {
		for (int j = 0; j < InfoMemory::detectedEnviSol[b].size(); ++j) {
			x[k] += fabs(InfoMemory::detectedEnviSol[b][j].x[k] - InfoMemory::detectedEnviSol[a][j].x[k]);
		}
		x[k] /= InfoMemory::detectedEnviSol[b].size();
	}
	InfoMemory::hist_move.push_back(x);
}

#endif