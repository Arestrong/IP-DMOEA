#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "TrainingPoint.h"
using namespace std;
using namespace Eigen;

TrainingPoint::TrainingPoint(){

}
TrainingPoint::TrainingPoint(int *tx, int *ty, int numx, int numy){
	this->x.resize(numx);
	this->orignalx.resize(numx);
	this->orignaly.resize(numy);
	this->y.resize(numy);
	for (int i = 0; i < x.size(); ++i){
		x[i] = tx[i];
		orignalx[i] = tx[i];
	}
	for (int i = 0; i < y.size(); ++i){
		orignaly[i] = ty[i];
		y[i] = ty[i];
	}
}
void TrainingPoint::setData(VectorXd a, VectorXd b, int anum, int bnum){
	if (x.size() != a.size()) x.resize(a.size());
	if (y.size() != b.size()) y.resize(b.size());
	if (orignalx.size() != a.size()) orignalx.resize(a.size());
	if (orignaly.size() != b.size()) orignaly.resize(b.size());
	x = a;
	y = b;
	orignalx = a;
	orignaly = b;
}

void TrainingPoint::normilizeXMinMax(vector<double> maxv, vector<double> minv, int dim){
	if (dim != orignalx.size()){
		cout << "The dimension of the input vecotor is wrong... " << orignalx.size() << " != " << dim << endl;
		assert(false);
	}
	for (int i = 0; i < orignalx.size(); ++i){
		x[i] = 2 * (orignalx[i] - minv[i]) / (maxv[i] - minv[i]) - 1;
	}
}
void TrainingPoint::normilizeYMinMax(vector<double> maxv, vector<double> minv, int dim){
	if (dim != orignaly.size()){
		cout << "The dimension of the input vecotor is wrong... " << orignaly.size() << " != " << dim << endl;
		assert(false);
	}
	for (int i = 0; i < orignaly.size(); ++i){
		y[i] = 2 * (orignaly[i] - minv[i]) / (maxv[i] - minv[i]) - 1;
	}
}
void TrainingPoint::normilizeXMeanStd(vector<double> mean, vector<double> std, int dim){
	if (dim != orignalx.size()){
		cout << "The dimension of the input vecotor is wrong... " << orignalx.size() << " != " << dim << endl;
		assert(false);
	}
	for (int i = 0; i < orignalx.size(); ++i){
		x[i] = (orignalx[i] - mean[i]) / std[i];
	}
}
void TrainingPoint::normilizeYMeanStd(vector<double> mean, vector<double> std, int dim){
	if (dim != orignaly.size()){
		cout << "The dimension of the input vecotor is wrong... " << orignaly.size() << " != " << dim << endl;
		assert(false);
	}
	for (int i = 0; i < orignaly.size(); ++i){
		y[i] = (orignaly[i] - mean[i]) / std[i];
	}
}

void TrainingPoint::restorePYMaxMin(const vector<double> maxv, const vector<double> minv, const int ydim){
	if (maxv.size() != ydim || minv.size() != ydim) assert(false);
	for (int j = 0; j < ydim; ++j){
		y[j] = (y[j] + 1) / 2 * (maxv[j] - minv[j]) + minv[j];
	}
}
void TrainingPoint::restorePYMeanStd(const vector<double> mean, const vector<double> std, const int ydim){
	if (mean.size() != ydim || std.size() != ydim) assert(false);
	for (int j = 0; j < ydim; ++j){
		y[j] = y[j] * std[j] + mean[j];
	}
}

bool TrainingPoint::operator<(const TrainingPoint &p){
	if (y.size() == 0) { cout << "wrong in TrainingPoint::operator<(): the size of y is 0\n"; assert(false); }
	if (y.size() == 1){
		return y[0] < p.y[0];
	}
	else{
		//multiple objectives
		int smaller = 0;
		int equal = 0;
		int larger = 0;
		for (int j = 0; j < y.size(); ++j){
			if (y[j] < p.y[j]){
				smaller++;
			}
			else if (y[j] > p.y[j]) larger++;
			else equal++;
		}
		if (smaller > 0 && larger == 0) return true;
		else return false;
	}
}

TrainingPoint& TrainingPoint::operator=(const TrainingPoint &p){
	if (orignalx.size() != p.orignalx.size()){
		orignalx.resize(p.orignalx.size());
		x.resize(p.x.size());
	}
	if (orignaly.size() != p.orignaly.size()){
		orignaly.resize(p.orignaly.size());
		y.resize(p.y.size());
	}
	for (int j = 0; j < orignalx.size(); ++j){
		orignalx[j] = p.orignalx[j];
	}
	for (int j = 0; j < x.size(); ++j) x[j] = p.x[j];
	for (int j = 0; j < orignaly.size(); ++j) orignaly[j] = p.orignaly[j];
	for (int j = 0; j < y.size(); ++j) y[j] = p.y[j];
	return *this;
}

double TrainingPoint::xdistance(const TrainingPoint &p) {
	double dis = 0;
	if (orignalx.size() != p.orignalx.size()) assert(false);
	for (int j = 0; j < orignalx.size(); ++j) {
		dis += (orignalx[j] - p.orignalx[j])* (orignalx[j] - p.orignalx[j]);
	}
	dis = sqrt(dis);
	return dis;
}

double TrainingPoint::ydistance(const TrainingPoint &p) {
	double dis = 0;
	if (orignaly.size() != p.orignaly.size()) assert(false);
	for (int j = 0; j < orignaly.size(); ++j) {
		dis += (orignaly[j] - p.orignaly[j])* (orignaly[j] - p.orignaly[j]);
	}
	dis = sqrt(dis);
	return dis;
}
