///trainingpoint class
#ifndef TRAININGPOINT_H
#define TRAININGPOINT_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
using namespace Eigen;
using namespace std;

class TrainingPoint{
public:
	VectorXd x;
	VectorXd y;
	VectorXd orignalx;
	VectorXd orignaly;
public:
	TrainingPoint();
	TrainingPoint(int *tx, int *ty, int numx, int numy);
	void setData(VectorXd a, VectorXd b, int anum, int bnum);
	void normilizeXMinMax(vector<double> maxv, vector<double> minv, int dim);
	void normilizeYMinMax(vector<double> maxv, vector<double> minv, int dim);
	void normilizeXMeanStd(vector<double> mean, vector<double> std, int dim);
	void normilizeYMeanStd(vector<double> mean, vector<double> std, int dim);
	void restorePYMaxMin(const vector<double> maxv, const vector<double> minv, const int ydim);
	void restorePYMeanStd(const vector<double> mean, const vector<double> std, const int ydim);
	bool operator<(const TrainingPoint &p);
	TrainingPoint& operator=(const TrainingPoint &p);
	double xdistance(const TrainingPoint &p);
	double ydistance(const TrainingPoint &p);
};


#endif