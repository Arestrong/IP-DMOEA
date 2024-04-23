#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "TrainingPoint.h"
using namespace Eigen;
using namespace std;

#define NORMMAXMIN   //use the max min to normilize the input and output of the training set

/*
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
	void normilizeXMinMax(vector<double> maxv, vector<double> minv,int dim);
	void normilizeYMinMax(vector<double> maxv, vector<double> minv, int dim);
	void normilizeXMeanStd(vector<double> mean, vector<double> std, int dim);
	void normilizeYMeanStd(vector<double> mean, vector<double> std, int dim);
	void restorePYMaxMin(const vector<double> maxv, const vector<double> minv, const int ydim);
	void restorePYMeanStd(const vector<double> mean, const vector<double> std, const int ydim);
};
*/

enum activefun { sigmodfun, linearfun, tansigfun, ReLUfun, softmaxfun,tanhfun };

class NeuralNetwork{
public:
	int numLayer; //the number of total layers M + 1: 0, ....,M; not include the input layer
	vector<int> numNeuroEachLayer; //include the input layer
	VectorXd input; //the input variant of the NN
	VectorXd output; //the output vector of the NN;

	vector<VectorXd> bais; //the bias in each neuron in each layer;
	vector<MatrixXd> weights; //the weight matrix in each layer;
	
	//train
	vector<VectorXd> s; //the sensitivities s
	vector<VectorXd> nvalue; //the value of the neuro in each layer=weight sum of the input
	vector<VectorXd> avalue; //the output of the neuro in each layer avalue=f(nvalue)
	vector<MatrixXd> deviationOfN; //the deviation of the sigmo function for the weight sum of input

	double learnrate; //the learn rate of the steepest gradient
	int totalParaNum; // the total number of weights and bias in hidden and output layers in the network
	int afunHiddenLayer;  // the active function used in the hidden layer
	int afunOutputLayer;  //the active function used in the output layer

	//the parameter that minimize the error obtained during the training process
	vector<VectorXd> minbais; //the bias in each neuron in each layer;
	vector<MatrixXd> minweights; //the weight matrix in each layer;
	double lower;
	double upper;
public:
	
	void gaussianInitializePara();
	void randomInitializePara(); //initilize the parameter of the nn
	void initilizeParaNgWi90(); //initilize the parameter of the nn with the method of NgWi90
	void setStructure(int numlayer, vector<int> numNeuroEachLayer, int hiddenafun, int outputafun);
	void setPara(double para[], int pnum); //set the parameter of the nn, weights and bias
	void getOutput(double x[], double y[], int xnum, int ynum);
	void setParaBound(double lower, double upper);

	double sigmodDeviation(double n, double a);
	double linearDeviation(double n);
	double tansigDeviation(double n, double a);
	double ReLUDeviation(double n, double a);
	double softmaxDeviation(double n, double a);
	double tanhDeviation(double n, double a);
	double deviation(int whichlayer, double n, double a); //deviation of (eTe)
	double eDeviation(int whichlayer, double n, double a); //deviation of e

	double activeFun(int whichlayer, double x);
	double sigmod(double x);
	double linear(double x);
	double tansig(double x);
	double ReLU(double x);
	double softmax(double x);
	double tanh(double x);

	void showParam();

	///train the nn, get the parameters of the nn
	void trainBySDBPIncreLearn(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int maxEpochs);
	void trainBySDBPBatch(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int maxEpochs);
	void trainBySDBPMiniBatch(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int batchSize, int maxEpochs);
	//batching; the parameters are updated only after the entire training set has been presented
	void trainByMomentumBP(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double initLearnRate, double momentumCoeff, int maxEpochs); //MOBP
	void trainByVarLearnRateBP(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double initLearnRate, double percentage, double momenCoeff, double incFactor, double decFactor, int maxEpochs); //VLBP
	//n: the number of generation to reset the search direction p as the steepest gradiant
	void trainByConjugateGradientBP(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double initLearnRate, double tstepsize, int maxEpochs, int n, double tol, const vector<double> ymax, const vector<double> ymin);//CGBP
	void trainByLevenbergMarquardtBP(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double mu, double factor, int maxEpochs, const vector<double> ymax, const vector<double> ymin);//LMBP
	void trainByGPSO(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int popsize, double w, double c1, double c2, int generation);
	double calErrorWithPara(const vector<double> para, const vector<TrainingPoint> data, const int datanum, const int xnum, const int ynum);
	void trainByDE(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int popsize, double F, double CR, int generation);
	void trainByLMBPCorssValidation(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double mu, double factor, int maxEpochs, const vector<double> ymax, const vector<double> ymin);//LMBP
	void trainByBayesianRegularization(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double mu, double factor, int maxEpochs, const vector<double> ymax, const vector<double> ymin);//LMBP

	void outNormilizePredictFile(vector<TrainingPoint> &testData, int datanum, int xnum, int ynum, const vector<double> ymax, const vector<double> ymin, const char filename[]);
	double regressionOutput(vector<TrainingPoint> &testData, int datanum, int xnum, int ynum, const vector<double> ymax, const vector<double> ymin);//return coefficient factor

	//simple steepest descent backpropagation (SDBP)
	void predict(VectorXd x, VectorXd &y, int xnum, int ynum);
	void predictData(); //output the simulated data into file for visiual
	double gaussian(double mean= 0, double std = 1.0);
	double random(double low, double upper);
	int random_int(int low, int high);
	double getPrectErr(vector<TrainingPoint> testData,int datanum, int xnum, int ynum);
	double getSumSquareErr(vector<TrainingPoint> testData, int datanum, int xnum, int ynum);
	double getRegularationErr(double afa, double beta, vector<TrainingPoint> testData, int datanum, int xnum, int ynum);
	double getParaErr();
	void outputSimuFunToFile(vector<TrainingPoint> simuSet, int datanum, char filename[]);
	double getErrWithPara(vector<MatrixXd> weights, vector<VectorXd> bais, vector<TrainingPoint> testData, int datanum, int xnum, int ynum);
	void predictWithNNPara(vector<MatrixXd> weights, vector<VectorXd> bais, VectorXd x, VectorXd &y, int xnum, int ynum);
	void setNNParaValue(const vector<double> para);
	void outputNNParaToFile(fstream &ft);

	//update the minweights and minbais as the current weights and bais
	void updateMinErrPara();
	void setParaAsMiniPara();
	double predictDataSet(vector<TrainingPoint> &testData, int datanum, int xnum, int ynum, bool normilized, const vector<double> ymax, const vector<double> ymin);

	
	NeuralNetwork& operator=(const NeuralNetwork &p);

};


#endif