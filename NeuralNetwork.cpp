#include "NeuralNetwork.h"
#include "TrainingPoint.h"
//#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>
#include <time.h>

#ifdef UNIX
#include <unistd.h>
#include <sys/types.h>  
#include <sys/stat.h> 
#include <sys/time.h>
#endif
using namespace std;
using namespace Eigen;

//#define DEBUG
/*
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
*/
double NeuralNetwork::gaussian(double mean, double std) {
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if (phase == 0) {
		do {
			double U1 = (double)rand() / ((double)RAND_MAX + 1);
			double U2 = (double)rand() / ((double)RAND_MAX + 1);

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;
	return X * std + mean;

	std::random_device rd;
	std::mt19937 gen(rd());
	if (std <= 0) { cout << mean << "\t" << std << endl; }
	// values near the mean are the most likely
	// standard deviation affects the dispersion of generated values from the mean
	std::normal_distribution<double> d(mean, std);
	return d(gen);
}

double NeuralNetwork::random(double lower, double upper) {
	return lower + (double)rand() / RAND_MAX * (upper - lower);
}

int NeuralNetwork::random_int(int low, int high) {
	int r = low + (double)rand() / RAND_MAX * (high + 1 - low);
	if (r < low) assert(false);
	if (r > high) r = high;
	return r;
}

void NeuralNetwork::setParaBound(double lower, double upper) {
	this->lower = lower;
	this->upper = upper;
}

///
void NeuralNetwork::randomInitializePara() { //set the initialze weight and bias of the NN

	for (int i = 0; i < weights.size(); ++i) {
		for (int j = 0; j < weights[i].rows(); ++j) {
			for (int k = 0; k < weights[i].cols(); ++k) {
				//weights[i](j,k) = gaussian();
				weights[i](j, k) = random(-0.5, 0.5);
			}
		}
	}
	for (int i = 0; i < bais.size(); ++i) {
		for (int j = 0; j < bais[i].rows(); ++j) {
			for (int k = 0; k < bais[i].cols(); ++k) {
				//	bais[i](j, k) = gaussian();
				bais[i](j, k) = random(-0.5, 0.5);
			}
		}
	}
	learnrate = 0.5;
}

void NeuralNetwork::initilizeParaNgWi90() {
	for (int i = 0; i < weights.size(); ++i) {
		for (int j = 0; j < weights[i].rows(); ++j) {
			for (int k = 0; k < weights[i].cols(); ++k) {
				//weights[i](j,k) = gaussian();
				weights[i](j, k) = random(-0.5, 0.5);
			}
		}
	}
	for (int i = 0; i < bais.size(); ++i) {
		for (int j = 0; j < bais[i].rows(); ++j) {
			for (int k = 0; k < bais[i].cols(); ++k) {
				//	bais[i](j, k) = gaussian();
				bais[i](j, k) = random(-0.5, 0.5);
			}
		}
	}
	//the first leyer connected to the input layer
	double magnitude = 0.7*pow(weights[1].rows(), 1.0 / input.size());
	for (int j = 0; j < weights[1].rows(); ++j) {
		///
		VectorXd rw(weights[1].cols());
		for (int i = 0; i < rw.size(); ++i) rw[i] = random(-0.5, 0.5);
		double length = 0;
		for (int i = 0; i < rw.size(); ++i) length += rw[i] * rw[i];
		length = sqrt(length);
		for (int i = 0; i < rw.size(); ++i) rw[i] = rw[i] / length * magnitude;
		for (int k = 0; k < weights[1].cols(); ++k) {
			weights[1](j, k) = rw[k];
		}
	}
	//
	for (int j = 0; j < bais[1].size(); ++j) {
		bais[1][j] = random(-magnitude, magnitude);
	}
	learnrate = 0.5;
}

void NeuralNetwork::gaussianInitializePara() { //set the initialze weight and bias of the NN

	for (int i = 0; i < weights.size(); ++i) {
		for (int j = 0; j < weights[i].rows(); ++j) {
			for (int k = 0; k < weights[i].cols(); ++k) {
				weights[i](j, k) = gaussian();
				//	weights[i](j, k) = random(-0.5, 0.5);
			}
		}
	}
	for (int i = 0; i < bais.size(); ++i) {
		for (int j = 0; j < bais[i].rows(); ++j) {
			for (int k = 0; k < bais[i].cols(); ++k) {
				bais[i](j, k) = gaussian();
				//bais[i](j, k) = random(-0.5, 0.5);
			}
		}
	}
	learnrate = 0.5;
}

///numlayer = input + hidden + output
//numLayer = input + hidden + output
void NeuralNetwork::setStructure(int numlayer, vector<int> numNeuroEachLayer, int hiddenafun, int outputafun) {
	numLayer = numlayer;
	if (numNeuroEachLayer.size() == 0) return;
	this->numNeuroEachLayer.resize(numNeuroEachLayer.size());
	if (numNeuroEachLayer.size() != numlayer) {
		cout << "numNeuroEachLayer.size() != numlayer = 1\n";
		assert(false);
	}
	for (int i = 0; i < this->numNeuroEachLayer.size(); ++i) {
		this->numNeuroEachLayer[i] = numNeuroEachLayer[i];
	}
	//set the input neuroal
	int outlayerindex = this->numNeuroEachLayer.size() - 1;
	input.resize(this->numNeuroEachLayer[0]);
	output.resize(this->numNeuroEachLayer[outlayerindex]);

	bais.resize(numLayer);
	weights.resize(numLayer);
	for (int i = 0; i < bais.size(); ++i) {
		bais[i].resize(this->numNeuroEachLayer[i]);
		if (i >= 1)
			weights[i].resize(this->numNeuroEachLayer[i], this->numNeuroEachLayer[i - 1]);
	}

	minbais.resize(numLayer);
	minweights.resize(numLayer);
	for (int i = 0; i < minbais.size(); ++i) {
		minbais[i].resize(this->numNeuroEachLayer[i]);
		if (i >= 1)
			minweights[i].resize(this->numNeuroEachLayer[i], this->numNeuroEachLayer[i - 1]);
	}

	//set parameter for training
	s.resize(numLayer);
	nvalue.resize(numLayer);
	avalue.resize(numLayer);
	deviationOfN.resize(numLayer);

	for (int i = 0; i < s.size(); ++i) {
		s[i].resize(this->numNeuroEachLayer[i]);
		nvalue[i].resize(this->numNeuroEachLayer[i]);
		avalue[i].resize(this->numNeuroEachLayer[i]);
		deviationOfN[i].resize(this->numNeuroEachLayer[i], this->numNeuroEachLayer[i]);
	}

	afunHiddenLayer = hiddenafun;
	afunOutputLayer = outputafun;

	//
	int count = 0;
	for (int i = 1; i < numLayer; ++i) {
		count += this->numNeuroEachLayer[i];
		count += this->numNeuroEachLayer[i] * this->numNeuroEachLayer[i - 1];
	}
	totalParaNum = count;

	setParaBound(-2000, 2000);
}

//set the weight and bias into the hidden and output layer
//weight + bias in para[]
void NeuralNetwork::setPara(double para[], int pnum) {
	if (pnum != totalParaNum) {
		cout << "the input para[] with (" << pnum <<
			") is not enough to set the parameters" << totalParaNum << endl;
	}
	int count = 0;
	for (int i = 1; i < numLayer; ++i) {
		for (int j = 0; j < weights[i].rows(); ++j) {
			for (int k = 0; k < weights[i].cols(); ++k) {
				weights[i](j, k) = para[count];
				count++;
			}
		}
	}
	for (int i = 1; i < numLayer; ++i) {
		for (int j = 0; j < bais[i].size(); ++j) {
			bais[i][j] = para[count];
			count++;
		}
	}
	if (count != totalParaNum) {
		cout << "error in setPara in NeuralNetwork\n";
		assert(false);
	}
}

double NeuralNetwork::sigmod(double x) {
	return 1.0 / (1 + exp(-x));
}
double NeuralNetwork::linear(double x) {
	return x;
}
double NeuralNetwork::tansig(double x) {
	return 2.0 / (1 + exp(-2 * x)) - 1;
}
double NeuralNetwork::tanh(double x) {
	return (1 - exp(-2 * x)) / (1 + exp(-2 * x));
}
double NeuralNetwork::ReLU(double x) {
	if (x <= 0) return 0;
	else return x;
}
double NeuralNetwork::softmax(double x) {
	return log(1 + exp(x));
}

void NeuralNetwork::showParam() {
	for (int i = 1; i < numLayer; ++i)
		//	cout << "w" << i <<"\n" << weights[i] << endl;
		for (int i = 1; i < numLayer; ++i) {
			//	cout << "b" << i << "\n" << bais[i] << endl;
		}
	///
	for (int i = 1; i < numLayer; ++i) {
		cout << "layer" << i << endl;
		cout << "w:\n";
		for (int j = 0; j < numNeuroEachLayer[i]; ++j) {
			for (int k = 0; k < numNeuroEachLayer[i - 1]; ++k) {
				cout << weights[i](j, k) << "\t";// weights[-bais[i][j] / weights[i](j, k) << "\t";
			}
			cout << endl;
		}
		cout << "bais: ";
		for (int j = 0; j < numNeuroEachLayer[i]; ++j) {
			cout << bais[i][j] << "\t";
		}
		cout << endl;
	}
}

void NeuralNetwork::outputNNParaToFile(fstream &ft) {
	for (int i = 1; i < numLayer; ++i)
		ft << "w" << i << "\n" << weights[i] << endl;
	for (int i = 1; i < numLayer; ++i) {
		ft << "b" << i << "\n" << bais[i] << endl;
	}
}

void NeuralNetwork::getOutput(double x[], double y[], int xnum, int ynum) {
	if (xnum != input.size()) { cout << "input size is wrong " << xnum << " " << input.size() << endl; assert(false); }
	if (ynum != output.size()) assert(false);

	for (int i = 0; i < input.size(); ++i) input[i] = x[i];
	vector<VectorXd> xa(numLayer);
	xa[0] = input;
	for (int i = 1; i < numLayer; ++i) {
		xa[i] = weights[i] * xa[i - 1] + bais[i];
		for (int j = 0; j < xa[i].size(); ++j) xa[i][j] = activeFun(i, xa[i][j]);
	}
	output = xa[numLayer - 1];
	for (int j = 0; j < output.size(); ++j) y[j] = output[j];
}

void NeuralNetwork::predict(VectorXd x, VectorXd &y, int xnum, int ynum) {
	if (xnum != input.size()) { cout << "input size is wrong " << xnum << " " << input.size() << endl; assert(false); }
	if (ynum != output.size()) { cout << "the ydim of the predict varible " << ynum << " != " << "the dim of the neuralnetwork " << output.size() << "\n"; assert(false); }

	/*	input = x;
		vector<VectorXd> xa(numLayer);
		xa[0] = input;
		for (int i = 1; i < numLayer; ++i){
			xa[i] = weights[i] * xa[i - 1] + bais[i];
			for (int j = 0; j < xa[i].size(); ++j) xa[i][j] = activeFun(i, xa[i][j]);
		}
		output = xa[numLayer - 1];
		for (int j = 0; j < output.size(); ++j) y[j] = output[j];*/
		//showParam();
	//	cout << "x:\n"; cout << x << endl;
	input = x;
	//the output of the neuro in each layer
	avalue[0] = input;
	for (int k = 1; k < numLayer; ++k) {
		nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
		for (int l = 0; l < nvalue[k].size(); ++l)
			avalue[k][l] = activeFun(k, nvalue[k][l]);
		//	cout << "a" << k << "\n"; cout << avalue[k] << endl;
	}
	output = avalue[numLayer - 1];

	y = output;
	//cout << output[0] << " " << y[0] << endl;
}

void NeuralNetwork::predictWithNNPara(vector<MatrixXd> weights, vector<VectorXd> bais, VectorXd x, VectorXd &y, int xnum, int ynum) {
	if (xnum != input.size()) { cout << "input size is wrong " << xnum << " " << input.size() << endl; assert(false); }
	if (ynum != output.size()) assert(false);

	input = x;
	//the output of the neuro in each layer
	avalue[0] = input;
	for (int k = 1; k < numLayer; ++k) {
		nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
		for (int l = 0; l < nvalue[k].size(); ++l)
			avalue[k][l] = activeFun(k, nvalue[k][l]);
		//	cout << "a" << k << "\n"; cout << avalue[k] << endl;
	}
	output = avalue[numLayer - 1];
	y = output;
	//cout << output[0] << " " << y[0] << endl;
}

double NeuralNetwork::sigmodDeviation(double n, double a) {
	return (1 - a)*a;
}
double NeuralNetwork::linearDeviation(double n) {
	return 1;
}
double NeuralNetwork::tansigDeviation(double n, double a) {
	return 1 - a * a;
	//return (1 - a)*a;
}
double NeuralNetwork::ReLUDeviation(double n, double a) {
	if (n < 0) return 0;
	else return 1;
}
double NeuralNetwork::softmaxDeviation(double n, double a) {
	return (1 - a)*a;
}
double NeuralNetwork::tanhDeviation(double n, double a) {
	return 1 - a * a;
}

double NeuralNetwork::deviation(int whichlayer, double n, double a) {
	int whichfun = 0;
	if (whichlayer < numLayer - 1) {
		whichfun = afunHiddenLayer;
	}
	else whichfun = afunOutputLayer;
	if (whichfun == 0) return sigmodDeviation(n, a);
	else  if (whichfun == 1) return linearDeviation(n);
	else if (whichfun == 2) return tansigDeviation(n, a);
	else if (whichfun == 3) return ReLUDeviation(n, a);
	else if (whichfun == 4) return softmaxDeviation(n, a);
	else if (whichfun == 5) return tanhDeviation(n, a);
	else {
		cout << "There is no active function " << whichfun << " for devation....\n";
		assert(false);
	}
}
double NeuralNetwork::eDeviation(int whichlayer, double n, double a) {
	int whichfun = 0;
	if (whichlayer < numLayer - 1) {
		whichfun = afunHiddenLayer;
	}
	else whichfun = afunOutputLayer;
	if (whichfun == 0) return sigmodDeviation(n, a);
	else  if (whichfun == 1) return linearDeviation(n);
	else if (whichfun == 2) return tansigDeviation(n, a);
	else if (whichfun == 3) return ReLUDeviation(n, a);
	else if (whichfun == 4) return softmaxDeviation(n, a);
	else if (whichfun == 5) return tanhDeviation(n, a);
	else {
		cout << "There is no active function " << whichfun << " for devation....\n";
		assert(false);
	}
}
double NeuralNetwork::activeFun(int whichlayer, double x) {
	int whichfun = 0;
	if (whichlayer < numLayer - 1) {
		whichfun = afunHiddenLayer;
	}
	else whichfun = afunOutputLayer;
	if (whichfun == 0) return sigmod(x);
	else if (whichfun == 1) return linear(x);
	else if (whichfun == 2) return tansig(x);
	else if (whichfun == 3) return ReLU(x);
	else if (whichfun == 4) return softmax(x);
	else if (whichfun == 5) return tanh(x);
	else {
		cout << "There is no acitvfun " << whichfun << endl;
		assert(false);
	}
}

void NeuralNetwork::updateMinErrPara() {
	minweights = weights;
	minbais = bais;
}
void NeuralNetwork::setParaAsMiniPara() {
	weights = minweights;
	bais = minbais;
}
//
void NeuralNetwork::trainBySDBPIncreLearn(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int maxEpochs) {
	//the less the training sample in one time, the faster the convergence
	//incremental training: single point cycle 
	learnrate = 0.1;
	int ite = 0;
	double error = 0;
	const double accurate = 0.01;
	bool finish = false;
	double preerror = getPrectErr(data, datanum, xnum, ynum);
	updateMinErrPara();

	//	int maxEpochs = 10;
	while (!finish) {
		for (int j = 0; j < datanum; ++j) {

			input = data[j].x;
			//	cout << "forward .....\n";
				//the output of the neuro in each layer
			avalue[0] = input;
#ifdef DEBUG
			//	cout << "a0: \n" << avalue[0] << endl;
#endif
			for (int k = 1; k < numLayer; ++k) {
				//cout << weights[k].rows() << " " << weights[k].cols() << " " << avalue[k-1].size() << endl;
				nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
				for (int l = 0; l < nvalue[k].size(); ++l)
					avalue[k][l] = activeFun(k, nvalue[k][l]);
#ifdef DEBUG
				//	cout << "n" << k << ":\n";
				//	cout << nvalue[k] << "\n";
				//	cout << "a" << k << ":\n";
				//	cout << avalue[k] << "\n";
#endif
			}
#ifdef DEBUG
			//	cout << "error e: " <<  data[j].y[0] - avalue[numLayer - 1][0] << endl;
#endif
		//	cout << "bp----deviation....\n";
			///backward propagation
			//calculate deviationOfN
			for (int k = 1; k < numLayer; ++k) {
				//deviationOfN[k].Zero();
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					for (int l = 0; l < deviationOfN[k].cols(); ++l)
						deviationOfN[k](j, l) = 0;
				}
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
				}
#ifdef DEBUG
				//	cout << "f'(" << k << "):\n";
				//	cout << deviationOfN[k];
#endif
			}
			//
		//	cout << "sensities ...\n";
			s[numLayer - 1] = -2 * deviationOfN[numLayer - 1] * (data[j].y - avalue[numLayer - 1]);
#ifdef DEBUG
			//cout << "s" << numLayer - 1 << ":\n";
			//cout << s[numLayer - 1] << "\n";
#endif
			//cout << "output layer..";
			for (int k = numLayer - 2; k >= 1; --k) {
				s[k] = deviationOfN[k] * (weights[k + 1].transpose())*s[k + 1];
#ifdef DEBUG
				//	cout << "s" << k << ":\n";
				//	cout << s[k] << "\n";
#endif
			}
			//	cout << "update weight and bais....\n";
				//update weight: approximate steepest decent
			for (int k = numLayer - 1; k >= 1; --k) {
				weights[k] = weights[k] - learnrate * s[k] * avalue[k - 1].transpose();
				bais[k] = bais[k] - learnrate * s[k];
#ifdef DEBUG
				//cout << "w" << k << ":\n";
			//	cout << weights[k] << "\n";
			//	cout << "b" << k << ":\n";
			//	cout << bais[k] << "\n";
#endif
			}

			//	cout << "calculate prediction errors ...\n";
			VectorXd predictY(output.size());
			error = 0;
			//calculate the error of the total samples 
			for (int l = 0; l < datanum; ++l) {
				getOutput(&data[l].x[0], &predictY[0], xnum, ynum);
				error += ((predictY - data[l].y).transpose())*(predictY - data[l].y);
				//error += e[0];
			}
			error = error / datanum;// /= sqrt(datanum / ynum);
		//	cout << error << endl;
		//	system("PAUSE");
			if (error <= accurate) {
				finish = true;
				break;
			}

			//	showParam();
			//	system("PAUSE");
		}
#ifdef DEBUG
		cout << ite << "\t" << error << endl;
#endif
		if (error < preerror) {
			updateMinErrPara();
		}
		preerror = error;

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();
}

void NeuralNetwork::trainBySDBPBatch(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int maxEpochs) {
	//cout << "SDBP begin ....\n";
	//the whole training set
	int ite = 0;
	double error = 0;
	const double accurate = 0.01;
	bool finish = false;
	double preerror = getPrectErr(data, datanum, xnum, ynum);
	double minerror = preerror;
	updateMinErrPara();

	//int maxEpochs = 200;
	while (!finish) {
		///the total 
		vector<MatrixXd> tsa(numLayer); //the total sum of sensitivities s for each training sample
		vector<VectorXd> ts(numLayer); //the total sum of s*avalue output of the neuro in each layer avalue=f(nvalue)
		for (int j = 1; j < tsa.size(); ++j) tsa[j].resize(weights[j].rows(), weights[j].cols());
		for (int j = 1; j < ts.size(); ++j) ts[j].resize(bais[j].size());
		//	cout << "ts ...\n";
		for (int j = 0; j < datanum; ++j) {
			input = data[j].x;
			avalue[0] = input;
			for (int k = 1; k < numLayer; ++k) {
				nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
				for (int l = 0; l < nvalue[k].size(); ++l)
					avalue[k][l] = activeFun(k, nvalue[k][l]);
			}
			//cout << "a :\n";
			///backward propagation
			//calculate deviationOfN
			for (int k = 1; k < numLayer; ++k) {
				//deviationOfN[k].Zero();
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					for (int l = 0; l < deviationOfN[k].cols(); ++l)
						deviationOfN[k](j, l) = 0;
				}
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
				}
			}
			//cout << "d :\n";
			//
			s[numLayer - 1] = -2 * deviationOfN[numLayer - 1] * (data[j].y - avalue[numLayer - 1]);
			for (int k = numLayer - 2; k >= 1; --k) {
				s[k] = deviationOfN[k] * weights[k + 1].transpose() *s[k + 1];
			}

			for (int k = numLayer - 1; k >= 1; --k) {
				if (j == 0) {
					tsa[k] = s[k] * avalue[k - 1].transpose();
					ts[k] = s[k];
				}
				else {
					tsa[k] += s[k] * avalue[k - 1].transpose();
					ts[k] += s[k];
				}
			}
			//cout << "s: \n";
		}
		//update weight: approximate steepest decent
		for (int k = numLayer - 1; k >= 1; --k) {
			weights[k] = weights[k] - learnrate / datanum * tsa[k];
			bais[k] = bais[k] - learnrate / datanum * ts[k];
		}

		error = getPrectErr(data, datanum, xnum, ynum);

		if (error < minerror) {
			updateMinErrPara();
			minerror = error;
		}

#ifdef DEBUG
		cout << ite << "\t" << error << endl;
#endif
		if (error <= accurate) {
			finish = true;
			break;
		}

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();
}

void NeuralNetwork::trainBySDBPMiniBatch(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int batchSize, int maxEpochs) {
	//batch training: after all inputs are applied to the network, the complete gradient is computed
//#define DEBUG
	int ite = 0;
	double error = 0;
	const double accurate = 0.01;
	bool finish = false;
	//	int maxEpochs = 200;
		//int batchSize = 4;// 50;
	vector<int> batchIndex(data.size());
	for (int j = 0; j < batchIndex.size(); ++j) batchIndex[j] = j;
	random_shuffle(batchIndex.begin(), batchIndex.end());

	int batchNum = data.size() / batchSize;

	int trainnum = data.size() / batchSize * batchSize; //训练样本数（batch_size）可以取2的整数次方，如64,128等等
	int predictnum = data.size() - trainnum;

	double minerror = getPrectErr(data, datanum, xnum, ynum);
	updateMinErrPara();

	while (!finish) {
		//minibatch
		for (int whichbatch = 0; whichbatch < batchNum; ++whichbatch) {
			vector<MatrixXd> tsa(numLayer); //the total sum of sensitivities s for each training sample
			vector<VectorXd> ts(numLayer); //the total sum of s*avalue output of the neuro in each layer avalue=f(nvalue)
			for (int j = 1; j < tsa.size(); ++j) tsa[j].resize(weights[j].rows(), weights[j].cols());
			for (int j = 1; j < ts.size(); ++j) ts[j].resize(bais[j].size());

			for (int index = 0; index < batchSize; ++index) {
				int j = batchIndex[index + whichbatch * batchSize];
				if (j >= datanum) { cout << j << " " << datanum << endl; assert(false); }
				input = data[j].x;
				//	cout <<"----" << whichbatch <<" " << j << " : x" << input <<" " << "y: "  << data[j].y<< endl;

					//the output of the neuro in each layer
				avalue[0] = input;
				for (int k = 1; k < numLayer; ++k) {
					nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
					for (int l = 0; l < nvalue[k].size(); ++l)
						avalue[k][l] = activeFun(k, nvalue[k][l]);
					//	cout << "a" << k << "\n";
					//	cout << avalue[k] << endl;
				}
				///backward propagation
				//calculate deviationOfN
				for (int k = 1; k < numLayer; ++k) {
					//deviationOfN[k].Zero();
					for (int j = 0; j < deviationOfN[k].rows(); ++j) {
						for (int l = 0; l < deviationOfN[k].cols(); ++l)
							deviationOfN[k](j, l) = 0;
					}
					for (int j = 0; j < deviationOfN[k].rows(); ++j) {
						deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
					}
					//	cout << "f'(" << k << ")" << "\n";
					//	cout << deviationOfN[k] << endl;
				}
				//
				s[numLayer - 1] = -2 * deviationOfN[numLayer - 1] * (data[j].y - avalue[numLayer - 1]);
				for (int k = numLayer - 2; k >= 1; --k) {
					s[k] = deviationOfN[k] * weights[k + 1].transpose()*s[k + 1];
				}

				for (int k = numLayer - 1; k >= 1; --k) {
					//	cout << "s" << k <<"\n";
					//	cout << s[k] << endl;
					//	cout << "tsa" << k << endl;
					//	cout << tsa[k] << endl;
					//	cout << "ts" << k << endl;
					//	cout << ts[k] << endl;
					if (index == 0) {
						tsa[k] = s[k] * avalue[k - 1].transpose();
						ts[k] = s[k];
					}
					else {
						tsa[k] += s[k] * avalue[k - 1].transpose();
						ts[k] += s[k];
					}
				}
			}
			//showParam();
			//update weight: approximate steepest decent
			for (int k = numLayer - 1; k >= 1; --k) {
				//	cout << "tsa:\n";
				//	cout << tsa[k] << endl;
				weights[k] = weights[k] - learnrate / batchSize * tsa[k];
				bais[k] = bais[k] - learnrate / batchSize * ts[k];
			}
#ifdef DEBUG
			//	showParam();
			//	system("PAUSE");
#endif
		//	cout << whichbatch << " ";
		}
		//cout << endl;

		error = getPrectErr(data, datanum, xnum, ynum);

		if (error < minerror) {
			updateMinErrPara();
			minerror = error;
		}

#ifdef DEBUG
		cout << ite << "\t" << error << endl;
#endif

		if (error <= accurate) {
			finish = true;
			break;
		}

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();
}

void NeuralNetwork::trainByMomentumBP(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double initLearnRate, double momentumCoeff, int maxEpochs) { //MOBP
	//the whole training set
	learnrate = initLearnRate;
	int ite = 0;
	double error = 0;
	const double accurate = 0.01;
	bool finish = false;
	//int maxEpochs = 200;
	//detaW, detaB
	vector<VectorXd> detaBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> detaWeights(bais.size()); //the weight matrix in each layer;
	for (int i = 1; i < detaBais.size(); ++i) {
		detaBais[i].resize(bais[i].size());
		for (int j = 0; j < detaBais[i].size(); ++j) detaBais[i][j] = 0;
	}
	for (int i = 1; i < detaWeights.size(); ++i) {
		detaWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < detaWeights[i].rows(); ++j) {
			for (int k = 0; k < detaWeights[i].cols(); ++k) {
				detaWeights[i](j, k) = 0;
			}
		}
	}

	double minerror = getPrectErr(data, datanum, xnum, ynum);
	updateMinErrPara();

	//double eta
	while (!finish) {
		///the total 
		vector<MatrixXd> tsa(numLayer); //the total sum of sensitivities s for each training sample
		vector<VectorXd> ts(numLayer); //the total sum of s*avalue output of the neuro in each layer avalue=f(nvalue)

		for (int j = 0; j < datanum; ++j) {
			input = data[j].x;
			avalue[0] = input;
			for (int k = 1; k < numLayer; ++k) {
				nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
				for (int l = 0; l < nvalue[k].size(); ++l)
					avalue[k][l] = activeFun(k, nvalue[k][l]);
			}
			///backward propagation
			//calculate deviationOfN
			for (int k = 1; k < numLayer; ++k) {
				//deviationOfN[k].Zero();
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					for (int l = 0; l < deviationOfN[k].cols(); ++l)
						deviationOfN[k](j, l) = 0;
				}
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
				}
			}
			//
			s[numLayer - 1] = -2 * deviationOfN[numLayer - 1] * (data[j].y - avalue[numLayer - 1]);
			for (int k = numLayer - 2; k >= 1; --k) {
				s[k] = deviationOfN[k] * weights[k + 1].transpose() *s[k + 1];
			}

			for (int k = numLayer - 1; k >= 1; --k) {
				if (j == 0) {
					tsa[k] = s[k] * avalue[k - 1].transpose();
					ts[k] = s[k];
				}
				else {
					tsa[k] += s[k] * avalue[k - 1].transpose();
					ts[k] += s[k];
				}
			}
		}
		//update weight: approximate steepest decent
		for (int k = numLayer - 1; k >= 1; --k) {
			detaWeights[k] = momentumCoeff * detaWeights[k] - (1 - momentumCoeff)*learnrate / datanum * tsa[k];
			detaBais[k] = momentumCoeff * detaBais[k] - (1 - momentumCoeff)*learnrate / datanum * ts[k];
			weights[k] = weights[k] + detaWeights[k];// -(1 - momentumCoeff)*learnrate / datanum*tsa[k];
			bais[k] = detaBais[k] + detaBais[k];// momentumCoeff*bais[k] - (1 - momentumCoeff)*learnrate / datanum*ts[k];
		}

		error = getPrectErr(data, datanum, xnum, ynum);
		//preerror = error;
#ifdef DEBUG
		cout << ite << "\t" << error << endl;
#endif

		if (error < minerror) {
			updateMinErrPara();
			minerror = error;
		}

		if (error <= accurate) {
			finish = true;
			break;
		}

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();
}

void NeuralNetwork::trainByVarLearnRateBP(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double initLearnRate, double percentage, double momenCoeff, double incFactor, double decFactor, int maxEpochs) { //VLBP
	//the whole training set
	learnrate = initLearnRate;
	int ite = 0;
	double error = 0;
	double preerror = getPrectErr(data, datanum, xnum, ynum);
	const double accurate = 0.01;
	double momentumCoeff = momenCoeff;
	bool finish = false;
	//int maxEpochs = 200;
	//detaW, detaB
	vector<VectorXd> detaBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> detaWeights(bais.size()); //the weight matrix in each layer;
	for (int i = 1; i < detaBais.size(); ++i) {
		detaBais[i].resize(bais[i].size());
		for (int j = 0; j < detaBais[i].size(); ++j) detaBais[i][j] = 0;
	}
	for (int i = 1; i < detaWeights.size(); ++i) {
		detaWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < detaWeights[i].rows(); ++j) {
			for (int k = 0; k < detaWeights[i].cols(); ++k) {
				detaWeights[i](j, k) = 0;
			}
		}
	}
	vector<VectorXd> preBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> preWeights(bais.size()); //the weight matrix in each layer;
	for (int i = 1; i < preBais.size(); ++i) {
		preBais[i].resize(bais[i].size());
		for (int j = 0; j < preBais[i].size(); ++j) preBais[i][j] = bais[i][j];
	}
	for (int i = 1; i < preWeights.size(); ++i) {
		preWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < preWeights[i].rows(); ++j) {
			for (int k = 0; k < preWeights[i].cols(); ++k) {
				preWeights[i](j, k) = weights[i](j, k);
			}
		}
	}

	double minerror = getPrectErr(data, datanum, xnum, ynum);
	updateMinErrPara();

	//double eta
	while (!finish) {
		///the total 
		vector<MatrixXd> tsa(numLayer); //the total sum of sensitivities s for each training sample
		vector<VectorXd> ts(numLayer); //the total sum of s*avalue output of the neuro in each layer avalue=f(nvalue)

		for (int j = 0; j < datanum; ++j) {
			input = data[j].x;
			avalue[0] = input;
			for (int k = 1; k < numLayer; ++k) {
				nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
				for (int l = 0; l < nvalue[k].size(); ++l)
					avalue[k][l] = activeFun(k, nvalue[k][l]);
			}
			///backward propagation
			//calculate deviationOfN
			for (int k = 1; k < numLayer; ++k) {
				//deviationOfN[k].Zero();
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					for (int l = 0; l < deviationOfN[k].cols(); ++l)
						deviationOfN[k](j, l) = 0;
				}
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
				}
			}
			//
			s[numLayer - 1] = -2 * deviationOfN[numLayer - 1] * (data[j].y - avalue[numLayer - 1]);
			for (int k = numLayer - 2; k >= 1; --k) {
				s[k] = deviationOfN[k] * weights[k + 1].transpose() *s[k + 1];
			}

			for (int k = numLayer - 1; k >= 1; --k) {
				if (j == 0) {
					tsa[k] = s[k] * avalue[k - 1].transpose();
					ts[k] = s[k];
				}
				else {
					tsa[k] += s[k] * avalue[k - 1].transpose();
					ts[k] += s[k];
				}
			}
		}
		//update weight: approximate steepest decent
		for (int k = numLayer - 1; k >= 1; --k) {
			detaWeights[k] = momentumCoeff * detaWeights[k] - (1 - momentumCoeff)*learnrate / datanum * tsa[k];
			detaBais[k] = momentumCoeff * detaBais[k] - (1 - momentumCoeff)*learnrate / datanum * ts[k];
			weights[k] = weights[k] + detaWeights[k];// -(1 - momentumCoeff)*learnrate / datanum*tsa[k];
			bais[k] = detaBais[k] + detaBais[k];// momentumCoeff*bais[k] - (1 - momentumCoeff)*learnrate / datanum*ts[k];
		}
		error = getPrectErr(data, datanum, xnum, ynum);

		if (error - preerror > percentage*preerror) {
			weights = preWeights;
			bais = preBais;
			learnrate = decFactor * learnrate;
			momentumCoeff = 0;
		}
		else if (error < preerror) {
			learnrate *= incFactor;
			if (momentumCoeff == 0)momentumCoeff = momenCoeff;
		}
		else {
			if (momentumCoeff == 0)momentumCoeff = momenCoeff;
		}

		//copy the weights and bais into the preWeights and preBais
		preWeights = weights;
		preBais = bais;
		preerror = error;

#ifdef DEBUG
		cout << ite << "\t" << learnrate << "\t" << error << endl;
#endif
		if (error < minerror) {
			updateMinErrPara();
			minerror = error;
		}

		if (error <= accurate) {
			finish = true;
			break;
		}

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();
}

//tol: the acceracy of the interval [a,b] to stop further reduce
void NeuralNetwork::trainByConjugateGradientBP(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double initLearnRate, double tstepsize, int maxEpochs, int n, double tol, const vector<double> ymax, const vector<double> ymin) {//CGBP
	//the whole training set
	double varduration = 0;
#ifdef UNIX
	struct timeval starttime, endtime;
	struct timezone tz;
	gettimeofday(&starttime, &tz);
#else
	double starttime = clock();
#endif

	learnrate = initLearnRate;
	int ite = 0;
	double error = 0;
	double preerror = getPrectErr(data, datanum, xnum, ynum);
	const double accurate = 0.01;
	bool finish = false;
	//int maxEpochs = 200;
	//int n = 100;// the search direction is reset as the steepest descent direction after n iterations
	//detaW, detaB
	vector<VectorXd> detaBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> detaWeights(bais.size()); //the weight matrix in each layer;
	for (int i = 1; i < detaBais.size(); ++i) {
		detaBais[i].resize(bais[i].size());
		for (int j = 0; j < detaBais[i].size(); ++j) detaBais[i][j] = 0;
	}
	for (int i = 1; i < detaWeights.size(); ++i) {
		detaWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < detaWeights[i].rows(); ++j) {
			for (int k = 0; k < detaWeights[i].cols(); ++k) {
				detaWeights[i](j, k) = 0;
			}
		}
	}

	//define length along the search directions by using 4 search points
	vector<vector<VectorXd> > intervalPointsBais(4); //the four points
	vector<vector<MatrixXd> > intervalPointsWeights(4);
	for (int l = 0; l < intervalPointsBais.size(); ++l) {
		intervalPointsBais[l].resize(bais.size());
		intervalPointsWeights[l].resize(weights.size());
		for (int i = 1; i < intervalPointsBais[l].size(); ++i) {
			intervalPointsBais[l][i].resize(bais[i].size());
			intervalPointsWeights[l][i].resize(weights[i].rows(), weights[i].cols());
		}
	}

	///the current search direction p(k)
	vector<VectorXd> pBais(bais.size());
	vector<MatrixXd> pWeights(weights.size());
	for (int i = 1; i < pBais.size(); ++i) pBais[i].resize(bais[i].size());
	for (int i = 1; i < pWeights.size(); ++i) pWeights[i].resize(weights[i].rows(), weights[i].cols());

	//the last search direction p(k-1)
	vector<VectorXd> lastdetaBais(bais.size());
	vector<MatrixXd> lastdetaWeights(weights.size());
	for (int i = 1; i < lastdetaBais.size(); ++i) lastdetaBais[i].resize(bais[i].size());
	for (int i = 1; i < lastdetaWeights.size(); ++i) lastdetaWeights[i].resize(weights[i].rows(), weights[i].cols());

	double minerror = getPrectErr(data, datanum, xnum, ynum);
	updateMinErrPara();

	//double eta
	while (!finish) {
		///the total 
		vector<MatrixXd> tsa(numLayer); //the total sum of sensitivities s for each training sample
		vector<VectorXd> ts(numLayer); //the total sum of s*avalue output of the neuro in each layer avalue=f(nvalue)

		for (int j = 0; j < datanum; ++j) {
			input = data[j].x;
			//the output of the neuro in each layer
			avalue[0] = input;
			for (int k = 1; k < numLayer; ++k) {
				nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
				for (int l = 0; l < nvalue[k].size(); ++l)
					avalue[k][l] = activeFun(k, nvalue[k][l]);
			}
			///backward propagation
			//calculate deviationOfN
			for (int k = 1; k < numLayer; ++k) {
				//deviationOfN[k].Zero();
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					for (int l = 0; l < deviationOfN[k].cols(); ++l)
						deviationOfN[k](j, l) = 0;
				}
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
				}
			}
			//
			s[numLayer - 1] = -2 * deviationOfN[numLayer - 1] * (data[j].y - avalue[numLayer - 1]);
			for (int k = numLayer - 2; k >= 1; --k) {
				s[k] = deviationOfN[k] * weights[k + 1].transpose() *s[k + 1];
			}

			for (int k = numLayer - 1; k >= 1; --k) {
				if (j == 0) {
					tsa[k] = s[k] * avalue[k - 1].transpose();
					ts[k] = s[k];
				}
				else {
					tsa[k] += s[k] * avalue[k - 1].transpose();
					ts[k] += s[k];
				}
			}
		}
		//copy the lastdetaWeights

		for (int k = numLayer - 1; k >= 1; --k) {
			lastdetaBais[k] = detaBais[k];
			lastdetaWeights[k] = detaWeights[k];
			detaWeights[k] = tsa[k] / datanum;     //gk(the gradient of the problem)
			detaBais[k] = ts[k] / datanum;
		}
		//set the search direction
		if (ite == 0) {
			//init search direction
			for (int k = numLayer - 1; k >= 1; --k) {
				pBais[k] = -detaBais[k];
				pWeights[k] = -detaWeights[k];
			}
		}

		//find the length of the search direction to achieve the local optima
		//1. interval location [a,b] //a in 0, b in 3
		double stepsize = tstepsize;//0.1;
		int index = 1;
		vector<double> fvalue(intervalPointsBais.size());
		fvalue[0] = preerror;
		for (int k = numLayer - 1; k >= 1; --k) {
			intervalPointsBais[0][k] = bais[k];
			intervalPointsWeights[0][k] = weights[k];
		}
		while (true) {
			//update the
		//	int index = intervalPointsBais.size() - 1;
			for (int k = numLayer - 1; k >= 1; --k) {
				intervalPointsBais[index][k] = bais[k] + stepsize * pBais[k];
				intervalPointsWeights[index][k] = weights[k] + stepsize * pWeights[k];
			}
			fvalue[index] = getErrWithPara(intervalPointsWeights[index], intervalPointsBais[index], data, datanum, xnum, ynum);
			if (fvalue[index] >= fvalue[index - 1]) {
				break;
			}
			index++;
			if (index >= 4) {
				for (int k = 2; k < intervalPointsBais.size(); ++k) {
					intervalPointsBais[k - 1] = intervalPointsBais[k];
					intervalPointsWeights[k - 1] = intervalPointsWeights[k];
					fvalue[k - 1] = fvalue[k];
				}
				index = 3;
			}
			stepsize *= 2;
#ifdef DEBUG
			//cout << stepsize << " " << fvalue[0] << " " << fvalue[index] << endl;
#endif
		}
		//set the b point
		if (index != 3) {
			intervalPointsBais[3] = intervalPointsBais[index];
			intervalPointsWeights[3] = intervalPointsWeights[index];
			fvalue[3] = fvalue[index];
		}
		if (index == 3) {
			intervalPointsBais[0] = intervalPointsBais[1];
			intervalPointsWeights[0] = intervalPointsWeights[1];
			fvalue[0] = fvalue[1];
		}
		//2. interval reduction by golden section search; find the point minimize the error
		//double tol = 0;
		double rate = 0.618;
		for (int k = numLayer - 1; k >= 1; --k) {
			intervalPointsBais[1][k] = intervalPointsBais[0][k] + (1 - rate)*(intervalPointsBais[3][k] - intervalPointsBais[0][k]);
			intervalPointsWeights[1][k] = intervalPointsWeights[0][k] + (1 - rate)*(intervalPointsWeights[3][k] - intervalPointsWeights[0][k]);
			intervalPointsBais[2][k] = intervalPointsBais[3][k] - (1 - rate)*(intervalPointsBais[3][k] - intervalPointsBais[0][k]);
			intervalPointsWeights[2][k] = intervalPointsWeights[3][k] - (1 - rate)*(intervalPointsWeights[3][k] - intervalPointsWeights[0][k]);
		}
		fvalue[2] = getErrWithPara(intervalPointsWeights[2], intervalPointsBais[2], data, datanum, xnum, ynum);
		fvalue[1] = getErrWithPara(intervalPointsWeights[1], intervalPointsBais[1], data, datanum, xnum, ynum);
		double d = intervalPointsBais[3][1][0] - intervalPointsBais[0][1][0];//length of the interval
#ifdef DEBUG
		cout << d << " " << fvalue[0] << "\t" << fvalue[3] << "\n";
#endif
		int caldtime = 0;
		while (d >= tol) {
			if (fvalue[1] < fvalue[2]) { //c < d
				intervalPointsBais[3] = intervalPointsBais[2];  //b = d
				intervalPointsWeights[3] = intervalPointsWeights[2];  //b = d
				fvalue[3] = fvalue[2];
				intervalPointsBais[2] = intervalPointsBais[1];    //d = c
				intervalPointsWeights[2] = intervalPointsWeights[1];
				fvalue[2] = fvalue[1];
				//update c = a + (1-rate)(b - a)
				for (int k = numLayer - 1; k >= 1; k--) {
					intervalPointsBais[1][k] = intervalPointsBais[0][k] + (1 - rate)*(intervalPointsBais[3][k] - intervalPointsBais[0][k]);
					intervalPointsWeights[1][k] = intervalPointsWeights[0][k] + (1 - rate)*(intervalPointsWeights[3][k] - intervalPointsWeights[0][k]);
				}
				fvalue[1] = getErrWithPara(intervalPointsWeights[1], intervalPointsBais[1], data, datanum, xnum, ynum);
			}
			else {
				intervalPointsBais[0] = intervalPointsBais[1];  //a = c
				intervalPointsWeights[0] = intervalPointsWeights[1];
				fvalue[0] = fvalue[1];
				intervalPointsBais[1] = intervalPointsBais[2];   //c = d
				intervalPointsWeights[1] = intervalPointsWeights[2];
				fvalue[1] = fvalue[2];
				//
				for (int k = numLayer - 1; k >= 1; k--) {
					intervalPointsBais[2][k] = intervalPointsBais[3][k] - (1 - rate)*(intervalPointsBais[3][k] - intervalPointsBais[0][k]);
					intervalPointsWeights[2][k] = intervalPointsWeights[3][k] - (1 - rate)*(intervalPointsWeights[3][k] - intervalPointsWeights[0][k]);
				}
				fvalue[2] = getErrWithPara(intervalPointsWeights[2], intervalPointsBais[2], data, datanum, xnum, ynum);
			}
			d = 0;
			for (int k = 1; k < numLayer; ++k) {
				for (int i = 0; i < intervalPointsWeights[3][k].rows(); ++i) {
					for (int j = 0; j < intervalPointsWeights[3][k].cols(); ++j) {
						d += (intervalPointsWeights[3][k](i, j) - intervalPointsWeights[0][k](i, j))*
							(intervalPointsWeights[3][k](i, j) - intervalPointsWeights[0][k](i, j));
					}
				}
				for (int i = 0; i < intervalPointsBais[3][k].size(); ++i) {
					d += (intervalPointsBais[3][k][i] - intervalPointsBais[0][k][i])*
						(intervalPointsBais[3][k][i] - intervalPointsBais[0][k][i]); //different between a and b

				}
			}
			d = sqrt(d);
			//cout << d << "\n";
		}

		//update weight: approximate steepest decent
		int pointIndex = 0;
		for (int i = 0; i < 4; ++i) if (fvalue[i] < fvalue[pointIndex]) pointIndex = i;


		for (int k = numLayer - 1; k >= 1; --k) {
			weights[k] = intervalPointsWeights[pointIndex][k];//weights[k] + detaWeights[k];// -(1 - momentumCoeff)*learnrate / datanum*tsa[k];
			bais[k] = intervalPointsBais[pointIndex][k];//detaBais[k] + detaBais[k];// momentumCoeff*bais[k] - (1 - momentumCoeff)*learnrate / datanum*ts[k];
		}

		//Polak-Ribiere rule detaG*G(t)/(G(t-1))
		double gsum = 0;
		double lastgsum = 0;
		for (int k = numLayer - 1; k >= 1; --k) {
			for (int j = 0; j < bais[k].size(); ++j) {
				gsum += (detaBais[k][j] - lastdetaBais[k][j])* detaBais[k][j];
				lastgsum += lastdetaBais[k][j] * lastdetaBais[k][j];
			}
			for (int j = 0; j < weights[k].rows(); ++j) {
				for (int l = 0; l < weights[k].cols(); ++l) {
					gsum += (detaWeights[k](j, l) - lastdetaWeights[k](j, l))*detaWeights[k](j, l);
					lastgsum += lastdetaWeights[k](j, l)*lastdetaWeights[k](j, l);
				}
			}
		}
		double beta;
		if (ite == 0) beta = 1;
		else beta = gsum / lastgsum;
		if (pointIndex == 0 || ite > 0 && ite % n == 0) {   //reset the search direction p
			for (int k = numLayer - 1; k >= 1; k--) {
				pBais[k] = -detaBais[k];
				pWeights[k] = -detaWeights[k];
			}
		}
		else {
			for (int k = numLayer - 1; k >= 1; --k) {
				pBais[k] = -detaBais[k] + beta * pBais[k];
				pWeights[k] = -detaWeights[k] + beta * pWeights[k];
			}
		}

		error = getPrectErr(data, datanum, xnum, ynum);
		preerror = error;

		if (error < minerror) {
			updateMinErrPara();
			minerror = error;
		}

		//cout << ite << "\t" << error << endl;

#ifdef DEBUG
		cout << ite << "\t" << error << endl;
#endif
		if (error <= accurate) {
			finish = true;
			break;
		}

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();
	//cout << "train by the CGBP: " << ite << "\t" << minerror << endl;
	//regission on the dataset
	double totalSetR = regressionOutput(data, datanum, xnum, ynum, ymax, ymin);
	//cout << "regression R2 for trainSet : " << totalSetR << endl;

#ifdef UNIX
	gettimeofday(&endtime, &tz);
	varduration = (endtime.tv_sec - starttime.tv_sec) + (endtime.tv_usec - starttime.tv_usec) / 1000000.0;
#else
	double endtime = clock();
	varduration = 1.0*(endtime - starttime) / CLOCKS_PER_SEC;
#endif
	//cout << varduration << "s\n";
}

//LMBP
void NeuralNetwork::trainByLevenbergMarquardtBP(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double mu, double factor, int maxEpochs, const vector<double> ymax, const vector<double> ymin) {//LMBP
	//mu = 0.01; factor = 10;
	//the whole training set
	double varduration = 0;
#ifdef UNIX
	struct timeval starttime, endtime;
	struct timezone tz;
	gettimeofday(&starttime, &tz);
#else
	double starttime = clock();
#endif

	int ite = 0;
	double error = 0;
	//cout << "begin ....." << "\t";
	double preerror = getPrectErr(data, datanum, xnum, ynum);
	if (preerror >= 0.01) initilizeParaNgWi90();
	//cout << datanum << "\t" << "init error:\t" << preerror << "\t";
	const double accurate = 1e-7;
	bool finish = false;
	//	int maxEpochs = 200;

		//check the total parameter number
	int pcount = 0;
	//cout << bais.size() << "+ " << numLayer << endl;
	//for (int i = 0; i < bais.size(); ++i)
	//	cout << bais[i].size() << "-";
//	cout << endl;
	for (int k = numLayer - 1; k >= 1; --k) {
		//	cout  bais[k].size() << " " << weights[k].rows() << " " << weights[k].cols() << endl;
		pcount += bais[k].size() + weights[k].rows()*weights[k].cols();// [k - 1].size() *bais[k].size();
	}
	if (pcount != totalParaNum) assert(false);

	//cout << "totalParaNum : " << totalParaNum << endl;

	//detaW, detaB
	vector<VectorXd> detaBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> detaWeights(bais.size()); //the weight matrix in each layer;
	for (int i = 1; i < detaBais.size(); ++i) {
		detaBais[i].resize(bais[i].size());
		for (int j = 0; j < detaBais[i].size(); ++j) detaBais[i][j] = 0;
	}
	for (int i = 1; i < detaWeights.size(); ++i) {
		detaWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < detaWeights[i].rows(); ++j) {
			for (int k = 0; k < detaWeights[i].cols(); ++k) {
				detaWeights[i](j, k) = 0;
			}
		}
	}

	vector<VectorXd> preBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> preWeights(weights.size()); //the weight matrix in each layer;
	for (int i = 1; i < preBais.size(); ++i) {
		preBais[i].resize(bais[i].size());
		preWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < preBais[i].size(); ++j) preBais[i][j] = bais[i][j];
	}

	//Jacobian matrix
	MatrixXd J;//
	int N = datanum * ynum;
	J.resize(N, totalParaNum);
	VectorXd v(N);

	double minerror = getPrectErr(data, datanum, xnum, ynum);
	updateMinErrPara();

	//cout << v.size() << " " << J.cols() << " " << J.rows() << endl;
	int continuousTimes = 0;

	while (!finish) {

		//calculate the J matrix
		for (int i = 0; i < J.rows(); ++i) {
			for (int j = 0; j < J.cols(); ++j)
				J(i, j) = 0;
		}
		int pindex = 0;
		for (int j = 0; j < datanum; ++j) {
			input = data[j].x;
			avalue[0] = input;
			for (int k = 1; k < numLayer; ++k) {
				nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
				for (int l = 0; l < nvalue[k].size(); ++l)
					avalue[k][l] = activeFun(k, nvalue[k][l]);
			}
			//calcualte the error vector v
			pindex = j * avalue[numLayer - 1].size();
			for (int l = 0; l < avalue[numLayer - 1].size(); ++l) {
				v[pindex] = data[j].y[l] - avalue[numLayer - 1][l];
				pindex++;
				if (pindex > v.size()) { cout << pindex << ">" << v.size() << endl; assert(false); }
			}
			///backward propagation
			//calculate deviationOfN
			for (int k = 1; k < numLayer; ++k) {
				//deviationOfN[k].Zero();
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					for (int l = 0; l < deviationOfN[k].cols(); ++l)
						deviationOfN[k](j, l) = 0;
				}
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
				}
			}
			//deviation 
			//
			for (int k = 0; k < s[numLayer - 1].size(); ++k)
				s[numLayer - 1][k] = -deviationOfN[numLayer - 1](k, k);
			//s[numLayer - 1] = -deviationOfN[numLayer - 1];
			for (int k = numLayer - 2; k >= 1; --k) {
				s[k] = deviationOfN[k] * weights[k + 1].transpose() *s[k + 1];
			}
			pindex = 0;
			for (int k = 1; k <= numLayer - 1; ++k) {
				//weights
				for (int i = 0; i < weights[k].rows(); ++i) {
					for (int l = 0; l < weights[k].cols(); ++l) {
						J(j, pindex) = s[k][i] * avalue[k - 1][l]; //s(k)*a(k-1)
						pindex++;
					}
				}
				//bais
				for (int i = 0; i < bais[k].size(); ++i) {
					J(j, pindex) = s[k][i];
					pindex++;
				}
			}
			if (pindex != totalParaNum) {
				cout << pindex << " " << totalParaNum << " " << J.cols() << endl;
				assert(false);
			}
		}

		for (int i = 1; i < preBais.size(); ++i) {
			for (int j = 0; j < preBais[i].size(); ++j) preBais[i][j] = bais[i][j];
		}
		for (int i = 1; i < preWeights.size(); ++i) {
			for (int j = 0; j < preWeights[i].rows(); ++j) {
				for (int k = 0; k < preWeights[i].cols(); ++k) {
					preWeights[i](j, k) = weights[i](j, k);
				}
			}
		}

		//
		MatrixXd IdenM(J.cols(), J.cols());
		for (int i = 0; i < IdenM.rows(); ++i) {
			for (int j = 0; j < IdenM.cols(); ++j)
				IdenM(i, j) = 0;
			IdenM(i, i) = 1;
		}
		//	IdenM.Identity();	
		int m = 1;
		while (true) {

			MatrixXd detap = -(J.transpose()*J + mu * IdenM).inverse()*J.transpose()*v;
			if (detap.cols() > 1) assert(false);

			//update weight: approximate steepest decent
			int pindex = 0;
			for (int k = 1; k <= numLayer - 1; ++k) {
				for (int i = 0; i < bais[k].size(); ++i) {
					for (int j = 0; j < bais[k - 1].size(); ++j) {
						weights[k](i, j) = preWeights[k](i, j) + detap(pindex, 0);
						pindex++;
					}
				}
				for (int i = 0; i < bais[k].size(); ++i) {
					bais[k][i] = preBais[k][i] + detap(pindex, 0);
					pindex++;
				}
			}
			error = getPrectErr(data, datanum, xnum, ynum);
			if (error <= preerror) {
				//cout << "LMBP" << " " << mu << " " << preerror << " " << error << endl;
				mu /= factor;
				break;
			}
			else {
				mu *= factor;
				if (m > 5) {  //reset the weights and bais as the previous values
					break;
				}
				m = m + 1;
				//	if (mu >= 10000) break;
				//	cout << "LMBP" << " " << mu << " " << preerror << " " << error << endl;
			}
		}


		if (error < minerror) {
			updateMinErrPara();
			minerror = error;
		}
		if (error == preerror) {
			continuousTimes++;
		}
		//restart parameter for training
		if (continuousTimes >= 5) {
			initilizeParaNgWi90();
			continuousTimes = 0;
			error = getPrectErr(data, datanum, xnum, ynum);
			if (error < minerror) {
				updateMinErrPara();
				minerror = error;
			}
		}

		preerror = error;
		//cout << ite << "\t" <<"LMBP\t" << error << "\t" << minerror << endl;
#ifdef DEBUG
		cout << ite << "\t" << error << endl;
#endif
		if (error <= accurate) {
			finish = true;
			break;
		}

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();
	//cout << "train by LMBP: " << ite << "\t" << minerror << "\n";
		//regission on the dataset
	//double totalSetR = regressionOutput(data, datanum, xnum, ynum, ymax, ymin);
	//	cout << "regression R2: " << totalSetR << "\n";
		//cout << "LMBP " << ite << " R2 " << totalSetR << "\n";
#ifdef UNIX
	gettimeofday(&endtime, &tz);
	varduration = (endtime.tv_sec - starttime.tv_sec) + (endtime.tv_usec - starttime.tv_usec) / 1000000.0;
#else
	double endtime = clock();
	varduration = 1.0*(endtime - starttime) / CLOCKS_PER_SEC;
#endif
	//cout << varduration << "s\n";

}

//LMBP with cross-validation
void NeuralNetwork::trainByLMBPCorssValidation(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double mu, double factor, int maxEpochs, const vector<double> ymax, const vector<double> ymin) {//LMBP
	double varduration = 0;
#ifdef UNIX
	struct timeval starttime, endtime;
	struct timezone tz;
	gettimeofday(&starttime, &tz);
#else
	double starttime = clock();
#endif

	//mu = 0.01; factor = 10;
	//the whole training set
	int trainnum = data.size()*(0.7 / (0.85));
	int validationnum = data.size()*(0.15 / 0.85);
	vector<TrainingPoint> trainSet(trainnum);
	vector<TrainingPoint> validationSet(validationnum);
	vector<int> visit(data.size());
	for (int i = 0; i < data.size(); ++i) {
		visit[i] = i;
	}
	random_shuffle(visit.begin(), visit.end());
	for (int i = 0; i < trainSet.size(); ++i) {
		trainSet[i] = data[visit[i]];
	}
	for (int i = 0; i < validationSet.size(); ++i) {
		validationSet[i] = data[visit[i + trainnum]];
	}

	int ite = 0;
	double error = 0;
	//cout << "begin ....." << endl;
	double preerror = getPrectErr(validationSet, validationnum, xnum, ynum);
	//cout << preerror << endl;
	const double accurate = 1e-10;
	bool finish = false;
	//	int maxEpochs = 200;

	//check the total parameter number
	int pcount = 0;
	//cout << bais.size() << "+ " << numLayer << endl;
	//for (int i = 0; i < bais.size(); ++i)
	//	cout << bais[i].size() << "-";
	//	cout << endl;
	for (int k = numLayer - 1; k >= 1; --k) {
		//	cout  bais[k].size() << " " << weights[k].rows() << " " << weights[k].cols() << endl;
		pcount += bais[k].size() + weights[k].rows()*weights[k].cols();// [k - 1].size() *bais[k].size();
	}
	if (pcount != totalParaNum) assert(false);

	//cout << "totalParaNum : " << totalParaNum << endl;

	//detaW, detaB
	vector<VectorXd> detaBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> detaWeights(bais.size()); //the weight matrix in each layer;
	for (int i = 1; i < detaBais.size(); ++i) {
		detaBais[i].resize(bais[i].size());
		for (int j = 0; j < detaBais[i].size(); ++j) detaBais[i][j] = 0;
	}
	for (int i = 1; i < detaWeights.size(); ++i) {
		detaWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < detaWeights[i].rows(); ++j) {
			for (int k = 0; k < detaWeights[i].cols(); ++k) {
				detaWeights[i](j, k) = 0;
			}
		}
	}

	vector<VectorXd> preBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> preWeights(weights.size()); //the weight matrix in each layer;
	for (int i = 1; i < preBais.size(); ++i) {
		preBais[i].resize(bais[i].size());
		preWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < preBais[i].size(); ++j) preBais[i][j] = bais[i][j];
	}

	//Jacobian matrix
	MatrixXd J;//
	int N = trainnum * ynum;// datanum*ynum;
	J.resize(N, totalParaNum);
	VectorXd v(N);

	double minerror = getPrectErr(validationSet, validationnum, xnum, ynum);//getPrectErr(data, datanum, xnum, ynum);
	updateMinErrPara();

	int valErrIte = 0;
	//cout << v.size() << " " << J.cols() << " " << J.rows() << endl;
	/*cout <<"s " << s.size() << endl;
	for (int i = 0; i < s.size(); ++i){
		cout << "s" << i << ": " << s[i].size() << endl;
	}
	cout << "d " << deviationOfN.size() << endl;
	for (int i = 0; i < deviationOfN.size(); ++i){
		cout << "d" << i << ": " << deviationOfN[i].rows() << " " << deviationOfN[i].cols() << endl;
	}*/
	while (!finish) {

		//calculate the J matrix
		for (int i = 0; i < J.rows(); ++i) {
			for (int j = 0; j < J.cols(); ++j)
				J(i, j) = 0;
		}
		int pindex = 0;
		for (int j = 0; j < trainnum; ++j) {
			input = trainSet[j].x;
			avalue[0] = input;
			for (int k = 1; k < numLayer; ++k) {
				nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
				for (int l = 0; l < nvalue[k].size(); ++l)
					avalue[k][l] = activeFun(k, nvalue[k][l]);
			}
			//calcualte the error vector v
			pindex = j * avalue[numLayer - 1].size();
			for (int l = 0; l < avalue[numLayer - 1].size(); ++l) {
				v[pindex] = trainSet[j].y[l] - avalue[numLayer - 1][l];
				pindex++;
				if (pindex > v.size()) { cout << pindex << ">" << v.size() << endl; assert(false); }
			}
			///backward propagation
			//calculate deviationOfN
			for (int k = 1; k < numLayer; ++k) {
				//deviationOfN[k].Zero();
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					for (int l = 0; l < deviationOfN[k].cols(); ++l)
						deviationOfN[k](j, l) = 0;
				}
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
				}
			}
			//deviation 
		//	s[numLayer - 1] = -deviationOfN[numLayer - 1];
			//cout << "s";
			for (int k = 0; k < s[numLayer - 1].size(); ++k)
				s[numLayer - 1][k] = -deviationOfN[numLayer - 1](k, k);
			//cout << "--";
			for (int k = numLayer - 2; k >= 1; --k) {
				s[k] = deviationOfN[k] * weights[k + 1].transpose() *s[k + 1];
			}
			//cout << "J";
			pindex = 0;
			for (int k = 1; k <= numLayer - 1; ++k) {
				//weights
				for (int i = 0; i < weights[k].rows(); ++i) {
					for (int l = 0; l < weights[k].cols(); ++l) {
						J(j, pindex) = s[k][i] * avalue[k - 1][l]; //s(k)*a(k-1)
						pindex++;
					}
				}
				//bais
				for (int i = 0; i < bais[k].size(); ++i) {
					J(j, pindex) = s[k][i];
					pindex++;
				}
			}
			if (pindex != totalParaNum) {
				cout << pindex << " " << totalParaNum << " " << J.cols() << endl;
				assert(false);
			}
		}

		for (int i = 1; i < preBais.size(); ++i) {
			for (int j = 0; j < preBais[i].size(); ++j) preBais[i][j] = bais[i][j];
		}
		for (int i = 1; i < preWeights.size(); ++i) {
			for (int j = 0; j < preWeights[i].rows(); ++j) {
				for (int k = 0; k < preWeights[i].cols(); ++k) {
					preWeights[i](j, k) = weights[i](j, k);
				}
			}
		}

		//
		MatrixXd IdenM(J.cols(), J.cols());
		for (int i = 0; i < IdenM.rows(); ++i) {
			for (int j = 0; j < IdenM.cols(); ++j)
				IdenM(i, j) = 0;
			IdenM(i, i) = 1;
		}
		//	IdenM.Identity();	
		int m = 1;
		while (true) {

			MatrixXd detap = -(J.transpose()*J + mu * IdenM).inverse()*J.transpose()*v;
			if (detap.cols() > 1) assert(false);

			//update weight: approximate steepest decent
			int pindex = 0;
			for (int k = 1; k <= numLayer - 1; ++k) {
				for (int i = 0; i < bais[k].size(); ++i) {
					for (int j = 0; j < bais[k - 1].size(); ++j) {
						weights[k](i, j) = preWeights[k](i, j) + detap(pindex, 0);
						pindex++;
					}
				}
				for (int i = 0; i < bais[k].size(); ++i) {
					bais[k][i] = preBais[k][i] + detap(pindex, 0);
					pindex++;
				}
			}
			error = getPrectErr(validationSet, validationnum, xnum, ynum);
			if (error <= preerror) {
				//cout << "LMBP" << " " << mu << " " << preerror << " " << error << endl;
				mu /= factor;
				break;
			}
			else {
				mu *= factor;
				if (m > 5) {  //reset the weights and bais as the previous values
					break;
				}
				m += 1;

				//	if (mu >= 10000) break;
				//	cout << "LMBP" << " " << mu << " " << preerror << " " << error << endl;
			}
		}
		preerror = error;

		if (error < minerror) {
			updateMinErrPara();
			minerror = error;
			valErrIte = 0;
		}

		if (error > minerror) valErrIte++;
		if (valErrIte > 100) {
			finish = true;
			break;
		}

		cout << ite << "\t" << error << endl;
#ifdef DEBUG
		cout << ite << "\t" << error << endl;
#endif
		//double trainerr = getPrectErr(trainSet, trainnum, xnum, ynum);
		//double validationerr = getPrectErr(validationSet, validationnum, xnum, ynum);
	//	cout << ite << "\t" << trainerr << "\t" << validationerr << "\t" << minerror << "\n";
		if (error <= accurate) {
			finish = true;
			break;
		}

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();

	///
	cout << "train by the LMBPCrossoverValidation: " << ite << "\t" << minerror << endl;
	//regission on the dataset
	double trainR = regressionOutput(trainSet, trainnum, xnum, ynum, ymax, ymin);
	double validationR = regressionOutput(validationSet, validationnum, xnum, ynum, ymax, ymin);
	double totalSetR = regressionOutput(data, datanum, xnum, ynum, ymax, ymin);
	cout << "regression R2 for train, validation, and totalSet : " << trainR << "\t" << validationR << "\t" << totalSetR << "\t";

#ifdef UNIX
	gettimeofday(&endtime, &tz);
	varduration = (endtime.tv_sec - starttime.tv_sec) + (endtime.tv_usec - starttime.tv_usec) / 1000000.0;
#else
	double endtime = clock();
	varduration = 1.0*(endtime - starttime) / CLOCKS_PER_SEC;
#endif
	cout << varduration << "s\n";

}

void NeuralNetwork::trainByBayesianRegularization(vector<TrainingPoint> data, int datanum, int xnum, int ynum, double mu, double factor, int maxEpochs, const vector<double> ymax, const vector<double> ymin) {
	//mu = 0.01; factor = 10;
	//the whole training set
	int ite = 0;
	double error = 0;
	//cout << "begin ....." << endl;
	int N = datanum * ynum;
	//cout << N << endl;
	double afa, beta;
	double ganma = totalParaNum; //the effective number of parameters
	double Ew = getParaErr();
	//cout << Ew << endl;
	double Ed = getSumSquareErr(data, datanum, xnum, ynum);
	//cout <<N <<"\t" << Ew << "\t" << Ed << endl;
	afa = ganma / (2 * Ew);
	beta = (N - ganma) / (2 * Ed);
	double preerror = getRegularationErr(afa, beta, data, datanum, xnum, ynum);
	//getPrectErr(data, datanum, xnum, ynum);
//cout << preerror << endl;
	const double accurate = 1e-7;
	bool finish = false;
	//	int maxEpochs = 200;

	//check the total parameter number
	int pcount = 0;
	//cout << bais.size() << "+ " << numLayer << endl;
	//for (int i = 0; i < bais.size(); ++i)
	//	cout << bais[i].size() << "-";
	//	cout << endl;
	for (int k = numLayer - 1; k >= 1; --k) {
		//	cout  bais[k].size() << " " << weights[k].rows() << " " << weights[k].cols() << endl;
		pcount += bais[k].size() + weights[k].rows()*weights[k].cols();// [k - 1].size() *bais[k].size();
	}
	if (pcount != totalParaNum) assert(false);

	//cout << "totalParaNum : " << totalParaNum << endl;

	//detaW, detaB
	vector<VectorXd> detaBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> detaWeights(bais.size()); //the weight matrix in each layer;
	for (int i = 1; i < detaBais.size(); ++i) {
		detaBais[i].resize(bais[i].size());
		for (int j = 0; j < detaBais[i].size(); ++j) detaBais[i][j] = 0;
	}
	for (int i = 1; i < detaWeights.size(); ++i) {
		detaWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < detaWeights[i].rows(); ++j) {
			for (int k = 0; k < detaWeights[i].cols(); ++k) {
				detaWeights[i](j, k) = 0;
			}
		}
	}

	vector<VectorXd> preBais(bais.size()); //the bias in each neuron in each layer;
	vector<MatrixXd> preWeights(weights.size()); //the weight matrix in each layer;
	for (int i = 1; i < preBais.size(); ++i) {
		preBais[i].resize(bais[i].size());
		preWeights[i].resize(weights[i].rows(), weights[i].cols());
		for (int j = 0; j < preBais[i].size(); ++j) preBais[i][j] = bais[i][j];
	}

	//Jacobian matrix
	MatrixXd J;//
	J.resize(N, totalParaNum);
	VectorXd v(N);


	double minerror = //getRegularationErr(afa, beta, data, datanum, xnum, ynum);
		getSumSquareErr(data, datanum, xnum, ynum);
	//double preEd = minerror;
	updateMinErrPara();
	int conIte = 0;
	//cout << v.size() << " " << J.cols() << " " << J.rows() << endl;

	while (!finish) {

		//calculate the J matrix
		for (int i = 0; i < J.rows(); ++i) {
			for (int j = 0; j < J.cols(); ++j)
				J(i, j) = 0;
		}
		int pindex = 0;
		for (int j = 0; j < datanum; ++j) {
			input = data[j].x;
			avalue[0] = input;
			for (int k = 1; k < numLayer; ++k) {
				nvalue[k] = weights[k] * avalue[k - 1] + bais[k];
				for (int l = 0; l < nvalue[k].size(); ++l)
					avalue[k][l] = activeFun(k, nvalue[k][l]);
			}
			//calcualte the error vector v
			pindex = j * avalue[numLayer - 1].size();
			for (int l = 0; l < avalue[numLayer - 1].size(); ++l) {
				v[pindex] = data[j].y[l] - avalue[numLayer - 1][l];
				pindex++;
				if (pindex > v.size()) { cout << pindex << ">" << v.size() << endl; assert(false); }
			}
			///backward propagation
			//calculate deviationOfN
			for (int k = 1; k < numLayer; ++k) {
				//deviationOfN[k].Zero();
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					for (int l = 0; l < deviationOfN[k].cols(); ++l)
						deviationOfN[k](j, l) = 0;
				}
				for (int j = 0; j < deviationOfN[k].rows(); ++j) {
					deviationOfN[k](j, j) = deviation(k, nvalue[k][j], avalue[k][j]);
				}
			}
			//deviation 
			//s[numLayer - 1] = -deviationOfN[numLayer - 1];
			for (int k = 0; k < s[numLayer - 1].size(); ++k)
				s[numLayer - 1][k] = -deviationOfN[numLayer - 1](k, k);
			for (int k = numLayer - 2; k >= 1; --k) {
				s[k] = deviationOfN[k] * weights[k + 1].transpose() *s[k + 1];
			}
			pindex = 0;
			for (int k = 1; k <= numLayer - 1; ++k) {
				//weights
				for (int i = 0; i < weights[k].rows(); ++i) {
					for (int l = 0; l < weights[k].cols(); ++l) {
						J(j, pindex) = s[k][i] * avalue[k - 1][l]; //s(k)*a(k-1)
						pindex++;
					}
				}
				//bais
				for (int i = 0; i < bais[k].size(); ++i) {
					J(j, pindex) = s[k][i];
					pindex++;
				}
			}
			if (pindex != totalParaNum) {
				cout << pindex << " " << totalParaNum << " " << J.cols() << endl;
				assert(false);
			}
		}

		for (int i = 1; i < preBais.size(); ++i) {
			for (int j = 0; j < preBais[i].size(); ++j) preBais[i][j] = bais[i][j];
		}
		for (int i = 1; i < preWeights.size(); ++i) {
			for (int j = 0; j < preWeights[i].rows(); ++j) {
				for (int k = 0; k < preWeights[i].cols(); ++k) {
					preWeights[i](j, k) = weights[i](j, k);
				}
			}
		}

		//
		MatrixXd IdenM(J.cols(), J.cols());
		for (int i = 0; i < IdenM.rows(); ++i) {
			for (int j = 0; j < IdenM.cols(); ++j)
				IdenM(i, j) = 0;
			IdenM(i, i) = 1;
		}
		//	IdenM.Identity();	
		int m = 1;
		while (true) {
			VectorXd p(totalParaNum);
			int pindex = 0;
			for (int k = 1; k <= numLayer - 1; ++k) {
				for (int i = 0; i < bais[k].size(); ++i) {
					for (int j = 0; j < bais[k - 1].size(); ++j) {
						p[pindex] = weights[k](i, j);
						pindex++;
					}
				}
				for (int i = 0; i < bais[k].size(); ++i) {
					p[pindex] = bais[k][i];
					pindex++;
				}
			}
			//cout << "p ";
			//the devaiation for the regulazation terms
			MatrixXd detap = -(2 * beta*J.transpose()*J + 2 * afa*IdenM + mu * IdenM).inverse()*(2 * beta*J.transpose()*v + 2 * afa*p);
			//MatrixXd detap = -(J.transpose()*J + mu*IdenM).inverse()*J.transpose()*v;
		//	cout << m << "--";
			if (detap.cols() > 1) assert(false);

			//update weight: approximate steepest decent
			pindex = 0;
			for (int k = 1; k <= numLayer - 1; ++k) {
				for (int i = 0; i < bais[k].size(); ++i) {
					for (int j = 0; j < bais[k - 1].size(); ++j) {
						weights[k](i, j) = preWeights[k](i, j) + detap(pindex, 0);
						pindex++;
					}
				}
				for (int i = 0; i < bais[k].size(); ++i) {
					bais[k][i] = preBais[k][i] + detap(pindex, 0);
					pindex++;
				}
			}
			error = getRegularationErr(afa, beta, data, datanum, xnum, ynum);// getPrectErr(data, datanum, xnum, ynum);
			//cout << error << "\t" << afa << "\t" << beta << "\t"<< ganma << endl;
			if (error <= preerror) {
				//cout << "LMBP" << " " << mu << " " << preerror << " " << error << endl;
				mu /= factor;
				break;
			}
			else {
				mu *= factor;
				if (m > 5) {  //reset the weights and bais as the previous values
					break;
				}
				m += 1;
				//	if (mu >= 10000) break;
				//	cout << "LMBP" << " " << mu << " " << preerror << " " << error << endl;
			}
		}
		preerror = error;
		//the second-order deviation
		MatrixXd H(J.rows(), J.cols());
		H = 2 * beta*J.transpose()*J + 2 * afa*IdenM;
		ganma = totalParaNum - 2 * afa* H.inverse().trace();

		//update regularization parameters
		Ew = getParaErr();
		Ed = getSumSquareErr(data, datanum, xnum, ynum);
		afa = ganma / (2 * Ew);
		beta = (N - ganma) / (2 * Ed);

		cout << ite << "\t+" << afa << " " << beta << " " << afa / beta << "\t+ " << ganma << "\t+ " << Ed / datanum << "\t" << Ed << "\t" << Ew << "\t" << error << endl;

		if (Ed < minerror) {
			updateMinErrPara();
			minerror = Ed;
			conIte = 0;
		}
		else {
			conIte++;
		}

		if (conIte >= 50) {
			finish = true;
			break;
		}

#ifdef DEBUG
		cout << ite << "\t" << error << endl;
#endif
		if (Ed / datanum <= accurate) {
			finish = true;
			break;
		}

		ite++;
		if (ite >= maxEpochs) break;
	}
	setParaAsMiniPara();
	double meanSquareErr = getPrectErr(data, datanum, xnum, ynum);
	cout << "afa/beta " << afa / beta << "\tganma(" << ganma << ")\t" << "n(" << totalParaNum << ")\t" << meanSquareErr << endl;
	double totalSetR = regressionOutput(data, datanum, xnum, ynum, ymax, ymin);
	cout << "regression R2 for the totalSet : " << totalSetR << endl;

}

//return the regression coefficients to know the correlation between network output a and the real value t
//R*R represents the proportion of the variablility in a data set that is accounted for by the linear regression
double NeuralNetwork::regressionOutput(vector<TrainingPoint> &testData, int datanum, int xnum, int ynum, const vector<double> ymax, const vector<double> ymin) {//return coefficient factor
	VectorXd predictY(output.size());
	//showParam();
	//calculate the error of the total samples 
	MatrixXd py(datanum, ynum);
	for (int j = 0; j < datanum; ++j) {
		//getOutput(&testData[l].x[0], &predictY[0], xnum, ynum);
		//	cout << "x:-----" << testData[j].x.size() << " " << testData[j].x << endl;
		predict(testData[j].x, predictY, xnum, ynum);
		//py.row(j) = (predictY + 1).*(ymax - ymin) / 2 + ymin;
		//reflect normilize the predictY
		for (int k = 0; k < ynum; ++k) py(j, k) = (predictY[k] + 1)*(ymax[k] - ymin[k]) / 2 + ymin[k];
		//	cout << j << " " << error << endl;
	}
	//
	VectorXd meant(ynum);
	VectorXd meana(ynum);
	for (int k = 0; k < meant.size(); ++k) {
		meant[k] = 0;
		meana[k] = 0;
	}
	for (int j = 0; j < datanum; ++j) {
		meant += testData[j].orignaly;
		for (int k = 0; k < ynum; ++k) meana[k] += py(j, k);
	}
	meant /= datanum;
	meana /= datanum;

	//cout << "meant: " << meant << endl;
	//cout << "meana: " << meana << endl;

	VectorXd stdt(ynum);
	VectorXd stda(ynum);
	for (int k = 0; k < stdt.size(); ++k) {
		stdt[k] = 0;
		stda[k] = 0;
	}
	//cout << "regission " << py.rows() << "\t" << py.cols() << "\t" << meana.size() << "\t" << stda.size() << endl;
	//cout << testData.size() << " " << datanum << " " << testData[0].orignaly.size() << endl;
	for (int j = 0; j < datanum; ++j) {
		//	cout << j << " " << testData[j].orignaly.size() << " " << meant.size() << " ";
		VectorXd temp(ynum);
		temp = (testData[j].orignaly - meant);
		for (int k = 0; k < ynum; ++k) {
			temp[k] = temp[k] * temp[k];
		}
		stdt += temp;
		//	cout << j << " ";
		for (int k = 0; k < ynum; ++k) {
			temp[k] = py(j, k) - meana[k];
		}
		//temp = (py.row(j) - meana);
		for (int k = 0; k < ynum; ++k) {
			temp[k] = temp[k] * temp[k];
		}
		stda += temp;
		//	cout << "+";
	}
	stdt = stdt / (datanum - 1);
	stda = stda / (datanum - 1);
	for (int k = 0; k < ynum; ++k) stdt[k] = sqrt(stdt[k]);
	for (int k = 0; k < ynum; ++k) stda[k] = sqrt(stda[k]);

	//cout << "stdt: " << endl;
	//cout << "stda: " << stda << endl;

	double R = 0;
	for (int j = 0; j < datanum; ++j) {
		for (int k = 0; k < ynum; ++k) {
			R += (testData[j].orignaly[k] - meant[k])*(py(j, k) - meana[k]);
		}
		//R += ((testData[j].orignaly - meant).transpose()*(py.row(j) - meana))[0];
	}
	//cout << "R " << endl;
	R = R / ((datanum - 1)*(stdt.transpose()*stda)[0]);
	return R * R;
}


void NeuralNetwork::outNormilizePredictFile(vector<TrainingPoint> &testData, int datanum, int xnum, int ynum, const vector<double> ymax, const vector<double> ymin, const char filename[]) {

	fstream ft(filename, ios::out);
	if (ft.fail()) {
		cout << "cannot open file " << filename << endl;
		assert(false);
	}
	VectorXd predictY(output.size());
	ft << "n\t";
	for (int k = 0; k < xnum; ++k) ft << "nx" << k + 1 << "\t";
	for (int k = 0; k < ynum; ++k) ft << "ny" << k + 1 << "\t";
	for (int k = 0; k < ynum; ++k) ft << "py" << k + 1 << "\t";
	for (int k = 0; k < ynum; ++k) ft << "y" << k + 1 << "\t";
	for (int k = 0; k < ynum; ++k) ft << "py" << k + 1 << "\t";
	for (int k = 0; k < ynum; ++k) ft << "dy" << k + 1 << "\t";
	ft << "d(py,y)\t";
	ft << endl;
	//showParam();
	//calculate the error of the total samples 
	for (int j = 0; j < datanum; ++j) {
		int l = j;
		//getOutput(&testData[l].x[0], &predictY[0], xnum, ynum);
		//	cout << "x:-----" << testData[j].x.size() << " " << testData[j].x << endl;
		predict(testData[j].x, predictY, xnum, ynum);
		//	cout << predictY << "++++" << testData[l].y << endl;
	//	VectorXd e = ((predictY - testData[l].y).transpose())*(predictY - testData[l].y);
		ft << j + 1 << "\t";
		for (int k = 0; k < xnum; ++k) ft << testData[j].x[k] << "\t";
		//for (int k = 0; k < ynum; ++k) ft << 2 * (testData[j].orignaly[k] - ymin[k]) / (ymax[k] - ymin[k]) - 1 <<"\t";
		for (int k = 0; k < ynum; ++k) ft << testData[j].y[k] << "\t";
		for (int k = 0; k < ynum; ++k) ft << predictY[k] << "\t";
		for (int k = 0; k < ynum; ++k) ft << testData[j].orignaly[k] << "\t";
		for (int k = 0; k < ynum; ++k) ft << (predictY[k] + 1)*(ymax[k] - ymin[k]) / 2 + ymin[k] << "\t";
		for (int k = 0; k < ynum; ++k) ft << (predictY[k] + 1)*(ymax[k] - ymin[k]) / 2 + ymin[k] - testData[j].orignaly[k] << "\t";
		double length = 0;
		for (int k = 0; k < ynum; ++k) {
			length += ((predictY[k] + 1)*(ymax[k] - ymin[k]) / 2 + ymin[k] - testData[j].orignaly[k])*((predictY[k] + 1)*(ymax[k] - ymin[k]) / 2 + ymin[k] - testData[j].orignaly[k]);
		}
		length = sqrt(length);
		ft << length << "\t";
		ft << endl;
		//	cout << j << " " << error << endl;
	}
	ft.close();
}


void NeuralNetwork::setNNParaValue(const vector<double> para) {
	//set the parameter into the nn
	int pindex = 0;
	for (int k = 1; k < numLayer; ++k) {
		for (int i = 0; i < weights[k].rows(); ++i) {
			for (int j = 0; j < weights[k].cols(); ++j) {
				weights[k](i, j) = para[pindex];
				pindex++;
			}
		}
		for (int i = 0; i < bais[k].size(); ++i) {
			bais[k][i] = para[pindex];
			pindex++;
		}
	}
}

double NeuralNetwork::calErrorWithPara(const vector<double> para, const vector<TrainingPoint> data, const int datanum, const int xnum, const int ynum) {
	//set the parameter into the nn
	setNNParaValue(para);
	return getPrectErr(data, datanum, xnum, ynum);
}

void NeuralNetwork::trainByGPSO(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int popsize, double w, double c1, double c2, int generation) {
	///
	vector<vector<double> > pop(popsize);
	vector<vector<double> > velocity(popsize);
	vector<vector<double> > pbest(popsize);
	vector<double> fit(popsize);
	vector<double> pbestFit(popsize);
	//double lower = -2000; double upper = 2000;
	const int MAXVELOCITY = 0.2*(upper - lower);
	for (int i = 0; i < pop.size(); ++i) {
		pop[i].resize(totalParaNum);
		velocity[i].resize(totalParaNum);
		pbest[i].resize(totalParaNum);
		for (int j = 0; j < pop[i].size(); ++j) {
			pop[i][j] = random(lower, upper);
			pbest[i][j] = pop[i][j];
			velocity[i][j] = random(-MAXVELOCITY, MAXVELOCITY);
		}
		fit[i] = calErrorWithPara(pop[i], data, datanum, xnum, ynum);
		pbestFit[i] = fit[i];
	}
	int gbestIndex = 0;
	for (int i = 0; i < pop.size(); ++i) {
		if (pbestFit[i] < pbestFit[gbestIndex]) gbestIndex = i;
	}

	//evolve 
	w = 0.9;
	for (int ite = 0; ite < generation; ++ite) {
		w = 0.9 - (0.9 - 0.4)*ite / generation;
		//updete x
		for (int i = 0; i < pop.size(); ++i) {
			for (int j = 0; j < pop[i].size(); ++j) {
				velocity[i][j] = w * velocity[i][j] + c1 * random(0, 1)*(pbest[i][j] - pop[i][j]) + c2 * random(0, 1)*(pbest[gbestIndex][j] - pop[i][j]);
				if (velocity[i][j] < -MAXVELOCITY) velocity[i][j] = -MAXVELOCITY;
				if (velocity[i][j] > MAXVELOCITY) velocity[i][j] = MAXVELOCITY;
				pop[i][j] = pop[i][j] + velocity[i][j];
				if (pop[i][j] < lower) pop[i][j] = lower;
				if (pop[i][j] > upper) pop[i][j] = upper;
			}
			fit[i] = calErrorWithPara(pop[i], data, datanum, xnum, ynum);
		}
		//update pbest
		for (int i = 0; i < pop.size(); ++i) {
			if (fit[i] < pbestFit[i]) {
				for (int j = 0; j < pop[i].size(); ++j) {
					pbest[i][j] = pop[i][j];
				}
				pbestFit[i] = fit[i];
			}
			if (fit[i] < pbestFit[gbestIndex]) {
				gbestIndex = i;
			}
		}
#ifdef DEBUG
		cout << ite << " " << pbestFit[gbestIndex] << endl;
#endif
		///
	}
	//set the weight and parameter in the nn
	setNNParaValue(pbest[gbestIndex]);
}

void NeuralNetwork::trainByDE(vector<TrainingPoint> data, int datanum, int xnum, int ynum, int popsize, double F, double CR, int generation) {
	vector<vector<double> > pop(popsize);
	vector<vector<double> > u(popsize);
	vector<double> fit(popsize);
	vector<double> uFit(popsize);
	//double lower = -2000; double upper = 2000;
	//const int MAXVELOCITY = 0.2*(upper - lower);
	for (int i = 0; i < pop.size(); ++i) {
		pop[i].resize(totalParaNum);
		u[i].resize(totalParaNum);
		for (int j = 0; j < pop[i].size(); ++j) {
			pop[i][j] = random(lower, upper);
			u[i][j] = pop[i][j];
		}
		fit[i] = calErrorWithPara(pop[i], data, datanum, xnum, ynum);
	}
	int gbestIndex = 0;
	for (int i = 0; i < pop.size(); ++i) {
		if (fit[i] < fit[gbestIndex]) gbestIndex = i;
	}

	//evolve 
	for (int ite = 0; ite < generation; ++ite) {
		//updete x
		for (int i = 0; i < pop.size(); ++i) {
			//rand/1 scheme
			int r1, r2, r3;
			r1 = random_int(0, popsize - 1);
			do { r2 = random_int(0, popsize - 1); } while (r2 == r1);
			do { r3 = random_int(0, popsize - 1); } while (r3 == r2 || r3 == r1);
			int rj = random_int(0, pop[0].size() - 1);
			for (int j = 0; j < pop[i].size(); ++j) {
				if (j == rj || random(0, 1) <= CR) {
					u[i][j] = pop[r1][j] + F * (pop[r2][j] - pop[r3][j]);
				}
				else u[i][j] = pop[i][j];
				if (u[i][j] < lower) u[i][j] = lower;
				if (u[i][j] > upper) u[i][j] = upper;
			}
			uFit[i] = calErrorWithPara(u[i], data, datanum, xnum, ynum);
		}
		//update pbest
		for (int i = 0; i < pop.size(); ++i) {
			if (uFit[i] < fit[i]) {
				for (int j = 0; j < pop[i].size(); ++j) {
					pop[i][j] = u[i][j];
				}
				fit[i] = uFit[i];
			}
			if (fit[i] < fit[gbestIndex]) {
				gbestIndex = i;
			}
		}

#ifdef DEBUG
		cout << ite << " " << fit[gbestIndex] << endl;
#endif
		///
	}
	//set the weight and parameter in the nn
	setNNParaValue(pop[gbestIndex]);
}


double NeuralNetwork::getPrectErr(vector<TrainingPoint> testData, int datanum, int xnum, int ynum) {
	double error = 0;
	VectorXd predictY(output.size());
	//showParam();
	//calculate the error of the total samples 
	for (int j = 0; j < datanum; ++j) {
		int l = j;
		//getOutput(&testData[l].x[0], &predictY[0], xnum, ynum);
	//	cout << "x:-----" << testData[j].x.size() << " " << testData[j].x << endl;
		predict(testData[j].x, predictY, xnum, ynum);
		//	cout << predictY << "++++" << testData[l].y << endl;
		VectorXd e = ((predictY - testData[l].y).transpose())*(predictY - testData[l].y);
		if (e.size() != 1) assert(false);
		error += e[0];
		//	cout << j << " " << error << endl;
	}
	error = (error / datanum);
	return error;
}

double NeuralNetwork::getSumSquareErr(vector<TrainingPoint> testData, int datanum, int xnum, int ynum) {
	double error = 0;
	VectorXd predictY(output.size());
	//showParam();
	//calculate the error of the total samples 
	for (int j = 0; j < datanum; ++j) {
		int l = j;
		//getOutput(&testData[l].x[0], &predictY[0], xnum, ynum);
		//	cout << "x:-----" << testData[j].x.size() << " " << testData[j].x << endl;
		predict(testData[j].x, predictY, xnum, ynum);
		//	cout << predictY << "++++" << testData[l].y << endl;
		VectorXd e = ((predictY - testData[l].y).transpose())*(predictY - testData[l].y);
		if (e.size() != 1) assert(false);
		error += e[0];
		//	cout << j << " " << error << endl;
	}
	//error = (error / datanum);
	return error;
}

//get the error of the parameters
double NeuralNetwork::getParaErr() {
	double paraerr = 0;
	for (int i = 1; i < numLayer; ++i) {
		for (int j = 0; j < weights[i].rows(); ++j) {
			for (int k = 0; k < weights[i].cols(); ++k) {
				paraerr += weights[i](j, k)*weights[i](j, k);
			}
		}
		//
		for (int j = 0; j < bais[i].size(); ++j) {
			paraerr += bais[i][j] * bais[i][j];
		}
	}
	return paraerr;
}

double NeuralNetwork::getRegularationErr(double afa, double beta, vector<TrainingPoint> testData, int datanum, int xnum, int ynum) {
	double paraerr = getParaErr();
	double dataerr = getSumSquareErr(testData, datanum, xnum, ynum);
	double rerr = beta * dataerr + afa * paraerr;
	return rerr;
}

double NeuralNetwork::predictDataSet(vector<TrainingPoint> &testData, int datanum, int xnum, int ynum, bool normilized, const vector<double> ymax, const vector<double> ymin) {
	double error = 0;
	//showParam();
	//calculate the error of the total samples 
	for (int j = 0; j < datanum; ++j) {
		int l = j;
		//getOutput(&testData[l].x[0], &predictY[0], xnum, ynum);
		//	cout << "x:-----" << testData[j].x.size() << " " << testData[j].x << endl;
		predict(testData[j].x, testData[j].y, xnum, ynum);
		if (normilized) {
#ifdef NORMMAXMIN
			testData[j].restorePYMaxMin(ymax, ymin, ynum);
#else
			testData[j].restorePYMeanStd(ymean, ystd, ynum);
#endif
		}
		//	cout << predictY << "++++" << testData[l].y << endl;
		VectorXd e = ((testData[j].y - testData[l].orignaly).transpose())*(testData[j].y - testData[l].orignaly);
		if (e.size() != 1) assert(false);
		error += e[0];

	}
	error = (error / datanum);
	return error;
}

double NeuralNetwork::getErrWithPara(vector<MatrixXd> weights, vector<VectorXd> bais, vector<TrainingPoint> testData, int datanum, int xnum, int ynum) {
	double error = 0;
	VectorXd predictY(output.size());
	//showParam();
	//calculate the error of the total samples 
	for (int j = 0; j < datanum; ++j) {
		int l = j;
		//getOutput(&testData[l].x[0], &predictY[0], xnum, ynum);
		//	cout << "x:-----" << testData[j].x.size() << " " << testData[j].x << endl;
		predictWithNNPara(weights, bais, testData[j].x, predictY, xnum, ynum);
		//	cout << predictY << "++++" << testData[l].y << endl;
		VectorXd e = ((predictY - testData[l].y).transpose())*(predictY - testData[l].y);
		if (e.size() != 1) assert(false);
		error += e[0];
		//	cout << j << " " << error << endl;
	}
	error = (error / datanum);
	return error;
}

void NeuralNetwork::outputSimuFunToFile(vector<TrainingPoint> simuData, int datanum, char filename[]) {
	fstream ft;
	ft.open(filename, ios::out);
	ft << "num\t";
	for (int j = 0; j < simuData[0].x.size(); ++j) ft << "x" << j + 1 << "\t";
	for (int j = 0; j < simuData[0].y.size(); ++j) ft << "Py" << j + 1 << "\t";
	for (int j = 0; j < simuData[0].y.size(); ++j) ft << "y" << j + 1 << "\t";
	ft << "\n";
	//calculate the error of the total samples 
	for (int j = 0; j < datanum; ++j) {
		//	int l = j;
			//getOutput(&testData[l].x[0], &predictY[0], xnum, ynum);

		//	predict(simuData[j].x, predictY, simuData[j].x.size(), output.size());
		//	cout << j << " " << simuData[j].x << " " << simuData[j].y << " " << predictY << endl;
		ft << j << "\t";
		for (int k = 0; k < simuData[j].orignalx.size(); ++k) {
			ft << simuData[j].orignalx[k] << "\t";
		}
		for (int k = 0; k < simuData[j].y.size(); ++k) {
			ft << simuData[j].y[k] << "\t";
		}
		for (int k = 0; k < simuData[j].orignaly.size(); ++k)
			ft << simuData[j].orignaly[k] << "\t";
		ft << "\n";
	}

	ft.close();
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork &nn) {

	//numLayer = nn.numLayer;
	//numNeuroEachLayer = nn.numNeuroEachLayer;
	//afunHiddenLayer = nn.afunHiddenLayer;
	//afunOutputLayer = nn.afunOutputLayer;

	setStructure(nn.numLayer, nn.numNeuroEachLayer, nn.afunHiddenLayer, nn.afunOutputLayer);

	input = nn.input;
	output = nn.output;
	bais = nn.bais;
	weights = nn.weights;

	s = nn.s;
	nvalue = nn.nvalue;
	avalue == nn.nvalue;
	deviationOfN = nn.deviationOfN;

	learnrate = nn.learnrate;
	totalParaNum = nn.totalParaNum;
	afunHiddenLayer = nn.afunHiddenLayer;
	afunOutputLayer = nn.afunOutputLayer;

	minbais = nn.minbais;
	minweights = nn.minweights;

	lower = lower;
	upper = upper;
	return *this;
}
