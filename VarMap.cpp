#include "VarMap.h"
//#include "../Algorithms/PSO/Swarm.h"
//#include "../Algorithms/PSO/Particle.h"
//#include "rand.h"
#include "Global.h"
#include "mine.h"
#include "cppmine.h"
#include <vector>
#include <fstream>
using namespace std;

int domination(const vector<double> &a, const vector<double> &b) {
	int smaller = 0, equal = 0, large = 0;
	if (a.size() != b.size()) {
		cout << a.size() << "\t" << b.size() << endl;
		assert(false);
	}
	for (int i = 0; i < a.size(); ++i) {
		if (a[i] < b[i]) smaller++;
		else if (a[i] > b[i]) large++;
		else equal++;
	}
	if (smaller > 0 && large == 0) return 1;  //a is better
	else if (large > 0 && smaller == 0) return -1;
	else return 0;
}

double calDistance(const vector<double> a, const vector<double> b, int startIndex) {
	double d = 0;
	int dimNum = a.size();
	if (dimNum > b.size()) dimNum = b.size();
	for (int j = startIndex; j < dimNum; ++j) {
		d += fabs(a[j] - b[j]);// *(a[j] - b[j]);
	}
	//d = sqrt(d);
	return d;
}

void OutputVector(const vector<double> &x) {
	cout << "(";
	for (int j = 0; j < x.size(); ++j) cout << x[j] << ",";
	cout << ")\t";
}

vector<vector<double> > generate_weight(int nobj, int p) {
	double b;
	vector<int> visit(nobj);
	vector<int> stack(nobj);
	vector<int> ysum(nobj);

	vector<vector<double> > weights;

	int k = 0; // the index of the w
	b = 1.0 / p;
	int top = -1;
	stack[++top] = 0;
	ysum[top] = p;
	int v = 0;
	int sum = 0;
	for (int j = 0; j < visit.size(); ++j) visit[j] = 0;
	while (top != -1) {
		v = stack[top];
		//	printf("%d\t", stack[top]);
		if (top == nobj - 1) {
			stack[top] = ysum[top];
			visit[top] = true;
			top--;
			//the route of the w
			vector<double> temp_w(nobj);
			for (int j = 0; j < nobj; ++j) {
				temp_w[j] = (double)stack[j] * b;
			}
			weights.push_back(temp_w);
			k++;
			continue;
		}
		else {
			if (visit[top + 1] && v == ysum[top]) {
				visit[top] = true;
				top--;
				continue;
			}
			if (!visit[top + 1] && v == ysum[top]) {
				ysum[top + 1] = ysum[top] - v;
				top++;
				stack[top] = 0;
				for (int j = top + 1; j < nobj; ++j) {
					visit[j] = false;
				}
			}
			else if (visit[top + 1] && v < ysum[top]) {
				stack[top] += 1;
				stack[top + 1] = 0;
				ysum[top + 1] = ysum[top] - stack[top];

				for (int j = top + 1; j < nobj; ++j) {
					visit[j] = false;
				}
				top++;
			}
			else if (!visit[top + 1] && v < ysum[top]) {
				stack[top + 1] = 0;
				ysum[top + 1] = ysum[top] - stack[top];
				for (int j = top + 1; j < nobj; ++j) {
					visit[j] = false;
				}
				top++;
			}

		}
		//	printf("%d\t%d\t%d\n", top,stack[top],stack[top-1]);

	}
	return weights;
}

vector<double> define_refer_point(const vector<CSolution> & archive) {
	int num_obj = archive[0].f.size();
	if (num_obj <= 0) assert(false);
	vector<double> v(num_obj);
	for (int j = 0; j < archive.size(); ++j) {
		for (int k = 0; k < num_obj; ++k) {
			if (j == 0 || archive[j].f[k] < v[k]) {
				v[k] = archive[j].f[k];
			}
		}
	}
	return v;
}

double VectorNorm2(const vector <double> &vec1)
{
	int dim = vec1.size();
	double sum = 0;
	for (int n = 0; n < dim; n++)
		sum += vec1[n] * vec1[n];
	return sqrt(sum);
}

double innerProduct(const vector <double> &vec1, const vector <double> &vec2)
{
	int dim = vec1.size();
	double sum = 0;
	for (int n = 0; n < dim; n++)
		sum += vec1[n] * vec2[n];
	return sum;
}

double cal_pbi(const vector<double> &y_obj, const vector<double> &namda, const vector<double> &referencepoint, double &d1, double &d2) {
	int nobj = y_obj.size();
	double eta = 5;
	vector<double> vect_normal(nobj, 0);
	double namda_norm = VectorNorm2(namda);
	for (int n = 0; n < nobj; n++)
	{
		vect_normal[n] = namda[n] / namda_norm;
	}
	double d1_proj = innerProduct(y_obj, vect_normal);// abs(innerProduct(y_obj, vect_normal) - innerProduct(referencepoint, vect_normal));
	vector<double> thet_v(nobj);
	double temp = 0;
	for (int j = 0; j < nobj; ++j) {
		thet_v[j] = y_obj[j] - d1_proj * vect_normal[j];
		temp = temp + thet_v[j] * thet_v[j];
	}
	/*
	double d1_proj = abs(innerProduct(y_obj, vect_normal) - innerProduct(referencepoint, vect_normal));
	double temp = d1_proj * d1_proj - 2 * d1_proj*(innerProduct(y_obj, vect_normal) - innerProduct(referencepoint, vect_normal))
		+ (innerProduct(y_obj, y_obj) + innerProduct(referencepoint, referencepoint) - 2 * innerProduct(y_obj, referencepoint));
		*/
	double d2_pred = sqrt(temp);
	double scalar_obj = d1_proj + eta * d2_pred;
	d1 = d1_proj;
	d2 = d2_pred;
	return scalar_obj;

}

vector<vector<int> > determine_sol_weight(const vector<CSolution> &archive, const vector<vector<double> > &weights, vector<double> &pbi_value) {
	//
	vector<double> referpoint = define_refer_point(archive);
	//vector<double> pbi_value(archive.size());
	//vector<int> weight_arcindex(archive.size(), -1);
	vector<vector<int> > weight_arcindex(weights.size());
	double d1; double d2;
	for (int j = 0; j < archive.size(); ++j) {
		vector<double> value(weights.size());
		int min_index = -1;
		double perpen_dist = -1;
		for (int i = 0; i < weights.size(); ++i) {
			value[i] = cal_pbi(archive[j].f, weights[i], referpoint, d1, d2);
			//if (min_index == -1 || value[min_index] > value[i])
			if (min_index == -1 || perpen_dist > d2) {
				min_index = i;
				perpen_dist = d2;
			}
		}
		//取最小的点
		if (min_index == -1) assert(false);
		weight_arcindex[min_index].push_back(j);
	}
	return weight_arcindex;
}

vector<vector<int> > pair_sol_detected_change(const vector<CSolution> &a, const vector<CSolution> &b, const vector<vector<double> > &weights, bool use_ep, bool pbipair) {
	//bool use_ep = false;
	if (!pbipair && !use_ep) {
		vector<vector<int> > each_pair;// (weights.size());
		for (int j = 0; j < a.size(); ++j) {
			vector<int> pair(2);
			pair[0] = j;
			pair[1] = j;
			each_pair.push_back(pair);
		}
		return each_pair;
	}
	vector<CSolution> first_arc = a;
	vector<CSolution> sec_arc = b;
	vector<double> pbi_first_arc(first_arc.size());
	vector<double> pbi_sec_arc(sec_arc.size());
	vector<vector<int> > first_corindex = determine_sol_weight(first_arc, weights, pbi_first_arc);
	vector<vector<int> > sec_corindex = determine_sol_weight(sec_arc, weights, pbi_sec_arc);
	//determine the solution pair between two archives
	vector<vector<int> > each_pair;// (weights.size());
	//cout << "\nweights.size(): " <<  weights.size() << "\t";
	for (int j = 0; j < weights.size(); ++j) {
		//cout << j << "\t" << first_corindex[j].size() << "\t" << sec_corindex[j].size() << endl;
		if (first_corindex[j].size() == 0 || sec_corindex[j].size() == 0) continue;
		vector<int> visit_first(first_corindex[j].size(), 0);
		vector<int> visit_sec(sec_corindex[j].size(), 0);
		int ava_num = first_corindex[j].size();
		if (ava_num > sec_corindex[j].size()) ava_num = sec_corindex[j].size();
		//if (ava_num > 1) ava_num = 1;
		for (int k = 0; k < ava_num; ++k) {
			//find the min one in the first archive
			int min_index = -1;
			for (int i = 0; i < first_corindex[j].size(); ++i) {
				if (visit_first[i]) continue;
				if (min_index == -1 || pbi_first_arc[first_corindex[j][i]] < pbi_first_arc[first_corindex[j][min_index]])
					min_index = i;
			}
			visit_first[min_index] = 1;
			//find the min one in the second archive
			int min_sec_index = -1;
			for (int i = 0; i < sec_corindex[j].size(); ++i) {
				if (visit_sec[i]) continue;
				if (min_sec_index == -1 || pbi_sec_arc[sec_corindex[j][i]] < pbi_sec_arc[sec_corindex[j][min_sec_index]])
					min_sec_index = i;
			}
			visit_sec[min_sec_index] = 1;
			vector<int> pair(2);
			pair[0] = first_corindex[j][min_index];
			pair[1] = sec_corindex[j][min_sec_index];
			each_pair.push_back(pair);
			//break;
		}
	}

	//check self-variables and correlated variables in the archives;//using MIC
	int num_dim = first_arc[0].x.size();
	if (num_dim <= 0) assert(false);
	vector<double> error_dim(num_dim, 0);
	for (int j = 0; j < each_pair.size(); ++j) {
		//
		int first_index = each_pair[j][0];
		int sec_index = each_pair[j][1];
		vector<double> x = first_arc[first_index].x;
		vector<double> y = sec_arc[sec_index].x;
		for (int k = 0; k < num_dim; ++k) {
			error_dim[k] = error_dim[k] + fabs(x[k] - y[k]);
		}
	}
	/*
	cout << "change of x: ";
	vector<double> error(num_dim);
	int min_index = -1;
	for (int k = 0; k < num_dim; ++k) {
		error[k] = error_dim[k] / each_pair.size() / (GlobalPara::upperBound[k] - GlobalPara::lowBound[k]);
		cout << error[k] << ",";
		if (min_index == -1 || error[min_index] > error[k]) {
			min_index = k;
		}
	}
	cout << "\tdim " << min_index;
	cout << endl;
	*/
	return each_pair;
}

////////////////////////////////check ===========================================================================================================
vector<double> cal_x_change_bet_envi(const vector<solpair> &set, int num_dim) {
	vector<double> x_change(num_dim);
	for (int k = 0; k < set.size(); ++k) {
		for (int j = 0; j < num_dim; ++j) {
			x_change[j] = x_change[j] + fabs(set[k].x[j] - set[k].y[j]);
		}
	}
	for (int j = 0; j < num_dim; ++j) {
		x_change[j] = x_change[j] / set.size() / (DynPara::upperBound[j] - DynPara::lowBound[j]);
	}
	return x_change;
}

double cal_MIC(double *x, double *y, int n) {
	MINE mine = MINE(0.6, 15, EST_MIC_APPROX);
	double value = 0;
	/*
	try
	{
		mine = MINE(0.6, 15, EST_MIC_APPROX);
	}
	catch (char *s)
	{
		cout << "WARNING: " << s << "\n";
		cout << "MINE will be set with alpha=0.6 and c=15" << "\n";
		mine = MINE(0.6, 15, EST_MIC_APPROX);
	}
	*/
	//int n = x.size();
	try
	{
		mine.compute_score(x, y, n);
	}
	catch (char *s)
	{
		cout << "ERROR: " << s << "\n";
		return 1;
	}
	value = mine.mic();
	//delete mine;
	return value;
}

vector<vector<double> > cal_mic_bet_envi(const vector<solpair> &set, int num_dim) {
	double *x, *y;
	int n = set.size();
	x = new double[n];
	y = new double[n];

	vector<vector<double> > mic_value(num_dim);
	for (int j = 0; j < num_dim; ++j) mic_value[j].resize(num_dim);
	for (int j = 0; j < num_dim; ++j) {
		for (int i = 0; i < set.size(); ++i) {
			x[i] = set[i].x[j];
		}
		for (int k = 0; k < num_dim; ++k) {
			if (j == k) continue;
			for (int i = 0; i < set.size(); ++i) {
				y[i] = set[i].y[k];
			}
			mic_value[j][k] = cal_MIC(x, y, n);
			//mic_value[k][j] = mic_value[j][k];
			//cout << set.size() << "\t" << "dim " << j << "\t" << k << "\t" << mic_value[j][k] << endl;
		}
		//构造一个时间序列
		//for(int k = j + 1; k < num_dim; )
	}
	delete x;
	delete y;
	return mic_value;
}

vector<vector<double> > cal_mic_in_last_envi(int i, int num_dim) {
	int n = InfoMemory::detectedEnviSol[i].size();
	double *x = new double[n];
	double *y = new double[n];
	//int num_dim = GlobalPara::dimNum;
	vector<vector<double> > mic_last_envi(num_dim);
	for (int j = 0; j < num_dim; ++j) mic_last_envi[j].resize(num_dim);
	for (int j = 0; j < num_dim; ++j) {
		int count = 0;
		//int i = cur_envi - 1;
		for (int m = 0; m < InfoMemory::detectedEnviSol[i].size(); ++m) {
			x[count] = InfoMemory::detectedEnviSol[i][m].x[j];
			count = count + 1;
		}
		for (int k = j + 1; k < num_dim; ++k) {
			count = 0;
			for (int m = 0; m < InfoMemory::detectedEnviSol[i].size(); ++m) {
				y[count] = InfoMemory::detectedEnviSol[i][m].x[k];
				count = count + 1;
			}
			mic_last_envi[j][k] = cal_MIC(x, y, n);
			mic_last_envi[k][j] = mic_last_envi[j][k];
			//cout << set.size() << "\t" << "dim " << j << "\t" << k << "\t" << mic_value[j][k] << endl;
		}
		//构造一个时间序列
		//for(int k = j + 1; k < num_dim; )
	}
	delete x;
	delete y;
	return mic_last_envi;
}

vector<vector<double> > cal_mic_inside_all_envi(int cur_envi, int num_dim) {
	int n = 0;
	for (int j = 0; j < cur_envi; ++j) {
		n = n + InfoMemory::detectedEnviSol[j].size();
	}
	double *x = new double[n];
	double *y = new double[n];
	//int num_dim = GlobalPara::dimNum;
	vector<vector<double> > mic_inside_envi(num_dim);
	for (int j = 0; j < num_dim; ++j) mic_inside_envi[j].resize(num_dim);
	for (int j = 0; j < num_dim; ++j) {
		int count = 0;
		for (int i = 0; i < cur_envi; ++i) {
			for (int m = 0; m < InfoMemory::detectedEnviSol[i].size(); ++m) {
				x[count] = InfoMemory::detectedEnviSol[i][m].x[j];
				count = count + 1;
			}
		}
		for (int k = j + 1; k < num_dim; ++k) {
			count = 0;
			for (int i = 0; i < cur_envi; ++i) {
				for (int m = 0; m < InfoMemory::detectedEnviSol[i].size(); ++m) {
					y[count] = InfoMemory::detectedEnviSol[i][m].x[k];
					count = count + 1;
				}
			}
			mic_inside_envi[j][k] = cal_MIC(x, y, n);
			mic_inside_envi[k][j] = mic_inside_envi[j][k];
			//cout << set.size() << "\t" << "dim " << j << "\t" << k << "\t" << mic_value[j][k] << endl;
		}
		//构造一个时间序列
		//for(int k = j + 1; k < num_dim; )
	}
	delete x;
	delete y;
	return mic_inside_envi;
}

void find_connect_dim_in_envi(const vector<vector<int> >  &cor_matrix, vector<vector<int> > &connect_block, vector<int> &block_index, int num_dim) {
	for (int j = 0; j < num_dim; ++j) { connect_block[j].push_back(j); block_index[j] = j; }
	for (int j = 0; j < num_dim; ++j) {
		for (int k = j + 1; k < num_dim; ++k) {
			if (j == k) continue;
			if (cor_matrix[j][k] == 1) {
				int insert_index = block_index[j];
				int merge_index = block_index[k];
				if (connect_block[merge_index].size() == 0 || merge_index == insert_index) {
					continue;
				}
				else {
					for (int i = 0; i < connect_block[merge_index].size(); ++i) {
						connect_block[insert_index].push_back(connect_block[merge_index][i]);
					}
					connect_block[merge_index].clear();
					block_index[k] = insert_index;
				}
			}
		}
	}
	while (true) {
		bool flag = true;
		for (int j = 0; j < connect_block.size(); ++j) {
			if (connect_block[j].size() == 0) {
				connect_block.erase(connect_block.begin() + j);
				flag = false;
				break;
			}
		}
		if (flag) break;
	}
	for (int j = 0; j < connect_block.size(); ++j) {
		for (int k = 0; k < connect_block[j].size(); ++k) {
			block_index[connect_block[j][k]] = j;
		}
	}
}

vector<vector<int> > define_corrleration(const vector<vector<double> >& mic_last_envi, int num_dim, double threthold) {
	vector<vector<int> > cor_matrix(num_dim);
	for (int j = 0; j < num_dim; ++j) cor_matrix[j].resize(num_dim);
	for (int j = 0; j < num_dim; ++j) {
		for (int k = 0; k < num_dim; ++k) {
			cor_matrix[j][k] = 0;
			if (mic_last_envi[j][k] > threthold) cor_matrix[j][k] = 1;
		}
	}
	return cor_matrix;
}

vector<int> define_dimblock_ind_var(const vector<vector<int> > &connect_block, const vector<double> &x_change) {
	vector<int> indepen_var(connect_block.size());
	for (int j = 0; j < connect_block.size(); ++j) {
		int min_index = -1;
		for (int k = 0; k < connect_block[j].size(); ++k) {
			if (min_index == -1 || x_change[connect_block[j][k]] < x_change[connect_block[j][min_index]])
				min_index = k;
		}
		indepen_var[j] = connect_block[j][min_index];
	}
	return indepen_var;
}

vector<double> train_network(const vector<solpair> &set, const vector<vector<int> > &related_dims, int maxEpochs, int num_envi) {
	int num_dim = DynPara::dimNum;
	vector<vector<double> > test_set(set.size());
	for (int j = 0; j < set.size(); ++j) test_set[j].resize(num_dim);
	vector<double> train_time(num_dim);
	for (int j = 0; j < num_dim; ++j) {
		vector<int> sdim = related_dims[j]; //再加一个时间维度
		//cout << "dim " << j << "\t:";
		//for (int k = 0; k < sdim.size(); ++k) cout << sdim[k] << "\t";
		//cout << endl;
		InfoMemory::train_set.clear();
		for (int i = 0; i < set.size(); ++i) {
			TrainingPoint np;
			int tdim = sdim.size();
			if (use_time_condition) tdim += 1;
			VectorXd x(tdim);
			VectorXd y(1);
			for (int l = 0; l < sdim.size(); ++l) {
				x[l] = (set[i].x[sdim[l]] - DynPara::lowBound[sdim[l]]) / (DynPara::upperBound[sdim[l]] - DynPara::lowBound[sdim[l]]);
			}
			if (use_time_condition) x[sdim.size()] = (double)set[i].t / num_envi;
			y[0] = (set[i].y[j] - DynPara::lowBound[j])/(DynPara::upperBound[j] - DynPara::lowBound[j]);
		//	if (i == 0) cout << "\t*" << j << "*" << set[i].y[j] <<"," << y[0] << "->";
			np.setData(x, y, x.size(), 1);
			//cout << "train " << i << "\tx(";
			//for (int l = 0; l < x.size(); ++l) {
			//	cout << x[l] << ",";
		//	}
		//	cout << ")\t->\t" << y[0] << endl;
			InfoMemory::train_set.push_back(np);
		}
		int xdim = sdim.size();
		if (use_time_condition) xdim += 1;

		int hiddenNeuNum = 1;
		int b = InfoMemory::train_set.size() / ((xdim + 1) * 10);
		if (b < hiddenNeuNum) b = hiddenNeuNum;

		//InfoMemory::NNEachDim.resize(num_dim);
		int numlayer = 3;
		vector<int> numNeuEachLayer(numlayer);
		numNeuEachLayer[0] = xdim;
		numNeuEachLayer[1] = b;  //9, 10
		numNeuEachLayer[2] = 1;
		

		if (InfoMemory::NNEachDim[j].numNeuroEachLayer[0] != xdim || InfoMemory::NNEachDim[j].numNeuroEachLayer[1] != b) {
			InfoMemory::NNEachDim[j].numNeuroEachLayer[0] = xdim;
			InfoMemory::NNEachDim[j].numNeuroEachLayer[1] = b;
			InfoMemory::NNEachDim[j].setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
			InfoMemory::NNEachDim[j].initilizeParaNgWi90();
		}
		//normilizeDataXWithGivenBound(InfoMemory::train_set, InfoMemory::xmax, InfoMemory::xmin, xdim);
		//normilizeDataYWithGivenBound(InfoMemory::train_set, InfoMemory::xmax, InfoMemory::xmin, 1);

		//train
		char BPname[10];
		char simufile[100];
		strcpy(BPname, "LMBP");
		double mu = 0.01; //small value in the begining
		double factor = 10;
		vector<double> xmax;
		vector<double> xmin;

		clock_t start_time = clock();
		InfoMemory::NNEachDim[j].trainByLevenbergMarquardtBP(InfoMemory::train_set, InfoMemory::train_set.size(), xdim, 1, mu, factor, maxEpochs, xmax, xmin);
		clock_t end_time = clock();
		double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;
		train_time[j] = duration;

		int batch_size = 32;
		while (batch_size >= InfoMemory::train_set.size()) {
			batch_size = batch_size / 2;
		}
		/*if (batch_size >= InfoMemory::train_set.size()) batch_size = 16;
		if (batch_size >= InfoMemory::train_set.size()) batch_size = 8;
		if (batch_size >= InfoMemory::train_set.size()) batch_size = 4;
		if (batch_size >= InfoMemory::train_set.size()) batch_size = 2;
		if (batch_size >= InfoMemory::train_set.size()) batch_size = 1;*/
		//InfoMemory::NNEachDim[j].trainBySDBPMiniBatch(InfoMemory::train_set, InfoMemory::train_set.size(), xdim, 1, batch_size, maxEpochs);
		//check the training cases and results    //data, int datanum, int xnum, int ynum, int batchSize, int maxEpochs
		//cout << "dim " << j << ":\n";
		
		for (int k = 0; k < InfoMemory::train_set.size(); ++k) {
			//cout << k << "\tx(";
			//for (int m = 0; m < InfoMemory::train_set[k].x.size(); ++m) {
			//	cout << InfoMemory::train_set[k].x[m] << ",";
			//}
			//cout << ")\t";
			//cout << "y(" << InfoMemory::train_set[k].y[0] << ")\t";
			VectorXd ny(1);
			InfoMemory::NNEachDim[j].predict(InfoMemory::train_set[k].x, ny, xdim, 1);
			
			test_set[k][j] = ny[0]*(DynPara::upperBound[j] - DynPara::lowBound[j]) + DynPara::lowBound[j];
			if (test_set[k][j] < DynPara::lowBound[j]) test_set[k][j] = DynPara::lowBound[j];
			if (test_set[k][j] > DynPara::upperBound[j]) test_set[k][j] = DynPara::upperBound[j];
			//if (k == 0) cout << ny[0] << "," << test_set[k][j] <<  "\t";
			//cout << "predicty(" << ny[0] << ")\n";
		}
		
	}
	double avg_time = 0;
	for (int j = 0; j < num_dim; ++j)
		avg_time += train_time[j];
	avg_time /= num_dim;
	InfoMemory::train_time[InfoMemory::cur_run].push_back(avg_time);

	//training errors
	vector<double> train_error(num_dim, 0);
	for (int j = 0; j < num_dim; ++j) {
		for (int k = 0; k < set.size(); ++k) {
			train_error[j] += fabs(set[k].y[j] - test_set[k][j]);
		}
		train_error[j] /= set.size();
	}
	return train_error;
	//cout << endl;
	//
	/*
	double train_error = 0;
	for (int j = 0; j < test_set.size(); ++j) {
		double error = 0;
		for (int k = 0; k < num_dim; ++k) {
			error = error + fabs(test_set[j][k] - set[j].y[k]);// *(test_set[j][k] - set[j].y[k]);
		}
		train_error = train_error + error;
		if (j == 0) {
			cout << error << "\t";
			cout << "target y:(";
			for (int k = 0; k < num_dim; ++k) {
				cout << set[j].y[k] << ",";
			}
			cout << ")\t->\t(";
			for (int k = 0; k < num_dim; ++k) {
				cout << test_set[j][k] << ",";
			}
			cout << ")\n";
		}
	}
	cout << "train error:\t" << train_error/test_set.size() << endl;
	*/
}

vector<double> predict_a_sol_by_NN(int cur_envi, int envi_num, const vector<vector<int> > &relateddim,const vector<double> &x, bool sameIndep) {
	int i = InfoMemory::detectedEnviSol.size();
	int num_dim = DynPara::dimNum;
	vector<double> y(num_dim);
	for (int k = 0; k < num_dim; ++k) {
		if (sameIndep && relateddim[k].size() == 1) {
			int xindex = relateddim[k][0];
			y[k] = x[xindex];
		}
		else {
			int xdim = relateddim[k].size();
			if (use_time_condition) xdim += 1;
			TrainingPoint newpoint;
			VectorXd nx(xdim);
			VectorXd ny(1);
			for (int m = 0; m < relateddim[k].size(); ++m) {
				int xindex = relateddim[k][m];
				nx[m] = (x[xindex] - DynPara::lowBound[xindex]) / (DynPara::upperBound[xindex] - DynPara::lowBound[xindex]);
			}
			if (use_time_condition) nx[xdim - 1] = (double)(i) / envi_num;
			newpoint.setData(nx, ny, xdim, 1);
			InfoMemory::NNEachDim[k].predict(newpoint.x, ny, xdim, 1);
			y[k] = ny[0];
			y[k] = y[k] * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
			if (y[k] < DynPara::lowBound[k]) y[k] = DynPara::lowBound[k];
			if (y[k] > DynPara::upperBound[k]) y[k] = DynPara::upperBound[k];
		}
	}
	return y;
}

vector<vector<double> > predict_pop_by_network(const vector<CSolution>& pop, int i, int envi_num, const vector<vector<int> >& relateddim, const vector<int>& self_variable, bool main_indep_var){
	// = true; 
	int num_dim = DynPara::dimNum;
	vector<vector<double> > new_sol(pop.size());
	for (int j = 0; j < new_sol.size(); ++j) {
		new_sol[j].resize(num_dim);
	}
	for (int k = 0; k < num_dim; ++k) {

		int xdim = relateddim[k].size();//
		if(use_time_condition ) xdim += 1;
		for (int j = 0; j < pop.size(); ++j) {
			if (main_indep_var && relateddim[k].size() == 1) {
				int xindex = relateddim[k][0];
				new_sol[j][k] = pop[j].x[xindex];
			}
			else {
				TrainingPoint newpoint;
				VectorXd nx(xdim);
				VectorXd ny(1);
				for (int m = 0; m < relateddim[k].size(); ++m) {
					int xindex = relateddim[k][m];
					nx[m] = (pop[j].x[xindex] - DynPara::lowBound[xindex]) / (DynPara::upperBound[xindex] - DynPara::lowBound[xindex]);
				}
				if(use_time_condition) nx[xdim - 1] = (double)(i + 1) / envi_num;
				newpoint.setData(nx, ny, xdim, 1);
				//newpoint.normilizeXMinMax(InfoMemory::xmax, InfoMemory::xmin, xdim);
				//cout << "predict " << j << "(";
				//for (int j = 0; j < nx.size(); ++j) {
				//	cout << nx[j] << ",";
				//}
				//cout << ")\t->\t";
				InfoMemory::NNEachDim[k].predict(newpoint.x, ny, xdim, 1);
				double y = ny[0];
				//cout << y << "\n";
				/*
				y = y * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
				//newpoint.restorePYMaxMin(InfoMemory::xmax, InfoMemory::xmin, 1);
				if (y < DynPara::lowBound[k]) y = DynPara::lowBound[k];
				if (y > DynPara::upperBound[k]) y = DynPara::upperBound[k];
				*/
				new_sol[j][k] = y;
			}
			//
		}
	}
	//
	for (int j = 0; j < new_sol.size(); ++j) {
		for (int k = 0; k < num_dim; ++k) {
			new_sol[j][k] = new_sol[j][k] * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
			if (new_sol[j][k] < DynPara::lowBound[k]) new_sol[j][k] = DynPara::lowBound[k];//  (DynPara::lowBound[k] + InfoMemory::detectedEnviSol[i][j].x[k]) / 2;
			if (new_sol[j][k] > DynPara::upperBound[k]) new_sol[j][k] = DynPara::upperBound[k];// (DynPara::upperBound[k] + InfoMemory::detectedEnviSol[i][j].x[k]) / 2;
		}
	}

	if (keep_self_variable) {
		//check the diversity of the self-variables dimensions
		vector<double> diversity(num_dim, 0); //check the maximum and minmum value
		vector<double> min_value(num_dim, 0);
		vector<double> max_value(num_dim, 0);
		for (int k = 0; k < num_dim; ++k) {
			for (int i = 0; i < pop.size(); ++i) {
				if (i == 0 || pop[i].x[k] < min_value[k])
					min_value[k] = pop[i].x[k];
				if (i == 0 || pop[i].x[k] > max_value[k])
					max_value[k] = pop[i].x[k];
			}
		}
		for (int k = 0; k < num_dim; ++k) {
			for (int i = 0; i < pop.size(); ++i) {
				diversity[k] += pop[i].x[k];
			}
			diversity[k] /= pop.size();
		}
		vector<int> is_correct(num_dim, 1);
		for (int k = 0; k < self_variable.size(); ++k) {
			int d = self_variable[k];
			if (max_value[d] - min_value[d] > fabs(DynPara::upperBound[d] - DynPara::lowBound[d]) / 2) {
				is_correct[d] = 0;
			}
		}
		for (int j = 0; j < new_sol.size(); ++j) {
			for (int k = 0; k < num_dim; ++k) {
				if (is_correct[k] == 0) {
					new_sol[j][k] = pop[j].x[k];
				}
			}
		}
	}

	return new_sol;
}

vector<vector<double> > predict_sol_by_network(int cur_envi, int envi_num, const vector<vector<int> > &relateddim, bool main_indep_var) {
	// = true; 
	int i = InfoMemory::detectedEnviSol.size() - 1;
	if(cur_envi == -1) i = InfoMemory::detectedEnviSol.size() - 2;
	int num_dim = DynPara::dimNum;
	vector<vector<double> > new_sol(InfoMemory::detectedEnviSol[i].size());
	for (int j = 0; j < new_sol.size(); ++j) {
		new_sol[j].resize(num_dim);
	}
	for (int k = 0; k < num_dim; ++k) {
		
		int xdim = relateddim[k].size() + 1;
		for (int j = 0; j < InfoMemory::detectedEnviSol[i].size(); ++j) {
			if (main_indep_var && relateddim[k].size() == 1) {
				int xindex = relateddim[k][0];
				new_sol[j][k] = InfoMemory::detectedEnviSol[i][j].x[xindex];
			}
			else {
				TrainingPoint newpoint;
				VectorXd nx(xdim);
				VectorXd ny(1);
				for (int m = 0; m < relateddim[k].size(); ++m) {
					int xindex = relateddim[k][m];
					nx[m] = (InfoMemory::detectedEnviSol[i][j].x[xindex] - DynPara::lowBound[xindex]) / (DynPara::upperBound[xindex] - DynPara::lowBound[xindex]);
				}
				nx[xdim - 1] = (double)(i + 1) / envi_num;
				newpoint.setData(nx, ny, xdim, 1);
				//newpoint.normilizeXMinMax(InfoMemory::xmax, InfoMemory::xmin, xdim);
				//cout << "predict " << j << "(";
				//for (int j = 0; j < nx.size(); ++j) {
				//	cout << nx[j] << ",";
				//}
				//cout << ")\t->\t";
				InfoMemory::NNEachDim[k].predict(newpoint.x, ny, xdim, 1);
				double y = ny[0];
				//cout << y << "\n";
				/*
				y = y * (DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
				//newpoint.restorePYMaxMin(InfoMemory::xmax, InfoMemory::xmin, 1);
				if (y < DynPara::lowBound[k]) y = DynPara::lowBound[k];
				if (y > DynPara::upperBound[k]) y = DynPara::upperBound[k];
				*/
				new_sol[j][k] = y;
			}
			//
		}
	}
	for (int j = 0; j < new_sol.size(); ++j) {
		for (int k = 0; k < num_dim; ++k) {
			new_sol[j][k] = new_sol[j][k] *(DynPara::upperBound[k] - DynPara::lowBound[k]) + DynPara::lowBound[k];
			if (new_sol[j][k] < DynPara::lowBound[k]) new_sol[j][k] = DynPara::lowBound[k];//  (DynPara::lowBound[k] + InfoMemory::detectedEnviSol[i][j].x[k]) / 2;
			if (new_sol[j][k] > DynPara::upperBound[k]) new_sol[j][k] = DynPara::upperBound[k];// (DynPara::upperBound[k] + InfoMemory::detectedEnviSol[i][j].x[k]) / 2;
		}
	}
	return new_sol;
}

void initialNNInfo(int i) {
	//bool useFunNN = true;
	//
	int num_dim = DynPara::dimNum;
	
	if (i == 0 || num_dim > InfoMemory::xmax.size()) {
		InfoMemory::xmax.resize(num_dim);
		InfoMemory::xmin.resize(num_dim);
		for (int j = 0; j < num_dim; ++j) {
			InfoMemory::xmax[j] = 1;// DynPara::upperBound[j];
			InfoMemory::xmin[j] = 0;// DynPara::lowBound[j];
		}
	}
	if (i == 0) {
		InfoMemory::ymax.resize(1);
		InfoMemory::ymin.resize(1);
	}

	if (i >= 1 && (i == 1 || InfoMemory::NNEachDim.size() != num_dim)) {
		InfoMemory::NNEachDim.resize(num_dim);
		int numlayer = 3;
		vector<int> numNeuEachLayer(numlayer);
		numNeuEachLayer[0] = num_dim;
		numNeuEachLayer[1] = 6;  //9, 10
		numNeuEachLayer[2] = 1;
		for (int j = 0; j < InfoMemory::NNEachDim.size(); ++j) {
			InfoMemory::NNEachDim[j].setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
			InfoMemory::NNEachDim[j].initilizeParaNgWi90();
		}
		//all dimension to all dimension
		//all dimension model
		numNeuEachLayer[0] = num_dim;
		numNeuEachLayer[1] = 5;
		numNeuEachLayer[2] = 1;
		//InfoMemory::curVarNN.setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
		//InfoMemory::curVarNN.initilizeParaNgWi90();
	}
}

void normilizeDataXWithGivenBound(vector<TrainingPoint> &dataSet, vector<double> xmax, vector<double> xmin, int dim) {
	for (int i = 0; i < dataSet.size(); ++i) {
		dataSet[i].normilizeXMinMax(xmax, xmin, dim);
	}
}

void normilizeDataYWithGivenBound(vector<TrainingPoint> &dataSet, vector<double> ymax, vector<double> ymin, int dim) {
	for (int i = 0; i < dataSet.size(); ++i) {
		dataSet[i].normilizeYMinMax(ymax, ymin, dim);
	}
}

#ifdef VV

/*
bool fitBetter(double a, double b){
	if (DynPara::optimization_type == MIN) return a < b;
	else return a > b;
}
*/


void updateYMaxMin(){
	if (InfoMemory::solInCurEnvi.size() <= 0) assert(false);
	for (int j = 0; j < InfoMemory::solInCurEnvi.size(); ++j){
		if (j == 0 || InfoMemory::ymax[0] < InfoMemory::solInCurEnvi[j].orignaly[0])
			InfoMemory::ymax[0] = InfoMemory::solInCurEnvi[j].orignaly[0];
		if (j == 0 || InfoMemory::ymin[0] > InfoMemory::solInCurEnvi[j].orignaly[0])
			InfoMemory::ymin[0] = InfoMemory::solInCurEnvi[j].orignaly[0];
	}
}



void normilizeDataY(vector<TrainingPoint> &dataSet, const int ydim, vector<double> &maxyvalue, vector<double> &minyvalue){
	for (int i = 0; i < dataSet.size(); ++i){
		for (int j = 0; j < ydim; ++j){
			if (i == 0){
				maxyvalue[j] = dataSet[i].orignaly[j];
				minyvalue[j] = dataSet[i].orignaly[j];
			}
			else{
				if (maxyvalue[j] < dataSet[i].orignaly[j])
					maxyvalue[j] = dataSet[i].orignaly[j];
				if (minyvalue[j] > dataSet[i].orignaly[j])
					minyvalue[j] = dataSet[i].orignaly[j];
			}
		}
	}
	//#define NORMMAXMIN
	///
	for (int i = 0; i < dataSet.size(); ++i){
		dataSet[i].normilizeYMinMax(maxyvalue, minyvalue, ydim);
	}
}

void outputConOptima(int curenvi, int num_run){

}

double tDistance(TrainingPoint a, TrainingPoint b){
	double d = 0;
	if (a.orignalx.size() != b.orignalx.size()) { cout << "woring in dimension of trainingpoint\n"; assert(false); }
	for (int j = 0; j < a.orignalx.size(); ++j){
		d += (a.orignalx[j] - b.orignalx[j])*(a.orignalx[j] - b.orignalx[j]);
	}
	d = sqrt(d);
	return d;
}

//delete the same solutions 
void recordConOptima(int curenvi){
	//	CSwarm::pre_optima[0];
	//delete the same optima in the pre_optima
	
	int oldsize = InfoMemory::conOptEachEnvi[curenvi].size();
	if (oldsize == 0) return;
	double bound = (DynPara::upperBound[0] - DynPara::lowBound[0])*1e-2;
	vector<int> deleteIndex(InfoMemory::conOptEachEnvi[curenvi].size());
	for (int j = 0; j < deleteIndex.size(); ++j) deleteIndex[j] = 0;
	bool same = false;
	vector<int> visit(InfoMemory::conOptEachEnvi[curenvi].size());
	vector<int> wdeleteIndex;
	for (int i = 0; i < visit.size(); ++i) visit[i] = 0;
	for (int i = 0; i < InfoMemory::conOptEachEnvi[curenvi].size(); ++i){
		if (deleteIndex[i]) continue;
		same = false;
		int BestOne = i;
		for (int j = 0; j < visit.size(); ++j) visit[j] = 0;
		visit[i] = 1;
		for (int j = i + 1; j < InfoMemory::conOptEachEnvi[curenvi].size(); ++j){
			if (deleteIndex[j]) continue;
			double dis = tDistance(InfoMemory::conOptEachEnvi[curenvi][i], InfoMemory::conOptEachEnvi[curenvi][j]);
			if (dis <= bound){
				visit[j] = 1;
				//	CSwarm::pre_optima[0].Delete_Optimum(j);
				same = true;
				if (fitBetter(InfoMemory::conOptEachEnvi[curenvi][j].orignaly[0], InfoMemory::conOptEachEnvi[curenvi][BestOne].orignaly[0])){
					BestOne = j;
				}
			}
		}
		if (same){
			//
			for (int j = 0; j < visit.size(); ++j){
				if (visit[j]) deleteIndex[j] = 1;
			}
			deleteIndex[BestOne] = 0;
			wdeleteIndex.push_back(BestOne);
		}
	}
	int dnum = 0;
	for (int j = 0; j < deleteIndex.size(); ++j) if (deleteIndex[j])dnum++;
	int rnum = deleteIndex.size() - dnum;
	sort(wdeleteIndex.begin(), wdeleteIndex.end());
	for (int j = 0; j < wdeleteIndex.size() - 1; ++j) if (wdeleteIndex[j] == wdeleteIndex[j + 1]) assert(false);
	for (int j = 0; j < wdeleteIndex.size(); ++j){
		int index = wdeleteIndex[j] - j;
		InfoMemory::conOptEachEnvi[curenvi].erase(InfoMemory::conOptEachEnvi[curenvi].begin() + index);
	}

	if (InfoMemory::conOptEachEnvi[curenvi].size() == 0 || InfoMemory::conOptEachEnvi[curenvi].size() != rnum)
	{
		cout << oldsize << "\t" << InfoMemory::conOptEachEnvi[curenvi].size() << " " << rnum << "\n"; assert(false);
	}
}

void clusterSol(const vector<TrainingPoint> &dataSet, vector<int> &index, vector<int> &ceterIndex, const int clstnum){
	int firstcenter = random_int(0, dataSet.size() - 1);
	vector<double> dis(dataSet.size());
	if (ceterIndex.size() < clstnum) ceterIndex.resize(clstnum);
	ceterIndex[0] = firstcenter;
	for (int k = 1; k < clstnum; ++k){
		ceterIndex[k] = -1;
		double sum = 0;
		for (int j = 0; j < dataSet.size(); ++j){
			dis[j] = 0;
			double len = 0;
			for (int cindex = 0; cindex < k; ++cindex){
				len = 0;
				for (int k = 0; k < dataSet[j].orignalx.size(); ++k){
					len += (dataSet[j].orignalx[k] - dataSet[ceterIndex[cindex]].orignalx[k])*(dataSet[j].orignalx[k] - dataSet[ceterIndex[cindex]].orignalx[k]);
				}
				len = sqrt(len);
				if (cindex == 0 || len < dis[j]){
					dis[j] = len;
				}
			}
			sum += dis[j];
		}
		//
		double r = random(0, sum);
		double psum = 0;
		for (int j = 0; j < dataSet.size(); ++j){
			psum += dis[j];
			if (r <= psum) {
				bool beselected = false;
				for (int l = 0; l < k; ++l){
					if (ceterIndex[l] == j){ beselected = true; break; }
				}
				if (beselected) continue;
				ceterIndex[k] = j;
				break;
			}
		}
		if (r > psum || ceterIndex[k] == -1) { assert(false); }
	}

	///clust the solutions according to cluster centers
	for (int j = 0; j < dataSet.size(); ++j){
		double len = 0;
		for (int cindex = 0; cindex < clstnum; ++cindex){
			len = 0;
			for (int k = 0; k < dataSet[j].orignalx.size(); ++k){
				len += (dataSet[j].orignalx[k] - dataSet[ceterIndex[cindex]].orignalx[k])*(dataSet[j].orignalx[k] - dataSet[ceterIndex[cindex]].orignalx[k]);
			}
			len = sqrt(len);
			if (cindex == 0 || len < dis[j]){
				index[j] = cindex;
				dis[j] = len;
			}
		}
	}
}

//i: the current environment
//g: the current generation in SPSO
//num: the number of solutions map pairs
void outputMapSolutionsToFile(int i, int g, const vector<int> mapIndex, const vector<int> mapCIndex, const int num, const vector<int> lastClusterId, const vector<int> curClusterId, const vector<double> lminfit, const vector<double> lmaxfit, const vector<double> ecminfit, const vector<double> ecmaxfit){
	//output the solution map set into file
	//cout << num << " " << dd"\n";
	char mapfilename[50];
	sprintf(mapfilename, "solMapDataSet_%d_%d.out", i, g);
	fstream fsol;
	fsol.open(mapfilename, ios::out);
	if (fsol.fail()){
		cout << "cannot open file :" << mapfilename << endl; assert(false);
	}
	int num_sol = 0;
	vector<double> realcurmaxf(num_sol);
	for (int j = 0; j < realcurmaxf.size(); ++j) {
		realcurmaxf[j] = 1;// MovingPeak::peak[j][DynPara::dimNum + 1];
		fsol << j << "\t" << lminfit[j] << "\t" << lmaxfit[j] << "->\t" << ecminfit[j] << "\t" << ecmaxfit[j] << "\t" << realcurmaxf[j] << "\t";
		fsol << "(" << (realcurmaxf[j] - ecmaxfit[j]) / (ecmaxfit[j] - ecminfit[j]) << ")\t";
		fsol << endl;
	}
	fsol << "n\t";
	for (int k = 0; k < DynPara::dimNum; ++k) fsol << "x" << k + 1 << "\t";
	fsol << "fx\t";
	fsol << "nrank\t";
	for (int k = 0; k < DynPara::dimNum; ++k) fsol << "y" << k + 1 << "\t";
	fsol << "fy\t";
	fsol << "nrank\t";
	fsol << "realrank\t";
	fsol << "d(x,y)\n";
	//cout << InfoMemory::solMapBetEnvi.size() <<" " << num << endl; 
	for (int j = 0; j < num; ++j){
		//cout << j <<" ";
		fsol << j << "\t";
		if (InfoMemory::solMapBetEnvi[j].orignalx.size() != DynPara::dimNum){
			cout << InfoMemory::solMapBetEnvi[j].orignalx.size() << " " << DynPara::dimNum << " " << mapIndex[j] << endl;
			assert(false);
		}
		if (InfoMemory::solMapBetEnvi[j].orignaly.size() != DynPara::dimNum){
			cout << InfoMemory::solMapBetEnvi[j].orignaly.size() << " " << DynPara::dimNum << endl;
			assert(false);
		}
		for (int k = 0; k < DynPara::dimNum; ++k) fsol << InfoMemory::solMapBetEnvi[j].orignalx[k] << "\t";
		fsol << "(" << InfoMemory::solInLastEnvi[mapIndex[j]].orignaly[0] << ")\t";
		int lastclst = lastClusterId[j];
		fsol << "<" << (InfoMemory::solInLastEnvi[mapIndex[j]].orignaly[0] - lminfit[lastclst]) / (lmaxfit[lastclst] - lminfit[lastclst]) << ">\t";
		//	cout << mapIndex[j] << " " << mapCIndex[j] << endl;
		for (int k = 0; k < DynPara::dimNum; ++k) fsol << InfoMemory::solMapBetEnvi[j].orignaly[k] << "\t";
		fsol << "(" << InfoMemory::solInCurEnvi[mapCIndex[j]].orignaly[0] << ")\t";
		int curclst = curClusterId[j];
		if (curclst >= ecmaxfit.size()) {
			cout << curclst << "\tis out of the size " << ecmaxfit.size() << endl;
		}
		fsol << "<" << (InfoMemory::solInCurEnvi[mapCIndex[j]].orignaly[0] - ecminfit[curclst]) / (ecmaxfit[curclst] - ecminfit[curclst]) << ">\t";
		fsol << "<" << (InfoMemory::solInCurEnvi[mapCIndex[j]].orignaly[0] - ecminfit[curclst]) / (realcurmaxf[curclst] - ecminfit[curclst]) << ">\t";
		double length = 0;
		int lowdim = min(DynPara::dimNum, DynPara::dimNum);
		for (int k = 0; k < lowdim; ++k)
			length += (InfoMemory::solMapBetEnvi[j].orignalx[k] - InfoMemory::solMapBetEnvi[j].orignaly[k])*
			(InfoMemory::solMapBetEnvi[j].orignalx[k] - InfoMemory::solMapBetEnvi[j].orignaly[k]);
		length = sqrt(length);
		fsol << length << endl;
	}
	fsol.close();
}


void outputMapSolutionEachEnviToFile(int i, const int num){
	char mapfilename[50];
	sprintf(mapfilename, "solMapDataSet_%d.out", i);
	fstream fsol;
	fsol.open(mapfilename, ios::out);
	if (fsol.fail()){
		cout << "cannot open file :" << mapfilename << endl; assert(false);
	}
	fsol << "n\t";
	for (int k = 0; k < DynPara::dimNum; ++k) fsol << "x" << k + 1 << "\t";
	fsol << "fx\t";
	for (int k = 0; k < DynPara::dimNum; ++k) fsol << "y" << k + 1 << "\t";
	fsol << "fy\t";
	fsol << "d(x,y)\n";
	for (int j = 0; j < num; ++j){
		fsol << j << "\t";
		for (int k = 0; k < DynPara::dimNum; ++k) fsol << InfoMemory::solMapBetEnvi[j].orignalx[k] << "\t";
		fsol << InfoMemory::solInLastEnvi[j].orignaly[0] << "\t";
		for (int k = 0; k < DynPara::dimNum; ++k) fsol << InfoMemory::solMapBetEnvi[j].orignaly[k] << "\t";
		fsol << InfoMemory::solInCurEnvi[j].orignaly[0] << "\t";
		double length = 0;
		int lowdim = DynPara::dimNum;
		if (lowdim > DynPara::dimNum) lowdim = DynPara::dimNum;
		for (int k = 0; k < lowdim; ++k)
			length += (InfoMemory::solMapBetEnvi[j].orignalx[k] - InfoMemory::solMapBetEnvi[j].orignaly[k])*
			(InfoMemory::solMapBetEnvi[j].orignalx[k] - InfoMemory::solMapBetEnvi[j].orignaly[k]);
		length = sqrt(length);
		fsol << length << "\t";
		fsol << "\n";
	}
	fsol.close();
}

void outputConOptMapToFile(int i, int num, const vector<double> lastXFit, const vector<double> curXFit, const vector<int> lastpeak, const vector<int> curpeak){
	char mapfilename[50];
	int number_of_peaks = DynPara::num_peakorfun;
	sprintf(mapfilename, "%sP%d_conOptMap.out", DynPara::route,number_of_peaks);
	fstream fsol;
	if (i <= 1) fsol.open(mapfilename, ios::out);
	else fsol.open(mapfilename, ios::out | ios::app);
	if (fsol.fail()){
		cout << "cannot open file :" << mapfilename << endl; assert(false);
	}
	fsol << "envi\t" << i - 1 << " --->\t" << i << endl;
	fsol << "n\t";
	for (int k = 0; k < DynPara::dimNum; ++k) fsol << "x" << k + 1 << "\t";
	fsol << "fx\t";
	for (int k = 0; k < DynPara::dimNum; ++k) fsol << "y" << k + 1 << "\t";
	fsol << "fy\t";
	fsol << "d(x,y)\n";
	for (int j = 0; j < num; ++j){
		fsol << j << "\t";
		for (int k = 0; k < DynPara::dimNum; ++k) fsol << InfoMemory::solMapBetEnvi[j].orignalx[k] << "\t";
		fsol << "<" << lastpeak[j] << ">\t";
		fsol << "(" << lastXFit[j] << ")\t";
		for (int k = 0; k < DynPara::dimNum; ++k) fsol << InfoMemory::solMapBetEnvi[j].orignaly[k] << "\t";
		fsol << "<" << curpeak[j] << ">\t";
		fsol << "(" << curXFit[j] << ")\t";
		double length = 0;
		int lowdim = DynPara::dimNum;
		if (lowdim > DynPara::dimNum) lowdim = DynPara::dimNum;
		for (int k = 0; k < lowdim; ++k)
			length += (InfoMemory::solMapBetEnvi[j].orignalx[k] - InfoMemory::solMapBetEnvi[j].orignaly[k])*
			(InfoMemory::solMapBetEnvi[j].orignalx[k] - InfoMemory::solMapBetEnvi[j].orignaly[k]);
		length = sqrt(length);
		fsol << length << "\t";
		fsol << "\n";
	}
	fsol.close();
}

//clsuter the data set according to converged optima obtained in the environment (whichEnviOpt)
//precondition: the dataSet is sorted according to their fitness values                                   Local_Optima center; const int whichEnviOpt
//clstSize: in each cluster
void clstSolByOptima(vector<TrainingPoint> &dataSet, vector<int> &belongCluster, vector<vector<int> > &dataEachClst, vector<TrainingPoint> &center, const int clstnum, vector<vector<int> > &clstSize, bool clstSizeLimit){

	if (clstnum != center.size()){
		cout << clstnum << "\t" << center.size()<< endl;
	}
	///clust the solutions according to cluster centers
	double mindis = 0;
	if (dataEachClst.size() != clstnum){
		cout << "dataeachClst size " << dataEachClst.size() << " != " << clstnum << endl;
		assert(false);
	}
	for (int j = 0; j < dataEachClst.size(); ++j){
		dataEachClst[j].clear();
	}
	if (belongCluster.size() != dataSet.size()){
		belongCluster.resize(dataSet.size());
	}
	//cout << "begin clst ....\n";
	vector<double> dis(dataSet.size()); //the cloest distance of each point to the center
	//cout << dataSet.size() <<" " << center.numbers <<" " << clstnum <<" " << dataEachClst.size()  << "\n";
	for (int j = 0; j < dataSet.size(); ++j){
		//cout << j << " ";
		double len = 0;
		for (int cindex = 0; cindex < clstnum; ++cindex){
			len = 0;
			for (int k = 0; k < dataSet[j].orignalx.size(); ++k){
				len += (dataSet[j].orignalx[k] - center[cindex].orignalx[k])
					*(dataSet[j].orignalx[k] - center[cindex].orignalx[k]);
			}
			len = sqrt(len);
			if (cindex == 0 || len < mindis){
				belongCluster[j] = cindex;
				mindis = len;
				dis[j] = mindis;
			}
		}
		//cout << belongCluster[j] << "(" << dataEachClst[belongCluster[j]].size() << ") ";
		dataEachClst[belongCluster[j]].push_back(j); ///
		//	cout << dataEachClst.size() << " ";
	}
	//	cout << "\n";
	//cout << dataEachClst.size() << endl;
	//
	//the size of each cluster is limited by the clstSize
	//the individuals with larger distances are deleted from the clst and find another closer clst
	if (clstSizeLimit){
		if (clstSize.size() != clstnum){			cout << clstSize.size() << " " << clstnum << endl; assert(false);		}
		if (dataEachClst.size() != clstnum){ cout << dataEachClst.size() << " " << clstnum << endl; assert(false); }
		//	for(int j  =0; j < clstSize.size(); ++ j) cout << clstSize[j].size() <<" "; cout << endl;
		//	for(int j = 0; j < dataEachClst.size(); ++ j) cout << dataEachClst[j].size() <<" "; cout << endl;
		for (int j = 0; j < clstnum; ++j){
			//	getchar();
			//cout << j <<" " << clstSize[j].size() << " " << dataEachClst[j].size() <<" ";
			//getchar();
			if (clstSize[j].size() >= dataEachClst[j].size()) continue;
			//getchar();
			//find the index with larger distance
			vector<int> visit(dataEachClst[j].size());
			for (int k = 0; k < visit.size(); ++k){
				visit[k] = 0;
			}
			for (int k = clstSize[j].size(); k < dataEachClst[j].size(); ++k){
				////
				int largest = -1;
				for (int l = 0; l < visit.size(); ++l){
					if (visit[l])continue;
					if (largest == -1 || dis[dataEachClst[j][l]] > dis[dataEachClst[j][largest]]){
						largest = l;
					}
				}
				////
				visit[largest] = 1;
			}
			//
			int left = 0; int right = dataEachClst[j].size() - 1;
			while (left < right){
				while (!visit[left] && left < dataEachClst[j].size()) left++;
				while (visit[right] && right >= 0) right--;
				if (left >= dataEachClst[j].size() || right < 0) break;
				//////
				if (left >= right) break;
				int temp = dataEachClst[j][right];
				dataEachClst[j][right] = dataEachClst[j][left];
				dataEachClst[j][left] = temp;
				visit[left] = 0;
				visit[right] = 1;
			}
			for (int k = clstSize[j].size(); k < dataEachClst[j].size(); ++k){
				if (!visit[k])  assert(false);
				int newclst = -1;
				double minlen = -1;
				int index = dataEachClst[j][k];
				for (int cindex = 0; cindex < clstnum; ++cindex){
					if (clstSize[cindex].size() <= dataEachClst[cindex].size()) continue;
					double len = 0;
					for (int dim = 0; dim < dataSet[index].orignalx.size(); ++dim){
						len += (dataSet[index].orignalx[dim] - center[cindex].orignalx[dim])*(dataSet[index].orignalx[dim] - center[cindex].orignalx[dim]);
					}
					len = sqrt(len);
					if (newclst == -1 || len < minlen){
						newclst = cindex;
						minlen = len;
					}
				}
				///
				if (newclst == -1){
					cout << "clstnum = " << clstnum << ": " << j << " " << clstSize[j].size() << " " << dataEachClst[j].size() << "\n";
					for (int cindex = 0; cindex < center.size(); ++cindex){
						cout << cindex << " ";
						for (int dim = 0; dim < dataSet[index].orignalx.size(); ++dim){
							cout << center[cindex].orignalx[dim] << ",";
						}
						cout << endl;
					}
					for (int cindex = 0; cindex < clstnum; ++cindex){
						if (clstSize[cindex].size() <= dataEachClst[cindex].size()) continue;
						double len = 0;
						for (int dim = 0; dim < dataSet[index].orignalx.size(); ++dim){
							len += (dataSet[index].orignalx[dim] - center[cindex].orignalx[dim])*(dataSet[index].orignalx[dim] - center[cindex].orignalx[dim]);
						}
						len = sqrt(len);
						if (newclst == -1 || len < minlen){
							newclst = cindex;
							minlen = len;
						}
						cout << cindex << " " << len << endl;
					}
					assert(false);
				}
				dataEachClst[newclst].push_back(index);
			}
			while (dataEachClst[j].size() > clstSize[j].size()) dataEachClst[j].pop_back();
		}
		//	getchar();
		//sort the ind in each clst according to their fitness values
		for (int j = 0; j < clstnum; ++j){
			sort(dataEachClst[j].begin(), dataEachClst[j].end());
		}
	}
}

//vector<vector<int> > &cIndexInEachClst: the cluster result of the solutions in the current environment
//vector<vector<int> > indexInEachClst: the cluster result of the solutions in the last environment
void trainCurVarNN(int i, bool useVarNN, int maxEpochs, vector<int> &cbelongCluster, vector<vector<int> > &cIndexInEachClst, vector<vector<int> > indexInEachClst, int clstnum){
	if (i >= 1 && useVarNN && useNN){  //last-->current (all dimensions)
		sort(InfoMemory::solInCurEnvi.begin(), InfoMemory::solInCurEnvi.end());
		for (int j = 0; j < cbelongCluster.size(); ++j) cbelongCluster[j] = j;
		clstSolByOptima(InfoMemory::solInCurEnvi, cbelongCluster, cIndexInEachClst, InfoMemory::conOptEachEnvi[i - 1], clstnum, indexInEachClst, limitIndNumInClst);  //cluster the solutions
		InfoMemory::solMapBetEnvi.clear();
		
		//map between the sol in two environment according to partition of the cluster and their corresponding fitness values
		vector<int> mapIndex(InfoMemory::solInCurEnvi.size());
		vector<int> mapCIndex(InfoMemory::solInCurEnvi.size());
		int mindex = 0;
		vector<double> lastminfit(clstnum);
		vector<double> lastmaxfit(clstnum);
		vector<double> curminfit(clstnum);
		vector<double> curmaxfit(clstnum);
		vector<int> lastclusterId(InfoMemory::solInCurEnvi.size());
		vector<int> curclusterId(InfoMemory::solInCurEnvi.size());

		for (int j = 0; j < clstnum; ++j) {
			if (indexInEachClst[j].size() == 0 || cIndexInEachClst[j].size() == 0) { continue; }
			//the fitness values of the solutions in the last environment
			double lminfit = InfoMemory::solInLastEnvi[indexInEachClst[j][0]].orignaly[0];
			double lmaxfit = InfoMemory::solInLastEnvi[indexInEachClst[j].back()].orignaly[0];
			//the max and min of the solutions in current environment
			double cminfit = InfoMemory::solInCurEnvi[cIndexInEachClst[j][0]].orignaly[0];
			double cmaxfit = InfoMemory::solInCurEnvi[cIndexInEachClst[j].back()].orignaly[0];
			/////expand the bound of the fitness 
			double ecminfit = min(cminfit, lminfit);// -(cmaxfit - cminfit)*0.05;
			double ecmaxfit = cmaxfit + (cmaxfit - ecminfit)* DynPara::nfscale;
			double normcmaxfit = (cmaxfit - ecminfit) / (ecmaxfit - ecminfit);
			lastminfit[j] = lminfit; lastmaxfit[j] = lmaxfit;
			curminfit[j] = ecminfit; curmaxfit[j] = ecmaxfit;
			DynPara::ecmaxfit[j] = ecmaxfit;
		//	cout << j <<"\t" << DynPara::nfscale << "\t" << ecminfit << "\t" << cmaxfit << "\t"  << ecmaxfit << endl;
			int lastoneindex = -1;
			double sdis = 0;
			bool mapinorder = true;
			if (cIndexInEachClst[j].size() > indexInEachClst[j].size()) { mapinorder = false; }
			for (int l = 0; l < cIndexInEachClst[j].size(); ++l) {
				//find the max index		
				int r = lastoneindex + 1;
				int curIndex = cIndexInEachClst[j][l];
				double normcfit = (InfoMemory::solInCurEnvi[curIndex].orignaly[0] - ecminfit) / (ecmaxfit - ecminfit);
				double minabsf = 1000;
				int maxavaIndex = indexInEachClst[j].size() - cIndexInEachClst[j].size() + l + 1;
				double snormlfit = 0;
				bool first_one = true;
				if (mapinorder) {
					//find one to map according to the fitness value (=)
					for (int sr = r; sr < maxavaIndex; ++sr) {
						if (sr >= indexInEachClst[j].size()) {
							cout << r << "\t" << maxavaIndex << "\t" << sr << "\t" << indexInEachClst[j].size() << "\t" << indexInEachClst[j].size() << "\t" << cIndexInEachClst[j].size() << endl;
							assert(false);
						}
						int lastIndex = indexInEachClst[j][sr];
						double normlfit = (InfoMemory::solInLastEnvi[lastIndex].orignaly[0] - lminfit) / (lmaxfit - lminfit);
						double xdis = InfoMemory::solInLastEnvi[lastIndex].xdistance(InfoMemory::solInCurEnvi[curIndex]);
						double fdis = fabs(normcfit - normlfit);

						//find the one with cloest fitness values in the avalible range
						if (mapByFitness) {
							if (first_one || fdis < minabsf) {// - InfoMemory::solInLastEnvi[indexInEachClst[j][sr]].orignaly[0]) < minabsf) {
								minabsf = fdis;
								r = sr;
								sdis = xdis;
								snormlfit = normlfit;
								first_one = false;
							}
						}
						else if (mapByDistance) {
							if (first_one || xdis < minabsf) {
								minabsf = xdis;
								r = sr;
								sdis = xdis;
								snormlfit = normlfit;
								first_one = false;
							}
						}
					}
					if (r >= indexInEachClst[j].size() - cIndexInEachClst[j].size() + l + 1) assert(false);
					if (first_one) assert(false);
				}
				else {
					//select the cloest one
					int selectone = -1;
					double minfdis = 0;
					for (int sr = 0; sr < indexInEachClst[j].size(); ++sr) {
						int lastIndex = indexInEachClst[j][sr];
						double normlfit = (InfoMemory::solInLastEnvi[lastIndex].orignaly[0] - lminfit) / (lmaxfit - lminfit);
						double xdis = InfoMemory::solInLastEnvi[lastIndex].xdistance(InfoMemory::solInCurEnvi[curIndex]);
						double fdis = fabs(normcfit - normlfit);
						if (selectone == -1 || fdis < minfdis) {
							minfdis = fdis;
							selectone = sr;
						}
					}
					r = selectone;
				}
				lastoneindex = r;
				if (r < 0 || r >= indexInEachClst[j].size()) { cout << r << " " << indexInEachClst[j].size() << endl; assert(false); }
				if (cIndexInEachClst[j][l] < 0 || cIndexInEachClst[j][l] >= InfoMemory::solInCurEnvi.size()) {
					cout << cIndexInEachClst[j][l] << " "; assert(false);
				}
				if (indexInEachClst[j][r] < 0 || indexInEachClst[j][r] >= InfoMemory::solInLastEnvi.size()) {
					cout << indexInEachClst[j][r] << " "; assert(false);
				}
				mapIndex[mindex] = indexInEachClst[j][r];
				mapCIndex[mindex] = cIndexInEachClst[j][l];
				lastclusterId[mindex] = j;
				curclusterId[mindex] = j;
				mindex++;
				//cout<< j <<" "<< l << "<-" << r <<" " <<mindex <<" " << px.size() << " " << py.size() << endl;
				TrainingPoint p;
				VectorXd px(DynPara::dimNumAllEnvi[i - 1]);
				VectorXd py(DynPara::dimNum);
				for (int k = 0; k < px.size(); ++k) px[k] = InfoMemory::solInLastEnvi[indexInEachClst[j][r]].orignalx[k];
				for (int k = 0; k < py.size(); ++k) py[k] = InfoMemory::solInCurEnvi[cIndexInEachClst[j][l]].orignalx[k];
				p.setData(px, py, DynPara::dimNumAllEnvi[i - 1], DynPara::dimNum);
				InfoMemory::solMapBetEnvi.push_back(p);
			}
		}
		if (InfoMemory::solMapBetEnvi.size() == 0) { cout << "no solution pairs for NN training ......\n"; assert(false); }
		//output the solutions map in each generation into file
#ifndef PARALLEL
		int num = InfoMemory::solMapBetEnvi.size();
#ifndef UNIX
		outputMapSolutionsToFile(i, 0, mapIndex, mapCIndex, num, lastclusterId,curclusterId, lastminfit,lastmaxfit,curminfit, curmaxfit);
#endif
		//outputMapSolutionsToFile(i, g, mapIndex, mapCIndex, num);
#endif
		normilizeDataXWithGivenBound(InfoMemory::solMapBetEnvi, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
		normilizeDataYWithGivenBound(InfoMemory::solMapBetEnvi, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);

		bool lastToNewInAllDim = false;// true; //train the variable between two environment in all dimensions
		if (lastToNewInAllDim) {
			//	cout << "train...: " << number_dimension[i - 1] << "->" << DynPara::dimNum << "\t" << InfoMemory::solMapBetEnvi.size() << endl;
			//InfoMemory::curVarNN.trainByLevenbergMarquardtBP(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), number_dimension[i - 1],
			//	DynPara::dimNum,mu,factor,maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
			//	InfoMemory::curVarNN.initilizeParaNgWi90();
			char BPname[10];
			char simufile[100];
			strcpy(BPname, "LMBP");
			double mu = 0.01; //small value in the begining
			double factor = 10;
			//	int  maxEpochs = 100;
			//	InfoMemory::curVarNN.initilizeParaNgWi90();
			//cout << "curVarNN train: ";
			InfoMemory::curVarNN.trainByLMBPCorssValidation(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), DynPara::dimNum, DynPara::dimNum, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
			sprintf(simufile, "trainset_varNN_%s_%d.out", BPname, i);
			InfoMemory::curVarNN.outNormilizePredictFile(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), DynPara::dimNum, DynPara::dimNum, InfoMemory::xmax, InfoMemory::xmin, simufile);

			//
			/*InfoMemory::curVarNN.initilizeParaNgWi90();
			strcpy(BPname, "LMBP-BR");
			InfoMemory::curVarNN.trainByBayesianRegularization(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), number_dimension[i - 1], DynPara::dimNum, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
			sprintf(simufile, "trainset_varNN_%s_%d.out", BPname, i);
			InfoMemory::curVarNN.outNormilizePredictFile(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), number_dimension[i - 1], DynPara::dimNum, InfoMemory::xmax, InfoMemory::xmin, simufile);
			*/

			/*	double learnrate = 0.5;
			double stepsize = 0.1; //0.075
			int resetIte = 15; //the number of iteration to reset
			double intervalAcc = 0.01;
			//	InfoMemory::curVarNN.initilizeParaNgWi90();
			//InfoMemory::curVarNN.trainByConjugateGradientBP(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), DynPara::number_dimension[i - 1], DynPara::dimNum, learnrate, stepsize, maxEpochs, resetIte, intervalAcc, InfoMemory::xmax, InfoMemory::xmin);
			*/
			//construct test data set to validate the simulation abality of the cur var NN
			vector<TrainingPoint> testData(10);
			//sample
			//int rnum = InfoMemory::solInCurEnvi.size();
			int index;
			for (int j = 0; j < testData.size(); ++j) {
				VectorXd x(DynPara::dimNum);
				VectorXd y(DynPara::dimNum);
				if (j < testData.size() / 2) {
					int rclst = random_int(0, clstnum - 1);
					while (cIndexInEachClst[rclst].size() == 0) rclst = random_int(0, clstnum - 1);
					int ty = random_int(cIndexInEachClst[rclst].size() / 2, cIndexInEachClst[rclst].size() - 1);//cIndexInEachClst[rclst].size() + random_int(0,10);
					if (ty >= indexInEachClst[rclst].size()) ty = indexInEachClst[rclst].size() - 1;
					index = indexInEachClst[rclst][ty];//InfoMemory::solInCurEnvi.size() + j;
					if (index >= InfoMemory::solInLastEnvi.size()) {
						index = random_int(0, InfoMemory::solInLastEnvi.size() - 1);
					}
					for (int k = 0; k < x.size(); ++k) {
						x[k] = InfoMemory::solInLastEnvi[index].orignalx[k];
					}
					///////				
				}
				else {
					index = random_int(0, InfoMemory::solInLastEnvi.size() - 1);
					for (int k = 0; k < x.size(); ++k) {
						x[k] = InfoMemory::solInLastEnvi[index].orignalx[k];
					}
				}
				for (int k = 0; k < y.size(); ++k) {
					y[k] = 0;
				}
				testData[j].setData(x, y, x.size(), y.size());
			}
			normilizeDataXWithGivenBound(testData, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
			int betternum = 0;
			cout << "varNN in each generation test:\n";
			for (int j = 0; j < testData.size(); ++j) {
				InfoMemory::curVarNN.predict(testData[j].x, testData[j].y, DynPara::dimNum, DynPara::dimNum);
				testData[j].restorePYMaxMin(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
				vector<double> x(DynPara::dimNum);
				vector<double> y(DynPara::dimNum);
				for (int k = 0; k < x.size(); ++k) x[k] = testData[j].orignalx[k];
				for (int k = 0; k < y.size(); ++k) {
					y[k] = testData[j].y[k];
					if (y[k] < DynPara::lowBound[k]) y[k] = DynPara::lowBound[k];
					if (y[k] > DynPara::upperBound[k]) y[k] = DynPara::upperBound[k];
				}
				double xfit = eval_movpeaks(&x[0], false);
				double yfit = eval_movpeaks(&y[0], false);
				cout << j << " ";
				if (fitBetter(yfit, xfit)) cout << "+ "; else cout << "  ";
				//	for (int k = 0; k < x.size(); ++k) cout << x[k] << ","; cout << " ";
				cout << xfit << " -> " << yfit;// << "\n";
				if (fitBetter(yfit, xfit)) betternum++;
				double length = 0;
				for (int k = 0; k < y.size(); ++k) {
					length += (y[k] - x[k])*(y[k] - x[k]);
				}
				length = sqrt(length);
				cout << " d(x,y) " << length << " ";// << endl;
				VectorXd py(1);
				InfoMemory::curFunNN.predict(testData[j].x, py, DynPara::dimNum, 1);
				py[0] = (py[0] + 1) / 2 * (InfoMemory::ymax[0] - InfoMemory::ymin[0]) + InfoMemory::ymin[0];
				cout << "* " << py[0];// << "";
				InfoMemory::curFunNN.predict(testData[j].y, py, DynPara::dimNum, 1);
				py[0] = (py[0] + 1) / 2 * (InfoMemory::ymax[0] - InfoMemory::ymin[0]) + InfoMemory::ymin[0];
				cout << " (" << py[0] << ") " << fabs(py[0] - yfit) << "\n";
			}
			cout << "err: " << (double)betternum / testData.size();

		}
		else {
			//train the variable for the new environment one dimension by one NN
			vector<TrainingPoint> dataSet(InfoMemory::solMapBetEnvi.size());
		//	vector<vector<double> > predictOut(InfoMemory::solMapBetEnvi.size());
		//	for (int j = 0; j < predictOut.size(); ++j) predictOut[j].resize(DynPara::dimNum);
			int hiddenNeuNum = 1;
			int b = InfoMemory::solMapBetEnvi.size() / ((DynPara::dimNum + 1) * 10);
			if (b < hiddenNeuNum) b = hiddenNeuNum;
			for (int k = 0; k < DynPara::dimNum; ++k) {
				if (InfoMemory::varNNEachDim[k].numNeuroEachLayer[1] != b) {
					InfoMemory::varNNEachDim[k].numNeuroEachLayer[1] = b;
					InfoMemory::varNNEachDim[k].initilizeParaNgWi90();
				}
			}
			VectorXd y(1);
			for (int k = 0; k < DynPara::dimNum; ++k) {
			//	cout << k << "\t";
				//construct the train data				
				for (int j = 0; j < dataSet.size(); ++j) {
					y[0] = InfoMemory::solMapBetEnvi[j].orignaly[k];
					dataSet[j].setData(InfoMemory::solMapBetEnvi[j].orignalx, y, DynPara::dimNum, 1);
				}
				normilizeDataXWithGivenBound(dataSet, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
				normilizeDataYWithGivenBound(dataSet, InfoMemory::xmax, InfoMemory::xmin, 1);
				//train
				char BPname[10];
				char simufile[100];
				strcpy(BPname, "LMBP");
				double mu = 0.01; //small value in the begining
				double factor = 10;
				InfoMemory::varNNEachDim[k].trainByLevenbergMarquardtBP(dataSet, dataSet.size(), DynPara::dimNum, 1, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
				
			}
			//calculate the training errors
		}
	}
}


void initConOptNNStruct(const int i, const int samplenum, bool useConOptVarNN){

	//bool useConOptVarNN = true;
	if (useNN && useConOptVarNN){
		//7x + 1 : 
		//x: 1-3
		int prepeaknum = samplenum / i;
		int maxneunum = (samplenum - 1) / 7;
		int sinputnum = DynPara::dimNum;
		int soutnum = 1;
		int scalingFactor = 5;  //5-10
		int neunuma = (double)((double)samplenum / (sinputnum + soutnum)) / scalingFactor;
		int neunumb = (double)((double)samplenum / (sinputnum + soutnum)) / 10;
		int neunum = 1;
		if (neunumb < 1) neunum = 1;
		else neunum = neunumb;
	//	if (neunum > 3) neunum = 3;
		//	else neunum = neunuma;


		if (i >= 1 && (i == 1 || InfoMemory::conOptVarNNEachDim.size() != DynPara::dimNum)){
			//each dimension model
			InfoMemory::conOptVarNNEachDim.resize(DynPara::dimNum);
			int numlayer = 3;
			vector<int> numNeuEachLayer(numlayer);
			numNeuEachLayer[0] = DynPara::dimNum;
			numNeuEachLayer[1] = neunum;
			numNeuEachLayer[2] = 1;
			if (InfoMemory::conOptVarNNEachDim[0].numNeuroEachLayer[1] != neunum){
				for (int j = 0; j < InfoMemory::conOptVarNNEachDim.size(); ++j){
					InfoMemory::conOptVarNNEachDim[j].setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
					InfoMemory::conOptVarNNEachDim[j].initilizeParaNgWi90();
				}
			}
			int sinputnum = DynPara::dimNum;
			int soutnum = DynPara::dimNum;
			int neunuma = (double)((double)samplenum / (sinputnum + soutnum)) / scalingFactor;
			int neunumb = (double)((double)samplenum / (sinputnum + soutnum)) / 10;
			int neunum = 1;
			if (neunumb < 1) neunum = 1;
			else neunum = neunumb;
			if (neunum > 3) neunum = 3;

			//all dimension model
			numNeuEachLayer[0] = DynPara::dimNum;
			numNeuEachLayer[1] = neunum;
			numNeuEachLayer[2] = 1;
			InfoMemory::curConOptVarNN.setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
			InfoMemory::curConOptVarNN.initilizeParaNgWi90();
		}
	}

}

void trainConOptVarNN(int i, bool useConOptVarNN, bool accumulateOptMap, bool lastToNewAllDim, int  maxEpochs){
	if (i == 0 || !useConOptVarNN) return;
	// true;  //all the y are trained in only one nn
	double varErr = -1;

	if (i == 1){
		InfoMemory::solMapBetEnvi.clear();
		InfoMemory::nonCorMapPair.clear();
	}
	//accumulateOptMap = false;
	//function variation for NN structure
	if (i >= 1 && useConOptVarNN && useNN){
		//construct the solution map between two environments to train the varNN : the converged solutions obtained in two environments			

		vector<double> lastXFit;
		vector<double> curXFit;
		vector<int> lastpeak;
		vector<int> curpeak;
		int index = 0;
		int oldsize = 0;// InfoMemory::solMapBetEnvi.size();
		if (accumulateOptMap){
			//delete the nonCorMap pairs in the last environment
			int j = i - DynPara::EW;
			if (j >= 1){
				for (int k = 0; k < DynPara::pairNumInEachEnvi[j]; ++k){
					InfoMemory::solMapBetEnvi.erase(InfoMemory::solMapBetEnvi.begin());
				}
			}
		//	InfoMemory::nonCorMapPair.clear();
			oldsize = InfoMemory::solMapBetEnvi.size();
			int tcount = 0;
			for (int k = i - DynPara::EW + 1; k <= i - 1; ++k){
				if (k < 1)continue;
				tcount += DynPara::pairNumInEachEnvi[k];
			}
			if (tcount != oldsize){ assert(false); }
		}
		else{
			oldsize = 0;
			InfoMemory::solMapBetEnvi.clear();
		}
		int num = max(InfoMemory::conOptEachEnvi[i - 1].size(), InfoMemory::conOptEachEnvi[i].size());
		int comnum = min(InfoMemory::conOptEachEnvi[i - 1].size(), InfoMemory::conOptEachEnvi[i].size());
		if (onlyuseclosepair) num = comnum;
		DynPara::pairNumInEachEnvi[i] = num;
		InfoMemory::solMapBetEnvi.resize(oldsize + num);

		if (accumulateOptMap){
			//reset the size of the NN
		//	initConOptNNStruct(i, oldsize + num, useConOptVarNN);
		}
		//cout << InfoMemory::solMapBetEnvi.size() << "\t";

		VectorXd px(DynPara::dimNum);
		VectorXd py(DynPara::dimNum);
		MatrixXd dis(InfoMemory::conOptEachEnvi[i - 1].size(), InfoMemory::conOptEachEnvi[i].size());

		for (int j = 0; j < dis.rows(); ++j){
			for (int k = 0; k < dis.cols(); ++k){
				dis(j, k) = 0;
				double d = 0;
				for (int l = 0; l < DynPara::dimNum; ++l){
					d += (InfoMemory::conOptEachEnvi[i - 1][j].orignalx[l] - InfoMemory::conOptEachEnvi[i][k].orignalx[l])*(InfoMemory::conOptEachEnvi[i - 1][j].orignalx[l] - InfoMemory::conOptEachEnvi[i][k].orignalx[l]);
				}
				d = sqrt(d);
				dis(j, k) = d;
			}
		}
		int sindex = 0;
		//select the cloest pair ./
		vector<int> lastVisit(InfoMemory::conOptEachEnvi[i - 1].size());
		vector<int> curVisit(InfoMemory::conOptEachEnvi[i].size());
		for (int j = 0; j < lastVisit.size(); ++j) lastVisit[j] = 0;
		for (int j = 0; j < curVisit.size(); ++j) curVisit[j] = 0;
		for (int j = 0; j < comnum; ++j){
			int minlindex = -1;
			int minrindex = -1;
			for (int l = 0; l < dis.rows(); ++l){
				for (int r = 0; r < dis.cols(); ++r){
					if (lastVisit[l] || curVisit[r]) continue;
					if (minlindex == -1 || dis(l, r) < dis(minlindex, minrindex)){
						minlindex = l;
						minrindex = r;
					}
				}
			}
			minlindex = j; 
			minrindex = j;
			lastVisit[minlindex] = 1;
			curVisit[minrindex] = 1;
			for (int k = 0; k < px.size(); ++k) px[k] = InfoMemory::conOptEachEnvi[i - 1][minlindex].orignalx[k];
			for (int k = 0; k < py.size(); ++k){
				py[k] = InfoMemory::conOptEachEnvi[i][minrindex].orignalx[k];
			}
			lastXFit.push_back(InfoMemory::conOptEachEnvi[i - 1][minlindex].y[0]);
			curXFit.push_back(InfoMemory::conOptEachEnvi[i][minrindex].y[0]);
			lastpeak.push_back(InfoMemory::conOptEachEnvi[i - 1][minlindex].x[0]);// InfoMemory::conOptEachEnvi[i - 1][minlindex].refervSize);
			curpeak.push_back(InfoMemory::conOptEachEnvi[i][minrindex].x[0]);// InfoMemory::conOptEachEnvi[i][minrindex].refervSize);
		//	cout << InfoMemory::conOptEachEnvi[i][minrindex].x[0] << "," << InfoMemory::conOptEachEnvi[i][minrindex].y[0] << "\t";
		//	cout << InfoMemory::conOptEachEnvi[i - 1][minlindex].x[0] << "," << InfoMemory::conOptEachEnvi[i - 1][minlindex].y[0] << "\t";
			sindex = j + oldsize;
			InfoMemory::solMapBetEnvi[sindex].setData(px, py, DynPara::dimNum, DynPara::dimNum);
		}
		//
		index = oldsize + comnum;
		if (!onlyuseclosepair){
			if (InfoMemory::conOptEachEnvi[i].size() > InfoMemory::conOptEachEnvi[i - 1].size()){
				//
				for (int j = 0; j < curVisit.size(); ++j){
					if (curVisit[j])continue;
					int cloestIndex = -1;
					double mindis = 0;
					for (int k = 0; k < InfoMemory::solInLastEnvi.size(); ++k){
						double d = 0;
						for (int index = 0; index < DynPara::dimNum; ++index){
							d += (InfoMemory::solInLastEnvi[k].orignalx[index] - InfoMemory::conOptEachEnvi[i][j].orignalx[index])*
								(InfoMemory::solInLastEnvi[k].orignalx[index] - InfoMemory::conOptEachEnvi[i][j].orignalx[index]);
						}
						d = sqrt(d);
						if (cloestIndex == -1 || d < mindis && d > 1e-2){
							cloestIndex = k;
							mindis = d;
						}
					}
					for (int k = 0; k < px.size(); ++k) px[k] = InfoMemory::solInLastEnvi[cloestIndex].orignalx[k];
					for (int k = 0; k < py.size(); ++k){
						py[k] = InfoMemory::conOptEachEnvi[i][j].orignalx[k];
					}
					InfoMemory::solMapBetEnvi[index].setData(px, py, DynPara::dimNum, DynPara::dimNum);
					InfoMemory::nonCorMapPair.push_back(index);
					index++;
					lastXFit.push_back(InfoMemory::solInLastEnvi[cloestIndex].orignaly[0]);
					lastpeak.push_back(-1);
					curpeak.push_back(InfoMemory::conOptEachEnvi[i][j].x[0]);// conOPtEachEnvi[i].optima[j].refervSize);
					curXFit.push_back(InfoMemory::conOptEachEnvi[i][j].y[0]);

				}
				if (index != oldsize + num){ cout << index << " " << num << " @ map solution construction in SPSO()" << endl; assert(false); }
			}
			else{
				for (int j = 0; j < lastVisit.size(); ++j){
					if (lastVisit[j])continue;
					int cloestIndex = -1;
					double mindis = 0;
					for (int k = 0; k < InfoMemory::solInCurEnvi.size(); ++k){
						double d = 0;
						for (int index = 0; index < DynPara::dimNum; ++index){
							d += (InfoMemory::solInCurEnvi[k].orignalx[index] - InfoMemory::conOptEachEnvi[i - 1][j].orignalx[index])*
								(InfoMemory::solInCurEnvi[k].orignalx[index] - InfoMemory::conOptEachEnvi[i - 1][j].orignalx[index]);
						}
						d = sqrt(d);
						if (cloestIndex == -1 || d < mindis && d > 1e-2){
							cloestIndex = k;
							mindis = d;
						}
					}
					for (int k = 0; k < px.size(); ++k)  px[k] = InfoMemory::conOptEachEnvi[i - 1][j].orignalx[k];// px[k] = InfoMemory::solInLastEnvi[cloestIndex].orignalx[k];
					for (int k = 0; k < py.size(); ++k){
						py[k] = InfoMemory::solInCurEnvi[cloestIndex].orignalx[k];// conOPtEachEnvi[i].optima[j].location.x[k];
					}
					InfoMemory::solMapBetEnvi[index].setData(px, py, DynPara::dimNum, DynPara::dimNum);
					InfoMemory::nonCorMapPair.push_back(index);
					index++;
					lastXFit.push_back(InfoMemory::conOptEachEnvi[i - 1][j].y[0]);
					lastpeak.push_back(InfoMemory::conOptEachEnvi[i - 1][j].x[0]);// conOPtEachEnvi[i - 1].optima[j].refervSize);
					curXFit.push_back(InfoMemory::solInCurEnvi[cloestIndex].orignaly[0]);
					curpeak.push_back(-1);
				}
				if (index != oldsize + num){ cout << index << " " << num << " @ map solution construction in trainConOptVarNN(i)" << endl; assert(false); }
			}
		}

#ifndef PARALLEL
	//	outputConOptMapToFile(i, num, lastXFit, curXFit, lastpeak, curpeak);
#endif
		normilizeDataXWithGivenBound(InfoMemory::solMapBetEnvi, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
		normilizeDataYWithGivenBound(InfoMemory::solMapBetEnvi, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);

		if (lastToNewAllDim){
			char BPname[10];
			char simufile[100];
			/*	strcpy(BPname, "LMBP");
			InfoMemory::curVarNN.initilizeParaNgWi90();
			InfoMemory::curVarNN.trainByLMBPCorssValidation(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), number_dimension[i - 1], DynPara::dimNum, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
			sprintf(simufile, "trainset_varNN_%s_%d.out", BPname, i);
			InfoMemory::curVarNN.outNormilizePredictFile(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), number_dimension[i - 1], DynPara::dimNum, InfoMemory::xmax, InfoMemory::xmin, simufile);

			InfoMemory::curVarNN.initilizeParaNgWi90();
			strcpy(BPname, "LMBP-BR");
			InfoMemory::curVarNN.trainByBayesianRegularization(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), number_dimension[i - 1], DynPara::dimNum, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
			sprintf(simufile, "trainset_varNN_%s_%d.out", BPname, i);
			InfoMemory::curVarNN.outNormilizePredictFile(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), number_dimension[i - 1], DynPara::dimNum, InfoMemory::xmax, InfoMemory::xmin, simufile);
			*/
			//InfoMemory::curConOptVarNN.initilizeParaNgWi90();
			strcpy(BPname, "CGBP");
			double learnrate = 0.5;
			double stepsize = 0.1; //0.075
			int resetIte = 5; //the number of iteration to reset
			double intervalAcc = 0.01;
			InfoMemory::curConOptVarNN.trainByConjugateGradientBP(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), DynPara::dimNum,
				DynPara::dimNum, learnrate, stepsize, maxEpochs, resetIte, intervalAcc, InfoMemory::xmax, InfoMemory::xmin);
			//	sprintf(simufile, "trainset_varNN_%s_%d.out", BPname, i);
			//	cout << "varNN trainset predict output.\n";
			//	InfoMemory::curVarNN.outNormilizePredictFile(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), DynPara::number_dimension[i - 1], DynPara::dimNum, InfoMemory::xmax, InfoMemory::xmin, simufile);
			//	cout << "varNN train end.....\n";
		}
		else{
			//	cout << "convarNN each dimension ....\n";
			if (DynPara::adaptHiddenNeuro) initConOptNNStruct(i, InfoMemory::solMapBetEnvi.size(), useConOptVarNN);
			for (int k = 0; k < DynPara::dimNum; ++k){
				//construct the train data
				vector<TrainingPoint> dataSet(InfoMemory::solMapBetEnvi.size());
				VectorXd y(1);
				for (int j = 0; j < dataSet.size(); ++j){
					y[0] = InfoMemory::solMapBetEnvi[j].orignaly[k];
					dataSet[j].setData(InfoMemory::solMapBetEnvi[j].orignalx, y, DynPara::dimNum, 1);
				}
				normilizeDataXWithGivenBound(dataSet, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
				normilizeDataYWithGivenBound(dataSet, InfoMemory::xmax, InfoMemory::xmin, 1);
				//train
				char BPname[10];
				char simufile[100];
				strcpy(BPname, "LMBP");
				double mu = 0.01; //small value in the begining
				double factor = 10;
				InfoMemory::conOptVarNNEachDim[k].trainByLevenbergMarquardtBP(dataSet, dataSet.size(), DynPara::dimNum, 1, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
				//varNNEachDim[k].trainByLMBPCorssValidation(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), DynPara::number_dimension[i - 1], DynPara::dimNum, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
				//	sprintf(simufile, "trainset_varNN_%s_%d.out", BPname, i);
				//	InfoMemory::varNNEachDim[k].outNormilizePredictFile(dataSet, dataSet.size(), DynPara::number_dimension[i - 1], 1, InfoMemory::xmax, InfoMemory::xmin, simufile);
			}
			//	getchar();
#ifdef DEBUG
			/*
			//construct test data set to validate the simulation abality of the cur var NN
			cout << "evaluate varNN by x(solutions in last environment):\n";
			vector<TrainingPoint> testData(10);
			//sample
			int index;
			for (int j = 0; j < testData.size(); ++j){
			VectorXd x(DynPara::number_dimension[i - 1]);
			VectorXd y(DynPara::dimNum);
			index = random_int(0, InfoMemory::solInLastEnvi.size() - 1);
			for (int k = 0; k < x.size(); ++k){
			x[k] = InfoMemory::solInLastEnvi[index].orignalx[k];
			}
			for (int k = 0; k < y.size(); ++k){
			y[k] = 0;
			}
			testData[j].setData(x, y, x.size(), y.size());
			}
			normilizeDataXWithGivenBound(testData, InfoMemory::xmax, InfoMemory::xmin, DynPara::number_dimension[i - 1]);
			//
			VectorXd py(1);
			double err = 0;
			for (int j = 0; j < testData.size(); ++j){
			for (int k = 0; k < DynPara::dimNum; ++k){
			InfoMemory::conOptVarNNEachDim[k].predict(testData[j].x, py, DynPara::number_dimension[i - 1], 1);
			testData[j].y[k] = py[0];
			}
			testData[j].restorePYMaxMin(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);

			vector<double> x(DynPara::dimNum);
			vector<double> y(DynPara::dimNum);
			for (int k = 0; k < x.size(); ++k) x[k] = testData[j].orignalx[k];
			for (int k = 0; k < y.size(); ++k){
			y[k] = testData[j].y[k];
			if (y[k] < DynPara::boundary.lower) y[k] = DynPara::boundary.lower;
			if (y[k] > DynPara::boundary.upper) y[k] = DynPara::boundary.upper;
			}
			double xfit = eval_movpeaks(&x[0], false);
			double yfit = eval_movpeaks(&y[0], false);
			cout << j << " ";
			if (fitBetter(yfit, xfit)) cout << "+ "; else cout << "  ";
			cout << xfit << "->" << yfit << endl;
			}
			*/
#endif

		}
	}
}

void addNewPointToCradleByVarNN(int i, int &fes, bool useConOptVarNN, bool useVarNN, bool lastToNewAllDim, vector<vector<double> > &newpointset, const int newpointnum) {
	if (useNN && ((i >= 2 && useConOptVarNN) || (i >= 1 && useVarNN))) {
		if (useConOptVarNN && useVarNN) assert(false);
		//use the converged point in last environment to move
		vector<int> sindex(newpointnum);
		newpointset.resize(newpointnum);
		for (int j = 0; j < sindex.size(); ++j) sindex[j] = j;
		//randomly select the converged optima to generate new points
		if (newpointnum < InfoMemory::conOptEachEnvi[i - 1].size()) {
			vector<int> visit(InfoMemory::conOptEachEnvi[i - 1].size());
			for (int j = 0; j < visit.size(); ++j) visit[j] = j;
			random_shuffle(visit.begin(), visit.end());
			for (int j = 0; j < sindex.size(); ++j) sindex[j] = visit[j];
		}
		for (int j = 0; j < newpointnum; ++j) {
			TrainingPoint newpoint;
			VectorXd nx(DynPara::dimNum);
			VectorXd ny(DynPara::dimNum);
			if (j < InfoMemory::conOptEachEnvi[i - 1].size()) {
				for (int k = 0; k < DynPara::dimNum; ++k) {
					nx[k] = InfoMemory::conOptEachEnvi[i - 1][sindex[j]].orignalx[k];
				}
			}
			else {
				int index = random_int(InfoMemory::solInLastEnvi.size() / 2, InfoMemory::solInLastEnvi.size());
				for (int k = 0; k < DynPara::dimNum; ++k) {
					nx[k] = InfoMemory::solInLastEnvi[index].orignalx[k];
				}
			}
			newpoint.setData(nx, ny, DynPara::dimNum, DynPara::dimNum);
			newpoint.normilizeXMinMax(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
			if (lastToNewAllDim) {
				if (useConOptVarNN) InfoMemory::curConOptVarNN.predict(newpoint.x, newpoint.y, DynPara::dimNum, DynPara::dimNum);
				if (useVarNN) InfoMemory::curVarNN.predict(newpoint.x, newpoint.y, DynPara::dimNum, DynPara::dimNum);
				newpoint.restorePYMaxMin(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
			}
			else {
				VectorXd py(1);
				for (int k = 0; k < DynPara::dimNum; ++k) {
					if (useConOptVarNN) InfoMemory::conOptVarNNEachDim[k].predict(newpoint.x, py, DynPara::dimNum, 1);
					if (useVarNN) InfoMemory::varNNEachDim[k].predict(newpoint.x, py, DynPara::dimNum, 1);
					newpoint.y[k] = py[0];
				}
				newpoint.restorePYMaxMin(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
			}
			///add the new point into cradle
			vector<double> newx(DynPara::dimNum);
			for (int k = 0; k < DynPara::dimNum; ++k) {
				newx[k] = newpoint.y[k];
				if (newx[k] < DynPara::lowBound[k]) newx[k] = DynPara::lowBound[k];
				if (newx[k] > DynPara::upperBound[k]) newx[k] = DynPara::upperBound[k];
			}
			newpointset[j].resize(DynPara::dimNum);
			for (int k = 0; k < DynPara::dimNum; ++k) newpointset[j][k] = newx[k];

#ifdef DEBUG
			///ox nx
			vector<double> ox(DynPara::dimNum);
			for (int k = 0; k < ox.size(); ++k) {
				ox[k] = nx[k];
			}
			double fox = eval_movpeaks(&nx[0], false);
			double fnewx = eval_movpeaks(&newx[0], false);
			//cout <<j <<"\t" <<  fox << "\t" << "-->\t" << fnewx << endl;

			for (int k = 0; k < DynPara::dimNum; ++k) {
				cout << ox[k] << ",";
			}
			cout << "\t(" << fox << ")\t--->\t";
			for (int k = 0; k < DynPara::dimNum; ++k) {
				cout << newpointset[j][k] << ",";
			}
			cout << "\t(" << fnewx << ")\n";

			cout << j << "\t" << fox << " --->\t" << cradle.pop[cradle.popsize - 1].pself.fitness;
			if (fitBetter(cradle.pop[cradle.popsize - 1].pself.fitness, fox))cout << "\t+";
			cout << endl;
#endif
		}
	}
}

void addNewPointToMpSingleByVarNN(int i, bool useConOptVarNN, bool lastToNewAllDim, vector<vector<double> > &mp_single, const int newpointnum){
	if (i >= 2 && useConOptVarNN && useNN){
		//use the converged point in last environment to move
		vector<int> sindex(newpointnum);
		for (int j = 0; j < sindex.size(); ++j) sindex[j] = j;
		//randomly select the converged optima to generate new points
		if (newpointnum < InfoMemory::conOptEachEnvi[i - 1].size()){
			vector<int> visit(InfoMemory::conOptEachEnvi[i - 1].size());
			for (int j = 0; j < visit.size(); ++j) visit[j] = j;
			random_shuffle(visit.begin(), visit.end());
			for (int j = 0; j < sindex.size(); ++j) sindex[j] = visit[j];
		}
		for (int j = 0; j < newpointnum; ++j){
			TrainingPoint newpoint;
			VectorXd nx(DynPara::dimNum);
			VectorXd ny(DynPara::dimNum);
			if (j < InfoMemory::conOptEachEnvi[i - 1].size()){
				for (int k = 0; k < DynPara::dimNum; ++k){
					nx[k] = InfoMemory::conOptEachEnvi[i - 1][sindex[j]].orignalx[k];
				}
			}
			else{
				int index = random_int(InfoMemory::solInLastEnvi.size() / 2, InfoMemory::solInLastEnvi.size());
				for (int k = 0; k < DynPara::dimNum; ++k){
					nx[k] = InfoMemory::solInLastEnvi[index].orignalx[k];
				}
			}
			newpoint.setData(nx, ny, DynPara::dimNum, DynPara::dimNum);
			newpoint.normilizeXMinMax(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
			if (lastToNewAllDim){
				InfoMemory::curConOptVarNN.predict(newpoint.x, newpoint.y, DynPara::dimNum, DynPara::dimNum);
				newpoint.restorePYMaxMin(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
			}
			else{
				VectorXd py(1);
				for (int k = 0; k < DynPara::dimNum; ++k){
					InfoMemory::conOptVarNNEachDim[k].predict(newpoint.x, py, DynPara::dimNum, 1);
					newpoint.y[k] = py[0];
				}
				newpoint.restorePYMaxMin(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
			}
			///add the new point into cradle
			vector<double> newx(DynPara::dimNum);
			for (int k = 0; k < DynPara::dimNum; ++k){
				newx[k] = newpoint.y[k];
				if (newx[k] < DynPara::lowBound[k]) newx[k] = DynPara::lowBound[k];
				if (newx[k] > DynPara::upperBound[k]) newx[k] = DynPara::upperBound[k];
			}
		/*	Particle p;// (newx);
			for (int k = 0; k < DynPara::dimNum; ++k){
				p.m_pself.getGene<double>(k) = newx[k];
			}
			p.m_pself.evaluate(true);
			p.m_pbest = p.m_pself;*/
			mp_single.resize(mp_single.size() + 1);
			mp_single[mp_single.size() - 1].resize(DynPara::dimNum);
			int lastone = mp_single.size() - 1;
			for (int k = 0; k < DynPara::dimNum; ++k) mp_single[lastone][k] = newx[k];
		//	mp_single.push_back(p);
			///ox nx
			vector<double> ox(DynPara::dimNum);
			for (int k = 0; k < ox.size(); ++k){
				ox[k] = nx[k];
			}
			double fox = eval_movpeaks(&nx[0], false);
#ifdef DEBUG
			cout << j << "\t" << fox << " --->\t" << cradle.pop[cradle.popsize - 1].pself.fitness;
			if (fitBetter(cradle.pop[cradle.popsize - 1].pself.fitness, fox))cout << "\t+";
			cout << endl;
#endif
		}
	}
}

void trainCurFunNN(int i, bool useFunNN, int maxEpochs){
	//sort(InfoMemory::solInCurEnvi.begin(), InfoMemory::solInCurEnvi.end());
	//sort(InfoMemory::solInLastEnvi.begin(), InfoMemory::solInLastEnvi.end());
	//	bool useFunNN = true;
	//int maxEpochs = 10;
	double mu = 0.01; //small value in the begining
	double factor = 10;
	double funErr;
	vector<TrainingPoint> testData(10);
	if (useFunNN && useNN){
		//function NN structure and parameter initialization
		//train
		if (i == 0) updateYMaxMin();
		normilizeDataXWithGivenBound(InfoMemory::solInCurEnvi, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
		normilizeDataY(InfoMemory::solInCurEnvi, 1, InfoMemory::ymax, InfoMemory::ymin);
		//cout << "curFun simulation: " << InfoMemory::solInCurEnvi.size() << endl;
		//cout << "ymax: ";
		//for (int j = 0; j < InfoMemory::ymax.size(); ++j) cout << InfoMemory::ymax[j] << "\t";
		//cout << "\n";
		InfoMemory::curFunNN.trainByLevenbergMarquardtBP(InfoMemory::solInCurEnvi, InfoMemory::solInCurEnvi.size(), DynPara::dimNum, 1, mu, factor, maxEpochs, InfoMemory::ymax, InfoMemory::ymin);
		//	InfoMemory::curFunNN.trainByLMBPCorssValidation(InfoMemory::solInCurEnvi, InfoMemory::solInCurEnvi.size(), DynPara::dimNum, 1, mu, factor, maxEpochs, InfoMemory::ymax, InfoMemory::ymin);

		/*
		//test
		//construct test data set to validate the simulation abality of the cur fun NN
		sampleTestDataSet(testData, DynPara::dimNum, 1, DynPara::boundary.lower, DynPara::boundary.upper);
		normilizeDataXWithGivenBound(testData, InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
		funErr = InfoMemory::curFunNN.predictDataSet(testData, testData.size(), DynPara::dimNum, 1, true, InfoMemory::ymax, InfoMemory::ymin);
		char simufile[100];
		sprintf(simufile, "sim_curFun_%d.out", i);
		InfoMemory::curFunNN.outputSimuFunToFile(testData, testData.size(), simufile);
		cout << funErr << endl;
		char parafile[100];
		sprintf(parafile, "NNPara_curFun_%d.out", i);
		fstream fp(parafile, ios::out);
		if (fp.fail()) { cout << "cannot open file " << parafile << endl; assert(false); }
		InfoMemory::curFunNN.outputNNParaToFile(fp);
		fp.close();
		*/
	}
}

void trainInverseFToXNN(int i, bool useInverseFToXModel, int maxEpochs){
	//construct data set
	vector<TrainingPoint> dataSet(InfoMemory::solInCurEnvi.size());
	VectorXd y(1);
	VectorXd x(1);
	for (int k = 0; k < DynPara::dimNum; ++k){
		//construct the train data	
		for (int j = 0; j < dataSet.size(); ++j){
			y[0] = InfoMemory::solInCurEnvi[j].orignaly[0]; // the fitness value
			x[0] = InfoMemory::solInCurEnvi[j].orignalx[k]; //the variable x
			dataSet[j].setData(y, x, 1, 1);
		}
		normilizeDataXWithGivenBound(dataSet, InfoMemory::ymax, InfoMemory::ymin, 1);
		normilizeDataYWithGivenBound(dataSet, InfoMemory::xmax, InfoMemory::xmin, 1);
		//train
		char BPname[10];
		char simufile[100];
		strcpy(BPname, "LMBP");
		double mu = 0.01; //small value in the begining
		double factor = 10;
		//	int  maxEpochs = 20;
		//	InfoMemory::curVarNN.initilizeParaNgWi90();
		//	cout << "curVarNN train: ";
		InfoMemory::inverseFunToXEachDim[k].trainByLevenbergMarquardtBP(dataSet, dataSet.size(), DynPara::dimNum, 1, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
		//varNNEachDim[k].trainByLMBPCorssValidation(InfoMemory::solMapBetEnvi, InfoMemory::solMapBetEnvi.size(), DynPara::number_dimension[i - 1], DynPara::dimNum, mu, factor, maxEpochs, InfoMemory::xmax, InfoMemory::xmin);
		//	sprintf(simufile, "trainset_varNN_%s_%d.out", BPname, i);
		//	InfoMemory::varNNEachDim[k].outNormilizePredictFile(dataSet, dataSet.size(), DynPara::number_dimension[i - 1], 1, InfoMemory::xmax, InfoMemory::xmin, simufile);
	}
}

void initialNNInfo(int i, bool useFunNN, bool useVarNN, bool useConOptVarNN, bool useInverseFToXModel){
	//bool useFunNN = true;
	if (useNN){
		if (i == 0 || DynPara::dimNum > InfoMemory::xmax.size()){
			InfoMemory::xmax.resize(DynPara::dimNum);
			InfoMemory::xmin.resize(DynPara::dimNum);
			for (int j = 0; j < DynPara::dimNum; ++j){
				InfoMemory::xmax[j] = DynPara::upperBound[j];
				InfoMemory::xmin[j] = DynPara::lowBound[j];
			}
		}
		if (i == 0){
			InfoMemory::ymax.resize(1);
			InfoMemory::ymin.resize(1);
		}
	}
	if (useNN && useFunNN){
		//if (i >= 1) updateYMaxMin();
		if (i == 0 || DynPara::dimNum != DynPara::dimNum)// InfoMemory::curFunNN = InfoMemory::lastFunNN;
		{
			if (DynPara::dimNum != DynPara::dimNum){
				int numlayer = 3;
				vector<int> numNeuEachLayer(numlayer);
				numNeuEachLayer[0] = DynPara::dimNum;
				numNeuEachLayer[1] = 15;
				numNeuEachLayer[2] = 1;
				InfoMemory::curFunNN.setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
			}
			InfoMemory::curFunNN.initilizeParaNgWi90();
		}
	}
	//bool useVarNN = true;
	if (useNN && useVarNN){
		if (i >= 1 && (i == 1 || InfoMemory::varNNEachDim.size() != DynPara::dimNum)){
			InfoMemory::varNNEachDim.resize(DynPara::dimNum);
			int numlayer = 3;
			vector<int> numNeuEachLayer(numlayer);
			numNeuEachLayer[0] = DynPara::dimNum;
			numNeuEachLayer[1] = 6;  //9, 10
			numNeuEachLayer[2] = 1;
			for (int j = 0; j < InfoMemory::varNNEachDim.size(); ++j){
				InfoMemory::varNNEachDim[j].setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
				InfoMemory::varNNEachDim[j].initilizeParaNgWi90();
			}
			//all dimension to all dimension
			//all dimension model
			numNeuEachLayer[0] = DynPara::dimNum;
			numNeuEachLayer[1] = 5;
			numNeuEachLayer[2] = 1;
			InfoMemory::curVarNN.setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
			InfoMemory::curVarNN.initilizeParaNgWi90();
		}
	}
	//bool useConOptVarNN = true;
	if (useNN && useConOptVarNN){
		if (i >= 1 && (i == 1 || InfoMemory::conOptVarNNEachDim.size() != DynPara::dimNum)){
			//each dimension model
			InfoMemory::conOptVarNNEachDim.resize(DynPara::dimNum);
			int numlayer = 3;
			vector<int> numNeuEachLayer(numlayer);
			numNeuEachLayer[0] = DynPara::dimNum;
			numNeuEachLayer[1] = DynPara::numHiddenNeuros;
			numNeuEachLayer[2] = 1;
			for (int j = 0; j < InfoMemory::conOptVarNNEachDim.size(); ++j){
				InfoMemory::conOptVarNNEachDim[j].setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
				InfoMemory::conOptVarNNEachDim[j].initilizeParaNgWi90();
			}
			//all dimension model
			numNeuEachLayer[0] = DynPara::dimNum;
			numNeuEachLayer[1] = 2;
			numNeuEachLayer[2] = 1;
			InfoMemory::curConOptVarNN.setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
			InfoMemory::curConOptVarNN.initilizeParaNgWi90();
		}
	}
	if (useNN && useInverseFToXModel){
		if (i == 0 || DynPara::dimNum != DynPara::dimNum){
			InfoMemory::inverseFunToXEachDim.resize(DynPara::dimNum);
			int numlayer = 3;
			vector<int> numNeuEachLayer(numlayer);
			numNeuEachLayer[0] = 1;
			numNeuEachLayer[1] = 5;
			numNeuEachLayer[2] = 1;
			for (int j = 0; j < InfoMemory::inverseFunToXEachDim.size(); ++j){
				InfoMemory::inverseFunToXEachDim[j].setStructure(numlayer, numNeuEachLayer, tansigfun, linearfun);
				InfoMemory::inverseFunToXEachDim[j].initilizeParaNgWi90();
			}
		}
	}
}


void recordCurNNToPast(int i){
	//the current memory is record in the last memory
	InfoMemory::solInLastEnvi = InfoMemory::solInCurEnvi;
	InfoMemory::solInCurEnvi.clear();
	InfoMemory::lastFunNN = InfoMemory::curFunNN;
	InfoMemory::lastVarNN = InfoMemory::curVarNN;
	InfoMemory::lastInverseFunToXEachDim = InfoMemory::inverseFunToXEachDim;
}

double evaluateByFunNN(int i, bool useFunNN, double *x){
	if (!useFunNN) { cout << "no NN for function simulation ...\n"; assert(false); }
	double ratio = 1;
	TrainingPoint p;
	VectorXd tx(DynPara::dimNum);
	for (int j = 0; j < DynPara::dimNum; ++j){
		tx[j] = x[j];
	}
	VectorXd py(1);
	p.setData(tx, py, DynPara::dimNum, 1);
	p.normilizeXMinMax(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
	InfoMemory::curFunNN.predict(p.x, p.y, DynPara::dimNum, 1);
	p.restorePYMaxMin(InfoMemory::ymax, InfoMemory::ymin, 1);
	double valueByCur = p.y[0];
	double valueByLast = 0;
	if (false && i > 0){
		InfoMemory::lastFunNN.predict(p.x, p.y, DynPara::dimNum, 1);
		p.restorePYMaxMin(InfoMemory::ymax, InfoMemory::ymin, 1);
		valueByLast = p.y[0];
		ratio = min((int)InfoMemory::solInCurEnvi.size(), 500) / 500.0;
	}

	ratio = 1; valueByLast = 0;

#ifdef DEBUG
	double realvalue = eval_movpeaks(x, false);
	double finalvalue = ratio*valueByCur + (1 - ratio)*valueByLast;
	cout << realvalue << "\t" << valueByCur << "\t" << valueByLast << "\t" << finalvalue << endl;
#endif

	return ratio*valueByCur + (1 - ratio)*valueByLast;
}



void updateFunAndVarNN(int i, int &fes, int ite, bool useFunNN, bool useVarNN, vector<vector<double> > &cradle, const int selectnum, const int maxEpochs, vector<int> &cbelongCluster, vector<vector<int>> &cIndexInEachClst, vector<vector<int>> indexInEachClst, const int clstnum){
	//int  maxEpochs = 10;// max((int)((double)InfoMemory::solInCurEnvi.size() / 5000 * 15), 5);
	if (i > 0 && (ite == 0 || ite == 5) && useNN && useFunNN){
		if (InfoMemory::solInCurEnvi.size() <10){
			cout << InfoMemory::solInCurEnvi.size() << endl;
			assert(false);
		}
		trainCurFunNN(i, useFunNN, maxEpochs);
	}
	if (i > 0 && (ite == 0 || ite == 5) && useNN && useVarNN){
		trainCurVarNN(i, useVarNN, maxEpochs, cbelongCluster, cIndexInEachClst, indexInEachClst, clstnum);
		//construct new points
		vector<Solution> newpoints(InfoMemory::solInLastEnvi.size() * 2);
		vector<double> px(DynPara::dimNum);
		for (int j = 0; j < InfoMemory::solInLastEnvi.size(); ++j){
			VectorXd perDim(1);
			InfoMemory::solInLastEnvi[j].normilizeXMinMax(InfoMemory::xmax, InfoMemory::xmin, DynPara::dimNum);
			for (int k = 0; k < DynPara::dimNum; ++k){
				InfoMemory::varNNEachDim[k].predict(InfoMemory::solInLastEnvi[j].x, perDim, DynPara::dimNum, 1);
				perDim[0] = (perDim[0] + 1)*(InfoMemory::xmax[k] - InfoMemory::xmin[k]) / 2 + InfoMemory::xmin[k];
				px[k] = perDim[0];
				if (px[k] < DynPara::lowBound[k]) px[k] = DynPara::lowBound[k];
				if (px[k] > DynPara::upperBound[k]) px[k] = DynPara::upperBound[k];
			}
			//evaluate the new solutions by fun NN
			double fpx = evaluateByFunNN(i, useFunNN, &px[0]);
			newpoints[j].setSolution(px, fpx);
		}
		//the orignalx
		int index;
		for (int j = InfoMemory::solInLastEnvi.size(); j < InfoMemory::solInLastEnvi.size() * 2; ++j){
			index = j - InfoMemory::solInLastEnvi.size();
			vector<double> ox(DynPara::dimNum);
			for (int k = 0; k < DynPara::dimNum; ++k){
				ox[k] = InfoMemory::solInLastEnvi[index].orignalx[k];
			}
			double fox = evaluateByFunNN(i, useFunNN, &ox[0]);
			newpoints[j].setSolution(ox, fox);
#ifdef DEBUG
			if (j % 100 == 0) getchar();
#endif
		}

#ifdef DEBUG
		int sum = 0;
		int consum = 0;
		double bestvalue = 0;
		double bestoxvalue = 0; int bestoxondex = 0;
		int bestnewx = 0;
		double DynPara_max = MovingPeak::DynPara_max;
		for (int j = 0; j < newpoints.size(); ++j){
			if (j >= InfoMemory::solInLastEnvi.size()) break;  //solutions in the environment
			double realfnewpoint = eval_movpeaks(&newpoints[j].x[0], false);
			double realfox = eval_movpeaks(&InfoMemory::solInLastEnvi[j].orignalx[0], false);
			double nnfox = evaluateByFunNN(i, useFunNN, &InfoMemory::solInLastEnvi[j].orignalx[0]);
			if (fitBetter(realfnewpoint, realfox)) sum++;
			if (fitBetter(realfnewpoint, realfox) && fitBetter(newpoints[j].f, nnfox)) consum++;
			if (j == 0 || fitBetter(realfnewpoint, bestvalue)){ bestvalue = realfnewpoint; bestnewx = j; }
			if (j == 0 || fitBetter(realfox, bestoxvalue)){ bestoxvalue = realfox; bestoxondex = j; }
			///	cout << endl;
			//	if (j % 500 == 0) getchar();
		}
		cout << "number of new x better than the old x: " << sum << "\t" << consum << "\t(" << bestvalue
			<< ") " << fabs(newpoints[bestnewx].f - DynPara_max) << "*"
			<< "\t(" << DynPara_max << ")\t";// << endl;
		cout << fabs(newpoints[InfoMemory::solInLastEnvi.size() + bestoxondex].f - DynPara_max) << "\n";
		//find the worst ind in the cradle and subpopulations
		cout << "worst value in cradle and subpopulation: ";
		if (cradle.popsize > 0){
			cout << " ++ " << fabs(DynPara_max - cradle.pop[cradle.Find_PBest()].pbest.fitness) << "," << fabs(DynPara_max - cradle.pop[cradle.Find_PWorst()].pbest.fitness) << "\t--- ";
		}
		for (int k = 0; k < CSwarm::pop_num; ++k){
			cout << fabs(DynPara_max - CSwarm::sub_swarm[k].pop[CSwarm::sub_swarm[k].Find_PBest()].pbest.fitness)
				<< "," << fabs(DynPara_max - CSwarm::sub_swarm[k].pop[CSwarm::sub_swarm[k].Find_PWorst()].pbest.fitness) << "\t";
		}
		cout << "\n";
#endif
		sort(newpoints.begin(), newpoints.end()); //select the top 10 points adding into the population cradle
#ifdef DEBUG
		cout << "top 200:\n";
		for (int j = 0; j < 200; ++j){
			double realvalue = eval_movpeaks(&newpoints[j].x[0], false);
			cout << j << "(" << fabs(DynPara_max - realvalue) << "," << DynPara_max - newpoints[j].f << ")\t";
			if (j % 8 == 0) cout << endl;
		}
		cout << endl;
#endif
		int topnum = 500;
		// = 5;
		vector<int> visit(topnum);
		cradle.resize(selectnum);
		for (int j = 0; j < visit.size(); ++j) visit[j] = 0;
		for (int j = 0; j < selectnum; ++j){
			int index = random_int(0, topnum - j - 1);
			while (visit[index]){
				index++;
				if (index == topnum) index = 0;
			}
			visit[index] = 1;
			cradle[j].resize(DynPara::dimNum);
			for (int k = 0; k < DynPara::dimNum; ++k){
				cradle[j][k] = newpoints[index].x[j];
			}
			/*
			Chromosome location;
			for (int j = 0; j < DynPara::dimNum; ++j){
				location.getGene<double>(j) = newpoints[index].x[j];
			}
			location.evaluate(true);
			Particle p;
			p.initialize(location,0,0);
			cradle.addIndividual(p);
			*/
			//fes++;
		}
	}
}

#endif
