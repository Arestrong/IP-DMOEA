#ifndef __TINDIVIDUAL_H_
#define __TINDIVIDUAL_H_

//#include "global.h"
#include "objective.h"
#include "SDP.h"
#include "DF.h"

class TIndividual{
public:
	TIndividual();
	TIndividual(const TIndividual& ind2);
	virtual ~TIndividual();

	vector <double> x_var;
	vector <double> y_obj;

	void   rnd_init();
	void   obj_eval(int mEvals, bool flag =false);

    bool   operator<(const TIndividual &ind2);
    bool   operator==(const TIndividual &ind2);
    void   operator=(const TIndividual &ind2);

	void show_objective();
	void show_variable();
	void reMemory();

	int    rank;

};

TIndividual::TIndividual()
{
	for(int i=0; i<DynPara::dimNum; i++)
		x_var.push_back(0.0);
	for(int n=0; n<DynPara::objNum; n++)
        y_obj.push_back(0.0);
	rank = 0;
}

TIndividual::TIndividual(const TIndividual& ind2)
{
	x_var.resize(ind2.x_var.size());
	y_obj.resize(ind2.y_obj.size());
	for (int i = 0; i < DynPara::dimNum; i++)
		x_var[i] = ind2.x_var[i];
	for (int n = 0; n < DynPara::objNum; n++)
		y_obj[n] = ind2.y_obj[n];
	rank = 0;
}

TIndividual::~TIndividual()
{

}

void TIndividual::rnd_init()
{
    for(int n=0;n<DynPara::dimNum;n++)
        x_var[n] = DynPara::lowBound[n] + random()*(DynPara::upperBound[n] - DynPara::lowBound[n]);

}

void TIndividual::obj_eval(int mEvals, bool flag)
{
    //objectives(x_var,y_obj);
	if(DynPara::test_SDP)
		objective(DynPara::proName.c_str(), x_var, y_obj);
	else if (DynPara::test_DF) {
		y_obj = cec2018_DF_eval(DynPara::proName, x_var, mEvals, DynPara::taut, DynPara::nt);
		//cout << mEvals << "\t";
		//for (int j = 0; j < y_obj.size(); ++j) cout << y_obj[j] << ",";
		//cout << endl;
	}
	else {
		assert(false);
	}
}


void TIndividual::show_objective()
{
    for(int n=0; n<DynPara::objNum; n++)
		printf("%f ",y_obj[n]);
	printf("\n");
}

void TIndividual::show_variable()
{
    for(int n=0; n<DynPara::dimNum; n++)
		printf("%f ",x_var[n]);
	printf("\n");
}

void TIndividual::operator=(const TIndividual &ind2)
{
    x_var = ind2.x_var;
	y_obj = ind2.y_obj;
	rank  = ind2.rank;
}

bool TIndividual::operator<(const TIndividual &ind2)
{
	bool dominated = true;
    for(int n=0; n<DynPara::objNum; n++)
	{
		if(ind2.y_obj[n]<y_obj[n]) return false;
	}
	if(ind2.y_obj==y_obj) return false;
	return dominated;
}


bool TIndividual::operator==(const TIndividual &ind2)
{
	if(ind2.y_obj==y_obj) return true;
	else return false;
}

void TIndividual::reMemory() {
	if (x_var.size() != DynPara::dimNum) {
		vector<double> ox = x_var;
		x_var.resize(DynPara::dimNum);
		for (int j = 0; j < DynPara::dimNum; ++j) {
			if (j < ox.size())
				x_var[j] = ox[j];
			else
				x_var[j] = DynPara::lowBound[j] + random()*(DynPara::upperBound[j] - DynPara::lowBound[j]);
		}
	}
	if (y_obj.size() != DynPara::objNum) {
		y_obj.resize(DynPara::objNum);
	}
}



class TSOP 
{
public:
	TSOP();
	virtual ~TSOP();

	void show();

	TIndividual     indiv;
	vector <double> namda;    
	vector <int>    table;     // the vector for the indexes of neighboring subproblems
	vector <int>    array;

    void  operator=(const TSOP&sub2);
};

TSOP::TSOP()
{
}

TSOP::~TSOP()
{
}


void TSOP::operator=(const TSOP&sub2)
{
    indiv  = sub2.indiv;
	table  = sub2.table;
	namda  = sub2.namda;
	array  = sub2.array;
}


#endif