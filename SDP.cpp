#include "SDP.h"
#include <iostream>
#include <sstream>
#include <assert.h>
using namespace std;
/*
int randVal; // save random integer
int randVal2;  // save random interger (used in SDP15)

int nobj;
//const int nobj_l = 3;
//const int nobj_u = 5;

int nvar;
//const int nvar_l = 10;
//const int nvar_u = 20;

int seed = 177;
long rnd_uni_init;

double rnd_rt, // random number between 0 and 1;
Tt,		// time instant;
sdp_y[256]; // preserve PS of SDP1 every time step;
*/
int getProDimNum() {
	return nvar;
}
int getProObjNum() {
	return nobj;
}

double getRnd_rtInSDP4() {
	return rnd_rt;
}

int getPtInSDP15() {
	return randVal2;
}

int getDtInSDP15() {
	return randVal;
}
double getTt() {
	return Tt;
}

void getOptimalSolution(string strTestInstance, const vector<double> &x, vector<double> &optS) {
	if (optS.size() != nvar)optS.resize(nvar);
	for (int k = 0; k < nobj; ++k) {
		optS[k] = x[k];
	}
	if (strTestInstance == "SDP1") {
		for (int k = nobj - 1; k < nvar; ++k)
			optS[k] = sdp_y[k];
	}
	else if (strTestInstance == "SDP2") {
		double x1 = 3 * x[0] + 1;
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = cos(Tt + 2 * x1) / 2 + 0.5;
		}
	}
	else if (strTestInstance == "SDP3") {
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = cos(Tt) / 2 + 0.5;
		}
	}
	else if (strTestInstance == "SDP4") {
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = cos(Tt + x[k - 1] + x[0]) / 2 + 0.5; // cos(Tt + x[0] + x[k - 1]);
		}
	}
	else if (strTestInstance == "SDP5") {
		double Gt = fabs(sin(0.5*pi*Tt));
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = 0.5*Gt*x[0];
		}
	}
	else if (strTestInstance == "SDP6") {
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = 0.5;
		}
	}
	else if (strTestInstance == "SDP7") {
		int pt = 1 + floor(5 * rnd_rt);
		for (int k = nobj - 1; k < nvar; ++k)
			optS[k] = 0.1 * 2 * (pt - 1);
	}
	else if (strTestInstance == "SDP8") {
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = sin(Tt*x[0]) / 2 + 0.5;
		}
	}
	else if (strTestInstance == "SDP9") {////////////////////
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = fabs(atan(SafeCot(3 * pi*Tt*Tt))) / pi;
			//optS[k] = sin(Tt*x[0]) / 2 + 0.5;
		}
	}
	else if (strTestInstance == "SDP10") {
		double s = 0;
		for (int k = 0; k < nobj - 1; ++k) {
			s += x[k] * x[k];
		}
		s /= (nobj - 1);
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = sin(x[0] + 0.5*pi*Tt) / 2 + 0.5;
		}
	}
	else if (strTestInstance == "SDP11") {
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = fabs(sin(0.5*pi*Tt));
		}
	}
	else if (strTestInstance == "SDP12") {
		double pt = nvar;///////////////////////////////
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = sin(pt*Tt)*sin(2 * pi*x[0]) / 2 + 0.5;
		}
	}
	else if (strTestInstance == "SDP13") {
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = (k*Tt) / (nobj + k * Tt);
		}
	}
	else if (strTestInstance == "SDP14") {
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = 0.5;
		}
	}
	else if (strTestInstance == "SDP15") {
		int dt = randVal;
		for (int k = nobj - 1; k < nvar; ++k) {
			optS[k] = dt / (nobj - 1);
		}
	}
	else {
		cout << "There is no function: " << strTestInstance << endl;
		assert(false);
	}
}

//set the parameters in the first environment
void initSDPSystem(string strTestInstance, int initObjNum, int initDimNum) {
	seed = 177;
	//	nvar = 10;
	nobj = initObjNum;
	nvar = initDimNum;
	Tt = 0.0;    //the environment index; starting from 0
	rnd_rt = 0.0;
	randVal = 0;
	randVal2 = 0;
	if ((strTestInstance == "SDP1"))
	{
		for (int i = nobj; i < nvar; i++)
		{
			sdp_y[i] = 1.0 * (i + 1) / nvar;
		}
	}

	if ((strTestInstance == "SDP13"))
	{
		nobj = 2;
	}

	if ((strTestInstance == "SDP15"))
	{
		randVal = nobj - 1; randVal2 = 0;
	}
	//cout << nobj << "\t" << nvar << endl;
}

//change the environment in MOP
//taut: the change frequency -- the number of generations in one environment
int changeEnvironments(string strTestInstance, int gen, int T0, int nt, int taut)
{
	seed = (seed + 23) % 1377;
	rnd_uni_init = -(long)seed;

	//    if (strTestInstance=="SDP13"){ nobj_l=2; nobj_u=5;} else nobj_l=nobj_u=3;

	//nobj = nobj_l;
	//nvar = nvar_l;

	randVal2 = 0;

	int g = (gen - T0) > 0 ? (gen - T0) : 0;
	if (g > 0)
		Tt = 1.0 / nt * (floor(1.0*(g - 1) / taut) + 1);
	else
	{
		Tt = 0.0;
		rnd_rt = 0.0;

		// set number of objectives and variables, change here if needed
	}

	if (g%taut == 1)
	{
		rnd_rt = rnd_uni(&rnd_uni_init);

		if ((strTestInstance == "SDP1")) {
			for (int i = nobj - 1; i < nvar; i++)
			{
				rnd_rt = rnd_uni(&rnd_uni_init);
				sdp_y[i] += 5 * (rnd_rt - 0.5)*sin(0.5*pi*Tt);
				if ((sdp_y[i] > 1.0) || (sdp_y[i] < 0.0))
					sdp_y[i] = rnd_rt;
			}
		}

		else if (strTestInstance == "SDP12") {
			nvar = nvar_l + floor(0.5 + rnd_rt * (nvar_u - nvar_l));
			randVal = nvar;
			return nvar;
		}
		else if (strTestInstance == "SDP13") {
			nobj = nobj_l + floor(0.5 + rnd_rt * (nobj_u - nobj_l));
			randVal = nobj;
			return nobj;
		}
		else if (strTestInstance == "SDP15") {
			randVal = 1 + floor(0.5 + rnd_rt * (nobj - 2));
			randVal2 = 1 + floor(0.5 + rnd_uni(&rnd_uni_init)*(nobj - 2));
			return nobj;
		}
	}
	else
	{
		if ((strTestInstance == "SDP1"))
		{
			for (int i = nobj; i < nvar; i++)
			{
				sdp_y[i] = 1.0 * (i + 1) / nvar;
			}
		}

		if ((strTestInstance == "SDP13"))
		{
			nobj = 2;
		}

		if ((strTestInstance == "SDP15"))
		{
			randVal = nobj - 1; randVal2 = 0;
		}

	}
	return 1;
}

//change the environment if the curFes is the first FEs in the new environment
//taut: the change frequency -- the number of generations in one environment
int changeEnviInNextFes(string strTestInstance, int curFes, int T0, int nt, int taut)
{
	seed = (seed + 23) % 1377;
	rnd_uni_init = -(long)seed;

	//    if (strTestInstance=="SDP13"){ nobj_l=2; nobj_u=5;} else nobj_l=nobj_u=3;

	//nobj = nobj_l;
	//nvar = nvar_l;
	int nextFes = curFes + 1;
	randVal2 = 0;

	int g = (nextFes - T0) > 0 ? (nextFes - T0) : 0;
	if (g > 0)
		Tt = 1.0 / nt * (floor(1.0*(g - 1) / taut) + 1);
	else
	{
		Tt = 0.0;
		rnd_rt = 0.0;

		// set number of objectives and variables, change here if needed
	}
	//cout << "changeEnviInNextFes:\t" << nextFes << "\t" << taut << "\t" << Tt << "\t" << rnd_rt << "\n";
	if (g%taut == 1)  //when the new environment strats, changes the the environment
	{
		rnd_rt = rnd_uni(&rnd_uni_init);

		if ((strTestInstance == "SDP1")) {
			/*cout << "sdp_y in SDP1:\t";
			for (int i = nobj - 1; i < nvar; i++)
			{
				cout << sdp_y[i] << ",";
			}
			cout << endl;
			cout << rnd_rt << "\t" << rnd_uni_init << endl;*/
			for (int i = nobj - 1; i < nvar; i++)
			{
				rnd_rt = rnd_uni(&rnd_uni_init);
				sdp_y[i] += 5 * (rnd_rt - 0.5)*sin(0.5*pi*Tt);
				if ((sdp_y[i] > 1.0) || (sdp_y[i] < 0.0))
					sdp_y[i] = rnd_rt;
			}
			/*cout << "sdp_y in SDP1:\t";
			for (int i = nobj - 1; i < nvar; i++)
			{
				cout << sdp_y[i] << ",";
			}
			cout << endl;
			cout << rnd_rt << "\t" << rnd_uni_init << endl;*/
		}

		else if (strTestInstance == "SDP12") {
			nvar = nvar_l + floor(0.5 + rnd_rt * (nvar_u - nvar_l));
			randVal = nvar;
			return nvar;
		}
		else if (strTestInstance == "SDP13") {

			nobj = nobj_l + floor(0.5 + rnd_rt * (nobj_u - nobj_l));
			randVal = nobj;
			//	cout << "SDP13 change environment:\t" << rnd_rt << "\t" << nobj << endl;
			return nobj;
		}
		else if (strTestInstance == "SDP15") {
			randVal = 1 + floor(0.5 + rnd_rt * (nobj - 2));
			randVal2 = 1 + floor(0.5 + rnd_uni(&rnd_uni_init)*(nobj - 2));
			return nobj;
		}
	}
	return 1;
}

//change the environment if the curFes is the first FEs in the new environment
//taut: the change frequency -- the number of generations in one environment
int changeToNextEnvi(string strTestInstance, int nextEnviIndex, int nt)
{
	seed = (seed + 23) % 1377;
	rnd_uni_init = -(long)seed;

	//    if (strTestInstance=="SDP13"){ nobj_l=2; nobj_u=5;} else nobj_l=nobj_u=3;

	//nobj = nobj_l;
	//nvar = nvar_l;

	randVal2 = 0;

	if (nextEnviIndex > 0)
		Tt = 1.0 / nt * nextEnviIndex;
	else
	{
		Tt = 0.0;
		rnd_rt = 0.0;

		// set number of objectives and variables, change here if needed
	}

	if (nextEnviIndex > 0)  //when the new environment strats, changes the the environment
	{
		rnd_rt = rnd_uni(&rnd_uni_init);

		if ((strTestInstance == "SDP1")) {
			for (int i = nobj - 1; i < nvar; i++)
			{
				rnd_rt = rnd_uni(&rnd_uni_init);
				sdp_y[i] += 5 * (rnd_rt - 0.5)*sin(0.5*pi*Tt);
				if ((sdp_y[i] > 1.0) || (sdp_y[i] < 0.0))
					sdp_y[i] = rnd_rt;
			}
		}

		else if (strTestInstance == "SDP12") {
			nvar = nvar_l + floor(0.5 + rnd_rt * (nvar_u - nvar_l));
			randVal = nvar;
			return nvar;
		}
		else if (strTestInstance == "SDP13") {
			nobj = nobj_l + floor(0.5 + rnd_rt * (nobj_u - nobj_l));
			randVal = nobj;
			return nobj;
		}
		else if (strTestInstance == "SDP15") {
			randVal = 1 + floor(0.5 + rnd_rt * (nobj - 2));
			randVal2 = 1 + floor(0.5 + rnd_uni(&rnd_uni_init)*(nobj - 2));
			return nobj;
		}
		//cout << "<" << rnd_rt << ">\t";
	}
	else {
		if ((strTestInstance == "SDP1"))
		{
			for (int i = nobj; i < nvar; i++)
			{
				sdp_y[i] = 1.0 * (i + 1) / nvar;
			}
		}

		if ((strTestInstance == "SDP13"))
		{
			nobj = 2;
		}

		if ((strTestInstance == "SDP15"))
		{
			randVal = nobj - 1; randVal2 = 0;
		}
		//cout << "(" <<  rnd_rt << ")\t";
	}

	return 1;
}


double SafeAcos(double x)
{
	if (x < -1.0) x = -1.0;
	else if (x > 1.0) x = 1.0;
	return acos(x);
}

double SafeCot(double x)
{
	if (sin(x) == 0)
		return INFINITY;
	else
		return cos(x) / sin(x);
}

double minPeaksOfSDP7(double x, double pt)
{
	if (pt < 1) pt = 1;
	if (pt > 5) pt = 5;
	double min = INFINITY;
	double tmp;
	for (int i = 1; i < 6; i++)
	{
		if (i == pt)
			tmp = 0.5 + 10 * pow(10 * x - 2 * (i - 1), 2.0);
		else
			tmp = i + 10 * pow(10 * x - 2 * (i - 1), 2.0);
		if (tmp <= min)
			min = tmp;
	}
	return min;
}

// the objective vector of SDP.
void objective(const char* strTestInstance, vector<double> &x_var, vector<double> &y_obj)
{
	
	double G = sin(0.5*pi*Tt);
	// SDP PROBLEMS
	if (strcmp(strTestInstance, "SDP1") == 0) //SDP1
	{
		double g = 0;

		// ps functions
		for (int i = nobj; i < nvar; i++)
		{
			g += pow(x_var[i] - sdp_y[i], 2.0);
		}

		// pf functions
		double prd = 1;
		for (int j = 0; j < nobj; j++) {
			double xx = 3 * x_var[j] + 1;
			prd *= xx;
		}
		for (int j = 0; j < nobj; j++)
		{
			double xx = 3 * x_var[j] + 1;
			y_obj[j] = (1 + g)*xx / pow(prd / xx, 1.0 / (nobj - 1));
		}
		//cout << "SDP1" << "\t";
		return;
	}

	if (strcmp(strTestInstance, "SDP2") == 0) //SDP2
	{
		double g = 0;

		double x1 = 3 * x_var[0] + 1;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double xx = 2 * (x_var[i] - 0.5) - cos(Tt + 2 * x1);
			g += xx * xx;
		}
		g *= sin(pi*x1 / 8);
		// pf functions
		double sm = 0;
		for (int j = 0; j < nobj - 1; j++)
		{
			double xx = 3 * x_var[j] + 1;
			sm += xx;
		}
		sm += 1;

		for (int j = 0; j < nobj - 1; j++)
		{
			double xx = 3 * x_var[j] + 1;
			y_obj[j] = (1 + g)*(sm - xx + Tt) / xx;
		}
		y_obj[nobj - 1] = (1 + g)*(sm - 1) / (1 + Tt);
		return;
	}

	if (strcmp(strTestInstance, "SDP3") == 0) //SDP3
	{
		double g = 0;
		int pt = floor(5 * fabs(sin(pi*Tt)));
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = 2 * (x_var[i] - 0.5) - cos(Tt);
			g += 4 * pow(y, 2.0) - cos(2 * pt*pi*y) + 1;
		}

		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			y_obj[j] = (1 + g)*(1 - x_var[j] + 0.05*sin(6 * pi*x_var[j]))*pd;
			pd *= x_var[j] + 0.05*sin(6 * pi*x_var[j]);
		}
		y_obj[nobj - 1] = (1 + g)*pd;
		return;
	}

	if (strcmp(strTestInstance, "SDP4") == 0) //SDP4
	{
		double w = (rnd_rt - 0.5); //----------------------
		if (w > 0)
			w = 1.0;
		else if (w < 0)
			w = -1.0;
		w = w * floor(6 * fabs(G));

		randVal = w; // save this random value

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = 2 * (x_var[i] - 0.5) - cos(Tt + x_var[i - 1] + x_var[0]);
			g += y * y;
		}

		// pf functions
		double sm = 0;
		for (int j = 0; j < nobj - 2; j++)
		{
			y_obj[j] = x_var[j];
			sm += x_var[j];
		}
		sm += x_var[nobj - 2];
		sm = sm / (nobj - 1);

		y_obj[nobj - 2] = (1 + g)*(sm + 0.05*sin(w*pi*sm));
		y_obj[nobj - 1] = (1 + g)*(1.0 - sm + 0.05*sin(w*pi*sm));
		return;
	}

	if (strcmp(strTestInstance, "SDP5") == 0) //SDP5
	{
		double Gt = fabs(G);
		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = x_var[i] - 0.5*Gt*x_var[0];
			g += y * y;
		}
		g += Gt;

		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			double y = pi * Gt / 6 + (pi / 2 - pi * Gt / 3)*x_var[j];
			y_obj[j] = (1 + g)*sin(y)*pd;
			pd = cos(y)*pd;
		}
		y_obj[nobj - 1] = (1 + g)*pd;
		return;
	}

	if (strcmp(strTestInstance, "SDP6") == 0) //SDP6
	{
		double at = 0.5*fabs(sin(pi*Tt));
		double pt = 10 * cos(2.5*pi*Tt);

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = x_var[i] - 0.5;
			g += y * y *(1 + fabs(cos(8 * pi*x_var[i])));
		}
		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			y_obj[nobj - 1 - j] = (1 + g)*sin(0.5*pi*x_var[j])*pd;
			pd *= cos(0.5*pi*x_var[j]);
		}
		y_obj[0] = (1 + g)*pd;


		if (x_var[0] < at)
			y_obj[nobj - 1] = (1 + g)*fabs(pt*(cos(0.5*pi*x_var[0]) - cos(0.5*pi*at)) + sin(0.5*pi*at));
		//		else
		//            y_obj[nobj - 1] = (1 + g)*sin(0.5*pi*x_var[j])
		return;
	}

	if (strcmp(strTestInstance, "SDP7") == 0) //SDP7
	{
		double at = 0.5*sin(pi*Tt);
		int pt = 1 + floor(5 * rnd_rt);

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			g += minPeaksOfSDP7(x_var[i], pt);
		}
		g /= (nvar - nobj + 1);

		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			y_obj[j] = (0.5 + g)*(1 - x_var[j])*pd;
			pd *= x_var[j];
		}
		y_obj[nobj - 1] = (0.5 + g)*pd;
		return;
	}

	if (strcmp(strTestInstance, "SDP9") == 0) //SDP8
	{
		double Gt = fabs(G);
		int pt = floor(6 * Gt);
		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = x_var[i] - fabs(atan(SafeCot(3 * pi*Tt*Tt))) / pi;
			g += y * y;
		}
		g += Gt;

		// pf functions
		double sm = 0;
		for (int j = 0; j < nobj - 1; j++)
		{
			y_obj[j] = (1 + g)*pow(cos(0.5*pi*x_var[j]), 2.0) + Gt;
			sm += pow(sin(0.5*pi*x_var[j]), 2.0) + sin(0.5*pi*x_var[j])*pow(cos(pt*pi*x_var[j]), 2.0);
		}
		y_obj[nobj - 1] = sm + Gt;
		return;
	}

	if (strcmp(strTestInstance, "SDP8") == 0) //SDP9
	{
		double kt = 10 * sin(pi*Tt);

		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
			pd *= sin(floor(kt*(2 * x_var[j] - 1.0))*pi / 2);

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = 2 * (x_var[i] - 0.5) - sin(Tt*x_var[0]);
			g += y * y;
		}
		g += fabs(pd);

		// pf functions
		pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			y_obj[nobj - 1 - j] = (1 + g)*sin(0.5*pi*x_var[j])*pd;
			pd *= cos(0.5*pi*x_var[j]);
		}
		y_obj[0] = (1 + g)*pd;
		return;
	}

	if (strcmp(strTestInstance, "SDP10") == 0) //SDP10
	{
		int r = floor(10 * fabs(G));

		double sm = 0;
		for (int j = 0; j < nobj - 1; j++)
			sm += pow(x_var[j], 2.0);
		sm /= (nobj - 1);

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = 2 * (x_var[i] - 0.5) - sin(0.5*pi*Tt + x_var[0]);
			g += y * y;
		}

		// pf functions
		for (int j = 0; j < nobj - 1; j++)
			y_obj[j] = x_var[j];

		y_obj[nobj - 1] = (1 + g)*(2 - sm - pow(sm, 0.5)*pow(-sin(2.5*pi*sm), r));
		return;
	}

	if (strcmp(strTestInstance, "SDP11") == 0) //SDP11
	{
		double at = 3 * Tt - floor(3 * Tt);
		double bt = 3 * Tt + 0.2 - floor(3 * Tt + 0.2);

		//double sm = 0;
		/*for (int j = 0; j < nobj - 1; j++)
		 sm += x_var[j];
		 sm /= (nobj - 1);*/


		double g = 0;
		double ps = 0;
		for (int i = 0; i < nobj - 1; i++)
			ps += x_var[i];

		if (ps >= at && ps < bt) {
			for (int i = nobj - 1; i < nvar; i++) {
				double p = x_var[i] - fabs(G);
				g += -0.9 * p * p + pow(fabs(p), 0.6);
			}
		}
		else {
			for (int i = nobj - 1; i < nvar; i++) {
				double p = x_var[i] - fabs(G);
				g += p * p;
			}
		}

		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			double yj = 0.5*pi*x_var[j];
			y_obj[j] = (1 + g)*sin(yj)*pd;
			pd *= cos(yj);
		}
		y_obj[nobj - 1] = (1 + g)*pd;
		return;
	}

	if (strcmp(strTestInstance, "SDP12") == 0) //SDP12 -change variables
	{
		//int nvar = nvar_l + floor(rnd_rt*(nvar_u - nvar_l));

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = 2 * (x_var[i] - 0.5) - sin(Tt)*sin(2 * pi*x_var[1]);
			g += y * y;
		}

		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			y_obj[j] = (1.0 + g)*(1 - x_var[j])*pd;
			pd *= x_var[j];
		}
		y_obj[nobj - 1] = (1.0 + g)*pd;
		return;
	}

	if (strcmp(strTestInstance, "SDP13") == 0) //SDP13-change objectives
	{
		// *** this is for many objectives only.
		//nobj = nobj_l + floor(rnd_rt*(nobj_u - nobj_l));

		randVal = nobj;  // save this random value.

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = x_var[i] - (i*Tt) / (nobj + i * Tt);
			g += pow(y, 2.0);
		}

		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj; j++)
		{
			double yj = pi * (x_var[j] + 1) / 6;
			y_obj[j] = (1 + g)*sin(yj)*pd;
			pd *= cos(yj);
		}

		for (int j = nobj; j < y_obj.size(); j++)
			y_obj[j] = 0;
		return;
	}

	if (strcmp(strTestInstance, "SDP14") == 0) //SDP14 -degenerate
	{
		double pt = 1 + floor(fabs((nobj - 2)*cos(0.5*pi*Tt)));

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			g += pow(x_var[i] - 0.5, 2.0);
		}


		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			double yj = x_var[j];
			if (j >= pt) yj = 0.5 + x_var[j] * g*fabs(G);
			y_obj[j] = (1 + g)*(1 + g - yj)*pd;
			pd *= yj;
		}
		y_obj[nobj - 1] = (1.0 + g)*pd;
		return;
	}

	if (strcmp(strTestInstance, "SDP15") == 0) //SDP15-degenerate
	{
		int dt = randVal;
		int pt = randVal2;

		double g = 0;
		for (int i = nobj - 1; i < nvar; i++)
		{
			double y = x_var[i] - dt / (nobj - 1);
			g += pow(y, 2.0);
		}

		vector<double> yk(nobj - 1, 0);
		for (int i = 1; i < nobj; i++)
		{
			int k = (pt + i - 1) % (nobj - 1);
			if (k <= dt) yk[k] = 0.5*pi*x_var[k];
			else yk[k] = SafeAcos(1.0 / (pow(2.0, 0.5)*(1.0 + x_var[k] * g*fabs(G))));
		}

		// pf functions
		double pd = 1;
		for (int j = 0; j < nobj - 1; j++)
		{
			y_obj[j] = pow(1 + g, j + 1)*sin(yk[j])*pd;
			pd *= cos(yk[j]);
		}
		y_obj[nobj - 1] = pow(1 + g, nobj)*pd;
		return;
	}

	if (!strcmp(strTestInstance, "FDA1"))
	{
		double g = 0;
		int lnvar = 20;
		for (int n = 1; n < nvar; n++)
		{
			double x = 2 * (x_var[n] - 0.5);
			g += pow(x - G, 2.0);
		}
		g = 1 + g;

		y_obj[0] = x_var[0];
		y_obj[1] = g * (1 - sqrt(y_obj[0] / g));

		return;
	}

	if (!strcmp(strTestInstance, "FDA2"))
	{// Modified version of FDA2, used in dNSGA-II
		double g = 0;
		double H = 2 * sin(0.5*pi*(Tt - 1));
		int lnvar = 31;
		for (int n = 1; n <= 5; n++)
		{
			double x = 2 * (x_var[n] - 0.5);
			g += pow(x, 2.0);
		}
		g = 1 + g;

		double s = H;
		for (int n = 6; n < nvar; n++)
		{
			double x = 2 * (x_var[n] - 0.5);
			s += pow(x - H / 4, 2.0);
		}
		y_obj[0] = x_var[0];
		y_obj[1] = g * (1 - pow(y_obj[0] / g, pow(2.0, s)));

		return;
	}

	if (!strcmp(strTestInstance, "rFDA2"))
	{
		int ge = (itt - T0) > 0 ? (itt - T0) : 0;
		if (ge > 0)
			Tt = 2 * ((ge - 1) / taut)*(1.0*taut / (max_gen - taut));
		else Tt = 0;
		double H = 2 * sin(0.5*pi*(Tt - 1));

		double g = 0;
		int lnvar = 13;
		for (int n = 1; n < 6; n++)
		{
			double x = 2 * (x_var[n] - 0.5);
			g += pow(x, 2.0);
		}
		g = 1 + g;

		double s = H;
		for (int n = 6; n < lnvar; n++)
		{
			double x = 2 * (x_var[n] - 0.5);
			s += pow(x - H / 4, 2.0);
		}
		y_obj[0] = x_var[0];
		y_obj[1] = g * (1 - pow(y_obj[0] / g, 1.0 / s));

		return;
	}

	if (!strcmp(strTestInstance, "FDA3"))
	{
		double g = 0, F = pow(10.0, 2 * G);
		int lnvar = 30;
		int nvar1 = 2;
		for (int n = nvar1; n < nvar; n++)
		{
			double x = 2 * (x_var[n] - 0.5);
			g += pow(x - fabs(G), 2.0);
		}
		g = 1 + fabs(G) + g;

		y_obj[0] = 0;
		for (int n = 0; n < nvar1; n++)
			y_obj[0] += pow(x_var[n], F);
		y_obj[0] /= nvar1;
		y_obj[1] = g * (1 - sqrt(y_obj[0] / g));

		return;
	}

	if (!strcmp(strTestInstance, "FDA4"))
	{
		double g = 0, prod;

		for (int n = 2; n < nvar; n++)
		{
			g += pow(x_var[n] - fabs(G), 2.0);
		}
		g = 1 + g;

		prod = g;
		for (int n = 0; n < nobj - 1; n++)
			prod *= cos(0.5*pi*x_var[n]);
		y_obj[0] = prod;

		for (int i = 1; i < nobj; i++)
		{
			prod = g;
			for (int j = 0; j < nobj - 1 - i; j++)
				prod *= cos(0.5*pi*x_var[j]);
			prod *= sin(0.5*pi*x_var[nobj - 1 - i]);
			y_obj[i] = prod;
		}

		return;
	}

	if (!strcmp(strTestInstance, "FDA5"))
	{
		double g = 0, prod;

		double F = 1.0 + 100.0*pow(G, 4.0);
		for (int n = 2; n < nvar; n++)
		{
			g += pow(x_var[n] - fabs(G), 2.0);
		}
		g = 1 + fabs(G) + g;

		prod = g;
		for (int n = 0; n < nobj - 1; n++)
			prod *= cos(0.5*pi*pow(x_var[n], F));
		y_obj[0] = prod;

		for (int i = 1; i < nobj; i++)
		{
			prod = g;
			for (int j = 0; j < nobj - 1 - i; j++)
				prod *= cos(0.5*pi*pow(x_var[j], F));
			prod *= sin(0.5*pi*pow(x_var[nobj - 1 - i], F));
			y_obj[i] = prod;
		}

		return;
	}


	if (!strcmp(strTestInstance, "dMOP1"))
	{
		double g = 0, H = 0.75*G + 1.25;
		int lnvar = 10;
		for (int n = 1; n < lnvar; n++)
		{
			g += pow(x_var[n], 2.0);
		}
		g = 1 + 9 * g;

		y_obj[0] = x_var[0];
		y_obj[1] = g * (1 - pow(y_obj[0] / g, H));

		return;
	}

	if (!strcmp(strTestInstance, "dMOP2"))
	{
		double g = 0, H = 0.75*G + 1.25;
		int lnvar = 10;
		for (int n = 1; n < lnvar; n++)
		{
			g += pow(x_var[n] - fabs(G), 2.0);
		}
		g = 1 + g;

		y_obj[0] = x_var[0];
		y_obj[1] = g * (1 - pow(y_obj[0] / g, H));

		return;
	}

	if (!strcmp(strTestInstance, "dMOP3"))
	{
		double g = 0;
		int lnvar = 10;
		for (int n = 0; n < lnvar; n++)
		{
			if (n == Parar) continue;
			g += pow(x_var[n] - fabs(G), 2.0);
		}
		g = 1 + g;

		y_obj[0] = x_var[Parar];
		y_obj[1] = g * (1 - sqrt(y_obj[0] / g));

		return;
	}

	//============F5-F10 used in PPS=============//
	if (!strcmp(strTestInstance, "ZJZ1")) //F5
	{
		// note that a and b follows the setting of pps, which
		// is different from the original paper
		double a = 2 * cos(0.5*pi*Tt) + 2;
		double b = 2 * sin(pi*Tt) + 2;
		double H = 1.25 + 0.75*sin(pi*Tt);
		double Gi;

		double s1 = 0, s2 = 0;
		int lnvar = 10;
		for (int n = 1; n < lnvar; n++)
		{
			double x = 5 * x_var[n];
			Gi = 1.0 - pow(fabs(x_var[0] - a), H + double(n + 1.0) / double(lnvar));
			if (n % 2)
				s1 += pow(x - b - Gi, 2.0);
			else
				s2 += pow(x - b - Gi, 2.0);
		}

		y_obj[0] = pow(fabs(5 * x_var[0] - a), H) + 0.5*s1;
		y_obj[1] = pow(fabs(5 * x_var[0] - a - 1), H) + 0.5*s2;

		return;
	}

	if (!strcmp(strTestInstance, "ZJZ2")) //F6
	{
		double a = 2 * cos(1.5*0.5*pi*Tt)* sin(0.5*0.5*pi*Tt) + 2;
		double b = 2 * cos(1.5*0.5*pi*Tt)* cos(0.5*0.5*pi*Tt) + 2;
		double H = 1.25 + 0.75*sin(pi*Tt);
		double Gi;

		double s1 = 0, s2 = 0;
		int lnvar = 10;
		for (int n = 1; n < lnvar; n++)
		{
			double x = 5 * x_var[n];
			Gi = 1.0 - pow(fabs(x_var[0] - a), H + double(n + 1.0) / double(lnvar));
			if (n % 2)
				s1 += pow(x - b - Gi, 2.0);
			else
				s2 += pow(x - b - Gi, 2.0);
		}

		y_obj[0] = pow(fabs(5 * x_var[0] - a), H) + 0.5*s1;
		y_obj[1] = pow(fabs(5 * x_var[0] - a - 1), H) + 0.5*s2;

		return;
	}

	if (!strcmp(strTestInstance, "ZJZ3")) //F7
	{
		double a = 1.7*(1 - sin(0.5*pi*Tt)*sin(0.5*pi*Tt) + 2.0);
		double b = 1.4*(1 - sin(0.5*pi*Tt)*cos(0.5*pi*Tt) + 1.5);
		double H = 1.25 + 0.75*sin(pi*Tt);
		double Gi;

		double s1 = 0, s2 = 0;
		int lnvar = 10;
		for (int n = 1; n < lnvar; n++)
		{
			double x = 5 * x_var[n]; //[0,5]
			Gi = 1.0 - pow(fabs(x_var[0] - a), H + double(n + 1.0) / double(lnvar));
			if (n % 2)
				s1 += pow(x - b - Gi, 2.0);
			else
				s2 += pow(x - b - Gi, 2.0);
		}

		y_obj[0] = pow(fabs(5 * x_var[0] - a), H) + 0.5*s1;
		y_obj[1] = pow(fabs(5 * x_var[0] - a - 1), H) + 0.5*s2;

		return;
	}

	if (!strcmp(strTestInstance, "ZJZ4")) //F8
	{
		double H = 1.25 + 0.75*sin(pi*Tt);


		double g = 0;
		int lnvar = 10;
		for (int n = 2; n < lnvar; n++)
		{
			double x = 3.0 * x_var[n] - 1.0; //[-1,2]
			g += pow(x - pow(0.5*(x_var[0] + x_var[1]), H + double(n + 1.0) / double(lnvar)) - G, 2.0);

		}

		y_obj[0] = (1 + g)*cos(0.5*pi*x_var[0])*cos(0.5*pi*x_var[1]);
		y_obj[1] = (1 + g)*cos(0.5*pi*x_var[0])*sin(0.5*pi*x_var[1]);
		y_obj[2] = (1 + g)*sin(0.5*pi*x_var[0]);

		return;
	}

	if (!strcmp(strTestInstance, "ZJZ5")) //F9
	{
		double Tt0 = Tt - floor(Tt);
		double a = 2 * cos(0.5*pi*Tt0) + 2;
		double b = 2 * sin(pi*Tt0) + 2;
		double H = 1.25 + 0.75*sin(pi*Tt);
		double Gi;

		double s1 = 0, s2 = 0;
		int lnvar = 10;
		for (int n = 1; n < lnvar; n++)
		{
			double x = 5 * x_var[n];
			Gi = 1.0 - pow(fabs(x_var[0] - a), H + double(n + 1.0) / double(lnvar));
			if (n % 2)
				s1 += pow(x - b - Gi, 2.0);
			else
				s2 += pow(x - b - Gi, 2.0);
		}

		y_obj[0] = pow(fabs(5 * x_var[0] - a), H) + 0.5*s1;
		y_obj[1] = pow(fabs(5 * x_var[0] - a - 1), H) + 0.5*s2;

		return;
	}

	if (!strcmp(strTestInstance, "ZJZ6")) //F10
	{
		double a = 2 * cos(0.5*pi*Tt) + 2;
		double b = 2 * sin(pi*Tt) + 2;
		double H = 1.25 + 0.75*sin(pi*Tt);
		double Gi;

		double s1 = 0, s2 = 0;
		int lnvar = 10;

		int old = ((int)(Tt*nt + 0.001)) % 2;
		for (int n = 1; n < lnvar; n++)
		{
			double x = 5 * x_var[n];
			if (old)
				Gi = pow(fabs(x_var[0] - a), H + double(n + 1.0) / double(lnvar));
			else
				Gi = 1.0 - pow(fabs(x_var[0] - a), H + double(n + 1.0) / double(lnvar));

			if (n % 2)
				s1 += pow(x - b - Gi, 2.0);
			else
				s2 += pow(x - b - Gi, 2.0);
		}

		y_obj[0] = pow(fabs(5 * x_var[0] - a), H) + 0.5*s1;
		y_obj[1] = pow(fabs(5 * x_var[0] - a - 1), H) + 0.5*s2;

		return;
	}

	if (!strcmp(strTestInstance, "UDF1")) //UDF1
	{
		unsigned int j, count1, count2;
		double sum1, sum2, yj;

		sum1 = sum2 = 0.0;
		count1 = count2 = 0;

		for (j = 2; j <= nvar; j++)
		{
			double x = 4 * x_var[j - 1] - 2.0;
			yj = x - sin(6.0*pi*x_var[0] + j * pi / nvar) - G;
			yj = yj * yj;
			if (j % 2 == 0)
			{
				sum2 += yj;
				count2++;
			}
			else
			{
				sum1 += yj;
				count1++;
			}
		}
		y_obj[0] = x_var[0] + 2.0 * sum1 / (double)count1 + fabs(G);
		y_obj[1] = 1.0 - x_var[0] + 2.0 * sum2 / (double)count2 + fabs(G);

		return;
	}

	if (!strcmp(strTestInstance, "UDF2")) //UDF2
	{
		unsigned int j, count1, count2;
		double sum1, sum2, prod1, prod2, yj;

		sum1 = sum2 = 0.0;
		count1 = count2 = 0;
		prod1 = prod2 = 1.0;

		for (j = 2; j <= nvar; j++)
		{
			double x = 3 * x_var[j - 1] - 1.0;
			yj = x - pow(x_var[0], 0.5*(2.0 + 3.0*(j - 2.0) / (nvar - 2.0) + G)) - G;
			yj = yj * yj;
			if (j % 2 == 0)
			{
				sum2 += yj;
				count2++;
			}
			else
			{
				sum1 += yj;
				count1++;
			}
		}
		y_obj[0] = x_var[0] + 2.0 * sum1 / (double)count1 + fabs(G);
		y_obj[1] = 1.0 - x_var[0] + 2.0 * sum2 / (double)count2 + fabs(G);

		return;
	}

	if (!strcmp(strTestInstance, "UDF3")) //UDF3
	{
		unsigned int j, count1, count2;
		double sum1, sum2, prod1, prod2, yj, pj, hj, N, E;

		sum1 = sum2 = 0.0;
		count1 = count2 = 0;
		prod1 = prod2 = 1.0;
		N = 10.0; E = 0.1;

		for (j = 2; j <= nvar; j++)
		{
			double x = 2 * x_var[j - 1] - 1.0;
			yj = x - sin(6.0*pi*x_var[0] + j * pi / nvar);
			pj = cos(20.0*yj*pi / sqrt(j + 0.0));
			if (j % 2 == 0)
			{
				sum2 += 2 * yj*yj;
				prod2 *= pj;
				count2++;
			}
			else
			{
				sum1 += 2 * yj*yj;
				prod1 *= pj;
				count1++;
			}
		}
		hj = (0.5 / N + E)*(sin(2.0*N*pi*x_var[0]) - 2.0*N*fabs(G));
		if (hj < 0.0) hj = 0.0;
		y_obj[0] = x_var[0] + hj + 2.0*(4.0*sum1 - 2.0*prod1 + 2.0) / (double)count1;
		y_obj[1] = 1.0 - x_var[0] + hj + 2.0*(4.0*sum2 - 2.0*prod2 + 2.0) / (double)count2;

		return;
	}

	if (!strcmp(strTestInstance, "UDF4")) //UDF4
	{
		unsigned int j, count1, count2;
		double sum1, sum2, yj, hj;
		double M = 0.5 + fabs(G);

		sum1 = sum2 = 0.0;
		count1 = count2 = 0;
		int K = ceil(nvar*G);
		for (j = 2; j <= nvar; j++)
		{
			double x = 2 * x_var[j - 1] - 1.0;
			yj = x - sin(6.0*pi*x_var[0] + (j + K)*pi / nvar);
			yj = yj * yj;
			if (j % 2 == 0)
			{
				sum2 += yj;
				count2++;
			}
			else
			{
				sum1 += yj;
				count1++;
			}
		}
		y_obj[0] = x_var[0] + 2.0 * sum1 / (double)count1;
		y_obj[1] = 1.0 - M * pow(x_var[0], M) + 2.0 * sum2 / (double)count2;

		return;
	}

	if (!strcmp(strTestInstance, "UDF5")) //UDF5
	{
		unsigned int j, count1, count2;
		double sum1, sum2, yj, pj;
		double M = 0.5 + fabs(G);

		sum1 = sum2 = 0.0;
		count1 = count2 = 0;

		for (j = 2; j <= nvar; j++)
		{
			double x = 3 * x_var[j - 1] - 1.0;
			yj = x - pow(x_var[0], 0.5*(2.0 + 3.0*(j - 2.0) / (nvar - 2.0) + G)) - G;
			yj = yj * yj;
			if (j % 2 == 0)
			{
				sum2 += yj;
				count2++;
			}
			else
			{
				sum1 += yj;
				count1++;
			}
		}
		y_obj[0] = x_var[0] + 2.0 * sum1 / (double)count1;
		y_obj[1] = 1.0 - M * pow(x_var[0], M) + 2.0 * sum2 / (double)count2;

		return;
	}

	if (!strcmp(strTestInstance, "UDF6")) //UDF6
	{
		unsigned int j, count1, count2;
		double sum1, sum2, yj, pj, hj, N, E;
		double M = 0.5 + fabs(G);

		sum1 = sum2 = 0.0;
		count1 = count2 = 0;
		N = 10.0; E = 0.1;

		for (j = 2; j <= nvar; j++)
		{
			double x = 2 * x_var[j - 1] - 1.0;
			yj = x - sin(6.0*pi*x_var[0] + j * pi / nvar);
			pj = 2 * yj*yj - cos(4.0*yj*pi) + 1.0;
			if (j % 2 == 0)
			{
				sum2 += pj;
				count2++;
			}
			else
			{
				sum1 += pj;
				count1++;
			}
		}
		hj = (0.5 / N + E)*fabs(sin(2.0*N*pi*x_var[0])) + fabs(G);
		y_obj[0] = x_var[0] + hj + 2.0 * sum1 / (double)count1;
		y_obj[1] = 1.0 - M * x_var[0] + hj + 2.0 * sum2 / (double)count2;

		return;
	}

	if (!strcmp(strTestInstance, "UDF7")) //UDF7
	{
		unsigned int j, count1, count2, count3;
		double sum1, sum2, sum3, yj;
		double R = 1.0 + fabs(G);

		sum1 = sum2 = sum3 = 0.0;
		count1 = count2 = count3 = 0;

		for (j = 3; j <= nvar; j++)
		{
			double x = 4 * x_var[j - 1] - 2.0;
			yj = x - 2 * x_var[1] * sin(2 * pi*x_var[0] + j * pi / nvar);
			yj = yj * yj;
			if (j % 3 == 0)
			{
				sum3 += yj;
				count3++;
			}
			if (j % 3 == 1)
			{
				sum2 += yj;
				count2++;
			}
			if (j % 3 == 2)
			{
				sum1 += yj;
				count1++;
			}

		}
		y_obj[0] = R * cos(0.5*pi*x_var[0])*cos(0.5*pi*x_var[1]) + 2.0 * sum1 / (double)count1 + G;
		y_obj[1] = R * cos(0.5*pi*x_var[0])*sin(0.5*pi*x_var[1]) + 2.0 * sum2 / (double)count2 + G;
		y_obj[2] = R * sin(0.5*pi*x_var[0]) + 2.0 * sum3 / (double)count3 + G;

		return;

	}

	cout << "There is no function ........\n";
	assert(false);
}


//the random generator in [0,1)
double rnd_uni(long *idum)
{
	long j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB];
	double temp;

	if (*idum <= 0)
	{
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB + 7; j >= 0; j--)
		{
			k = (*idum) / IQ1;
			*idum = IA1 * (*idum - k * IQ1) - k * IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1;
	*idum = IA1 * (*idum - k * IQ1) - k * IR1;
	if (*idum < 0) *idum += IM1;
	k = idum2 / IQ2;
	idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
	if (idum2 < 0) idum2 += IM2;
	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp = AM * iy) > RNMX) return RNMX;
	else return temp;

}/*------End of rnd_uni()--------------------------*/

void testRandInSDP() {
	vector<vector<double> > randNum(30);
	vector<vector<int> > randObjNum(30);
	vector<vector<int> > randDimNum(30);
	vector<vector<int> > randVal(30);
	vector<vector<int> > randValue2(30);
	for (int j = 0; j < randNum.size(); ++j) {
		randNum[j].resize(31);
		randObjNum[j].resize(31);
		randValue2[j].resize(31);
		randDimNum[j].resize(31);
		randVal[j].resize(31);
	}
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
	int objCaseNum = 4;
	//vector<int> objNum(objCaseNum);
	//objNum[0] = 2; objNum[1] = 3; objNum[2] = 5; objNum[3] = 10;
	const int numOfDim = 10;
	int numOfObj = 2;
	int nt = 10;
	for (int i = 1; i <= 15; ++i) {
		for (int j = 0; j < 30; ++j) {
			initSDPSystem(proInsName[i], numOfObj, numOfDim);
			for (int k = 0; k < 31; ++k) {
				randNum[j][k] = getRnd_rtInSDP4();
				randObjNum[j][k] = getProObjNum();
				randDimNum[j][k] = getProDimNum();
				randValue2[j][k] = getPtInSDP15();
				randVal[j][k] = getDtInSDP15();
				changeToNextEnvi(proInsName[i], k + 1, nt);
			}

		}
		//
		for (int k = 0; k < 31; ++k) {
			for (int j = 1; j < 30; ++j) {
				if (randNum[j][k] != randNum[0][k]) assert(false);
				if (randObjNum[j][k] != randObjNum[0][k]) assert(false);
				if (randDimNum[j][k] != randDimNum[0][k]) assert(false);
				if (randValue2[j][k] != randValue2[0][k]) assert(false);
				if (randVal[j][k] != randVal[0][k]) assert(false);

			}
		}
	}
	cout << "Test the random number chaning list in SDP problems.\n";
	system("PAUSE");
}