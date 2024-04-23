#ifndef __OBJECTIVE_H_
#define __OBJECTIVE_H_

#include "global.h"
#include "SDP.h"
#include "DF.h"
#include <assert.h>

//#define pi   3.1415926
#define SQR2  sqrt(2)

void objectives(vector<double> x_var, vector <double> &y_obj, int mEvals, bool flag=false)
{
	int numVariables = DynPara::dimNum;
	int nvar = DynPara::dimNum;
	int nobj = DynPara::objNum;
	if (DynPara::test_SDP) {
		objective(DynPara::proName.c_str(), x_var, y_obj);
	}
	else if(DynPara::test_DF){
		y_obj = cec2018_DF_eval(DynPara::proName, x_var, mEvals, DynPara::taut, DynPara::nt);
	}
	return;

	if(strTestInstance == "ZDT1")
	{
		double g = 0;
		for(int n=1;n<numVariables;n++)
			g+= x_var[n];
		g = 1 + 9*g/(numVariables-1);

		y_obj[0] = x_var[0];
		y_obj[1] = g*(1 - sqrt(y_obj[0]/g));
	}


	if(strTestInstance == "ZDT2")
	{
		double g = 0;
		for(int n=1;n<numVariables;n++)
			g+= x_var[n];
		g = 1 + 9*g/(numVariables-1);
		y_obj[0] = x_var[0];
		y_obj[1] = g*(1 - pow(y_obj[0]/g,2));
	}


	
	if(strTestInstance == "ZDT3")
	{
		double g = 0;
		for(int n=1;n<numVariables;n++)
			g+= x_var[n];
		g = 1 + 9*g/(numVariables-1);

		y_obj[0] = x_var[0];
		y_obj[1] = g*(1 - sqrt(x_var[0]/g) - x_var[0]*sin(10*pi*x_var[0])/g);
	}


	if(strTestInstance == "ZDT4")
	{
		double g = 0;
		for(int n=1;n<numVariables;n++)
		{
			double x = 10*(x_var[n] - 0.5);
			g+= x*x - 10*cos(4*pi*x);
		}
		g = 1 + 10*(numVariables-1) + g;
		y_obj[0] = x_var[0];
		y_obj[1] = g*(1- sqrt(y_obj[0]/g));
	}

	if(strTestInstance == "ZDT6")
	{
		double g = 0;
		for(int n=1;n<numVariables;n++)
			g+= x_var[n]/(numVariables - 1);
		g = 1 + 9*pow(g,0.25) ;

		y_obj[0] = 1 - exp(-4*x_var[0])*pow(sin(6*pi*x_var[0]),6);
		y_obj[1] = g*(1- pow(y_obj[0]/g,2));
	}

	// OKA 1
	if(strTestInstance == "OKA-1")
	{
		double x1 = 2*pi*(x_var[0] - 0.5);
		double x2 = (x_var[1] - 0.5)*10;
		y_obj[0] = x1;
		y_obj[1] = pi - x1 + fabs(x2 - 5*cos(x1));
	}

	if(strTestInstance == "OKA-2")
	{
		double x1 = 2*pow(pi,3)*(x_var[0] - 0.5);
		double x2 = (x_var[1] - 0.5)*10;
		double eta;
		if(x1>=0) eta = pow(x1,1.0/3);
		else      eta = -pow(-x1,1.0/3);

		y_obj[0] = eta;
		y_obj[1] = pi - eta + fabs(x2 - 5*cos(x1));
	}

	if(strTestInstance == "DTLZ1")
	{
		double g = 0;
		for(int n=2; n<numVariables;n++)				
			g = g + pow(x_var[n]-0.5,2) - cos(20*pi*(x_var[n] - 0.5));
		g = 100*(numVariables- 2 + g);
		y_obj[0] = (1 + g)*x_var[0]*x_var[1];
		y_obj[1] = (1 + g)*x_var[0]*(1 - x_var[1]);
		y_obj[2] = (1 + g)*(1 - x_var[0]);
	}



	if(strTestInstance == "DTLZ2")
	{
		double g = 0;
		double xx = (x_var[0] + x_var[1])/2.0;
		for(int n=2; n<numVariables;n++)				
		{
			double x = 2*(x_var[n] - 0.5);
			g = g + x*x;;
		}
		y_obj[0] = (1 + g)*cos(x_var[0]*pi/2)*cos(x_var[1]*pi/2);
		y_obj[1] = (1 + g)*cos(x_var[0]*pi/2)*sin(x_var[1]*pi/2);
		y_obj[2] = (1 + g)*sin(x_var[0]*pi/2);
	}

	double G = sin(0.5*pi*Tt);
	// SDP PROBLEMS
	if (strTestInstance == "SDP1") //SDP1
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
		return;
	}

	if (strTestInstance == "SDP2") //SDP2
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

	if (strTestInstance == "SDP3") //SDP3
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

	if (strTestInstance == "SDP4") //SDP4
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

	if (strTestInstance == "SDP5") //SDP5
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

	if (strTestInstance == "SDP6") //SDP6
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

	if (strTestInstance == "SDP7") //SDP7
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

	if (strTestInstance == "SDP9") //SDP8
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

	if (strTestInstance == "SDP8") //SDP9
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

	if (strTestInstance == "SDP10") //SDP10
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

	if (strTestInstance == "SDP11") //SDP11
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

	if (strTestInstance == "SDP12") //SDP12 -change variables
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

	if (strTestInstance == "SDP13") //SDP13-change objectives
	{
		// *** this is for many objectives only.
		//nobj = nobj_l + floor(rnd_rt*(nobj_u - nobj_l));

		randVal = nobj;  // save this random value.

		double g = 0;
		//if (nvar != DynPara::dimNum || nvar != x_var.size()) 
		//{ cout << nvar << "\t" << DynPara::dimNum << "\t" << x_var.size() << endl; assert(false); }
		for (int i = nobj - 1; i < nvar; i ++)
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

	if (strTestInstance == "SDP14") //SDP14 -degenerate
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

	if (strTestInstance == "SDP15") //SDP15-degenerate
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

	if (strTestInstance == "FDA1")
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
	cout << "cannot find function objective: " << strTestInstance << endl;
}

void getOptimalSolution(string strTestInstance, const vector<double>& x, vector<double>& optS, int mEvas) {

	if (DynPara::test_SDP) {
		getOptimalSolution(strTestInstance, x, optS);
	}
	else if (DynPara::test_DF) {

		optS = get_cec2018_DF_best(strTestInstance, x, mEvas, DynPara::taut, DynPara::nt);
	}
	else {
		assert(false);
	}
}


#endif