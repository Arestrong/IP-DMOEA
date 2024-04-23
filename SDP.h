#pragma once
//dynamic multiobjective problem
//a scalable test suite for continuous dynamic multiobjective optimization
//IEEE Trans. cynernetics

#pragma once
#ifndef __SDP_H_
#define __SDP_H_

#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>

#define pi 3.141592653589793238462643383279502884197169399375105

/*------Constants for rnd_uni()--------------------------------------------*/

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-07
#define RNMX (1.0-EPS)

using namespace std;

static int randVal; // save random integer
static int randVal2;  // save random interger (used in SDP15)

static int nobj;
const int nobj_l = 2;
const int nobj_u = 5;

static int nvar;
const int nvar_l = 10;
const int nvar_u = 20;

static int seed = 177;
static long rnd_uni_init;

static double   rnd_rt, // random number between 0 and 1;
Tt,		// time instant;
sdp_y[2000]; // preserve PS of SDP1 every time step;

static int nt = 10, taut = 10, T0 = 50; //the severity, change generations, and the generations in the first environment
static int Parar = 0;//  // unchanged index within a time window, 0 at the beginning
static double itt = 0;  //the current generations 
static double max_gen = 500;

double rnd_uni(long *idum);
double minPeaksOfSDP7(double x, double pt);
double SafeCot(double x);
double SafeAcos(double x);
int changeEnvironments(string strTestInstance, int gen, int T0, int nt, int taut);
int changeEnviInNextFes(string strTestInstance, int curFes, int T0, int nt, int taut);
void initSDPSystem(string strTestInstance, int initObjNum, int initDimNum);
int getProDimNum();
int getProObjNum();
double getRnd_rtInSDP4();
int getPtInSDP15();
int getDtInSDP15();
double getTt();

//change the environment if the curFes is the first FEs in the new environment
//taut: the change frequency -- the number of generations in one environment
int changeToNextEnvi(string strTestInstance, int nextEnviIndex, int nt);
double SafeAcos(double x);
double SafeCot(double x);
double minPeaksOfSDP7(double x, double pt);
// the objective vector of SDP.
void objective(const char* strTestInstance, vector<double> &x_var, vector<double> &y_obj);
//the random generator in [0,1)
double rnd_uni(long *idum);
void testRandInSDP();

void getOptimalSolution(string strTestInstance, const vector<double> &x, vector<double> &optS);

#endif
