#pragma once
#ifndef _HV_H_
#define _HV_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
using namespace std;

typedef double OBJECTIVE;

typedef struct
{
	OBJECTIVE *objectives;
} OBJPOINT;

typedef struct
{
	int nPoints;
	int n;
	OBJPOINT *points;
} FRONT;

typedef struct
{
	int nFronts;
	FRONT *fronts;
} FILECONTENTS;

static void trimLine(char line[]);
void printContents(FILECONTENTS *f);
int gcmp(const void *v1, const void *v2);
int greaterabbrev(const void *v1, const void *v2);
int dominates2way(OBJPOINT p, OBJPOINT q, int k);
bool dominates1way(OBJPOINT p, OBJPOINT q, int k);
void makeDominatedBit(FRONT ps, int p);
double hv2(FRONT ps, int k);
double inclhv(OBJPOINT p);
double inclhv2(OBJPOINT p, OBJPOINT q);
double inclhv3(OBJPOINT p, OBJPOINT q, OBJPOINT r);
double inclhv4(OBJPOINT p, OBJPOINT q, OBJPOINT r, OBJPOINT s);
double exclhv(FRONT ps, int p);
double hypervolumn(FRONT ps);

FILECONTENTS* readFile(const vector<vector<double> > &archive, const int M);
double hypervolume(const vector<vector<double> > &archive, int objNum, vector<double> referPoint);

#endif
