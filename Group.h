// Group.h
#pragma once
#include "Global.h"

class Group{
public:
	CSolution center;
	CSolution best;
	vector<CSolution> members;
	int ID;
	int numbers;
	double radius;
public:
	Group(const int num);
	Group();
	~Group();
	void Initial(const CSolution &p,const int id);
	void Initial(const int num,const int id);
	void Initial(const Group &g );
	bool operator ==(const Group &g);
	void Merge(const Group &g);
	Group & operator=(const Group &g);
	void Calculate_Radius();
};


