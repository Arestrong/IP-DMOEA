// Clustering.h

#pragma once
#include "Global.h"
#include "Group.h"

class Cluster{
public:
	vector<Group> group;
	int numbers;
	int initial_numbers;
	vector<vector<double> > dis;
public:
	Cluster();
	Cluster(vector<CSolution> p,const int num);
	void Initial(const Cluster &clst);
	void Initial(const vector<CSolution> &p,const int num);
	~Cluster();
	void Clear();
	Cluster &operator =(const Cluster &clst);
	void Calculate_Dis_Matrix();
	void Refine_Clustering();
	void Rough_Clustering(int final_num);
	double Group_Dis(const int from,const int to);
	double Group_dist(const Group &a, const Group &b);
	bool Nearest_Group(int & g1, int & g2,bool flag);
	void Delete_Group(const int id);

};


