

#include "Group.h"
#include "Global.h"

// implementation of Group class
Group::Group(){

}

Group::Group(const int num){
	numbers=num;
	members.resize(numbers);
}
Group::~Group(){
//delete [] members;
numbers=0;
}
void Group::Initial(const CSolution &p,const int id){
	numbers=1;
	members.resize(numbers);
	members[0]=p; 
	center=p;
	best=p;
	ID=id;
	radius=0;
}
void Group::Initial(const int num,const int id){
	numbers=num;
	members.resize(numbers);
	ID=id;
}
void Group::Initial(const Group &g ){
	numbers=g.numbers;
	members.resize(numbers);
	for(int i=0;i<numbers;i++)
		members[i]=g.members[i];
	center=g.center;
	ID=g.ID;
	best=g.best;
	radius=g.radius;
}
void Group::Merge(const Group &g){
	//合并两个组
	//if (g.members.size() != g.numbers) { cout << g.members.size() << "\t" << g.numbers << endl; assert(false); }
	//if (numbers != members.size()) { cout << members.size() << "\t" << numbers << endl; assert(false); }
	for (int j = 0; j < g.numbers; ++j) {
		members.push_back(g.members[j]);
	}
	numbers = members.size();
	/*
	int t_num=numbers+g.numbers;
	vector<CSolution> t_mem(t_num);
	for(int i=0;i<numbers;i++)
		t_mem[i]=members[i];
	for(int i=0;i<g.numbers;i++)
		t_mem[i+numbers]=g.members[i];
	numbers=t_num;
	//delete[] members;
	members=t_mem;
	*/
	double converd = 0, cc = 0;
	for (int j = 0; j < g.best.f.size(); ++j) {
		converd += g.best.f[j];
		cc += best.f[j];
	}
	

	if(cc > converd) best=g.best;
	ID=ID<g.ID?ID:g.ID;
	Calculate_Radius();
}
void Group::Calculate_Radius(){

	radius=0;
	if(numbers<2) return;
	int num_dim = members[0].x.size();
	vector<double> t(num_dim);
	for(int i=0;i< num_dim;i++)
		t[i]=0;
	for(int i=0;i< num_dim;i++){
		for(int j=0;j<numbers;j++)
			t[i]+=members[j].x[i];
		t[i]/=numbers;
	}
	center.x = t;
	num_dim = members[0].f.size();
	center.f.resize(num_dim);
	for (int i = 0; i < num_dim; ++i) {
		center.f[i] = 0;
		for (int j = 0; j < numbers; j++) {
			center.f[i] += members[j].f[i];
		}
		center.f[i] /= numbers;
	}
		//if(members[i].ID!=center.ID) radius+=center.pbest.Distance(members[i].pbest);
	//radius=radius/(numbers-1);
}
bool Group::operator ==(const Group &g){

	if(numbers!=g.numbers) return false;
	for(int i=0;i<numbers;i++){
		int j=0;
		while(j<numbers)j++;
		if(j==numbers) return false;

	}
	return true;
}
Group &Group::operator =(const Group &g){

	//if(this==&g) return *this;
	//if(numbers!=g.numbers) return *this;
	members = g.members;
	center=g.center;
	radius=g.radius;
	best=g.best;
	ID=g.ID;
	numbers = g.numbers;
	return *this;
}
