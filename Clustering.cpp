
#include "Clustering.h"
#include "Global.h"
// implementation of Clustering class
Cluster::Cluster():initial_numbers(0){
	numbers=0;
	group.clear();
	dis.clear();
}
Cluster::Cluster(const vector<CSolution> p, const int num):initial_numbers(num){
	numbers=num;
	group.resize(numbers);
	for(int i=0;i<numbers;i++)
		group[i].Initial(p[i],i);
	dis.resize(initial_numbers);
	for(int i=0;i<initial_numbers;i++)
		dis.resize(initial_numbers);
	Calculate_Dis_Matrix();
}
void Cluster::Initial( const vector<CSolution> &p,const int num){
	Clear();

	initial_numbers=num;
	numbers=num;
	group.resize(numbers);
	for(int i=0;i<numbers;i++)
		group[i].Initial(p[i],i);
	dis.resize(initial_numbers);
	for(int i=0;i<initial_numbers;i++)
		dis[i].resize(initial_numbers);
	Calculate_Dis_Matrix();
}
void Cluster::Initial(const Cluster &clst){
	if(clst.initial_numbers==0) return;

	initial_numbers=clst.initial_numbers;
	numbers=clst.numbers;
	group.resize(numbers);
	for(int i=0;i<numbers;i++)
		group[i].Initial(clst.group[i]);
	dis.resize(initial_numbers);
	for(int i=0;i<initial_numbers;i++)
		dis.resize(initial_numbers);
	for(int i=0;i<initial_numbers;i++)
		for(int j=0;j<initial_numbers;j++)
			dis[i][j]=clst.dis[i][j];
}
Cluster::~Cluster(){
	Clear();
}
void Cluster::Calculate_Dis_Matrix(){
	// only called by constructor
	if (dis.size() != numbers) {
		dis.resize(numbers);
		for (int j = 0; j < numbers; ++j)
			dis[j].resize(numbers);
	}
	for(int i=0;i<numbers;i++){
		dis[i][i]=-1; //infinite great
		for(int j=0;j<i;j++)
			dis[i][j]=dis[j][i]= Group_Dis(i, j); //group[i].center.Distance(group[j].center);
	}

}
void Cluster::Refine_Clustering(){
	while(1){
		int i=0;
		while(i<numbers&&group[i].numbers>1) i++;
		if(i==numbers) break;
		int g1,g2;
		if(!Nearest_Group(g1,g2,1)) break;
		group[g2].Merge(group[g1]);
		Delete_Group(g1);
	}
	for(int i=0;i<numbers;i++) group[i].Calculate_Radius();
}
void Cluster::Rough_Clustering(int final_num){
	//cout << "Rough_Clustering " << numbers << "\t" << group.size() << "\n";
	int init_num = numbers;
	for (int j = 0; j < group.size(); ++j) {
		if (group[j].numbers != 1) { cout << group[j].numbers << "\t"; assert(false); }
		if (group[j].members.size() != 1) { cout << group[j].members.size() << "\t";  assert(false); }
	}
	while(numbers > final_num){
		//int i=0;
		//while(i<numbers&&group[i].numbers>1) i++;
		//if(i==numbers) break;
		int g1,g2;
		Nearest_Group(g1, g2, 0);
		//if(!) break;
		//cout << final_num << "\t" << numbers << "\t" << g1 << "\t" << g2 << "\t" << group[g2].numbers << "\t" << group[g1].numbers << "\t->" << group[g2].numbers + group[g1].numbers << "\t";
		group[g2].Merge(group[g1]);
		//cout << group[g2].numbers << "\t";
		Delete_Group(g1);
		Calculate_Dis_Matrix();
		//cout << numbers << endl;
		/*
		cout << numbers  << endl;
		int count = 0;
		int mcount = 0;
		for (int j = 0; j < numbers; ++j) {
			count += group[j].numbers;
			mcount += group[j].members.size();
		}
		if (count != init_num || mcount != init_num) { cout << count << "\t" << mcount << endl; assert(false); }
		*/
	}
	for(int i=0;i<numbers;i++) group[i].Calculate_Radius();
}
void Cluster::Delete_Group(const int id){
	//cout << id << "(" << group[id].numbers << ")";
	int num=numbers-1;
	if(num<1){
		Clear();
		return;
	}
	//cout << "*" << id << "*\t" << group[id].numbers << "\t";
	group.erase(group.begin() + id);
	//for (int j = 0; j < group.size(); ++j) {
	//	cout << group[j].numbers << ",";
	//}
	//cout << "(" << group[id].numbers << ")";
	/*
	vector<Group> g(num);
	for(int i=0,j=0;i<num;i++,j++){
		if(j!=id){ 
			g[i].Initial(group[j].numbers,i);
			g[i]=group[j];
		}else
			i--;
	}
	//delete [] group;
	group=g;
	*/

	numbers--;
	if (group.size() != numbers) assert(false);
	
}

double Cluster::Group_Dis(const int from,const int to){
	double dist = 0;
	int num_obj = DynPara::objNum;
	double a = 0, b = 0, c = 0;
	for (int j = 0; j < num_obj; ++j) {
		a += group[from].center.f[j] * group[to].center.f[j];
		b += group[from].center.f[j] * group[from].center.f[j];
		c += group[to].center.f[j] * group[to].center.f[j];
	}
	dist = acos(a / (sqrt(b)*sqrt(c)));
	return dist;
	//return dis[group[from].center.index][group[to].center.index];
}
double Cluster::Group_dist(const Group &g1, const Group &g2) {
	double dist = 0;
	int num_obj = DynPara::objNum;
	double a = 0, b = 0, c = 0;
	for (int j = 0; j < num_obj; ++j) {
		a += g1.center.f[j] * g2.center.f[j];
		b += g1.center.f[j] * g1.center.f[j];
		c += g2.center.f[j] * g2.center.f[j];
	}
	dist = acos(a / (sqrt(b)*sqrt(c)));
	return dist;
}
bool Cluster::Nearest_Group(int & g1, int & g2,bool flag){
	double Min_dis = -1,dist;
	if(numbers==1) {
		g1=0;
		g2=0;
		return false;
	}
	bool flag_fail=true;
	bool first = true;
	//Min_dis=sqrt((Global::boundary.upper-Global::boundary.lower)*(Global::boundary.upper-Global::boundary.lower)*Global::num_dim);
	
	if(flag==1){// Refinement clustering
		for(int i=0;i<numbers;i++){
			for(int j=0;j<numbers;j++){
				if(j==i) continue;
				//if(group[i].numbers>=CSwarm::max_local_popsize&&group[j].numbers>=CSwarm::max_local_popsize) continue;
				dist = Group_Dis(i, j);

				if(first || Min_dis>dist){
						Min_dis=dist;
						g1=i;
						g2=j;
					   flag_fail=false;
					   first = false;
				}
				
			}
		}
	}else{// Rough Clustering
		for(int i=0;i<numbers;i++){
			//if(group[i].numbers>1) continue;// can't merge two groups whose numbers are both greater than 1
			for(int j=0;j<numbers;j++){
					if(j==i) continue;
					//if(group[i].numbers>=CSwarm::max_local_popsize&&group[j].numbers>=CSwarm::max_local_popsize) continue;
					dist=Group_Dis(i,j);
					if(first || Min_dis>dist){
						Min_dis=dist;
						g1=i;
						g2=j;
						flag_fail=false;
						first = false;
					}			
			}
		}
	}
	if(flag_fail) return false;
	else return true;
}
void Cluster::Clear(){
	if(numbers>0){
		group.clear();
		dis.clear();
		numbers=0;
		initial_numbers=0;
	}	
}
Cluster &Cluster::operator =(const Cluster &clst){
	if(clst.initial_numbers==0) return *this;
	Clear();
	Initial(clst);	
	return *this;
}


