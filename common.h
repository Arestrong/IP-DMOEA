#ifndef __COMMON_H_
#define __COMMON_H_


double distanceArray(double vec1[], double vec2[], int dim)
{
    double sum = 0;
	for(int n=0; n<dim; n++)
	    sum+= (vec1[n] - vec2[n])*(vec1[n] - vec2[n]);
	return sqrt(sum);
}

double distanceVector(vector <double> &vec1, vector <double> &vec2)
{
    double sum = 0;
	for(int n=0; n<vec1.size(); n++)
	    sum+=(vec1[n] - vec2[n])*(vec1[n] - vec2[n]);
	return sqrt(sum);
}


double norm_vector(vector <double> &x)
{
	double sum = 0;
	for(int i=0;i<x.size();i++)
        sum = sum + x[i]*x[i];
    return sqrt(sum);
}

double sum_vector(vector<double>&vec)
{
	double sum = 0;
	for(int i=0;i<vec.size();i++)
        sum = sum + vec[i];
    return sum;
}

double innerproduct(vector <double>&vec1, vector <double>&vec2)
{
    double sum = 0;
	for(int i=0; i<vec1.size(); i++)
		sum+= vec1[i]*vec2[i];
	return sum;
}

void minfastsort(double x[], int idx[], int n, int m)
{
    for(int i=0; i<m; i++)
	{
	    for(int j=i+1; j<n; j++)
			if(x[i]>x[j])
			{
			    double temp = x[i];
				x[i]        = x[j];
				x[j]        = temp;
				int id      = idx[i];
				idx[i]      = idx[j];
				idx[j]      = id;
			}
	}
}


#endif