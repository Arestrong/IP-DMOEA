#include "random.h"

double random()
 {
 return rand() % 1000 / 1000.0;
}

double gaussian(double mean, double std) {
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if (phase == 0) {
		do {
			double U1 = (double)random();// rand() / RAND_MAX;
			double U2 = (double)random();// rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;
	return X * std + mean;
	//return mean * X + std;
}