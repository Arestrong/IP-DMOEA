#include <cstdlib>
#include "mine.h"
#include "cppmine.h"
#include <string>
using namespace std;


MINE::MINE(double alpha, double c, int est)
{
  char* ret;

  param.alpha = alpha;
  param.c = c;
  param.est = est;
  score = NULL;

  ret = mine_check_parameter(&param);

}


MINE::~MINE()
{
  mine_free_score(&score);
}


void MINE::compute_score(double *x, double *y, int n)
{
  prob.x = x;
  prob.y = y;
  prob.n = n;

  mine_free_score(&score);
  score = mine_compute_score(&prob, &param);
  string ret = "error in mine_compute_score()";
  if (score == NULL)
    throw ret;
}


double MINE::mic()
{
  string ret = "no score computed";
  if (score == NULL)
    throw ret;

  return mine_mic(score);
}


double MINE::mas()
{
  string ret = "no score computed";
  if (score == NULL)
    throw ret;

  return mine_mas(score);
}


double MINE::mev()
{
  string ret = "no score computed";
  if (score == NULL)
    throw ret;

  return mine_mev(score);
}


double MINE::mcn(double eps)
{
  string ret = "no score computed";
  if (score == NULL)
    throw ret;

  return mine_mcn(score, eps);
}


double MINE::mcn_general()
{
  string ret = "no score computed";
  if (score == NULL)
    throw ret;

  return mine_mcn_general(score);
}


double MINE::tic(int norm)
{
	string ret = "no score computed";
  if (score == NULL)
    throw ret;

  return mine_tic(score, norm);
}


double MINE::gmic(double p)
{
  string ret = "no score computed";
  if (score == NULL)
    throw ret;

  return mine_gmic(score, p);
}
