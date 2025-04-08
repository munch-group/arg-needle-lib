// spa_score_test.h
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct SA_NULL_Model {
    VectorXd y;
    VectorXd mu;
    VectorXd res;
    VectorXd V;
    MatrixXd X1;
    MatrixXd XV;
    MatrixXd XXVX_inv;
    bool converged;
};

struct RootResult {
    double root;
    int n_iter;
    bool converged;
};

struct SaddleResult {
    double p_value;
    double p_value_na;
    bool converged;
    double score;
};

// --- Function Declarations ---
MatrixXd PrepareX1Matrix(const MatrixXd& X1_input);
bool logistic_regression(const MatrixXd& X, const VectorXd& y, VectorXd& beta, VectorXd& mu, int max_iter = 25, double tol = 1e-8);
VectorXd firth_logistic_regression(const MatrixXd& X, const VectorXd& y);
SA_NULL_Model FitNullModel(const MatrixXd& X1_raw, const VectorXd& y);

// SPA Core

// K functions
double Korg(double t, const VectorXd& mu, const VectorXd& g);
double K1_adj(double t, const VectorXd& mu, const VectorXd& g, double q);
double K2(double t, const VectorXd& mu, const VectorXd& g);

// Root solver
RootResult GetRoot_K1(double init, const VectorXd& mu, const VectorXd& g, double q,
                      double tol = std::sqrt(std::numeric_limits<double>::epsilon()), int max_iter = 1000);

// Saddle prob
double normal_cdf(double z);
double Get_Saddle_Prob(double zeta, const VectorXd& mu, const VectorXd& g, double q, bool log_p = false);

// SPA wrapper
SaddleResult Saddle_Prob(double q,
                         const VectorXd& mu,
                         const VectorXd& g,
                         double cutoff = 2.0,
                         double alpha = 5e-8,
                         bool log_p = false);

SaddleResult TestSPA(const VectorXd& G, const SA_NULL_Model& null_model,
                     double cutoff = 2.0, double alpha = 5e-8, bool log_p = false);
