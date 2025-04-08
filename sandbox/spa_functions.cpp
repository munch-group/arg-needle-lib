#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>

using namespace Eigen;

// Function to calculate the rank of a matrix
int matrixRank(const MatrixXd& mat) {
    JacobiSVD<MatrixXd> svd(mat);
    double tol = std::numeric_limits<double>::epsilon() * std::max(mat.rows(), mat.cols()) * svd.singularValues().array().abs()(0);
    return (svd.singularValues().array() > tol).count();
}

// Function to implement ScoreTest_wSaddleApprox_Get_X1
MatrixXd ScoreTest_wSaddleApprox_Get_X1(MatrixXd X1) {
    int q1 = X1.cols();

    if (q1 >= 2) {
        if ((X1.col(0) - X1.col(1)).array().abs().sum() == 0) {
            X1 = X1.block(0, 1, X1.rows(), q1 - 1); // Remove the second column
            q1 -= 1;
        }
    }

    int rank = matrixRank(X1);
    if (rank < q1) {
        JacobiSVD<MatrixXd> svd(X1, ComputeThinU | ComputeThinV);
        X1 = svd.matrixU().leftCols(rank);
    }

    return X1;
}

#include <RcppEigen.h>
#include <iostream>
#include <cmath>

using namespace Eigen;

// Function to calculate the logistic regression fitted values
VectorXd logisticRegression(const MatrixXd& X, const VectorXd& y) {
    int maxIter = 100; // Maximum number of iterations
    double tol = 1e-6; // Convergence tolerance
    VectorXd beta = VectorXd::Zero(X.cols());
    VectorXd eta = X * beta;
    VectorXd mu = 1 / (1 + (-eta).array().exp());
    VectorXd W = mu.array() * (1 - mu.array());
    MatrixXd XtW = X.transpose() * W.asDiagonal();
    MatrixXd XtWX = XtW * X;
    VectorXd z = eta + (y - mu).array() / W.array();

    for (int iter = 0; iter < maxIter; ++iter) {
        VectorXd betaNew = XtWX.ldlt().solve(XtW * z);
        if ((betaNew - beta).norm() < tol) {
            beta = betaNew;
            break;
        }
        beta = betaNew;
        eta = X * beta;
        mu = 1 / (1 + (-eta).array().exp());
        W = mu.array() * (1 - mu.array());
        XtW = X.transpose() * W.asDiagonal();
        XtWX = XtW * X;
        z = eta + (y - mu).array() / W.array();
    }

    return mu;
}

// Function to implement ScoreTest_wSaddleApprox_NULL_Model
// [[Rcpp::export]]
Rcpp::List ScoreTest_wSaddleApprox_NULL_Model(const Eigen::MatrixXd& X1, const Eigen::VectorXd& y) {
    // Perform logistic regression
    VectorXd mu = logisticRegression(X1, y);

    // Check convergence
    double meanMu = mu.mean();
    double meanY = y.mean();
    bool convflag = (meanMu / meanY > 0.001) && ((1 - meanMu) / (1 - meanY) > 0.001);

    if (!convflag) {
        Rcpp::stop("Null model did not converge properly.");
    }

    // Compute residuals and variance
    VectorXd res = y - mu;
    VectorXd V = mu.array() * (1 - mu.array());

    // Compute XV and XXVX_inv
    MatrixXd XV = X1.transpose() * V.asDiagonal();
    MatrixXd XVX_inv = (XV * X1).inverse();
    MatrixXd XXVX_inv = X1 * XVX_inv;

    // Return results as a list
    return Rcpp::List::create(
        Rcpp::Named("y") = y,
        Rcpp::Named("mu") = mu,
        Rcpp::Named("res") = res,
        Rcpp::Named("V") = V,
        Rcpp::Named("X1") = X1,
        Rcpp::Named("XV") = XV,
        Rcpp::Named("XXVX_inv") = XXVX_inv
    );
}

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Function to compute log(1 + exp(x)) safely
double log1pexp(double x) {
    if (x > 0) {
        return x + log1p(exp(-x));
    } else {
        return log1p(exp(x));
    }
}

// Function to compute add_logp
double add_logp(double p1, double p2) {
    p1 = -abs(p1);
    p2 = -abs(p2);
    double maxp = max(p1, p2);
    double minp = min(p1, p2);
    return maxp + log1pexp(minp - maxp);
}

// Function to compute Korg
vector<double> Korg(const vector<double>& t, const vector<double>& mu, const vector<double>& g) {
    size_t n_t = t.size();
    vector<double> out(n_t, 0.0);

    for (size_t i = 0; i < n_t; ++i) {
        double t1 = t[i];
        vector<double> temp(mu.size());
        for (size_t j = 0; j < mu.size(); ++j) {
            temp[j] = log(1 - mu[j] + mu[j] * exp(g[j] * t1));
        }
        out[i] = accumulate(temp.begin(), temp.end(), 0.0);
    }
    return out;
}

// Function to compute K1_adj
vector<double> K1_adj(const vector<double>& t, const vector<double>& mu, const vector<double>& g, double q) {
    size_t n_t = t.size();
    vector<double> out(n_t, 0.0);

    for (size_t i = 0; i < n_t; ++i) {
        double t1 = t[i];
        vector<double> temp1(mu.size());
        vector<double> temp2(mu.size());
        for (size_t j = 0; j < mu.size(); ++j) {
            temp1[j] = (1 - mu[j]) * exp(-g[j] * t1) + mu[j];
            temp2[j] = mu[j] * g[j];
        }
        double sum_temp = 0.0;
        for (size_t j = 0; j < mu.size(); ++j) {
            sum_temp += temp2[j] / temp1[j];
        }
        out[i] = sum_temp - q;
    }
    return out;
}

// Function to compute K2
vector<double> K2(const vector<double>& t, const vector<double>& mu, const vector<double>& g) {
    size_t n_t = t.size();
    vector<double> out(n_t, 0.0);

    for (size_t i = 0; i < n_t; ++i) {
        double t1 = t[i];
        vector<double> temp1(mu.size());
        vector<double> temp2(mu.size());
        for (size_t j = 0; j < mu.size(); ++j) {
            temp1[j] = pow((1 - mu[j]) * exp(-g[j] * t1) + mu[j], 2);
            temp2[j] = (1 - mu[j]) * mu[j] * pow(g[j], 2) * exp(-g[j] * t1);
        }
        double sum_temp = 0.0;
        for (size_t j = 0; j < mu.size(); ++j) {
            sum_temp += temp2[j] / temp1[j];
        }
        out[i] = sum_temp;
    }
    return out;
}

// Example usage
int main() {
    vector<double> t = {0.1, 0.2, 0.3};
    vector<double> mu = {0.5, 0.6, 0.7};
    vector<double> g = {1.0, 1.5, 2.0};
    double q = 1.0;

    // Test add_logp
    double p1 = -2.0, p2 = -3.0;
    cout << "add_logp: " << add_logp(p1, p2) << endl;

    // Test Korg
    vector<double> korg_result = Korg(t, mu, g);
    cout << "Korg: ";
    for (double val : korg_result) {
        cout << val << " ";
    }
    cout << endl;

    // Test K1_adj
    vector<double> k1_adj_result = K1_adj(t, mu, g, q);
    cout << "K1_adj: ";
    for (double val : k1_adj_result) {
        cout << val << " ";
    }
    cout << endl;

    // Test K2
    vector<double> k2_result = K2(t, mu, g);
    cout << "K2: ";
    for (double val : k2_result) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Function to compute Get_Normal_Spline
MatrixXd Get_Normal_Spline(double var, const VectorXd& nodes) {
    VectorXd y1 = var * nodes;
    VectorXd y2 = VectorXd::Constant(nodes.size(), var);
    MatrixXd result(nodes.size(), 2);
    result.col(0) = y1;
    result.col(1) = y2;
    return result;
}

// Function to compute Get_Saddle_Spline
MatrixXd Get_Saddle_Spline(const VectorXd& mu, const VectorXd& g, const VectorXd& nodes, 
                           const function<VectorXd(const VectorXd&, const VectorXd&, const VectorXd&, double)>& K1_adj,
                           const function<VectorXd(const VectorXd&, const VectorXd&, const VectorXd&)>& K2) {
    double m1 = (mu.array() * g.array()).sum();
    VectorXd y1 = K1_adj(nodes, mu, g, 0.0) - m1;
    VectorXd y2 = K2(nodes, mu, g);
    MatrixXd result(nodes.size(), 3);
    result.col(0) = nodes;
    result.col(1) = y1;
    result.col(2) = y2;
    return result;
}

// Function to compute getroot_K1
struct RootResult {
    double root;
    int n_iter;
    bool is_converge;
};

RootResult getroot_K1(double init, const VectorXd& mu, const VectorXd& g, double q, double m1, 
                      double tol = sqrt(numeric_limits<double>::epsilon()), int maxiter = 1000,
                      const function<VectorXd(const VectorXd&, const VectorXd&, const VectorXd&, double)>& K1_adj,
                      const function<VectorXd(const VectorXd&, const VectorXd&, const VectorXd&)>& K2) {
    double g_pos = g.array().maxCoeff();
    double g_neg = g.array().minCoeff();
    if (q >= g_pos || q <= g_neg) {
        return {numeric_limits<double>::infinity(), 0, true};
    }

    double t = init;
    double K1_eval = K1_adj(VectorXd::Constant(1, t), mu, g, q)(0);
    double prev_jump = numeric_limits<double>::infinity();
    int rep = 1;
    bool conv = false;

    while (true) {
        double K2_eval = K2(VectorXd::Constant(1, t), mu, g)(0);
        double tnew = t - K1_eval / K2_eval;

        if (isnan(tnew)) {
            conv = false;
            break;
        }
        if (abs(tnew - t) < tol) {
            conv = true;
            break;
        }
        if (rep == maxiter) {
            conv = false;
            break;
        }

        double newK1 = K1_adj(VectorXd::Constant(1, tnew), mu, g, q)(0);
        if (signbit(K1_eval) != signbit(newK1)) {
            if (abs(tnew - t) > prev_jump - tol) {
                tnew = t + copysign(prev_jump / 2, newK1 - K1_eval);
                newK1 = K1_adj(VectorXd::Constant(1, tnew), mu, g, q)(0);
                prev_jump /= 2;
            } else {
                prev_jump = abs(tnew - t);
            }
        }

        rep++;
        t = tnew;
        K1_eval = newK1;
    }

    return {t, rep, conv};
}

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <stdexcept>

using namespace Eigen;
using namespace std;

// Helper function for safe log(1 + exp(x))
double log1pexp(double x) {
    if (x > 0) {
        return x + log1p(exp(-x));
    } else {
        return log1p(exp(x));
    }
}

// Function to compute the normal cumulative distribution function (CDF)
double pnorm(double x, bool lower_tail = true, bool log_p = false) {
    double p = 0.5 * erfc(-x * M_SQRT1_2);
    if (!lower_tail) {
        p = 1.0 - p;
    }
    if (log_p) {
        p = log(p);
    }
    return p;
}

// Function to compute Get_Saddle_Prob
double Get_Saddle_Prob(double zeta, const VectorXd& mu, const VectorXd& g, double q, bool log_p = false) {
    // Compute k1 and k2
    double k1 = (mu.array() * (1 - mu.array()) * g.array().square() * exp(g.array() * zeta)).sum();
    double k2 = (mu.array() * g.array().square() * exp(g.array() * zeta)).sum();

    if (isfinite(k1) && isfinite(k2)) {
        double temp1 = zeta * q - k1;
        double w = copysign(sqrt(2 * temp1), zeta);
        double v = zeta * sqrt(k2);
        double Z_test = w + (1 / w) * log(v / w);

        if (Z_test > 0) {
            return pnorm(Z_test, false, log_p);
        } else {
            return -pnorm(Z_test, true, log_p);
        }
    } else {
        return log_p ? -INFINITY : 0.0;
    }
}

// Function to compute Saddle_Prob
struct SaddleProbResult {
    double p_value;
    double p_value_NA;
    bool is_converge;
    double score;
};

SaddleProbResult Saddle_Prob(double q, const VectorXd& mu, const VectorXd& g, double Cutoff = 2, double alpha = 0.05, bool log_p = false) {
    double m1 = (mu.array() * g.array()).sum();
    double var1 = (mu.array() * (1 - mu.array()) * g.array().square()).sum();
    double Score = q - m1;
    double qinv = -copysign(abs(q - m1), q - m1) + m1;

    // Compute p-value without adjustment
    double pval_noadj = pnorm((q - m1) / sqrt(var1), false, log_p);
    bool is_converge = true;

    if (abs(q - m1) / sqrt(var1) < Cutoff) {
        return {pval_noadj, pval_noadj, is_converge, Score};
    } else {
        // Root-finding for Saddle Approximation
        double root1 = 0.0; // Placeholder for root-finding logic
        double root2 = 0.0; // Placeholder for root-finding logic

        double p1 = Get_Saddle_Prob(root1, mu, g, q, log_p);
        double p2 = Get_Saddle_Prob(root2, mu, g, qinv, log_p);

        double pval = log_p ? log1pexp(p1 + p2) : abs(p1) + abs(p2);

        if (pval != 0 && pval_noadj / pval > 1e3) {
            return Saddle_Prob(q, mu, g, Cutoff * 2, alpha, log_p);
        } else {
            return {pval, pval_noadj, is_converge, Score};
        }
    }
}

// Example usage
int main() {
    // Example inputs
    VectorXd mu(3);
    mu << 0.5, 0.6, 0.7;
    VectorXd g(3);
    g << 1.0, 1.5, 2.0;
    double q = 1.0;

    // Test Get_Saddle_Prob
    double pval = Get_Saddle_Prob(0.5, mu, g, q, false);
    cout << "Get_Saddle_Prob: " << pval << endl;

    // Test Saddle_Prob
    SaddleProbResult result = Saddle_Prob(q, mu, g);
    cout << "Saddle_Prob:\n"
         << "p_value: " << result.p_value << "\n"
         << "p_value_NA: " << result.p_value_NA << "\n"
         << "is_converge: " << (result.is_converge ? "true" : "false") << "\n"
         << "score: " << result.score << endl;

    return 0;
}

// Example usage
int main() {
    // Example inputs
    VectorXd mu(3);
    mu << 0.5, 0.6, 0.7;
    VectorXd g(3);
    g << 1.0, 1.5, 2.0;
    VectorXd nodes(3);
    nodes << -1.0, 0.0, 1.0;

    // Example K1_adj and K2 functions
    auto K1_adj = [](const VectorXd& t, const VectorXd& mu, const VectorXd& g, double q) -> VectorXd {
        VectorXd result(t.size());
        for (int i = 0; i < t.size(); ++i) {
            double t1 = t(i);
            VectorXd temp1 = (1 - mu.array()) * exp(-g.array() * t1) + mu.array();
            VectorXd temp2 = mu.array() * g.array();
            result(i) = (temp2.array() / temp1.array()).sum() - q;
        }
        return result;
    };

    auto K2 = [](const VectorXd& t, const VectorXd& mu, const VectorXd& g) -> VectorXd {
        VectorXd result(t.size());
        for (int i = 0; i < t.size(); ++i) {
            double t1 = t(i);
            VectorXd temp1 = ((1 - mu.array()) * exp(-g.array() * t1) + mu.array()).square();
            VectorXd temp2 = (1 - mu.array()) * mu.array() * g.array().square() * exp(-g.array() * t1);
            result(i) = (temp2.array() / temp1.array()).sum();
        }
        return result;
    };

    // Test Get_Normal_Spline
    MatrixXd normal_spline = Get_Normal_Spline(2.0, nodes);
    cout << "Get_Normal_Spline:\n" << normal_spline << endl;

    // Test Get_Saddle_Spline
    MatrixXd saddle_spline = Get_Saddle_Spline(mu, g, nodes, K1_adj, K2);
    cout << "Get_Saddle_Spline:\n" << saddle_spline << endl;

    // Test getroot_K1
    RootResult root_result = getroot_K1(0.0, mu, g, 1.0, 0.0, 1e-6, 1000, K1_adj, K2);
    cout << "getroot_K1:\n"
         << "Root: " << root_result.root << "\n"
         << "Iterations: " << root_result.n_iter << "\n"
         << "Converged: " << (root_result.is_converge ? "Yes" : "No") << endl;

    return 0;
}

// Example usage
int main() {
    // Example input matrix
    MatrixXd X1(4, 3);
    X1 << 1, 2, 3,
          4, 5, 6,
          7, 8, 9,
          10, 11, 12;

    std::cout << "Original Matrix X1:\n" << X1 << "\n";

    MatrixXd result = ScoreTest_wSaddleApprox_Get_X1(X1);

    std::cout << "Processed Matrix X1:\n" << result << "\n";

    return 0;
}