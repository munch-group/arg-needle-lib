// spa_score_test.cpp
#include "spa_score_test.h"

// --- Matrix Prep ---
MatrixXd PrepareX1Matrix(const MatrixXd& X1_input) {
    MatrixXd X1 = X1_input;
    int q1 = X1.cols();
    if (q1 >= 2 && (X1.col(0) - X1.col(1)).cwiseAbs().sum() == 0) {
        MatrixXd X1_new(X1.rows(), q1 - 1);
        X1_new << X1.col(0), X1.rightCols(q1 - 2);
        X1 = X1_new;
        q1--;
    }
    Eigen::ColPivHouseholderQR<MatrixXd> qr(X1);
    int rank = qr.rank();
    if (rank < q1) {
        Eigen::BDCSVD<MatrixXd> svd(X1, Eigen::ComputeThinU);
        X1 = svd.matrixU().leftCols(rank);
    }
    return X1;
}

// --- Logistic Regression ---
bool logistic_regression(const MatrixXd& X, const VectorXd& y, VectorXd& beta, VectorXd& mu, int max_iter, double tol) {
    int p = X.cols();
    beta = VectorXd::Zero(p);
    mu = 1.0 / (1.0 + (-X * beta).array().exp());
    for (int i = 0; i < max_iter; ++i) {
        VectorXd W_diag = mu.array() * (1 - mu.array());
        MatrixXd W = W_diag.asDiagonal();
        MatrixXd XTWX = X.transpose() * W * X;
        VectorXd grad = X.transpose() * (y - mu);
        VectorXd delta = XTWX.ldlt().solve(grad);
        beta += delta;
        mu = 1.0 / (1.0 + (-X * beta).array().exp());
        if (delta.norm() < tol) return true;
    }
    return false;
}

VectorXd firth_logistic_regression(const MatrixXd& X, const VectorXd& y) {
    VectorXd beta, mu;
    logistic_regression(X, y, beta, mu);
    return beta;
}

SA_NULL_Model FitNullModel(const MatrixXd& X1_raw, const VectorXd& y) {
    SA_NULL_Model result;
    MatrixXd X1 = PrepareX1Matrix(X1_raw);
    VectorXd beta, mu;
    bool converged = logistic_regression(X1, y, beta, mu);

    if (converged) {
        double r1 = mu.mean() / y.mean();
        double r2 = (1 - mu.mean()) / (1 - y.mean());
        if (r1 < 0.001 || r2 < 0.001) converged = false;
    }

    if (!converged) {
        beta = firth_logistic_regression(X1, y);
        mu = 1.0 / (1.0 + (-X1 * beta).array().exp());
    }

    VectorXd res = y - mu;
    VectorXd V = mu.array() * (1.0 - mu.array());
    MatrixXd XV = (X1.array().colwise() * V.array()).transpose();
    MatrixXd XVX = X1.transpose() * (X1.array().colwise() * V.array());
    MatrixXd XVX_inv = XVX.completeOrthogonalDecomposition().pseudoInverse();
    MatrixXd XXVX_inv = X1 * XVX_inv;

    result.X1 = X1;
    result.y = y;
    result.mu = mu;
    result.res = res;
    result.V = V;
    result.XV = XV;
    result.XXVX_inv = XXVX_inv;
    result.converged = true;
    return result;
}

// --- Kernels ---
double Korg(double t, const VectorXd& mu, const VectorXd& g) {
    return ((1 - mu.array()) + mu.array() * (g.array() * t).exp()).array().log().sum();
}

double K1_adj(double t, const VectorXd& mu, const VectorXd& g, double q) {
    VectorXd num = mu.array() * g.array();
    VectorXd denom = (1 - mu.array()) * (-g.array() * t).exp() + mu.array();
    return (num.array() / denom.array()).sum() - q;
}

double K2(double t, const VectorXd& mu, const VectorXd& g) {
    VectorXd exp_neg = (-g.array() * t).exp();
    VectorXd denom = ((1 - mu.array()) * exp_neg + mu.array()).square();
    VectorXd numer = (1 - mu.array()) * mu.array() * g.array().square() * exp_neg;
    return (numer.array() / denom.array()).sum();
}

RootResult GetRoot_K1(double init, const VectorXd& mu, const VectorXd& g, double q,
                      double tol, int max_iter) {
    double g_pos = g.array().max(0.0).sum();
    double g_neg = g.array().min(0.0).sum();
    if (q >= g_pos || q <= g_neg) return {std::numeric_limits<double>::infinity(), 0, true};

    double t = init, prev_jump = std::numeric_limits<double>::infinity();
    int rep = 1;
    bool converged = false;
    double K1_eval = K1_adj(t, mu, g, q);

    while (rep <= max_iter) {
        double K2_eval = K2(t, mu, g);
        double t_new = t - K1_eval / K2_eval;
        if (!std::isfinite(t_new)) return {NAN, rep, false};
        if (std::abs(t_new - t) < tol) {
            converged = true;
            t = t_new;
            break;
        }
        double newK1 = K1_adj(t_new, mu, g, q);
        if (std::signbit(K1_eval) != std::signbit(newK1)) {
            if (std::abs(t_new - t) > prev_jump - tol) {
                t_new = t + std::copysign(prev_jump / 2.0, newK1 - K1_eval);
                newK1 = K1_adj(t_new, mu, g, q);
                prev_jump /= 2.0;
            } else {
                prev_jump = std::abs(t_new - t);
            }
        }
        t = t_new;
        K1_eval = newK1;
        ++rep;
    }
    return {t, rep, converged};
}

double normal_cdf(double z) {
    return 0.5 * std::erfc(-z / std::sqrt(2));
}

double Get_Saddle_Prob(double zeta, const VectorXd& mu, const VectorXd& g, double q, bool log_p) {
    double k1 = Korg(zeta, mu, g);
    double k2 = K2(zeta, mu, g);
    if (!std::isfinite(k1) || !std::isfinite(k2)) return log_p ? -INFINITY : 0.0;
    double temp1 = zeta * q - k1;
    double w = std::copysign(std::sqrt(2.0 * temp1), zeta);
    double v = zeta * std::sqrt(k2);
    double Z = w + (1.0 / w) * std::log(v / w);
    return log_p ? (Z > 0 ? std::log(normal_cdf(-Z)) : -std::log(normal_cdf(Z)))
                 : (Z > 0 ? normal_cdf(-Z) : -normal_cdf(Z));
}

SaddleResult Saddle_Prob(double q, const VectorXd& mu, const VectorXd& g, double cutoff, double alpha, bool log_p) {
    double m1 = (mu.array() * g.array()).sum();
    double var1 = (mu.array() * (1 - mu.array()) * g.array().square()).sum();
    double score = q - m1;
    double qinv = -std::copysign(std::abs(score), score) + m1;
    double z = score / std::sqrt(var1);
    double pval_na = log_p ? -std::log(2.0) + std::log(normal_cdf(-std::abs(z)))
                           : 2.0 * normal_cdf(-std::abs(z));
    if (std::abs(score) / std::sqrt(var1) < cutoff)
        return {pval_na, pval_na, true, score};

    RootResult r1 = GetRoot_K1(0.0, mu, g, q);
    RootResult r2 = GetRoot_K1(0.0, mu, g, qinv);
    if (r1.converged && r2.converged) {
        double p1 = Get_Saddle_Prob(r1.root, mu, g, q, log_p);
        double p2 = Get_Saddle_Prob(r2.root, mu, g, qinv, log_p);
        double pval = log_p ? std::max(p1, p2) + std::log1p(std::exp(std::min(p1, p2) - std::max(p1, p2)))
                            : std::abs(p1) + std::abs(p2);
        if (!log_p && pval_na / pval > 1e3)
            return Saddle_Prob(q, mu, g, cutoff * 2, alpha, log_p);
        return {pval, pval_na, true, score};
    }
    return {pval_na, pval_na, false, score};
}

SaddleResult TestSPA(const VectorXd& G, const SA_NULL_Model& null_model, double cutoff, double alpha, bool log_p) {
    VectorXd G_clean = G;
    if (G.sum() / (2.0 * G.size()) > 0.5) G_clean = 2.0 - G.array();
    VectorXd GX = null_model.XV * G_clean;
    VectorXd G1 = G_clean - null_model.XXVX_inv * GX;
    double q = G1.dot(null_model.y);
    return Saddle_Prob(q, null_model.mu, G1, cutoff, alpha, log_p);
}

