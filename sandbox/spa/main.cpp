#include <iostream>
#include "spa_score_test.h"

int main() {
    using namespace Eigen;

    // --- Simulated data ---
    // 5 samples, 2 covariates (intercept + 1 covariate)
    MatrixXd X1(5, 2);
    X1 << 1, 0,
          1, 1,
          1, 2,
          1, 3,
          1, 4;

    // Binary phenotype
    VectorXd y(5);
    y << 0, 1, 1, 0, 1;

    // Genotype vector (e.g., for a SNP)
    VectorXd G(5);
    G << 0, 1, 2, 1, 0;

    // --- Fit null model ---
    SA_NULL_Model null_model = FitNullModel(X1, y);
    if (!null_model.converged) {
        std::cerr << "Null model did not converge.\n";
        return 1;
    }

    // --- Run SPA test ---
    SaddleResult result = TestSPA(G, null_model);

    std::cout << "SPA p-value:       " << result.p_value << "\n";
    std::cout << "Normal p-value:    " << result.p_value_na << "\n";
    std::cout << "Converged:         " << std::boolalpha << result.converged << "\n";
    std::cout << "Score statistic:   " << result.score << "\n";

    return 0;
}
