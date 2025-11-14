#include <gs_math.hpp>

/*
#include <gs_autotests.hpp>

// AUTOTESTS
void lu_test()
{
    typedef gs_matrix<double, 4, 4> mat4x4f;

    // 
    mat4x4f matrix(1.f);
    mat4x4f eye(0);
    for (int i = 0; i < 4; i++)
    {
        eye[i][i] = 1.0;
        for (int j = 0; j < 4; j++)
            matrix[i][j] = gs_pseudo_random<double>(1.0, 100.0);
    }

    auto factorization = gs_matrix_factor_square(matrix);

    mat4x4f lowerTriangularMatrix;
    mat4x4f upperTriangularMatrix;
    mat4x4f permutedMatrix = matrix;

    for (int i = 0; i < 4; i++)
    {
        lowerTriangularMatrix[i][i] = 1.0;

        for (int j = 0; j < 4; j++)
        {
            if(j > i)
            {
                lowerTriangularMatrix[i][j] = factorization.Matrix[i][j];
            }
            else
            {
                upperTriangularMatrix[i][j] = factorization.Matrix[i][j];
            }

            // swap raws
            gs_swap(permutedMatrix[i][factorization.InverseRowsPermutations[j]], permutedMatrix[i][j]);
        }
    }

    auto multiplication = lowerTriangularMatrix * upperTriangularMatrix;
    auto substraction   = multiplication - permutedMatrix;

    printf("A\n");
    gs_print(matrix);
    printf("L * U\n");
    gs_print(multiplication);
    printf("P * A\n");
    gs_print(permutedMatrix);
    printf("L * U - P * A\n");
    gs_print(substraction);
    // printf("solution\n");
    // gs_print(solution);
    // printf("multiplication\n");
    // gs_print(multiplication);
}

void solve_test()
{
    typedef gs_matrix<double, 4, 4> mat4x4f;

    // 
    mat4x4f matrix(1.f);
    mat4x4f eye(0);
    for (int i = 0; i < 4; i++)
    {
        eye[i][i] = 1.0;
        for (int j = 0; j < 4; j++)
            matrix[i][j] = gs_pseudo_random<double>(1.0, 100.0);
    }

    auto inverse = gs_matrix_solve_square(matrix, eye);
    auto inverseByMatrix = inverse * matrix;
    auto matrixByInverse = matrix * inverse;

    printf("A\n");
    gs_print(matrix);
    printf("inv(A) * A\n");
    gs_print(inverseByMatrix);
    printf("A * inv(A)\n");
    gs_print(matrixByInverse);
}

void invert_test()
{
    typedef gs_matrix<double, 4, 4> mat4x4f;

    // 
    mat4x4f matrix(1.f);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            matrix[i][j] = gs_pseudo_random<double>(1.0, 100.0);
    }

    auto inverse = gs_matrix_invert_square(matrix);
    auto inverseByMatrix = inverse * matrix;
    auto matrixByInverse = matrix * inverse;

    printf("A\n");
    gs_print(matrix);
    printf("inv(A) * A\n");
    gs_print(inverseByMatrix);
    printf("A * inv(A)\n");
    gs_print(matrixByInverse);
}

*/

int main(int argc, char *argv[])
{
    //gs_autotests_vectors();
    // solve_test();
    // invert_test();

    return 0;
}
