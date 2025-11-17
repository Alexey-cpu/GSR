#include <gs_math.hpp>

int main(int argc, char *argv[])
{
    printf("running %s\n", GS_STRINGIFY(gs_autotests_matrixes()));

    typedef gs_matrix<float, 4, 4> mat4x4f;
    typedef gs_vector<float, 4>    vec4f;

    mat4x4f::value_type epsilon = 0.001f;
    mat4x4f::value_type maxDiff = 0.f;

    mat4x4f mat_a;
    mat4x4f mat_b;
    mat4x4f mat_c;
    mat4x4f mat_d;

    for (int i = 0; i < mat_a.columns(); ++i)
    {
        for (int j = 0; j < mat_a.rows(); ++j)
            mat_a[i][j] = gs_pseudo_random<float>(0, 100, 1000);
    }

    for (int i = 0; i < mat_b.columns(); ++i)
    {
        for (int j = 0; j < mat_a.rows(); ++j)
            mat_b[i][j] = gs_pseudo_random<float>(0, 100, 1000);
    }

    // +
    maxDiff = 0.f;
    mat_c = mat_a + mat_b;
    for (int i = 0; i < mat_c.columns(); ++i)
    {
        for (int j = 0; j < mat_c.rows(); ++j)
        {
            maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - (mat_a[i][j] + mat_b[i][j])));
            GS_ASSERT(maxDiff < epsilon);
        }
    }
    printf("gs_matrix<...> + gs_matrix<...> cuccess %f\n", maxDiff);

    // -
    maxDiff = 0.f;
    mat_c = mat_a - mat_b;
    for (int i = 0; i < mat_c.columns(); ++i)
    {
        for (int j = 0; j < mat_c.rows(); ++j)
        {
            maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - (mat_a[i][j] - mat_b[i][j])));
            GS_ASSERT(maxDiff < epsilon);
        }
    }
    printf("gs_matrix<...> - gs_matrix<...> cuccess %f\n", maxDiff);

    // +=
    mat_c  = mat_a;
    mat_c += mat_b;
    for (int i = 0; i < mat_c.columns(); ++i)
    {
        for (int j = 0; j < mat_c.rows(); ++j)
        {
            maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - (mat_a[i][j] + mat_b[i][j])));
            GS_ASSERT(maxDiff < epsilon);
        }
    }

    printf("gs_matrix<...> += gs_matrix<...> cuccess %f\n", maxDiff);

    // -=
    mat_c  = mat_a;
    mat_c -= mat_b;
    for (int i = 0; i < mat_c.columns(); ++i)
    {
        for (int j = 0; j < mat_c.rows(); ++j)
        {
            maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - (mat_a[i][j] - mat_b[i][j])));
            GS_ASSERT(maxDiff < epsilon);
        }
    }
    printf("gs_matrix<...> -= gs_matrix<...> cuccess %f\n", maxDiff);

    // *=
    mat_c  = mat_a;
    mat_c *= mat_b;
    mat_d  = mat_a * mat_b;
    for (int i = 0; i < mat_c.columns(); ++i)
    {
        for (int j = 0; j < mat_c.rows(); ++j)
        {
            maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - mat_d[i][j]));
            GS_ASSERT(maxDiff < epsilon);
        }
    }
    printf("gs_matrix<...> *= gs_matrix<...> cuccess %f\n", maxDiff);

    // gs_matrix_invert_square(gs_matrix<...>)
    auto inverted_mat_a = gs_matrix_invert_square(mat_a);

    mat_c = mat_a * inverted_mat_a;
    mat_d = inverted_mat_a * mat_a;

    maxDiff = 0.f;

    for (int i = 0; i < mat_c.columns(); i++)
    {
        for (int j = 0; j < mat_c.rows(); j++)
        {
            if(i == j)
            {
                GS_ASSERT(gs_abs(mat_c[i][j] - 1.0) < epsilon);
                GS_ASSERT(gs_abs(mat_d[i][j] - 1.0) < epsilon);
            }
            else
            {
                GS_ASSERT(gs_abs(mat_c[i][j] - 0.0) < epsilon);
                GS_ASSERT(gs_abs(mat_d[i][j] - 0.0) < epsilon);
            }
        }
    }

    auto diff = mat_c - mat_d;

    for (int i = 0; i < mat_c.columns(); i++)
    {
        for (int j = 0; j < mat_c.rows(); j++)
        {
            GS_ASSERT(gs_abs(diff[i][j] - 0.0) < epsilon);
        }
    }

    printf("gs_matrix_invert_square(gs_matrix<...>) cuccess \n");

    // gs_matrix<...> * gs_vector<...>
    gs_matrix<mat4x4f::value_type, 4, 1> mat_e;

    for (int i = 0; i < mat_e.columns(); ++i)
    {
        for (int j = 0; j < mat_e.rows(); ++j)
        {
            mat_e[i][j] = gs_pseudo_random<float>(0, 100, 1000);
        }
    }

    auto result = gs_matrix_solve_square(mat_a, mat_e);

    gs_vec4f x;
    for (int j = 0; j < mat_e.rows(); ++j)
        x[j] = result[0][j];

    auto b = mat_a * x;

    for (int j = 0; j < mat_e.rows(); ++j)
        GS_ASSERT(gs_abs(b[j] - mat_e[0][j]) < epsilon);
    printf("gs_matrix<...> * gs_vector<...> cuccess \n");

    return 0;
}