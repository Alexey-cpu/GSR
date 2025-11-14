#include <gs_math.hpp>

int main(int argc, char *argv[])
{
    printf("running %s\n", GS_STRINGIFY(gs_autotests_matrixes()));

    typedef gs_matrix<float, 4, 4> mat4x4f;
    typedef gs_matrix<float, 3, 2> mat3x2f;
    typedef gs_matrix<float, 2, 3> mat2x3f;

    mat4x4f::value_type maxDiff = 0.f;

    mat4x4f mat_a;
    mat4x4f mat_b;
    mat4x4f mat_c;

    for (int i = 0; i < mat_a.columns(); i++)
    {
        for (int j = 0; j < mat_a.rows(); j++)
            mat_a[i][j] = gs_pseudo_random<float>(0, 100, 1000);
    }

    for (int i = 0; i < mat_b.columns(); i++)
    {
        for (int j = 0; j < mat_a.rows(); j++)
            mat_b[i][j] = gs_pseudo_random<float>(0, 100, 1000);
    }

    // +
    maxDiff = 0.f;
    mat_c = mat_a + mat_b;
    for (int i = 0; i < mat_c.columns(); i++)
    {
        for (int j = 0; j < mat_c.rows(); j++)
        {
            maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - (mat_a[i][j] + mat_b[i][j])));
            GS_ASSERT(maxDiff < 1e-3);
        }
    }
    printf("gs_matrix<...> + gs_matrix<...> cuccess %f\n", maxDiff);

    // -
    maxDiff = 0.f;
    mat_c = mat_a - mat_b;
    for (int i = 0; i < mat_c.columns(); i++)
    {
        for (int j = 0; j < mat_c.rows(); j++)
        {
            maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - (mat_a[i][j] - mat_b[i][j])));
            GS_ASSERT(maxDiff < 1e-3);
        }
    }
    printf("gs_matrix<...> - gs_matrix<...> cuccess %f\n", maxDiff);

    // +=
    mat_c  = mat_a;
    mat_c += mat_b;
    for (int i = 0; i < mat_c.columns(); i++)
    {
        for (int j = 0; j < mat_c.rows(); j++)
        {
            maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - (mat_a[i][j] + mat_b[i][j])));
            GS_ASSERT(maxDiff < 1e-3);
        }
    }

    printf("gs_matrix<...> += gs_matrix<...> cuccess %f\n", maxDiff);

    // // -=
    // mat_c  = mat_a;
    // mat_c -= mat_b;
    // for (int i = 0; i < mat_c.columns(); i++)
    // {
    //     for (int j = 0; j < mat_c.rows(); j++)
    //     {
    //         maxDiff = gs_max(maxDiff, gs_abs(mat_c[i][j] - (mat_a[i][j] - mat_b[i][j])));
    //         GS_ASSERT(maxDiff < 1e-3);
    //     }
    // }
    // printf("gs_matrix<...> =- gs_matrix<...> cuccess %f\n", maxDiff);

    // printf("multiply non square matrixes\n");
    // {
    //     mat3x2f mat_a;
    //     mat_a[0][0] = 1.f;
    //     mat_a[0][1] = 2.f;
    //     mat_a[0][2] = 3.f;
    //     mat_a[1][0] = 4.f;
    //     mat_a[1][1] = 5.f;
    //     mat_a[1][2] = 6.f;

    //     mat2x3f mat_b;
    //     mat_b[0][0] = 1.f;
    //     mat_b[0][1] = 2.f;
    //     mat_b[1][0] = 3.f;
    //     mat_b[1][1] = 4.f;
    //     mat_b[2][0] = 5.f;
    //     mat_b[2][1] = 6.f;

    //     auto mat_c = mat_b * mat_a;

    //     printf("mat_a:\n");
    //     gs_print(mat_a);
    //     printf("mat_b:\n");
    //     gs_print(mat_b);
    //     printf("mat_c:\n");
    //     gs_print(mat_c);

    //     auto transposed = gs_matrix_transpose(mat_c);

    //     printf("transposed mat_c:\n");
    //     gs_print(transposed);
    // }

    return 0;
}