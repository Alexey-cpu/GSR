#include <gs_math.hpp>

// AUTOTESTS
void gs_autotests_gs_utilities()
{
    printf("running %s\n", GS_STRINGIFY(gs_autotests_gs_utilities()));

    // gs_max
    GS_ASSERT(gs_max(10, 12, 3, 0, 22, -1) == 22);
    printf("gs_max success...\n");

    // gs_min
    GS_ASSERT(gs_min(10, 12, 3, 0, 22, -1) == -1);
    printf("gs_min success... \n");

    GS_ASSERT(gs_sum_of_squares(2, 3, 4, 5) == (2 * 2 + 3 * 3 + 4 * 4 + 5 * 5));
    printf("gs_sum_of_squares success... \n");

    GS_ASSERT(gs_vector_length (2, 3, 4, 5) == sqrt(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5));
    printf("gs_vector_length success... \n");

    printf("\n");
}

void gs_autotests_gs_vector()
{
    printf("running %s\n", GS_STRINGIFY(gs_autotests_gs_vector()));

    const int Size2 = 2;
    const int Size3 = 3;
    const int Size4 = 4;

    typedef gs_vector<float, Size2> vector2f;
    typedef gs_vector<float, Size3> vector3f;
    typedef gs_vector<float, Size4> vector4f;

    vector4f vec_a;
    vector4f vec_b;
    vector4f vec_c;

    for (int i = 0; i < vec_a.length(); i++)
    {
        vec_a[i] = (vector4f::value_type)i;
        vec_b[i] = (vector4f::value_type)(i + vec_b.length());
    }

    vector4f vec_a_nrm = gs_vector_normalize(vec_a);
    for (int i = 0; i < vec_a.length(); i++)
        GS_ASSERT(gl_abs(vec_a_nrm[i] - vec_a[i] / gs_vector_length(vec_a)) < 1e-3);
    printf("gs_vector_normalize(gs_vector<%s, %d>) cuccess\n", typeid(vector4f::value_type).name(), Size4);

    // !=
    GS_ASSERT(vec_a != vec_b);
    printf("gs_vector<%s, %d> != cuccess\n", typeid(vector4f::value_type).name(), Size4);

    // ==
    GS_ASSERT((vec_c = vec_a) == vec_a);
    printf("gs_vector<%s, %d> == cuccess\n", typeid(vector4f::value_type).name(), Size4);

    // *
    vec_c = vec_a * vec_b;
    for (int i = 0; i < vec_c.length(); i++)
        GS_ASSERT(vec_c[i] == vec_a[i] * vec_b[i]);
    printf("gs_vector<%s, %d> * cuccess\n", typeid(vector4f::value_type).name(), Size4);

    // +
    vec_c = vec_a + vec_b;
    for (int i = 0; i < vec_c.length(); i++)
        GS_ASSERT(vec_c[i] == vec_a[i] + vec_b[i]);
    printf("gs_vector<%s, %d> + cuccess\n", typeid(vector4f::value_type).name(), Size4);

    // -
    vec_c = vec_a - vec_b;
    for (int i = 0; i < vec_c.length(); i++)
        GS_ASSERT(vec_c[i] == vec_a[i] - vec_b[i]);
    printf("gs_vector<%s, %d> - cuccess\n", typeid(vector4f::value_type).name(), Size4);

    // /
    vec_c = vec_a / vec_b;
    for (int i = 0; i < vec_c.length(); i++)
        GS_ASSERT(vec_c[i] == vec_a[i] / vec_b[i]);
    printf("gs_vector<%s, %d> / cuccess\n", typeid(vector4f::value_type).name(), Size4);

    // other operators
    {
        vector4f vec_a;
        vector4f vec_b;
        vector4f vec_c;
        vector4f vec_d;

        for (int i = 0; i < vec_a.length(); i++)
        {
            vec_a[i] = (vector4f::value_type)i;
            vec_b[i] = (vector4f::value_type)(i + vec_b.length());
        }

        // *=
        vec_c = vec_a * vec_b;
        vec_d = vec_a;
        vec_d *= vec_b;
        GS_ASSERT(vec_d == vec_c);
        printf("gs_vector<%s, %d> *= cuccess\n", typeid(vector4f::value_type).name(), Size4);

        // +=
        vec_c = vec_a + vec_b;
        vec_d = vec_a;
        vec_d += vec_b;
        GS_ASSERT(vec_d == vec_c);
        printf("gs_vector<%s, %d> += cuccess\n", typeid(vector4f::value_type).name(), Size4);

        // -=
        vec_c = vec_a - vec_b;
        vec_d = vec_a;
        vec_d -= vec_b;
        GS_ASSERT(vec_d == vec_c);
        printf("gs_vector<%s, %d> -= cuccess\n", typeid(vector4f::value_type).name(), Size4);

        // /=
        vec_c = vec_a / vec_b;
        vec_d = vec_a;
        vec_d /= vec_b;
        GS_ASSERT(vec_d == vec_c);
        printf("gs_vector<%s, %d> /= cuccess\n", typeid(vector4f::value_type).name(), Size4);
    }

    // dot
    double dot = 0;
    for (int i = 0; i < vec_a.length(); i++)
        dot += vec_a[i] * vec_b[i];

    GS_ASSERT(gs_vectors_dot(vec_a, vec_b) == dot);
    printf("gs_vector_dot(gs_vector<%s, %d>, gs_vector<%s, %d>) cuccess\n",
        typeid(vector4f::value_type).name(),
        Size4,
        typeid(vector4f::value_type).name(),
        Size4);

    float a = 1.f;
    float b = 2.f;
    vector2f vec_d = vector2f({a, b});
    GS_ASSERT(vec_d[0] == a);
    GS_ASSERT(vec_d[1] == b);

    vector3f vec_e = vec_d;
    GS_ASSERT(vec_e[0] == vec_d[0]);
    GS_ASSERT(vec_e[1] == vec_d[1]);
    vec_e[2] = 3.f;

    vector4f vec_f = vec_e;
    GS_ASSERT(vec_f[0] == vec_e[0]);
    GS_ASSERT(vec_f[1] == vec_e[1]);
    GS_ASSERT(vec_f[2] == vec_e[2]);

    printf("gs_vecrtor swizzle works fine...\n");
}

void gs_autotests_gs_matrix()
{
    printf("running %s\n", GS_STRINGIFY(gs_autotests_gs_matrix()));

    typedef gs_matrix<float, 4, 4> mat4x4f;
    typedef gs_matrix<float, 3, 2> mat3x2f;
    typedef gs_matrix<float, 2, 3> mat2x3f;

    mat4x4f mat_a(1.f);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            mat_a[i][j] = (float)(i + j);
    }

    mat4x4f mat_b(1.f);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            mat_b[i][j] = (float)(i + j);
    }

    mat4x4f mat_c = mat_a + mat_b;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            GS_ASSERT(mat_c[i][j] == mat_a[i][j] + mat_b[i][j]);
    }
    printf("gs_matrix<...> + cuccess\n");

    mat4x4f mat_d = mat_a - mat_b;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            GS_ASSERT(mat_d[i][j] == mat_a[i][j] - mat_b[i][j]);
    }
    printf("gs_matrix<...> - cuccess\n");

    {
        mat3x2f mat_a;
        mat_a[0][0] = 1.f;
        mat_a[0][1] = 2.f;
        mat_a[0][2] = 3.f;
        mat_a[1][0] = 4.f;
        mat_a[1][1] = 5.f;
        mat_a[1][2] = 6.f;

        mat2x3f mat_b;
        mat_b[0][0] = 1.f;
        mat_b[0][1] = 2.f;
        mat_b[1][0] = 3.f;
        mat_b[1][1] = 4.f;
        mat_b[2][0] = 5.f;
        mat_b[2][1] = 6.f;

        auto mat_c = mat_a * mat_b;

        printf("mat_a:\n");
        gs_matrix_print(mat_a);
        printf("mat_b:\n");
        gs_matrix_print(mat_b);
        printf("mat_c:\n");
        gs_matrix_print(mat_c);

        auto transposed = gs_matrix_transpose(mat_c);

        printf("transposed mat_c:\n");
        gs_matrix_print(transposed);
    }
}

template<typename Type, int Rows, int Columns>
gs_matrix<Type, Rows, Columns> gs_right_looking_lu_factor(const gs_matrix<Type, Rows, Columns>& _Matrix)
{
    // TODO: implement right-looking LU factorization here
    gs_matrix<Type, Rows, Columns> LU(1);
    return LU(1);
}

template<typename Type, int Rows, int Columns>
gs_matrix<Type, Rows, Columns> gs_invert(const gs_matrix<Type, Rows, Columns>& _Matrix)
{
    // TODO: implement matrix inversion using right-looking LU factorization here
    gs_matrix<Type, Rows, Columns> Inverted(1);
    return LU(1);
}

int main(int argc, char *argv[])
{
    // gs_autotests_gs_utilities();
    // gs_autotests_gs_vector();
    // gs_autotests_gs_matrix();

    gs_mat4f mat(1);

    gs_matrix_rotate(
        mat,
        gs_vec3f({(float)gs_to_radians(45), (float)gs_to_radians(45), (float)gs_to_radians(45)}));

    return 0;
}
