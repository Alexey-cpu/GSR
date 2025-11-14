#include <gs_math.hpp>

int main(int argc, char *argv[])
{
    printf("running %s\n", "gs_autotests_vectors.cpp");

    // initialize
    typedef gs_vector<float, 2> vector2f;
    typedef gs_vector<float, 3> vector3f;
    typedef gs_vector<float, 4> vector4f;

    vector4f vec_a;
    vector4f vec_b;
    vector4f vec_c;
    vector4f vec_d;

    for (int i = 0; i < vec_a.size(); i++)
    {
        vec_a[i] = (vector4f::value_type)i;
        vec_b[i] = (vector4f::value_type)(i + vec_b.size());
    }

    // !=
    GS_ASSERT(vec_a != vec_b);
    printf("gs_vector<%s, %d> != cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // ==
    GS_ASSERT((vec_c = vec_a) == vec_a);
    printf("gs_vector<%s, %d> == cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // *
    vec_c = vec_a * vec_b;
    for (int i = 0; i < vec_c.size(); i++)
        GS_ASSERT(vec_c[i] == vec_a[i] * vec_b[i]);
    printf("gs_vector<%s, %d> * cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // +
    vec_c = vec_a + vec_b;
    for (int i = 0; i < vec_c.size(); i++)
        GS_ASSERT(vec_c[i] == vec_a[i] + vec_b[i]);
    printf("gs_vector<%s, %d> + cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // -
    vec_c = vec_a - vec_b;
    for (int i = 0; i < vec_c.size(); i++)
        GS_ASSERT(vec_c[i] == vec_a[i] - vec_b[i]);
    printf("gs_vector<%s, %d> - cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // /
    vec_c = vec_a / vec_b;
    for (int i = 0; i < vec_c.size(); i++)
        GS_ASSERT(vec_c[i] == vec_a[i] / vec_b[i]);
    printf("gs_vector<%s, %d> / cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // self modifying operators tests
    for (int i = 0; i < vec_a.size(); i++)
    {
        vec_a[i] = (vector4f::value_type)i;
        vec_b[i] = (vector4f::value_type)(i + vec_b.size());
    }

    // *=
    vec_c = vec_a * vec_b;
    vec_d = vec_a;
    vec_d *= vec_b;
    GS_ASSERT(vec_d == vec_c);
    printf("gs_vector<%s, %d> *= cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // +=
    vec_c = vec_a + vec_b;
    vec_d = vec_a;
    vec_d += vec_b;
    GS_ASSERT(vec_d == vec_c);
    printf("gs_vector<%s, %d> += cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // -=
    vec_c = vec_a - vec_b;
    vec_d = vec_a;
    vec_d -= vec_b;
    GS_ASSERT(vec_d == vec_c);
    printf("gs_vector<%s, %d> -= cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // /=
    vec_c = vec_a / vec_b;
    vec_d = vec_a;
    vec_d /= vec_b;
    GS_ASSERT(vec_d == vec_c);
    printf("gs_vector<%s, %d> /= cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());

    // initialization
    float a = 1.f;
    float b = 2.f;
    vec_d = vector2f(a, b);
    GS_ASSERT(vec_d[0] == a);
    GS_ASSERT(vec_d[1] == b);

    printf("gs_vector<%s, %d>(a, b) sucess\n", typeid(vector2f::value_type).name(), vector2f().size());

    // swizzle
    vector3f vec_e = vec_d;
    GS_ASSERT(vec_e[0] == vec_d[0]);
    GS_ASSERT(vec_e[1] == vec_d[1]);
    vec_e[2] = 3.f;

    vector4f vec_f = vec_e;
    GS_ASSERT(vec_f[0] == vec_e[0]);
    GS_ASSERT(vec_f[1] == vec_e[1]);
    GS_ASSERT(vec_f[2] == vec_e[2]);

    printf("gs_vecrtor swizzle works fine...\n");

    // normalize
    {
        vector4f vec_a_nrm = gs_vector_normalize(vec_a);
        for (int i = 0; i < vec_a.size(); i++)
            GS_ASSERT(gs_abs(vec_a_nrm[i] - vec_a[i] / gs_vector_length(vec_a)) < 1e-3);
        printf("gs_vector_normalize(gs_vector<%s, %d>) cuccess\n", typeid(vector4f::value_type).name(), vector4f().size());
    }

    // dot product
    {
        // this test checks that vectors dot product formula is correct
        double dot = 0;
        for (int i = 0; i < vec_a.size(); i++)
            dot += vec_a[i] * vec_b[i];

        GS_ASSERT(gs_vectors_dot(vec_a, vec_b) == dot);
        printf("gs_vector_dot(gs_vector<%s, %d>, gs_vector<%s, %d>) cuccess\n",
            typeid(vector4f::value_type).name(),
            vector4f().size(),
            typeid(vector4f::value_type).name(),
            vector4f().size());

        // this additional test checks that dot product of two vectors is proportional to
        // the cosine of the angle between vectors:
        // dot(a, b) = |a| * |b| * cos(alpha)
        vector2f a = gs_vector_normalize(vector2f(cos(gs_to_radians(30.f)), sin(gs_to_radians(30.f))));
        vector2f b = gs_vector_normalize(vector2f(cos(gs_to_radians(60.f)), sin(gs_to_radians(60.f))));

        GS_ASSERT(gs_abs(gs_abs(gs_to_degrees(acos(gs_vectors_dot(a, b)))) - 30) < 1e-3);
        printf("gs_vectors_dot(gs_vector<%s, %d>, gs_vector<%s, %d>) cuccess\n",
            typeid(vector2f::value_type).name(),
            vector2f().size(),
            typeid(vector2f::value_type).name(),
            vector2f().size());
    }

    // cross product
    {
        // this test checks that vectors cross product of two 3D vectors
        // generates the third vector, which is perpendicular to the cross multiplierd vectors
        vector3f a = gs_vector_normalize(vector3f(2, 4, 6));
        vector3f b = gs_vector_normalize(vector3f(8, 16, -32));
        vector3f c = gs_vector_normalize(gs_vector_cross(a, b));
        GS_ASSERT(gs_abs(gs_vectors_dot(a, c) - 0) < 1e3);
        GS_ASSERT(gs_abs(gs_vectors_dot(b, c) - 0) < 1e3);

        printf("gs_vector_cross(gs_vector<%s, %d>, gs_vector<%s, %d>) cuccess\n",
            typeid(vector3f::value_type).name(),
            vector3f().size(),
            typeid(vector3f::value_type).name(),
            vector3f().size());

        // this additional test checks that cross product of two 2D vectors is proportional to
        // the sine of the angle between vectors:
        // dot(a, b) = |a| * |b| * sin(alpha)
        vector2f d = gs_vector_normalize(vector2f(cos(gs_to_radians(30.f)), sin(gs_to_radians(30.f))));
        vector2f e = gs_vector_normalize(vector2f(cos(gs_to_radians(60.f)), sin(gs_to_radians(60.f))));
        GS_ASSERT(gs_abs(gs_abs(gs_to_degrees(asin(gs_vector_cross(d, e)))) - 30) < 1e-3);
        printf("gs_vector_cross(gs_vector<%s, %d>, gs_vector<%s, %d>) cuccess\n",
            typeid(vector2f::value_type).name(),
            vector2f().size(),
            typeid(vector2f::value_type).name(),
            vector2f().size());
    }

    return 0;
}