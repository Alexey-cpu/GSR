// C
#include <cmath>
#include <cstdio>
#include <cassert>

// C++
#include <limits>
#include <typeinfo>

#define GS_ASSERT assert
#define GS_STRINGIFY(INPUT) #INPUT

#ifndef GS_TO_DEGREES_CONVERSION_MULTIPLYER__
#define GS_TO_DEGREES_CONVERSION_MULTIPLYER__ 57.295779513082320876798154814105
#endif

#ifndef GS_TO_RADIANS_CONVERSION_MULTIPLYER__
#define GS_TO_RADIANS_CONVERSION_MULTIPLYER__ 0.01745329251994329576923690768489
#endif

// Fortran analogues functions
template<typename Type> Type 
gs_digits()
{
    return std::numeric_limits<Type>::digits;
}

template<typename Type> Type gs_epsilon()
{
    return std::numeric_limits<Type>::epsilon();
}

template<typename Type> Type gs_huge()
{
    return std::numeric_limits<Type>::max();
}

template<typename Type> Type gs_maxexponent()
{
    return std::numeric_limits<Type>::max_exponent;
}

template<typename Type> Type gs_minexponent()
{
    return std::numeric_limits<Type>::min_exponent;
}

template<typename Type> Type gs_radix()
{
    return std::numeric_limits<Type>::radix;
}

template<typename Type> Type gs_tiny()
{
    return std::numeric_limits<Type>::min();
}

inline double gs_to_degrees(double _Angle)
{
    return _Angle * GS_TO_DEGREES_CONVERSION_MULTIPLYER__;
}

inline double gs_to_radians__(double _Angle)
{
    return _Angle * GS_TO_RADIANS_CONVERSION_MULTIPLYER__;
}

template<typename Type>
inline Type gl_abs(const Type& _A)
{
    return _A < 0 ? -_A : +_A;
}

template<typename Type>
inline Type gs_max(const Type& _A, const Type& _B)
{
    return _A > _B ? _A : _B;
}

template<typename Type, typename ... Args>
inline Type gs_max(const Type& _A, const Type& _B, Args... _Args)
{
    return gs_max(gs_max(_A, _B), _Args...);
}

template<typename Type>
inline Type gs_min(const Type& _A, const Type& _B)
{
    return _A < _B ? _A : _B;
}

template<typename Type, typename ... Args>
inline Type gs_min(const Type& _A, const Type& _B, Args... _Args)
{
    return gs_min(gs_min(_A, _B), _Args...);
}

template<typename Type>
inline void gs_swap(Type& _A, Type& _B)
{
    Type _C = _A;
    _A = _B;
    _B = _C;
}

template<typename Type, int Size>
struct gs_vector final
{
    typedef Type value_type;

    gs_vector()
    {
        for (int i = 0; i < Size; i++)
            Data[i] = 0;
    }

    gs_vector(const Type& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] = _Value;
    }

    gs_vector(const gs_vector<Type, Size>& _Other)
    {
        for (int i = 0; i < Size; i++)
            Data[i] = _Other[i];
    }

    template<int OtherSize>
    gs_vector(const gs_vector<Type, OtherSize>& _Other)
    {
        for (int i = 0; i < gs_min(OtherSize, Size); i++)
            Data[i] = _Other[i];
    }

    template<int OtherSize>
    gs_vector(Type const (&_Values)[OtherSize])
    {
        for (int i = 0; i < gs_min(OtherSize, Size); i++)
            Data[i] = _Values[i];
    }

    const int length() const
    {
        return Size;
    }

    // &[]
    Type& operator[](const unsigned char& _Index)
    {
        GS_ASSERT(_Index < Size);
        return Data[_Index];
    }

    // const Type[]&
    const Type& operator[](const unsigned char& _Index) const
    {
        GS_ASSERT(_Index < Size);
        return Data[_Index];
    }

    // !=
    bool operator!=(const gs_vector<Type, Size>& _Other)
    {
        bool value = false;
        for (int i = 0; i < Size; i++)
            value |= _Other[i] != Data[i];
        return value;
    }

    // ==
    bool operator==(const gs_vector<Type, Size>& _Other)
    {
        bool value = true;
        for (int i = 0; i < Size; i++)
            value &= _Other[i] == Data[i];
        return value;
    }

    // +
    gs_vector<Type, Size> operator+(const Type& _Value)
    {
        gs_vector<Type, Size> result;
        for (int i = 0; i < Size; i++)
            result[i] = Data[i] + _Value;
        return result;
    }

    gs_vector<Type, Size> operator+(const gs_vector<Type, Size>& _Value)
    {
        gs_vector<Type, Size> result;
        for (int i = 0; i < Size; i++)
            result[i] = Data[i] + _Value[i];
        return result;
    }

    // -
    gs_vector<Type, Size> operator-(const Type& _Value)
    {
        gs_vector<Type, Size> result;
        for (int i = 0; i < Size; i++)
            result[i] = Data[i] - _Value;
        return result;
    }

    gs_vector<Type, Size> operator-(const gs_vector<Type, Size>& _Value)
    {
        gs_vector<Type, Size> result;
        for (int i = 0; i < Size; i++)
            result[i] = Data[i] - _Value[i];
        return result;
    }

    // *
    gs_vector<Type, Size> operator*(const Type& _Value)
    {
        gs_vector<Type, Size> result;
        for (int i = 0; i < Size; i++)
            result[i] = Data[i] * _Value;
        return result;
    }

    gs_vector<Type, Size> operator*(const gs_vector<Type, Size>& _Value)
    {
        gs_vector<Type, Size> result;
        for (int i = 0; i < Size; i++)
            result[i] = Data[i] * _Value[i];
        return result;
    }

    // /
    gs_vector<Type, Size> operator/(const Type& _Value)
    {
        gs_vector<Type, Size> result;
        for (int i = 0; i < Size; i++)
            result[i] = Data[i] / _Value;
        return result;
    }

    gs_vector<Type, Size> operator/(const gs_vector<Type, Size>& _Value)
    {
        gs_vector<Type, Size> result;
        for (int i = 0; i < Size; i++)
            result[i] = Data[i] / _Value[i];
        return result;
    }

    // +=
    gs_vector<Type, Size> operator+=(const Type& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] += _Value;
        return *this;
    }

    gs_vector<Type, Size> operator+=(const gs_vector<Type, Size>& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] += _Value[i];
        return *this;
    }

    // -=
    gs_vector<Type, Size> operator-=(const Type& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] -= _Value;
        return *this;
    }

    gs_vector<Type, Size> operator-=(const gs_vector<Type, Size>& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] -= _Value[i];
        return *this;
    }

    // *=
    gs_vector<Type, Size> operator*=(const Type& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] *= _Value;
        return *this;
    }

    gs_vector<Type, Size> operator*=(const gs_vector<Type, Size>& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] *= _Value[i];
        return *this;
    }

    // /=
    gs_vector<Type, Size> operator/=(const Type& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] /= _Value;
        return *this;
    }

    gs_vector<Type, Size> operator/=(const gs_vector<Type, Size>& _Value)
    {
        for (int i = 0; i < Size; i++)
            Data[i] /= _Value[i];
        return *this;
    }

private:
    Type Data[Size]{0};
};

		// GLM_FUNC_DISCARD_DECL GLM_CONSTEXPR mat<3, 3, T, Q> & operator-=(mat<3, 3, U, Q> const& m);
		// template<typename U>
		// GLM_FUNC_DISCARD_DECL GLM_CONSTEXPR mat<3, 3, T, Q> & operator*=(U s);

template<typename Type, int Rows, int Columns>
struct gs_matrix final
{
    typedef Type value_type;

    gs_matrix()
    {
        for (int i = 0; i < Size; i++)
            Data[i] = 0;
    }

    gs_matrix(const Type& _Value)
    {
        for (int i = 0; i < Columns; ++i)
            Data[i * Columns + i] = _Value;
    }

    gs_matrix(const gs_matrix<Type, Rows, Columns>& _Matrix)
    {
        for (int i = 0; i < Size; ++i)
            Data[i] = _Matrix.Data[i]; 
    }

    int rows() const
    {
        return Rows;
    }

    int columns() const
    {
        return Columns;
    }

    Type* operator[](const int& _Column)
    {
        GS_ASSERT(_Column < Columns);
        return &Data[_Column * Rows];
    }

    const Type* operator[](const int& _Column) const
    {
        GS_ASSERT(_Column < Columns);
        return &Data[_Column * Rows];
    }

    // +
    gs_matrix<Type, Rows, Columns> operator+(const gs_matrix<Type, Rows, Columns>& _Mat) const
    {
        gs_matrix<Type, Rows, Columns> result;
        add_mat(_Mat, *this, result);
        return result;
    }

    // +=
    gs_matrix<Type, Rows, Columns> operator+=(const gs_matrix<Type, Rows, Columns>& _Mat) const
    {
        gs_matrix<Type, Rows, Columns> result;
        add_mat(_Mat, *this, result);
        *this = result;
        return *this;
    }

    // -
    gs_matrix<Type, Rows, Columns> operator-(const gs_matrix<Type, Rows, Columns>& _Mat) const
    {
        gs_matrix<Type, Rows, Columns> result;
        sub_mat(_Mat, *this, result);
        return result;
    }

    // -=
    gs_matrix<Type, Rows, Columns> operator-=(const gs_matrix<Type, Rows, Columns>& _Mat) const
    {
        gs_matrix<Type, Rows, Columns> result;
        sub_mat(_Mat, *this, result);
        *this = result;
        return *this;
    }

    // *
    template<int Dimention>
    gs_matrix<Type, Rows, Dimention> operator*(const gs_matrix<Type, Columns, Dimention>& _Mat) const
    {
        gs_matrix<Type, Rows, Dimention> result;
        mul_mat(*this, _Mat, result);
        return result;
    }

    // *=
    template<int Dimention>
    gs_matrix<Type, Rows, Dimention> operator*=(const gs_matrix<Type, Columns, Dimention>& _Mat) const
    {
        gs_matrix<Type, Rows, Dimention> result;
        mul_mat(*this, _Mat, result);
        *this = result;
        return *this;
    }

private:

    Type Data[Rows * Columns]{};
    int  Size{Rows * Columns};

    // service methods
    void add_mat(
        const gs_matrix<Type, Rows, Columns>& _A,
        const gs_matrix<Type, Rows, Columns>& _B,
        gs_matrix<Type, Rows, Columns>&       _C) const
    {
        for (int i = 0; i < Size; i++)
            _C.Data[i] = _A.Data[i] + _B.Data[i];
    }

    void sub_mat(
        const gs_matrix<Type, Rows, Columns>& _A,
        const gs_matrix<Type, Rows, Columns>& _B,
        gs_matrix<Type, Rows, Columns>&       _C) const
    {
        for (int i = 0; i < Size; i++)
            _C.Data[i] = _A.Data[i] - _B.Data[i];
    }

    template<int Dimention>
    void mul_mat(
        const gs_matrix<Type, Rows, Columns>&      _A,
        const gs_matrix<Type, Columns, Dimention>& _B,
        gs_matrix<Type, Rows, Dimention>&          _C) const
    {
        GS_ASSERT(_A.columns() == _B.rows());
        GS_ASSERT(_C.columns() == _B.columns());

        for (int i = 0; i < Dimention; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                for (int k = 0; k < Rows; k++)
                {
                    _C[i][k] += _A[j][k] * _B[i][j];
                }
            }
        }
    }
};

template<typename Type>
inline double gs_sum_of_squares(Type _A)
{
    return _A * _A;
}

template<typename Type, typename ... Args>
inline double gs_sum_of_squares(Type _A, Args... _Args)
{
    return gs_sum_of_squares(_A) + gs_sum_of_squares(_Args ...);
}

template<typename Type, int Size>
inline double gs_sum_of_squares(const gs_vector<Type, Size>& _Vector)
{
    double sumOfSquares = 0;
    for (int i = 0; i < Size; ++i)
        sumOfSquares += _Vector[i] * _Vector[i];
    return sumOfSquares;
}

template<typename Type, typename ... Args>
inline double gs_vector_length(Type _A, Type _B, Args... _Args)
{
    double sumOfSquares = gs_sum_of_squares(_A, _B, _Args ...);
    return sumOfSquares > 0 ? sqrt(sumOfSquares) : 0;
}

template<typename Type, int Size>
inline double gs_vector_length(const gs_vector<Type, Size>& _Vector)
{
    double sumOfSquares = gs_sum_of_squares<Type, Size>(_Vector);
    return sumOfSquares > 0 ? sqrt(sumOfSquares) : 0;
}

template<typename Type, int Size>
inline gs_vector<Type, Size> gs_vector_normalize(const gs_vector<Type, Size>& _Vector)
{
    gs_vector<Type, Size> result(0);
    double length = gs_vector_length(_Vector);

    if(length <= static_cast<Type>(0)) 
    {
        result[0] = static_cast<Type>(1);
        return result;
    }

    Type oneOverLen = static_cast<Type>(1) / static_cast<Type>(length);
    for (int i = 0; i < Size; i++)
        result[i] = _Vector[i] * oneOverLen;
    return result;
}

template<typename Type, int Size>
inline double gs_vectors_dot(const gs_vector<Type, Size>& _A, const gs_vector<Type, Size>& _B)
{
    double dot = 0;
    for (int i = 0; i < Size; i++)
        dot += _A[i] * _B[i];
    return dot;
}

template<typename Type, int Rows, int Columns>
void gs_matrix_print(const gs_matrix<Type, Rows, Columns>& _Matrix)
{
    printf("[");
    for (int i = 0; i < Rows; i++)
    {
        for (int j = 0; j < Columns; j++)
        {
            if(j < Columns - 1)
                printf("%.f,\t", _Matrix[j][i]);
            else
                printf("%.f", _Matrix[j][i]);
        }
        
        if(i < Rows - 1)
            printf("\n");
        else
            printf("]\n");
    }
}

template<typename Type, int Rows, int Columns>
inline gs_matrix<Type, Rows, Columns> gs_matrix_transpose(const gs_matrix<Type, Rows, Columns>& _Matrix)
{
    gs_matrix<Type, Columns, Rows> transposed(0);

    for (int i = 0; i < Columns; i++)
    {
        for (int j = 0; j < Rows; j++)
            transposed[i][j] = _Matrix[j][i];
    }
    
    return transposed;
}

template<typename Type>
inline gs_matrix<Type, 4, 4> gs_matrix_scale(const gs_matrix<Type, 4, 4>& _Matrix, const gs_vector<Type, 3>& _Scale)
{
    gs_matrix<Type, 4, 4> transform(1);
    transform[0][0] = _Scale[0];
    transform[1][1] = _Scale[1];
    transform[2][2] = _Scale[2];
    return _Matrix * transform;
}

template<typename Type>
inline gs_matrix<Type, 4, 4> gs_matrix_translate(const gs_matrix<Type, 4, 4>& _Matrix, const gs_vector<Type, 3>& _Translation)
{
    gs_matrix<Type, 4, 4> transform(1);
    transform[3][0] = _Translation[0];
    transform[3][1] = _Translation[1];
    transform[3][2] = _Translation[2];
    return _Matrix * transform;
}

// // cross product of 2D vectors returns the scalar value equal to the area of the parallelogram formed by two input vectors
// template<typename Type>
// inline gs_vector2<Type> gs_vector_cross(const gs_vector2<Type>& _A, const gs_vector2<Type>& _B)
// {
//     return _A.x * _B.y - _A.y * _B.x;
// }

// // cross product of 3D vectors returns the vector perpedicular to cross multiplied vectors and the length
// // of resulting vector is equal to the area of the parallelogram formed by two input vectors
// template<typename Type>
// inline gs_vector3<Type> gs_vector_cross(const gs_vector3<Type>& _A, const gs_vector3<Type>& _B)
// {
//     return gs_vector3<Type>(
//         _A.y * _B.z - _B.y * _A.z,
//         _A.z * _B.x - _B.z * _A.x,
//         _A.x * _B.y - _B.x * _A.y);
// }



// vectors typedefs
typedef gs_vector<float, 2> gs_vector2f;
typedef gs_vector<float, 3> gs_vector3f;
typedef gs_vector<float, 4> gs_vector4f;
typedef gs_vector<double, 2> gs_vector2d;
typedef gs_vector<double, 3> gs_vector3d;
typedef gs_vector<double, 4> gs_vector4d;
typedef gs_vector<int, 2> gs_vector2i;
typedef gs_vector<int, 3> gs_vector3i;
typedef gs_vector<int, 4> gs_vector4i;

// matrix typedefs
typedef gs_matrix<float, 2, 2> gs_matrix2x2f;
typedef gs_matrix<float, 3, 3> gs_matrix3x3f;
typedef gs_matrix<float, 4, 4> gs_matrix4x4f;
typedef gs_matrix<double, 2, 2> gs_matrix2x2d;
typedef gs_matrix<double, 3, 3> gs_matrix3x3d;
typedef gs_matrix<double, 4, 4> gs_matrix4x4d;
typedef gs_matrix<int, 2, 2> gs_matrix2x2i;
typedef gs_matrix<int, 3, 3> gs_matrix3x3i;
typedef gs_matrix<int, 4, 4> gs_matrix4x4i;

// undef all macro
#undef GS_TO_DEGREES_CONVERSION_MULTIPLYER__
#undef GS_TO_RADIANS_CONVERSION_MULTIPLYER__

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

int main(int argc, char *argv[])
{
    // gs_matrix4x4f mat4(1);

    // printf("initial matrix\n");
    // gs_matrix_print(mat4);

    // // scale, then rotate, then translate (SRT)

    // printf("translated matrix\n");
    // mat4 = gs_matrix_translate(mat4, gs_vector3f({100.f, 150.f, 200.f}));
    // gs_matrix_print(mat4);

    // printf("scaled matrix\n");
    // mat4 = gs_matrix_scale(mat4, gs_vector3f({2.f, 4.f, 8.f}));
    // gs_matrix_print(mat4);

    gs_autotests_gs_utilities();
    gs_autotests_gs_vector();
    gs_autotests_gs_matrix();

    return 0;
}
