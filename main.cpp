#include <cmath>
#include <cstdio>
#include <cassert>

#define GS_ASSERT assert
#define GS_STRINGIFY(INPUT) #INPUT

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

template<typename Type>
inline Type gs_max(Type _A, Type _B)
{
    return _A > _B ? _A : _B;
}

template<typename Type, typename ... Args>
inline Type gs_max(Type _A, Type _B, Args... _Args)
{
    return gs_max(gs_max(_A, _B), _Args...);
}

template<typename Type>
inline Type gs_min(Type _A, Type _B)
{
    return _A < _B ? _A : _B;
}

template<typename Type, typename ... Args>
inline Type gs_min(Type _A, Type _B, Args... _Args)
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

template<typename Type>
struct gs_vector2
{
    gs_vector2() : x(0), y(0){}
    gs_vector2(const Type& _X) : x(_X), y(_X){}
    gs_vector2(const Type& _X, const Type& _Y) : x(_X), y(_Y){}

    const int length() const
    {
        return 2;
    }

    // &[]
    Type& operator[](const unsigned char& _Index)
    {
        GS_ASSERT(_Index == 0 || _Index == 1);
        return _Index < 1 ? x : y;
    }

    // const Type[]&
    const Type& operator[](const unsigned char& _Index) const
    {
        GS_ASSERT(_Index == 0 || _Index == 1);
        return _Index < 1 ? x : y;
    }

    // !=
    bool operator!=(const gs_vector2<Type>& _Other)
    {
        return x != _Other.x || y != _Other.y;
    }

    // ==
    bool operator==(const gs_vector2<Type>& _Other)
    {
        return x == _Other.x && y == _Other.y;
    }

    // +
    gs_vector2<Type> operator+(const Type& _Value)
    {
        return gs_vector2<Type>(x + _Value, y + _Value);
    }

    gs_vector2<Type> operator+(const gs_vector2<Type>& _Value)
    {
        return gs_vector2<Type>(x + _Value.x, y + _Value.y);
    }

    // -
    gs_vector2<Type> operator-(const Type& _Value)
    {
        return gs_vector2<Type>(x - _Value, y - _Value);
    }

    gs_vector2<Type> operator-(const gs_vector2<Type>& _Value)
    {
        return gs_vector2<Type>(x - _Value.x, y - _Value.y);
    }

    // *
    gs_vector2<Type> operator*(const Type& _Value)
    {
        return gs_vector2<Type>(x * _Value, y * _Value);
    }

    gs_vector2<Type> operator*(const gs_vector2<Type>& _Value)
    {
        return gs_vector2<Type>(x * _Value.x, y * _Value.y);
    }

    // /
    gs_vector2<Type> operator/(const Type& _Value)
    {
        return gs_vector2<Type>(x / _Value, y / _Value);
    }

    gs_vector2<Type> operator/(const gs_vector2<Type>& _Value)
    {
        return gs_vector2<Type>(x / _Value.x, y / _Value.y);
    }

    // +=
    gs_vector2<Type> operator+=(const Type& _Value)
    {
        x += _Value;
        y += _Value;
        return *this;
    }

    gs_vector2<Type> operator+=(const gs_vector2<Type>& _Value)
    {
        x += _Value.x;
        y += _Value.y;
        return *this;
    }

    // -=
    gs_vector2<Type> operator-=(const Type& _Value)
    {
        x -= _Value;
        y -= _Value;
        return *this;
    }

    gs_vector2<Type> operator-=(const gs_vector2<Type>& _Value)
    {
        x -= _Value.x;
        y -= _Value.y;
        return *this;
    }

    // *=
    gs_vector2<Type> operator*=(const Type& _Value)
    {
        x *= _Value;
        y *= _Value;
        return *this;
    }

    gs_vector2<Type> operator*=(const gs_vector2<Type>& _Value)
    {
        x *= _Value.x;
        y *= _Value.y;
        return *this;
    }

    // /=
    gs_vector2<Type> operator/=(const Type& _Value)
    {
        x /= _Value;
        y /= _Value;
        return *this;
    }

    gs_vector2<Type> operator/=(const gs_vector2<Type>& _Value)
    {
        x /= _Value.x;
        y /= _Value.y;
        return *this;
    }

    Type x{0};
    Type y{0};
};

template<typename Type>
struct gs_vector3
{
    gs_vector3() : x(0), y(0), z(0){}
    gs_vector3(const Type& _X) : x(_X), y(_X), z(_X){}
    gs_vector3(const Type& _X, const Type& _Y) : x(_X), y(_Y), z(0){}
    gs_vector3(const Type& _X, const Type& _Y, const Type& _Z) : x(_X), y(_Y), z(_Z){}

    const int length() const
    {
        return 3;
    }

    // []
    Type& operator[](const unsigned char& _Index)
    {
        GS_ASSERT(_Index == 0 || _Index == 1 || _Index == 2);

        if(_Index < 1) return x;
        if(_Index < 2) return y;
        return z;
    }

    // const Type[]&
    const Type& operator[](const unsigned char& _Index) const
    {
        GS_ASSERT(_Index == 0 || _Index == 1 || _Index == 2);

        if(_Index < 1) return x;
        if(_Index < 2) return y;
        return z;
    }

    // !=
    bool operator!=(const gs_vector3<Type>& _Other)
    {
        return x != _Other.x || y != _Other.y || z != _Other.z;
    }

    // ==
    bool operator==(const gs_vector3<Type>& _Other)
    {
        return x == _Other.x && y == _Other.y && z == _Other.z;
    }

    // +
    gs_vector3<Type> operator+(const Type& _Value)
    {
        return gs_vector3<Type>(x + _Value, y + _Value, z + _Value);
    }

    gs_vector3<Type> operator+(const gs_vector3<Type>& _Value)
    {
        return gs_vector3<Type>(x + _Value.x, y + _Value.y, z + _Value.z);
    }

    // -
    gs_vector3<Type> operator-(const Type& _Value)
    {
        return gs_vector3<Type>(x - _Value, y - _Value, z - _Value);
    }

    gs_vector3<Type> operator-(const gs_vector3<Type>& _Value)
    {
        return gs_vector3<Type>(x - _Value.x, y - _Value.y, z - _Value.z);
    }

    // *
    gs_vector3<Type> operator*(const Type& _Value)
    {
        return gs_vector3<Type>(x * _Value, y * _Value, z * _Value);
    }

    gs_vector3<Type> operator*(const gs_vector3<Type>& _Value)
    {
        return gs_vector3<Type>(x * _Value.x, y * _Value.y, z * _Value.z);
    }

    // /
    gs_vector3<Type> operator/(const Type& _Value)
    {
        return gs_vector3<Type>(x / _Value, y / _Value, z / _Value);
    }

    gs_vector3<Type> operator/(const gs_vector3<Type>& _Value)
    {
        return gs_vector3<Type>(x / _Value.x, y / _Value.y, z / _Value.z);
    }

    // +=
    gs_vector3<Type> operator+=(const Type& _Value)
    {
        x += _Value;
        y += _Value;
        z += _Value;
        return *this;
    }

    gs_vector3<Type> operator+=(const gs_vector3<Type>& _Value)
    {
        x += _Value.x;
        y += _Value.y;
        z += _Value.z;
        return *this;
    }

    // -=
    gs_vector3<Type> operator-=(const Type& _Value)
    {
        x -= _Value;
        y -= _Value;
        z -= _Value;
        return *this;
    }

    gs_vector3<Type> operator-=(const gs_vector3<Type>& _Value)
    {
        x -= _Value.x;
        y -= _Value.y;
        z -= _Value.z;
        return *this;
    }

    // *=
    gs_vector3<Type> operator*=(const Type& _Value)
    {
        x *= _Value;
        y *= _Value;
        z *= _Value;
        return *this;
    }

    gs_vector3<Type> operator*=(const gs_vector3<Type>& _Value)
    {
        x *= _Value.x;
        y *= _Value.y;
        z *= _Value.z;
        return *this;
    }

    // /=
    gs_vector3<Type> operator/=(const Type& _Value)
    {
        x /= _Value;
        y /= _Value;
        z /= _Value;
        return *this;
    }

    gs_vector3<Type> operator/=(const gs_vector3<Type>& _Value)
    {
        x /= _Value.x;
        y /= _Value.y;
        z /= _Value.z;
        return *this;
    }

    Type x{0};
    Type y{0};
    Type z{0};
};

template<typename Type>
struct gs_vector4
{
    gs_vector4() : x(0), y(0), z(0), w(0){}
    gs_vector4(const Type& _X) : x(_X), y(_X), z(_X), w(_X){}
    gs_vector4(const Type& _X, const Type& _Y) : x(_X), y(_Y), z(0), w(0){}
    gs_vector4(const Type& _X, const Type& _Y, const Type& _Z) : x(_X), y(_Y), z(_Z), w(0){}
    gs_vector4(const Type& _X, const Type& _Y, const Type& _Z, const Type& _W) : x(_X), y(_Y), z(_Z), w(_W){}

    const int length() const
    {
        return 4;
    }

    // []
    Type& operator[](const unsigned char& _Index)
    {
        GS_ASSERT(_Index == 0 || _Index == 1 || _Index == 2 || _Index == 3);
        if(_Index < 1) return x;
        if(_Index < 2) return y;
        if(_Index < 3) return z;
        return w;
    }

    // const Type[]&
    const Type& operator[](const unsigned char& _Index) const
    {
        GS_ASSERT(_Index == 0 || _Index == 1 || _Index == 2 || _Index == 3);
        if(_Index < 1) return x;
        if(_Index < 2) return y;
        if(_Index < 3) return z;
        return w;
    }

    // !=
    bool operator!=(const gs_vector4<Type>& _Other)
    {
        return x != _Other.x || y != _Other.y || z != _Other.z || w != _Other.w;
    }

    // ==
    bool operator==(const gs_vector4<Type>& _Other)
    {
        return x == _Other.x && y == _Other.y && z == _Other.z && w == _Other.w;
    }

    // +
    gs_vector4<Type> operator+(const Type& _Value)
    {
        return gs_vector4<Type>(x + _Value, y + _Value, z + _Value, w + _Value);
    }

    gs_vector4<Type> operator+(const gs_vector4<Type>& _Value)
    {
        return gs_vector4<Type>(x + _Value.x, y + _Value.y, z + _Value.z, w + _Value.w);
    }

    // -
    gs_vector4<Type> operator-(const Type& _Value)
    {
        return gs_vector4<Type>(x - _Value, y - _Value, z - _Value, w - _Value);
    }

    gs_vector4<Type> operator-(const gs_vector4<Type>& _Value)
    {
        return gs_vector4<Type>(x - _Value.x, y - _Value.y, z - _Value.z, w - _Value.w);
    }

    // *
    gs_vector4<Type> operator*(const Type& _Value)
    {
        return gs_vector4<Type>(x * _Value, y * _Value, z * _Value, w * _Value);
    }

    gs_vector4<Type> operator*(const gs_vector4<Type>& _Value)
    {
        return gs_vector4<Type>(x * _Value.x, y * _Value.y, z * _Value.z, w * _Value.w);
    }

    // /
    gs_vector4<Type> operator/(const Type& _Value)
    {
        return gs_vector4<Type>(x / _Value, y / _Value, z / _Value, w / _Value);
    }

    gs_vector4<Type> operator/(const gs_vector4<Type>& _Value)
    {
        return gs_vector4<Type>(x / _Value.x, y / _Value.y, z / _Value.z, w / _Value.w);
    }

    // +=
    gs_vector4<Type> operator+=(const Type& _Value)
    {
        x += _Value;
        y += _Value;
        z += _Value;
        w += _Value;
        return *this;
    }

    gs_vector4<Type> operator+=(const gs_vector4<Type>& _Value)
    {
        x += _Value.x;
        y += _Value.y;
        z += _Value.z;
        w += _Value.w;
        return *this;
    }

    // -=
    gs_vector4<Type> operator-=(const Type& _Value)
    {
        x -= _Value;
        y -= _Value;
        z -= _Value;
        w -= _Value;
        return *this;
    }

    gs_vector4<Type> operator-=(const gs_vector4<Type>& _Value)
    {
        x -= _Value.x;
        y -= _Value.y;
        z -= _Value.z;
        w -= _Value.w;
        return *this;
    }

    // *=
    gs_vector4<Type> operator*=(const Type& _Value)
    {
        x *= _Value;
        y *= _Value;
        z *= _Value;
        w *= _Value;
        return *this;
    }

    gs_vector4<Type> operator*=(const gs_vector4<Type>& _Value)
    {
        x *= _Value.x;
        y *= _Value.y;
        z *= _Value.z;
        w *= _Value.w;
        return *this;
    }

    // /=
    gs_vector4<Type> operator/=(const Type& _Value)
    {
        x /= _Value;
        y /= _Value;
        z /= _Value;
        w /= _Value;
        return *this;
    }

    gs_vector4<Type> operator/=(const gs_vector4<Type>& _Value)
    {
        x /= _Value.x;
        y /= _Value.y;
        z /= _Value.z;
        w /= _Value.w;
        return *this;
    }

    Type x{0};
    Type y{0};
    Type z{0};
    Type w{0};
};

template<typename Type, int Rows, int Columns>
struct gs_matrix
{
public:
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
    gs_matrix<Type, Rows, Columns> operator+(const gs_matrix<Type, Rows, Columns>& _Mat)
    {
        gs_matrix<Type, Rows, Columns> result;
        add_mat(_Mat, *this, result);
        return result;
    }

    // +=
    gs_matrix<Type, Rows, Columns> operator+=(const gs_matrix<Type, Rows, Columns>& _Mat)
    {
        gs_matrix<Type, Rows, Columns> result;
        add_mat(_Mat, *this, result);
        *this = result;
        return *this;
    }

    // -
    gs_matrix<Type, Rows, Columns> operator-(const gs_matrix<Type, Rows, Columns>& _Mat)
    {
        gs_matrix<Type, Rows, Columns> result;
        sub_mat(_Mat, *this, result);
        return result;
    }

    // -=
    gs_matrix<Type, Rows, Columns> operator-=(const gs_matrix<Type, Rows, Columns>& _Mat)
    {
        gs_matrix<Type, Rows, Columns> result;
        sub_mat(_Mat, *this, result);
        *this = result;
        return *this;
    }

    // *
    template<int Dimention>
    gs_matrix<Type, Rows, Dimention> operator*(const gs_matrix<Type, Columns, Dimention>& _Mat)
    {
        gs_matrix<Type, Rows, Dimention> result;
        mul_mat(*this, _Mat, result);
        return result;
    }

    // *=
    template<int Dimention>
    gs_matrix<Type, Rows, Dimention> operator*=(const gs_matrix<Type, Columns, Dimention>& _Mat)
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
        gs_matrix<Type, Rows, Columns>&       _C)
    {
        for (int i = 0; i < Size; i++)
            _C.Data[i] = _A.Data[i] + _B.Data[i];
    }

    void sub_mat(
        const gs_matrix<Type, Rows, Columns>& _A,
        const gs_matrix<Type, Rows, Columns>& _B,
        gs_matrix<Type, Rows, Columns>&       _C)
    {
        for (int i = 0; i < Size; i++)
            _C.Data[i] = _A.Data[i] - _B.Data[i];
    }

    template<int Dimention>
    void mul_mat(
        const gs_matrix<Type, Rows, Columns>&      _A,
        const gs_matrix<Type, Columns, Dimention>& _B,
        gs_matrix<Type, Rows, Dimention>&          _C)
    {
        // multiply
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

typedef gs_vector2<float> gs_vector2f;
typedef gs_vector3<float> gs_vector3f;
typedef gs_vector4<float> gs_vector4f;

typedef gs_vector2<double> gs_vector2d;
typedef gs_vector3<double> gs_vector3d;
typedef gs_vector4<double> gs_vector4d;

typedef gs_vector2<int> gs_vector2i;
typedef gs_vector3<int> gs_vector3i;
typedef gs_vector4<int> gs_vector4i;

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

template<typename Type>
inline double gs_sum_of_squares(const gs_vector2<Type>& _Vector)
{
    return gs_sum_of_squares(_Vector.x, _Vector.y);
}

template<typename Type>
inline double gs_sum_of_squares(const gs_vector3<Type>& _Vector)
{
    return gs_sum_of_squares(_Vector.x, _Vector.y, _Vector.z);
}

template<typename Type>
inline double gs_sum_of_squares(const gs_vector4<Type>& _Vector)
{
    return gs_sum_of_squares(_Vector.x, _Vector.y, _Vector.z, _Vector.w);
}

template<typename Type, typename ... Args>
inline double gs_vector_length(Type _A, Type _B, Args... _Args)
{
    double sumOfSquares = gs_sum_of_squares(_A, _B, _Args ...);
    return sumOfSquares > 0 ? sqrt(sumOfSquares) : 0;
}

template<typename Type>
inline double gs_vector_length(const gs_vector2<Type>& _Vector)
{
    return gs_vector_length(_Vector.x, _Vector.y);
}

template<typename Type>
inline double gs_vector_length(const gs_vector3<Type>& _Vector)
{
    return gs_vector_length(_Vector.x, _Vector.y, _Vector.z);
}

template<typename Type>
inline double gs_vector_length(const gs_vector4<Type>& _Vector)
{
    return gs_vector_length(_Vector.x, _Vector.y, _Vector.z, _Vector.w);
}

template<typename Type>
inline double gs_vector_dot(const gs_vector2<Type>& _A, const gs_vector2<Type>& _B)
{
    return _A.x * _B.x + _A.y * _B.y;
}

template<typename Type>
inline double gs_vector_dot(const gs_vector3<Type>& _A, const gs_vector3<Type>& _B)
{
    return _A.x * _B.x + _A.y * _B.y + _A.z * _B.z;
}

template<typename Type>
inline double gs_vector_dot(const gs_vector4<Type>& _A, const gs_vector4<Type>& _B)
{
    return _A.x * _B.x + _A.y * _B.y + _A.z * _B.z + _A.w * _B.w;
}

// cross product of 2D vectors returns the scalar value equal to the area of the parallelogram formed by two input vectors
template<typename Type>
inline gs_vector2<Type> gs_vector_cross(const gs_vector2<Type>& _A, const gs_vector2<Type>& _B)
{
    return _A.x * _B.y - _A.y * _B.x;
}

// cross product of 3D vectors returns the vector perpedicular to cross multiplied vectors and the length
// of resulting vector is equal to the area of the parallelogram formed by two input vectors
template<typename Type>
inline gs_vector3<Type> gs_vector_cross(const gs_vector3<Type>& _A, const gs_vector3<Type>& _B)
{
    return gs_vector3<Type>(
        _A.y * _B.z - _B.y * _A.z,
        _A.z * _B.x - _B.z * _A.x,
        _A.x * _B.y - _B.x * _A.y);
}

template<typename Type, int Rows, int Columns>
void gs_print_matrix(const gs_matrix<Type, Rows, Columns>& _Matrix)
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

/*
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

void gs_autotests_gs_vector2()
{
    printf("running %s\n", GS_STRINGIFY(gs_autotests_gs_vector2()));

    // gs_vector2
    {
        float ax  = 1.f;
        float ay  = 2.f;
        float bx  = 3.f;
        float by  = 4.f;
        float val = 5.f;

        GS_ASSERT(gs_vector2<float>(ax).x == ax && gs_vector2<float>(ax).y == ax);
        printf("gs_vector2<T>(val) success... \n");

        gs_vector2<float> vec{ax, ay};
        GS_ASSERT(vec[0] == ax &&
                vec[1] == ay &&
                vec.x == ax &&
                vec.y == ay);
        printf("gs_vector2<T> success... \n");
    
        GS_ASSERT(gs_sum_of_squares(vec) == gs_sum_of_squares(vec.x, vec.y));
        printf("gs_sum_of_squares<gs_vector2<T>> success... \n");
    
        GS_ASSERT(gs_vector_length(vec) == gs_vector_length(vec.x, vec.y));
        GS_ASSERT("gs_vector_length<gs_vector2<T>> success... \n");

        gs_vector2<float> vec_a{ax, ay};
        gs_vector2<float> vec_b{bx, by};
        gs_vector2<float> vec_c;
        vec_c = vec_a;

        GS_ASSERT((vec_c == vec_a));
        printf("gs_vector2<T> = gs_vector2<T> success... \n");

        GS_ASSERT(!(vec_a == vec_b));
        printf("gs_vector2<T> == gs_vector2<T> success... \n");

        GS_ASSERT((vec_a != vec_b));
        printf("gs_vector2<T> != gs_vector2<T> success... \n");

        GS_ASSERT(((vec_a + val).x == (ax + val) && (vec_a + val).y == (ay + val)));
        GS_ASSERT(!((vec_a + val).x == (ax + gs_tiny<float>()) && (vec_a + gs_tiny<float>()).y == (ay + gs_tiny<float>())));
        printf("gs_vector2<T> + T success... \n");

        GS_ASSERT((vec_a - val).x == (ax - val) && (vec_a - val).y == (ay - val));
        GS_ASSERT(!((vec_a - val).x == (ax - gs_tiny<float>()) && (vec_a - val).y == (ay - gs_tiny<float>())));
        printf("gs_vector2<T> - T success... \n");

        GS_ASSERT((vec_a * val).x == (ax * val) && (vec_a * val).y == (ay * val));
        GS_ASSERT(!((vec_a * val).x == (ax * gs_tiny<float>()) && (vec_a * val).y == (ay * gs_tiny<float>())));
        printf("gs_vector2<T> * T success... \n");

        GS_ASSERT((vec_a / val).x == (ax / val) && (vec_a / val).y == (ay / val));
        GS_ASSERT(!((vec_a / val).x == (ax / gs_tiny<float>()) && (vec_a / val).y == (ay / gs_tiny<float>())));
        printf("gs_vector2<T> / T success... \n");

        GS_ASSERT((vec_a + vec_b).x == (ax + bx) && (vec_a + vec_b).y == (ay + by));
        GS_ASSERT(!((vec_a + vec_b).x == (ax + gs_tiny<float>()) && (vec_a + vec_b).y == (ay + gs_tiny<float>())));
        printf("gs_vector2<T> + gs_vector2<T> success... \n");

        GS_ASSERT((vec_a - vec_b).x == (ax - bx) && (vec_a - vec_b).y == (ay - by));
        GS_ASSERT(!((vec_a - vec_b).x == (ax - gs_tiny<float>()) && (vec_a - vec_b).y == (ay - gs_tiny<float>())));
        printf("gs_vector2<T> - gs_vector2<T> success... \n");

        GS_ASSERT((vec_a * vec_b).x == (ax * bx) && (vec_a * vec_b).y == (ay * by));
        GS_ASSERT(!((vec_a * vec_b).x == (ax * gs_tiny<float>()) && (vec_a * vec_b).y == (ay * gs_tiny<float>())));
        printf("gs_vector2<T> * gs_vector2<T> success... \n");

        GS_ASSERT((vec_a / vec_b).x == (ax / bx) && (vec_a / vec_b).y == (ay / by));
        GS_ASSERT(!((vec_a / vec_b).x == (ax / gs_tiny<float>()) && (vec_a / vec_b).y == (ay / gs_tiny<float>())));
        printf("gs_vector2<T> / gs_vector2<T> success... \n");

        GS_ASSERT((gs_vector2<float>(ax, ay) += val).x == (ax + val) &&
            (gs_vector2<float>(ax, ay) += val).y == (ay + val));

        GS_ASSERT(!((gs_vector2<float>(ax, ay) += val).x == (ax + gs_tiny<float>()) &&
            (gs_vector2<float>(ax, ay) += val).y == (ay + gs_tiny<float>())));
        printf("gs_vector2<T> += T success... \n");

        GS_ASSERT((gs_vector2<float>(ax, ay) -= val).x == (ax - val) &&
            (gs_vector2<float>(ax, ay) -= val).y == (ay - val));
        GS_ASSERT(!((gs_vector2<float>(ax, ay) -= val).x == (ax - gs_tiny<float>()) &&
            (gs_vector2<float>(ax, ay) -= val).y == (ay - gs_tiny<float>())));
        printf("gs_vector2<T> =- T success... \n");

        GS_ASSERT((gs_vector2<float>(ax, ay) *= val).x == (ax * val) &&
            (gs_vector2<float>(ax, ay) *= val).y == (ay * val));
        GS_ASSERT(!((gs_vector2<float>(ax, ay) *= val).x == (ax * gs_tiny<float>()) &&
            (gs_vector2<float>(ax, ay) *= val).y == (ay * gs_tiny<float>())));
        printf("gs_vector2<T> *= T success... \n");

        GS_ASSERT((gs_vector2<float>(ax, ay) /= val).x == (ax / val) &&
            (gs_vector2<float>(ax, ay) /= val).y == (ay / val));
        GS_ASSERT(!((gs_vector2<float>(ax, ay) /= val).x == (ax / gs_tiny<float>()) &&
            (gs_vector2<float>(ax, ay) /= val).y == (ay / gs_tiny<float>())));
        printf("gs_vector2<T> /= T success... \n");

        GS_ASSERT((gs_vector2<float>(ax, ay) += gs_vector2<float>(bx, by)).x == (ax + bx) &&
            (gs_vector2<float>(ax, ay) += gs_vector2<float>(bx, by)).y == (ay + by));
        GS_ASSERT(!((gs_vector2<float>(ax, ay) += gs_vector2<float>(bx, by)).x == (ax + gs_tiny<float>()) &&
            (gs_vector2<float>(ax, ay) += gs_vector2<float>(bx, by)).y == (ay + gs_tiny<float>())));
        printf("gs_vector2<T> += gs_vector2<T> success... \n");

        GS_ASSERT((gs_vector2<float>(ax, ay) -= gs_vector2<float>(bx, by)).x == (ax - bx) &&
            (gs_vector2<float>(ax, ay) -= gs_vector2<float>(bx, by)).y == (ay - by));
        GS_ASSERT(!((gs_vector2<float>(ax, ay) -= gs_vector2<float>(bx, by)).x == (ax - gs_tiny<float>()) &&
            (gs_vector2<float>(ax, ay) -= gs_vector2<float>(bx, by)).y == (ay - gs_tiny<float>())));
        printf("gs_vector2<T> =- gs_vector2<T> success... \n");

        GS_ASSERT((gs_vector2<float>(ax, ay) *= gs_vector2<float>(bx, by)).x == (ax * bx) &&
            (gs_vector2<float>(ax, ay) *= gs_vector2<float>(bx, by)).y == (ay * by));
        GS_ASSERT(!((gs_vector2<float>(ax, ay) *= gs_vector2<float>(bx, by)).x == (ax * gs_tiny<float>()) &&
            (gs_vector2<float>(ax, ay) *= gs_vector2<float>(bx, by)).y == (ay * gs_tiny<float>())));
        printf("gs_vector2<T> *= gs_vector2<T> success... \n");

        GS_ASSERT((gs_vector2<float>(ax, ay) /= gs_vector2<float>(bx, by)).x == (ax / bx) &&
            (gs_vector2<float>(ax, ay) /= gs_vector2<float>(bx, by)).y == (ay / by));
        GS_ASSERT(!((gs_vector2<float>(ax, ay) /= gs_vector2<float>(bx, by)).x == (ax / gs_tiny<float>()) &&
            (gs_vector2<float>(ax, ay) /= gs_vector2<float>(bx, by)).y == (ay / gs_tiny<float>())));
        printf("gs_vector2<T> /= gs_vector2<T> success... \n");
    }

    printf("\n");
}

void gs_autotests_gs_vector3()
{
    printf("running %s\n", GS_STRINGIFY(gs_autotests_gs_vector3()));

    // gs_vector2
    {
        float ax  = 1.f;
        float ay  = 2.f;
        float az  = 3.f;
        float bx  = 4.f;
        float by  = 5.f;
        float bz  = 6.f;
        float val = 7.f;

        GS_ASSERT(gs_vector3<float>(ax).x == ax &&
                gs_vector3<float>(ax).y == ax &&
                gs_vector3<float>(ax).z == ax);
        printf("gs_vector3<T>(val) success... \n");

        GS_ASSERT(gs_vector3<float>(ax, ay).x == ax &&
                gs_vector3<float>(ax, ay).y == ay &&
                gs_vector3<float>(ax, ay).z == 0.f);
        printf("gs_vector3<T>(val, val) success... \n");

        gs_vector3<float> vec{ax, ay, az};
        GS_ASSERT(vec[0] == ax  &&
                vec[1] == ay &&
                vec[2] == az &&
                vec.x == ax  &&
                vec.y == ay  &&
                vec.z == az);
        printf("gs_vector3<T> success... \n");
    
        GS_ASSERT(gs_sum_of_squares(vec) == gs_sum_of_squares(vec.x, vec.y, vec.z));
        printf("gs_sum_of_squares<gs_vector3<T>> success... \n");
    
        GS_ASSERT(gs_vector_length(vec) == gs_vector_length(vec.x, vec.y, vec.z));
        GS_ASSERT("gs_vector_length<gs_vector3<T>> success... \n");

        gs_vector3<float> vec_a{ax, ay, az};
        gs_vector3<float> vec_b{bx, by, bz};
        gs_vector3<float> vec_c;
        vec_c = vec_a;

        GS_ASSERT((vec_a == vec_c));
        printf("gs_vector3<T> = gs_vector3<T> success... \n");

        GS_ASSERT(!(vec_a == vec_b));
        printf("gs_vector3<T> == gs_vector3<T> success... \n");

        GS_ASSERT((vec_a != vec_b));
        printf("gs_vector3<T> != gs_vector3<T> success... \n");

        GS_ASSERT((vec_a + val).x == (ax + val) &&
                (vec_a + val).y == (ay + val) &&
                (vec_a + val).z == (az + val));
        printf("gs_vector3<T> + T success... \n");

        GS_ASSERT((vec_a - val).x == (ax - val) &&
                (vec_a - val).y == (ay - val) &&
                (vec_a - val).z == (az - val));
        printf("gs_vector3<T> - T success... \n");

        GS_ASSERT((vec_a * val).x == (ax * val) &&
                (vec_a * val).y == (ay * val) &&
                (vec_a * val).z == (az * val));
        printf("gs_vector3<T> * T success... \n");

        GS_ASSERT((vec_a / val).x == (ax / val) &&
                (vec_a / val).y == (ay / val) &&
                (vec_a / val).z == (az / val));
        printf("gs_vector3<T> / T success... \n");

        GS_ASSERT((vec_a + vec_b).x == (ax + bx) &&
                (vec_a + vec_b).y == (ay + by) &&
                (vec_a + vec_b).z == (az + bz));
        printf("gs_vector3<T> + gs_vector3<T> success... \n");

        GS_ASSERT((vec_a - vec_b).x == (ax - bx) &&
                (vec_a - vec_b).y == (ay - by) &&
                (vec_a - vec_b).z == (az - bz));
        printf("gs_vector3<T> - gs_vector3<T> success... \n");

        GS_ASSERT((vec_a * vec_b).x == (ax * bx) &&
                (vec_a * vec_b).y == (ay * by) &&
                (vec_a * vec_b).z == (az * bz));
        printf("gs_vector3<T> * gs_vector3<T> success... \n");

        GS_ASSERT((vec_a / vec_b).x == (ax / bx) &&
                (vec_a / vec_b).y == (ay / by) &&
                (vec_a / vec_b).z == (az / bz));
        printf("gs_vector3<T> / gs_vector3<T> success... \n");

        GS_ASSERT((gs_vector3<float>(ax, ay, az) += val).x == (ax + val) &&
                (gs_vector3<float>(ax, ay, az) += val).y == (ay + val) &&
                (gs_vector3<float>(ax, ay, az) += val).z == (az + val));
        printf("gs_vector3<T> += T success... \n");

        GS_ASSERT((gs_vector3<float>(ax, ay, az) -= val).x == (ax - val) &&
                (gs_vector3<float>(ax, ay, az) -= val).y == (ay - val) &&
                (gs_vector3<float>(ax, ay, az) -= val).z == (az - val));
        printf("gs_vector3<T> =- T success... \n");

        GS_ASSERT((gs_vector3<float>(ax, ay, az) *= val).x == (ax * val) &&
                (gs_vector3<float>(ax, ay, az) *= val).y == (ay * val) &&
                (gs_vector3<float>(ax, ay, az) *= val).z == (az * val));
        printf("gs_vector3<T> *= T success... \n");

        GS_ASSERT((gs_vector3<float>(ax, ay, az) /= val).x == (ax / val) &&
                (gs_vector3<float>(ax, ay, az) /= val).y == (ay / val) &&
                (gs_vector3<float>(ax, ay, az) /= val).z == (az / val));
        printf("gs_vector3<T> /= T success... \n");

        GS_ASSERT((gs_vector3<float>(ax, ay, az) += gs_vector3<float>(bx, by, bz)).x == (ax + bx) &&
                (gs_vector3<float>(ax, ay, az) += gs_vector3<float>(bx, by, bz)).y == (ay + by) &&
                (gs_vector3<float>(ax, ay, az) += gs_vector3<float>(bx, by, bz)).z == (az + bz));
        printf("gs_vector3<T> += gs_vector3<T> success... \n");

        GS_ASSERT((gs_vector3<float>(ax, ay, az) -= gs_vector3<float>(bx, by, bz)).x == (ax - bx) &&
                (gs_vector3<float>(ax, ay, az) -= gs_vector3<float>(bx, by, bz)).y == (ay - by) &&
                (gs_vector3<float>(ax, ay, az) -= gs_vector3<float>(bx, by, bz)).z == (az - bz));
        printf("gs_vector3<T> =- gs_vector3<T> success... \n");

        GS_ASSERT((gs_vector3<float>(ax, ay, az) *= gs_vector3<float>(bx, by, bz)).x == (ax * bx) &&
                (gs_vector3<float>(ax, ay, az) *= gs_vector3<float>(bx, by, bz)).y == (ay * by) &&
                (gs_vector3<float>(ax, ay, az) *= gs_vector3<float>(bx, by, bz)).z == (az * bz));
        printf("gs_vector3<T> *= gs_vector3<T> success... \n");

        GS_ASSERT((gs_vector3<float>(ax, ay, az) /= gs_vector3<float>(bx, by, bz)).x == (ax / bx) &&
                (gs_vector3<float>(ax, ay, az) /= gs_vector3<float>(bx, by, bz)).y == (ay / by) &&
                (gs_vector3<float>(ax, ay, az) /= gs_vector3<float>(bx, by, bz)).z == (az / bz));
        printf("gs_vector3<T> /= gs_vector3<T> success... \n");
    }

    printf("\n");
}
void gs_autotests_gs_vector4()
{
    printf("running %s\n", GS_STRINGIFY(gs_autotests_gs_vector4()));

    // gs_vector2
    {
        float ax  = 1.f;
        float ay  = 2.f;
        float az  = 3.f;
        float aw  = 4.f;
        float bx  = 5.f;
        float by  = 6.f;
        float bz  = 7.f;
        float bw  = 8.f;
        float val = 2.f;

        GS_ASSERT(gs_vector4<float>(ax).x == ax &&
                gs_vector4<float>(ax).y == ax &&
                gs_vector4<float>(ax).z == ax &&
                gs_vector4<float>(ax).w == ax);
        printf("gs_vector4<T>(val) success... \n");

        GS_ASSERT(gs_vector4<float>(ax, ay).x == ax &&
                gs_vector4<float>(ax, ay).y == ay  &&
                gs_vector4<float>(ax, ay).z == 0.f &&
                gs_vector4<float>(ax, ay).w == 0.f);
        printf("gs_vector4<T>(val, val) success... \n");

        GS_ASSERT(gs_vector4<float>(ax, ay, az).x == ax &&
                gs_vector4<float>(ax, ay, az).y == ay  &&
                gs_vector4<float>(ax, ay, az).z == az &&
                gs_vector4<float>(ax, ay).w == 0.f);
        printf("gs_vector4<T>(val, val, val) success... \n");

        gs_vector4<float> vec{ax, ay, az, aw};
        GS_ASSERT(vec[0] == ax  &&
                vec[1] == ay &&
                vec[2] == az &&
                vec[3] == aw &&
                vec.x == ax  &&
                vec.y == ay  &&
                vec.z == az &&
                vec.w == aw);
        printf("gs_vector4<T> success... \n");
    
        GS_ASSERT(gs_sum_of_squares(vec) == gs_sum_of_squares(vec.x, vec.y, vec.z, vec.w));
        printf("gs_sum_of_squares<gs_vector4<T>> success... \n");
    
        GS_ASSERT(gs_vector_length(vec) == gs_vector_length(vec.x, vec.y, vec.z, vec.w));
        GS_ASSERT("gs_vector_length<gs_vector4<T>> success... \n");

        gs_vector4<float> vec_a{ax, ay, az, aw};
        gs_vector4<float> vec_b{bx, by, bz, bw};
        gs_vector4<float> vec_c;
        vec_c = vec_a;

        GS_ASSERT((vec_a == vec_c));
        printf("gs_vector4<T> = gs_vector4<T> success... \n");

        GS_ASSERT(!(vec_a == vec_b));
        printf("gs_vector4<T> == gs_vector4<T> success... \n");

        GS_ASSERT((vec_a != vec_b));
        printf("gs_vector4<T> != gs_vector4<T> success... \n");

        GS_ASSERT((vec_a + val).x == (ax + val) &&
                (vec_a + val).y == (ay + val) &&
                (vec_a + val).z == (az + val) &&
                (vec_a + val).w == (aw + val));
        printf("gs_vector4<T> + T success... \n");

        GS_ASSERT((vec_a - val).x == (ax - val) &&
                (vec_a - val).y == (ay - val) &&
                (vec_a - val).z == (az - val) &&
                (vec_a - val).w == (aw - val));
        printf("gs_vector4<T> - T success... \n");

        GS_ASSERT((vec_a * val).x == (ax * val) &&
                (vec_a * val).y == (ay * val) &&
                (vec_a * val).z == (az * val) &&
                (vec_a * val).w == (aw * val));
        printf("gs_vector4<T> * T success... \n");

        GS_ASSERT((vec_a / val).x == (ax / val) &&
                (vec_a / val).y == (ay / val) &&
                (vec_a / val).z == (az / val) &&
                (vec_a / val).w == (aw / val));
        printf("gs_vector4<T> / T success... \n");

        GS_ASSERT((vec_a + vec_b).x == (ax + bx) &&
                (vec_a + vec_b).y == (ay + by) &&
                (vec_a + vec_b).z == (az + bz) &&
                (vec_a + vec_b).w == (aw + bw));
        printf("gs_vector4<T> + gs_vector4<T> success... \n");

        GS_ASSERT((vec_a - vec_b).x == (ax - bx) &&
                (vec_a - vec_b).y == (ay - by) &&
                (vec_a - vec_b).z == (az - bz) &&
                (vec_a - vec_b).w == (aw - bw));
        printf("gs_vector4<T> - gs_vector4<T> success... \n");

        GS_ASSERT((vec_a * vec_b).x == (ax * bx) &&
                (vec_a * vec_b).y == (ay * by) &&
                (vec_a * vec_b).z == (az * bz) &&
                (vec_a * vec_b).w == (aw * bw));
        printf("gs_vector4<T> * gs_vector4<T> success... \n");

        GS_ASSERT((vec_a / vec_b).x == (ax / bx) &&
                (vec_a / vec_b).y == (ay / by) &&
                (vec_a / vec_b).z == (az / bz) &&
                (vec_a / vec_b).w == (aw / bw));
        printf("gs_vector4<T> / gs_vector4<T> success... \n");

        GS_ASSERT((gs_vector4<float>(ax, ay, az, aw) += val).x == (ax + val) &&
                (gs_vector4<float>(ax, ay, az, aw) += val).y == (ay + val) &&
                (gs_vector4<float>(ax, ay, az, aw) += val).z == (az + val) &&
                (gs_vector4<float>(ax, ay, az, aw) += val).w == (aw + val));
        printf("gs_vector4<T> += T success... \n");

        GS_ASSERT((gs_vector4<float>(ax, ay, az, aw) -= val).x == (ax - val) &&
                (gs_vector4<float>(ax, ay, az, aw) -= val).y == (ay - val) &&
                (gs_vector4<float>(ax, ay, az, aw) -= val).z == (az - val) &&
                (gs_vector4<float>(ax, ay, az, aw) -= val).w == (aw - val));
        printf(", gs_vector4<T> =- T success... \n");

        GS_ASSERT((gs_vector4<float>(ax, ay, az, aw) *= val).x == (ax * val) &&
                (gs_vector4<float>(ax, ay, az, aw) *= val).y == (ay * val) &&
                (gs_vector4<float>(ax, ay, az, aw) *= val).z == (az * val) &&
                (gs_vector4<float>(ax, ay, az, aw) *= val).w == (aw * val));
        printf("gs_vector4<T> *= T success... \n");

        GS_ASSERT((gs_vector4<float>(ax, ay, az, aw) /= val).x == (ax / val) &&
                (gs_vector4<float>(ax, ay, az, aw) /= val).y == (ay / val) &&
                (gs_vector4<float>(ax, ay, az, aw) /= val).z == (az / val) &&
                (gs_vector4<float>(ax, ay, az, aw) /= val).w == (aw / val));
        printf("gs_vector4<T> /= T success... \n");

        GS_ASSERT((gs_vector4<float>(ax, ay, az, aw) += gs_vector4<float>(bx, by, bz, bw)).x == (ax + bx) &&
                (gs_vector4<float>(ax, ay, az, aw) += gs_vector4<float>(bx, by, bz, bw)).y == (ay + by) &&
                (gs_vector4<float>(ax, ay, az, aw) += gs_vector4<float>(bx, by, bz, bw)).z == (az + bz) &&
                (gs_vector4<float>(ax, ay, az, aw) += gs_vector4<float>(bx, by, bz, bw)).w == (aw + bw));
        printf("gs_vector4<T> += gs_vector4<T> success... \n");

        GS_ASSERT((gs_vector4<float>(ax, ay, az, aw) -= gs_vector4<float>(bx, by, bz, bw)).x == (ax - bx) &&
                (gs_vector4<float>(ax, ay, az, aw) -= gs_vector4<float>(bx, by, bz, bw)).y == (ay - by) &&
                (gs_vector4<float>(ax, ay, az, aw) -= gs_vector4<float>(bx, by, bz, bw)).z == (az - bz) &&
                (gs_vector4<float>(ax, ay, az, aw) -= gs_vector4<float>(bx, by, bz, bw)).w == (aw - bw));
        printf("gs_vector4<T> =- gs_vector4<T> success... \n");

        GS_ASSERT((gs_vector4<float>(ax, ay, az, aw) *= gs_vector4<float>(bx, by, bz, bw)).x == (ax * bx) &&
                (gs_vector4<float>(ax, ay, az, aw) *= gs_vector4<float>(bx, by, bz, bw)).y == (ay * by) &&
                (gs_vector4<float>(ax, ay, az, aw) *= gs_vector4<float>(bx, by, bz, bw)).z == (az * bz) &&
                (gs_vector4<float>(ax, ay, az, aw) *= gs_vector4<float>(bx, by, bz, bw)).w == (aw * bw));
        printf("gs_vector4<T> *= gs_vector4<T> success... \n");

        GS_ASSERT((gs_vector4<float>(ax, ay, az, aw) /= gs_vector4<float>(bx, by, bz, bw)).x == (ax / bx) &&
                (gs_vector4<float>(ax, ay, az, aw) /= gs_vector4<float>(bx, by, bz, bw)).y == (ay / by) &&
                (gs_vector4<float>(ax, ay, az, aw) /= gs_vector4<float>(bx, by, bz, bw)).z == (az / bz) &&
                (gs_vector4<float>(ax, ay, az, aw) /= gs_vector4<float>(bx, by, bz, bw)).w == (aw / bw));
        printf("gs_vector4<T> /= gs_vector4<T> success... \n");
    }

    printf("\n");
}
*/

int main(int argc, char *argv[])
{
    // gs_autotests_gs_utilities();
    // gs_autotests_gs_vector2();
    // gs_autotests_gs_vector3();
    // gs_autotests_gs_vector4();

    typedef gs_matrix<float, 2, 2> mat2x2f;
    typedef gs_matrix<float, 3, 2> mat3x2f;
    typedef gs_matrix<float, 2, 3> mat2x3f;

    // mat2x2f mat_a(1.f);
    // mat_a[0][0] = 1.f;
    // mat_a[0][1] = 2.f;
    // mat_a[1][0] = 3.f;
    // mat_a[1][1] = 4.f;

    // mat2x2f mat_b(1.f);
    // mat_b[0][0] = 5.f;
    // mat_b[0][1] = 6.f;
    // mat_b[1][0] = 7.f;
    // mat_b[1][1] = 8.f;

    // mat2x2f mat_c = mat_a + mat_b;
    // mat2x2f mat_d = mat_a - mat_b;

    // printf("mat_a:\n");
    // gs_print_matrix(mat_a);
    // printf("mat_b:\n");
    // gs_print_matrix(mat_b);
    // printf("mat_c:\n");
    // gs_print_matrix(mat_c);
    // printf("mat_d:\n");
    // gs_print_matrix(mat_d);

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
        gs_print_matrix(mat_a);
        printf("mat_b:\n");
        gs_print_matrix(mat_b);
        printf("mat_c:\n");
        gs_print_matrix(mat_c);
    }

    return 0;
}
