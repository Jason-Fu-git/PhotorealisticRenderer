/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef VECTOR_4F_H
#define VECTOR_4F_H

class Vector3f;

class Vector4f
{
public:

	__device__ __host__ Vector4f( float f = 0.f );
    __device__ __host__ Vector4f( float fx, float fy, float fz, float fw );
    __device__ __host__ Vector4f( float buffer[ 4 ] );


    __device__ __host__ Vector4f( const Vector3f& xyz, float w );
    __device__ __host__ Vector4f( float x, const Vector3f& yzw );

	// copy constructors
    __device__ __host__ Vector4f( const Vector4f& rv );

	// assignment operators
    __device__ __host__ Vector4f& operator = ( const Vector4f& rv );

	// no destructor necessary

	// returns the ith element
    __device__ __host__ const float& operator [] ( int i ) const;
    __device__ __host__ float& operator [] ( int i );

	__device__ __host__ float& x();
	__device__ __host__ float& y();
	__device__ __host__ float& z();
	__device__ __host__ float& w();

	__device__ __host__ float x() const;
	__device__ __host__ float y() const;
	__device__ __host__ float z() const;
	__device__ __host__ float w() const;

	__device__ __host__ Vector3f xyz() const;
	__device__ __host__ Vector3f yzw() const;
	__device__ __host__ Vector3f zwx() const;
	__device__ __host__ Vector3f wxy() const;

	__device__ __host__ Vector3f xyw() const;
	__device__ __host__ Vector3f yzx() const;
	__device__ __host__ Vector3f zwy() const;
	__device__ __host__ Vector3f wxz() const;

	__device__ __host__ float abs() const;
	__device__ __host__ float absSquared() const;
	__device__ __host__ void normalize();
	__device__ __host__ Vector4f normalized() const;

	// if v.z != 0, v = v / v.w
	__device__ __host__ void homogenize();
	__device__ __host__ Vector4f homogenized() const;

    __device__ __host__ void negate();

	// ---- Utility ----
	__device__ __host__ operator const float* () const; // automatic type conversion for OpenGL
	__device__ __host__ operator float* (); // automatic type conversion for OpenG
	__device__ __host__ void print() const;

	__device__ __host__ static float dot( const Vector4f& v0, const Vector4f& v1 );
	__device__ __host__ static Vector4f lerp( const Vector4f& v0, const Vector4f& v1, float alpha );

private:

	float m_elements[ 4 ];

};

// component-wise operators
__device__ __host__ Vector4f operator + ( const Vector4f& v0, const Vector4f& v1 );
__device__ __host__ Vector4f operator - ( const Vector4f& v0, const Vector4f& v1 );
__device__ __host__ Vector4f operator * ( const Vector4f& v0, const Vector4f& v1 );
__device__ __host__ Vector4f operator / ( const Vector4f& v0, const Vector4f& v1 );

// unary negation
__device__ __host__ Vector4f operator - ( const Vector4f& v );

// multiply and divide by scalar
__device__ __host__ Vector4f operator * ( float f, const Vector4f& v );
__device__ __host__ Vector4f operator * ( const Vector4f& v, float f );
__device__ __host__ Vector4f operator / ( const Vector4f& v, float f );

__device__ __host__ bool operator == ( const Vector4f& v0, const Vector4f& v1 );
__device__ __host__ bool operator != ( const Vector4f& v0, const Vector4f& v1 );

#endif // VECTOR_4F_H
