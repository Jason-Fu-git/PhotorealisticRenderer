/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef MATRIX4F_H
#define MATRIX4F_H

#include <cstdio>

class Matrix3f;
class Vector3f;
class Vector4f;

// 4x4 Matrix, stored in column major order (OpenGL style)
class Matrix4f
{
public:

    // Fill a 4x4 matrix with "fill".  Default to 0.
	__device__ __host__ Matrix4f( float fill = 0.f );
    __device__ __host__ Matrix4f( float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33 );
	
	// setColumns = true ==> sets the columns of the matrix to be [v0 v1 v2 v3]
	// otherwise, sets the rows
    __device__ __host__ Matrix4f( const Matrix4f& rm ); // copy constructor
    __device__ __host__ Matrix4f& operator = ( const Matrix4f& rm ); // assignment operator
    __device__ __host__ Matrix4f& operator/=(float d);
	// no destructor necessary

    __device__ __host__ const float& operator () ( int i, int j ) const;
    __device__ __host__ float& operator () ( int i, int j );


	// gets the 3x3 submatrix of this matrix to m
	// starting with upper left corner at (i0, j0)
    __device__ __host__ Matrix3f getSubmatrix3x3( int i0, int j0 ) const;


	// sets a 3x3 submatrix of this matrix to m
	// starting with upper left corner at (i0, j0)
    __device__ __host__ void setSubmatrix3x3( int i0, int j0, const Matrix3f& m );

    __device__ __host__ float determinant() const;
    __device__ __host__ Matrix4f inverse( bool* pbIsSingular = NULL, float epsilon = 0.f ) const;

    __device__ __host__ void transpose();
    __device__ __host__ Matrix4f transposed() const;

	// ---- Utility ----

	__device__ __host__ static Matrix4f ones();
	__device__ __host__ static Matrix4f identity();
	__device__ __host__ static Matrix4f translation( float x, float y, float z );
	__device__ __host__ static Matrix4f translation( const Vector3f& rTranslation );
	__device__ __host__ static Matrix4f rotateX( float radians );
	__device__ __host__ static Matrix4f rotateY( float radians );
	__device__ __host__ static Matrix4f rotateZ( float radians );
	__device__ __host__ static Matrix4f rotation( const Vector3f& rDirection, float radians );
	__device__ __host__ static Matrix4f scaling( float sx, float sy, float sz );
	__device__ __host__ static Matrix4f uniformScaling( float s );
	__device__ __host__ static Matrix4f lookAt( const Vector3f& eye, const Vector3f& center, const Vector3f& up );
	__device__ __host__ static Matrix4f orthographicProjection( float width, float height, float zNear, float zFar, bool directX );
	__device__ __host__ static Matrix4f orthographicProjection( float left, float right, float bottom, float top, float zNear, float zFar, bool directX );
	__device__ __host__ static Matrix4f perspectiveProjection( float fLeft, float fRight, float fBottom, float fTop, float fZNear, float fZFar, bool directX );
	__device__ __host__ static Matrix4f perspectiveProjection( float fovYRadians, float aspect, float zNear, float zFar, bool directX );
	__device__ __host__ static Matrix4f infinitePerspectiveProjection( float fLeft, float fRight, float fBottom, float fTop, float fZNear, bool directX );

	// Returns the rotation matrix represented by a quaternion
	// uses a normalized version of q
    __device__ __host__ static Matrix4f rotation( const Vector4f& q );

private:

	float m_elements[16];

};

// Matrix-Vector multiplication
// 4x4 * 4x1 ==> 4x1
__device__ __host__ Vector4f operator * ( const Matrix4f& m, const Vector4f& v );

// Matrix-Matrix multiplication
__device__ __host__ Matrix4f operator * ( const Matrix4f& x, const Matrix4f& y );

#endif // MATRIX4F_H
