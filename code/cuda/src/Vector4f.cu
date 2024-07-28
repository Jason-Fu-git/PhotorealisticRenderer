//
// Created by Jason Fu on 24-7-14.
//

#include "Vector4f.cuh"
#include "Vector3f.cuh"

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#include <cmath>
#include <cstdio>


__host__ __device__ Vector4f::Vector4f( float f )
{
    m_elements[ 0 ] = f;
    m_elements[ 1 ] = f;
    m_elements[ 2 ] = f;
    m_elements[ 3 ] = f;
}

__host__ __device__ Vector4f::Vector4f( float fx, float fy, float fz, float fw )
{
    m_elements[0] = fx;
    m_elements[1] = fy;
    m_elements[2] = fz;
    m_elements[3] = fw;
}

__host__ __device__ Vector4f::Vector4f( float buffer[ 4 ] )
{
    m_elements[ 0 ] = buffer[ 0 ];
    m_elements[ 1 ] = buffer[ 1 ];
    m_elements[ 2 ] = buffer[ 2 ];
    m_elements[ 3 ] = buffer[ 3 ];
}

__host__ __device__ Vector4f::Vector4f( const Vector3f& xyz, float w )
{
    m_elements[0] = xyz._x;
    m_elements[1] = xyz._y;
    m_elements[2] = xyz._z;
    m_elements[3] = w;
}

__host__ __device__ Vector4f::Vector4f( float x, const Vector3f& yzw )
{
    m_elements[0] = x;
    m_elements[1] = yzw._x;
    m_elements[2] = yzw._y;
    m_elements[3] = yzw._z;
}

__host__ __device__ Vector4f::Vector4f( const Vector4f& rv )
{
    m_elements[0] = rv.m_elements[0];
    m_elements[1] = rv.m_elements[1];
    m_elements[2] = rv.m_elements[2];
    m_elements[3] = rv.m_elements[3];
}

__host__ __device__ Vector4f& Vector4f::operator = ( const Vector4f& rv )
{
    if( this != &rv )
    {
        m_elements[0] = rv.m_elements[0];
        m_elements[1] = rv.m_elements[1];
        m_elements[2] = rv.m_elements[2];
        m_elements[3] = rv.m_elements[3];
    }
    return *this;
}

__host__ __device__ const float& Vector4f::operator [] ( int i ) const
{
    return m_elements[ i ];
}

__host__ __device__ float& Vector4f::operator [] ( int i )
{
    return m_elements[ i ];
}

__host__ __device__ float& Vector4f::x()
{
    return m_elements[ 0 ];
}

__host__ __device__ float& Vector4f::y()
{
    return m_elements[ 1 ];
}

__host__ __device__ float& Vector4f::z()
{
    return m_elements[ 2 ];
}

__host__ __device__ float& Vector4f::w()
{
    return m_elements[ 3 ];
}

__host__ __device__ float Vector4f::x() const
{
    return m_elements[0];
}

__host__ __device__ float Vector4f::y() const
{
    return m_elements[1];
}

__host__ __device__ float Vector4f::z() const
{
    return m_elements[2];
}

__host__ __device__ float Vector4f::w() const
{
    return m_elements[3];
}

__host__ __device__ Vector3f Vector4f::xyz() const
{
    return { m_elements[0], m_elements[1], m_elements[2] };
}

__host__ __device__ Vector3f Vector4f::yzw() const
{
    return { m_elements[1], m_elements[2], m_elements[3] };
}

__host__ __device__ Vector3f Vector4f::zwx() const
{
    return { m_elements[2], m_elements[3], m_elements[0] };
}

__host__ __device__ Vector3f Vector4f::wxy() const
{
    return { m_elements[3], m_elements[0], m_elements[1] };
}

__host__ __device__ Vector3f Vector4f::xyw() const
{
    return { m_elements[0], m_elements[1], m_elements[3] };
}

__host__ __device__ Vector3f Vector4f::yzx() const
{
    return { m_elements[1], m_elements[2], m_elements[0] };
}

__host__ __device__ Vector3f Vector4f::zwy() const
{
    return { m_elements[2], m_elements[3], m_elements[1] };
}

__host__ __device__ Vector3f Vector4f::wxz() const
{
    return { m_elements[3], m_elements[0], m_elements[2] };
}

__host__ __device__ float Vector4f::abs() const
{
    return sqrtf( m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] + m_elements[2] * m_elements[2] + m_elements[3] * m_elements[3] );
}

__host__ __device__ float Vector4f::absSquared() const
{
    return( m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] + m_elements[2] * m_elements[2] + m_elements[3] * m_elements[3] );
}

__host__ __device__ void Vector4f::normalize()
{
    float norm = sqrtf( m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] + m_elements[2] * m_elements[2] + m_elements[3] * m_elements[3] );
    m_elements[0] = m_elements[0] / norm;
    m_elements[1] = m_elements[1] / norm;
    m_elements[2] = m_elements[2] / norm;
    m_elements[3] = m_elements[3] / norm;
}

__host__ __device__ Vector4f Vector4f::normalized() const
{
    float length = abs();
    return {
                    m_elements[0] / length,
                    m_elements[1] / length,
                    m_elements[2] / length,
                    m_elements[3] / length
            };
}

__host__ __device__ void Vector4f::homogenize()
{
    if( m_elements[3] != 0 )
    {
        m_elements[0] /= m_elements[3];
        m_elements[1] /= m_elements[3];
        m_elements[2] /= m_elements[3];
        m_elements[3] = 1;
    }
}

__host__ __device__ Vector4f Vector4f::homogenized() const
{
    if( m_elements[3] != 0 )
    {
        return {
                        m_elements[0] / m_elements[3],
                        m_elements[1] / m_elements[3],
                        m_elements[2] / m_elements[3],
                        1
                };
    }
    else
    {
        return {
                        m_elements[0],
                        m_elements[1],
                        m_elements[2],
                        m_elements[3]
                };
    }
}

__host__ __device__ void Vector4f::negate()
{
    m_elements[0] = -m_elements[0];
    m_elements[1] = -m_elements[1];
    m_elements[2] = -m_elements[2];
    m_elements[3] = -m_elements[3];
}

__host__ __device__ Vector4f::operator const float* () const
{
    return m_elements;
}

__host__ __device__ Vector4f::operator float* ()
{
    return m_elements;
}

__host__ __device__ void Vector4f::print() const
{
    printf( "< %.4f, %.4f, %.4f, %.4f >\n",
            m_elements[0], m_elements[1], m_elements[2], m_elements[3] );
}

// static
__host__ __device__ float Vector4f::dot( const Vector4f& v0, const Vector4f& v1 )
{
    return v0.x() * v1.x() + v0.y() * v1.y() + v0.z() * v1.z() + v0.w() * v1.w();
}

// static
__host__ __device__ Vector4f Vector4f::lerp( const Vector4f& v0, const Vector4f& v1, float alpha )
{
    return alpha * ( v1 - v0 ) + v0;
}

//////////////////////////////////////////////////////////////////////////
// Operators
//////////////////////////////////////////////////////////////////////////

__host__ __device__ Vector4f operator + ( const Vector4f& v0, const Vector4f& v1 )
{
    return { v0.x() + v1.x(), v0.y() + v1.y(), v0.z() + v1.z(), v0.w() + v1.w() };
}

__host__ __device__ Vector4f operator - ( const Vector4f& v0, const Vector4f& v1 )
{
    return { v0.x() - v1.x(), v0.y() - v1.y(), v0.z() - v1.z(), v0.w() - v1.w() };
}

__host__ __device__ Vector4f operator * ( const Vector4f& v0, const Vector4f& v1 )
{
    return { v0.x() * v1.x(), v0.y() * v1.y(), v0.z() * v1.z(), v0.w() * v1.w() };
}

__host__ __device__ Vector4f operator / ( const Vector4f& v0, const Vector4f& v1 )
{
    return { v0.x() / v1.x(), v0.y() / v1.y(), v0.z() / v1.z(), v0.w() / v1.w() };
}

__host__ __device__ Vector4f operator - ( const Vector4f& v )
{
    return { -v.x(), -v.y(), -v.z(), -v.w() };
}

__host__ __device__ Vector4f operator * ( float f, const Vector4f& v )
{
    return { f * v.x(), f * v.y(), f * v.z(), f * v.w() };
}

__host__ __device__ Vector4f operator * ( const Vector4f& v, float f )
{
    return { f * v.x(), f * v.y(), f * v.z(), f * v.w() };
}

__host__ __device__ Vector4f operator / ( const Vector4f& v, float f )
{
    return { v[0] / f, v[1] / f, v[2] / f, v[3] / f };
}

__host__ __device__ bool operator == ( const Vector4f& v0, const Vector4f& v1 )
{
    return( v0.x() == v1.x() && v0.y() == v1.y() && v0.z() == v1.z() && v0.w() == v1.w() );
}

__host__ __device__ bool operator != ( const Vector4f& v0, const Vector4f& v1 )
{
    return !( v0 == v1 );
}
