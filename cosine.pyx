"""
checked under python 3.
"""

cimport cython

from libc.math cimport sqrt


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef double dot(double[:] v1, double[:] v2):
    cdef double s=0.0
    for i in range(v1.shape[0]):
        s += v1[i]*v2[i]
    return s

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double norm(double[:] v):
    cdef double s=0.0
    for i in range(v.shape[0]):
        s+=v[i]*v[i]
    return sqrt(s)

@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False) 
cpdef double cosine(double[:] v1, double[:] v2):
    return 1.0 - dot(v1,v2)/norm(v1)/norm(v2)