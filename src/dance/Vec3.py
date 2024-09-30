'''
Simple class for 3D vectors.
(c) Ilan Schnell, 2008
'''
import numpy as np
import math

_TINY = 1e-15

def _xyzto012(c):
    if c in 'xyz':
        return ord(c) - ord('x')
    else:
        raise AttributeError("Vec3 instance has no attribute '%s'" % c)


def _args2tuple(funcname, args):
    narg = len(args)
    if narg == 0:
        data = 3*(0,)
    elif narg == 1:
        data = args[0]
        if len(data) != 3:
            raise TypeError('Vec3.%s() takes sequence with 3 elements '
                            '(%d given),\n\t   when 1 argument is given' %
                             (funcname, len(data)))
    elif narg == 3:
        data = args
    else:
        raise TypeError('Vec3.%s() takes 0, 1 or 3 arguments (%d given)' %
                        (funcname, narg))
    assert len(data) == 3
    try:
        return tuple(map(float, data))
    except (TypeError, ValueError):
        raise TypeError("Vec3.%s() can't convert elements to float" % funcname)


class Vec3(np.ndarray):
    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], Vec3):
                return args[0].copy()
            if isinstance(args[0], np.matrix):
                return Vec3(args[0].flatten().tolist()[0])
        data = _args2tuple('__new__', args)
        arr = np.array(data, dtype=float, copy=True)
        return np.ndarray.__new__(cls, shape=(3,), buffer=arr)
    
    def __repr__(self):
        return 'Vec3' + repr(tuple(self))
    
    def __mul__(self, other):
        return np.dot(self, other)
    
    def __abs__(self):
        return math.sqrt(self * self)
    
    def __pow__(self, x):
        return (self * self) if x == 2 else pow(abs(self), x)
    
    def __eq__(self, other):
        return abs(self-other) < _TINY
    
    def __ne__(self, other):
        return not self == other
    
    def __getattr__(self, name):
        return self[_xyzto012(name)]
    
    def __setattr__(self, name, val):
        self[_xyzto012(name)] = val
    
    def get_spherical(self):
        r = abs(self)
        if r < _TINY:
            theta = phi = 0.0
        else:
            x, y, z = self
            theta = math.acos(z/r)
            phi = math.atan2(y, x)
        
        return r, theta, phi
    
    def set_spherical(self, *args):
        r, theta, phi = _args2tuple('set_spherical', args)
        self[0] = r * math.sin(theta) * math.cos(phi);
        self[1] = r * math.sin(theta) * math.sin(phi);
        self[2] = r * math.cos(theta);
    
    def get_cylindrical(self):
        x, y, z = self
        rho = math.sqrt(x*x + y*y)
        phi = math.atan2(y, x)
        return rho, phi, z
    
    def set_cylindrical(self, *args):
        rho, phi, z = _args2tuple('set_cylindrical', args)
        self[0] = rho * math.cos(phi);
        self[1] = rho * math.sin(phi);
        self[2] = z

    def __array__(self):
        # Personnalisez la conversion de votre objet en un tableau NumPy
        return np.array(self, dtype=float)


def dot(a,b):
    return a*b

def norm(a):
    return math.sqrt(a.x*a.x + a.y*a.y + a.z*a.z)

def cross(a, b):
    return Vec3(np.cross(a, b))


if __name__ == '__main__':
    a = Vec3(1,2,3)
    b = Vec3(4,5,6)
    c = Vec3(0.0, 0.2, 0.4)
    print(a)
    print(b)
    print(c)
    print("dot="+str(dot(a,b)))
    print("cross="+str(cross(a,b)))
    print(type(a))
    x = np.array(b, dtype=float)
    print(x)
    print(type(x))
    print(x.dtype)