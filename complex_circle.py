import numpy as np
import hashlib

class ComplexCircleFactory:
    """
    This class mimics a Riemannian manifold for unit-modulus complex numbers.
    It provides various utilities for optimization over this manifold.
    
    Parameters
    ----------
    n : int, optional
        Dimension of the complex circle. Default is 1.
    """
    def __init__(self, n=1):
        self.n = n
    
    def name(self):
        """Returns the name of the manifold."""
        return f'Complex circle (S^1)^{self.n}'
    
    def dim(self):
        """Returns the dimension of the manifold."""
        return self.n
    
    def inner(self, z, v, w):
        """
        Computes the inner product of two vectors v and w in the manifold.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
        v : array_like
            First vector.
        w : array_like
            Second vector.
            
        Returns
        -------
        float
            The inner product of v and w.
        """
        return np.real(np.vdot(v, w))
    
    def norm(self, x, v):
        """
        Computes the norm of a vector v in the manifold.
        
        Parameters
        ----------
        x : array_like
            Point on the manifold.
        v : array_like
            The vector.
            
        Returns
        -------
        float
            The norm of the vector v.
        """
        return np.linalg.norm(v)
    
    def dist(self, x, y):
        """
        Computes the Riemannian distance between two points x and y on the manifold.
        
        Parameters
        ----------
        x : array_like
            First point.
        y : array_like
            Second point.
            
        Returns
        -------
        float
            The Riemannian distance between x and y.
        """
        return np.linalg.norm(np.arccos(np.conj(x) * y))
    
    def typical_dist(self):
        """Returns a typical distance on the manifold."""
        return np.pi * np.sqrt(self.n)
    
    def proj(self, z, u):
        """
        Orthogonal projection of a vector u onto the tangent space at point z.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
        u : array_like
            The vector to be projected.
            
        Returns
        -------
        array_like
            The projected vector.
        """
        return u - np.real(np.conj(u) * z) * z
    
    def tangent(self, z, u):
        """
        Tanget projection of a vector z onto vector u.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
        u : array_like
            The vector to be projected.
            
        Returns
        -------
        array_like
            The projected vector.
        """
        return self.proj(z, u)
    
    def egrad2rgrad(self, z, egrad):
        """
        Converts a Euclidean gradient to a Riemannian gradient using orthogonal projection.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
        egrad : array_like
            Euclidean gradient.
            
        Returns
        -------
        array_like
            Riemannian gradient.
        """
        return self.proj(z, egrad)
    
    def ehess2rhess(self, z, egrad, ehess, zdot):
        """
        Convert Euclidean Hessian to Riemannian Hessian.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
        egrad : array_like
            Euclidean gradient at point z.
        ehess : array_like
            Euclidean Hessian at point z.
        zdot : array_like
            Tangent vector at point z.
            
        Returns
        -------
        array_like
            Riemannian Hessian.
        """
        return self.proj(z, ehess - np.real(z * np.conj(egrad)) * zdot)
    
    def exp(self, z, v, t=1.0):
        """
        Exponential map at a point z with a given tangent vector v and scaling factor t.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
        v : array_like
            Tangent vector at point z.
        t : float, optional
            Scaling factor. Default is 1.0.
            
        Returns
        -------
        array_like
            The resulting point on the manifold.
        """
        y = np.zeros((self.n, 1), dtype=np.complex128)
        tv = t * v
        nrm_tv = np.abs(tv)
        mask = nrm_tv > 1e-6
        y[mask] = z[mask] * np.cos(nrm_tv[mask]) + tv[mask] * (np.sin(nrm_tv[mask]) / nrm_tv[mask])
        y[~mask] = z[~mask] + tv[~mask]
        y[~mask] = y[~mask] / np.abs(y[~mask])
        return y
    
    def retr(self, z, v, t=1.0):
        """
        Retraction map, an approximation of the exponential map.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
        v : array_like
            Tangent vector at point z.
        t : float, optional
            Scaling factor. Default is 1.0.
            
        Returns
        -------
        array_like
            The resulting point on the manifold.
        """
        y = z + t * v
        return y / np.abs(y)
    
    def log(self, x1, x2):
        """
        Logarithm map at a point x1 towards another point x2.
        
        Parameters
        ----------
        x1 : array_like
            Point on the manifold.
        x2 : array_like
            Target point on the manifold.
            
        Returns
        -------
        array_like
            Tangent vector at x1 that points towards x2.
        """
        v = self.proj(x1, x2 - x1)
        di = self.dist(x1, x2)
        nv = self.norm(x1, v)
        return v * (di / nv)
    
    def hash(self, z):
        """
        Generate a hash for a point on the manifold.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
            
        Returns
        -------
        str
            Hash string.
        """
        m = hashlib.md5()
        m.update(np.real(z).tobytes())
        m.update(np.imag(z).tobytes())
        return 'z' + m.hexdigest()
    
    def random(self):
        """
        Generate a random point on the manifold.
        
        Returns
        -------
        array_like
            Random point on the manifold.
        """
        z = np.random.randn(self.n, 1) + 1j * np.random.randn(self.n, 1)
        return z / np.abs(z)
    
    def randomvec(self, z):
        """
        Generate a random tangent vector at a point on the manifold.
        
        Parameters
        ----------
        z : array_like
            Point on the manifold.
            
        Returns
        -------
        array_like
            Random tangent vector at the given point.
        """
        v = np.random.randn(self.n, 1) * (1j * z)
        return v / np.linalg.norm(v)
    
    def lincomb(self, x, a1, d1, a2=None, d2=None):
        """
        Linear combination of tangent vectors.
        
        Parameters
        ----------
        x : array_like
            Point on the manifold. (Not actually used in the computation)
        a1 : float
            Scalar multiplier for the first tangent vector.
        d1 : array_like
            First tangent vector.
        a2 : float, optional
            Scalar multiplier for the second tangent vector.
        d2 : array_like, optional
            Second tangent vector.
            
        Returns
        -------
        array_like
            Result of the linear combination.
        
        Raises
        ------
        ValueError
            If a2 and d2 are not both provided.
        """
        if a2 is None and d2 is None:
            return a1 * d1
        elif a2 is not None and d2 is not None:
            return a1 * d1 + a2 * d2
        else:
            raise ValueError("Bad use of lincomb.")