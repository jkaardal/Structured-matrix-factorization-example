import numpy as np
import GPy
import GPyOpt
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_cg
from scipy.optimize import check_grad

np.set_printoptions(precision=6, suppress=True)

def detect_infeasibility(x):
    # detect if the inequality constraints are violated after acquisition
    for i in range(x.size-1):
        if x[0, i] > x[0, i+1]:
            print "^ Infeasible point detected!"
            break
        
class Objective:

    def __init__(self, w, n, Y, Ycv, Z, scale=10000.0, m=10, iprint=1):
        self.n = n
        self.Y = Y
        self.Ycv = Ycv
        self.d = Y.shape[1]
        self.Z = Z
        self.w = w
        self.x = None
        self.fval = np.inf
        self.scale = scale
        self.m = m
        self.iprint = iprint
        
    def reshape_weights(self, w):
        # form left and right weight matrices of rank n
        Q = np.copy(w.reshape((self.n, 2*self.d)).T)
        U = Q[:self.d, :]
        V = Q[self.d:, :]

        return (U, V)

    def g(self, w, x):
        # l2-norm regularized least squares objective function
        U, V = self.reshape_weights(w)
        Y_fit = np.dot(U, V.T)

        lsq = 0.5 * np.sum(self.Z * (self.Y - Y_fit) ** 2)
        l2 = 0.5 * np.sum(np.dot(U ** 2, x.T) + np.dot(V ** 2, x.T))
        
        return (lsq + l2) * self.scale

    def gcv(self, w):
        # least squares objective function for cross-validation
        U, V = self.reshape_weights(w)
        Y_fit = np.dot(U, V.T)

        lsq = 0.5 * np.sum(self.Z * (self.Ycv - Y_fit) ** 2)
        
        return lsq * self.scale
        
    def dg(self, w, x):
        # gradient of the l2-norm regularized least squares objective function
        U, V = self.reshape_weights(w)
        Y_fit = np.dot(U, V.T)

        dlsq_U = np.dot(self.Z * (Y_fit - self.Y), V) * self.scale
        dlsq_V = np.dot((self.Z * (Y_fit - self.Y)).T, U) * self.scale
        dl2_U = np.dot(U, np.diag(x.ravel())) * self.scale
        dl2_V = np.dot(V, np.diag(x.ravel())) * self.scale

        return np.copy(np.concatenate([dlsq_U + dl2_U, dlsq_V + dl2_V], axis=0).T.reshape((2*self.n*self.d,)))
        
    def f(self, x):
        # cross-validation function to be optimized
        w, fnew, _ = fmin_l_bfgs_b(lambda w: self.g(w, x), 0.01*np.random.randn(self.w.size), fprime=lambda w: self.dg(w, x), m=self.m, iprint=self.iprint)
        fval = self.gcv(w)
        
        print x
        #print "f(x) = " + str(fval)
        detect_infeasibility(x)
        
        if fval < self.fval:
            self.w = np.copy(w)
            self.fval = np.copy(fval)
            self.x = np.copy(x)

        return fval.reshape((1, 1))



def main():
    # structured matrix factorization example
    d = 10      # square matrix order
    n = 8       # rank of factorization (number of regularization parameters)
    n_true = 10 # ground truth rank of the factorization
    
    m = 10                   # size of L-BFGS memory
    iprint = -1              # L-BFGS output level
    jitter = 0.01            # acquisition jitter to add white noise to kernel
    exploration_weight = 2.0 # exploration weight for lower confidence bound
    eps = 0.0                # minimum difference in x for convergence
    max_iter = 100           # maximum number of iterations

    # constructing cross-validation matrix from hermite polynomials
    z_axis = np.linspace(-2.0, 2.0, d)
    U = np.zeros((d, n_true))
    V = np.zeros((d, n_true))
    r = np.random.randn(n_true)
    c = np.zeros((n_true+1,))
    for i in range(n_true):
        c[i] = 0.0
        c[i+1] = 1.0
        U[:, i] = np.polynomial.hermite.hermval(z_axis, c)
        V[:, i] = np.polynomial.hermite.hermval(z_axis, c) * r[i]
    Ycv = np.dot(U, V.T)
    Ycv = 0.5 * (Ycv + Ycv.T)

    # feature normalization
    Ycv = Ycv/np.linalg.norm(Ycv.ravel())
    
    Y = np.copy(Ycv) + 0.01*np.random.randn(d, d) # training matrix (white noise added)
    Z = np.ones((d, d))                           # weighting importance of elements
    w = 0.01 * np.random.randn(2*n*d)             # weight initialization

    obj = Objective(w, n, Y, Ycv, Z, m=m, iprint=iprint) # initialize class with objective functions and gradient

    # test that the gradient is correct
    xtest = np.random.rand(1, n)
    err = check_grad(lambda w: obj.g(w, xtest), lambda w: obj.dg(w, xtest), w)
    print "grad. check of g and dg: " + str(err)
    print ""
    
    # construct domain
    domain = []
    for i in range(n):
        domain.append({'name': 'x_' + str(i+1) + '_b', 'type': 'continuous', 'domain': (0.0, 1.0)})

    # construct constraints
    constraints = []
    for i in range(n-1):
        constraints.append({'name': 'x_' + str(i+1) + '_' + str(i+2) + '_c', 'constrain': 'x[:,' + str(i) + '] - x[:,' + str(i+1) + ']'})

    print "domain specifications:"
    for i in range(n):
        print domain[i]
    print ""
        
    print "constraint specifications:"
    for i in range(n-1):
        print constraints[i]
    print ""

    print "gathering initial points"
    bayes_opt = GPyOpt.methods.BayesianOptimization(f=obj.f, domain=domain, constrains=constraints, acquisition_type='LCB', exact_feval=True, normalize_Y=True, acquisition_jitter=jitter, model_type='GP', exploration_weight=exploration_weight)
    print ""
    
    print "optimizing"
    bayes_opt.run_optimization(max_iter=max_iter, eps=eps, verbosity=True)
    print ""
    
    print "best function value: " + str(obj.fval)
    print "best value of x:"
    print obj.x
        


if __name__ == "__main__":
    main()
    
