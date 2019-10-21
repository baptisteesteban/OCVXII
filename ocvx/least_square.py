import numpy as np
import matplotlib.pyplot as plt

class LeastSquare:
    def __init__(self, X):
        """Constructor of the least_square.
           
           Parameter
           ---------
               X: A numpy array
                   Represents a matrix N x P which
                   N is the length of element and P number of features
        """
        self.X = X
        self.beta = np.zeros(X.shape[1])
        self.eps = np.random.randn(X.shape[0])
        self.y = self.X.dot(self.beta) + self.eps 
        for i in range(self.y.shape[0]):
            r = (self.y[i] - self.beta.dot(self.X[i,:]))
            self.eps[i] = r**2
        
        self.SSE = []
        self.max_iter = 0
        
       
    def compute_SSE(self):
        """Compute the SSE of the model.
           
           Return:
           -------
               integer which represents the SSE of the model.
        """
        return np.linalg.norm(self.y - self.X.dot(self.beta), ord=2)**2


    def tikhonov_reg(self, alpha, y, max_iter=10):
        """Constructor of the least_square.
           
           Parameters
           ----------
               alpha: float
                   Represents the regularisation paramter (lambda)
               y: vector
                   Reference of the matrix X
               max_iter: int
                   Set the maximum of iteration (default: 10)
            Return:
            -------
                the prediction
        """
        iteration = 0
        self.max_iter = max_iter
        while iteration < max_iter:
            self.y = np.dot(self.X, self.beta)
            self.eps = self.y - y
            self.beta = np.linalg.inv(np.transpose(self.X).dot(self.X) + np.identity(self.X.shape[1]) * alpha).dot(np.transpose(self.X).dot(self.eps))
            if iteration < 5 or iteration >= max_iter - 5:
                print("epoch: %d/%d" % (iteration + 1, max_iter))
                print("SSE = %f" % self.compute_SSE())
            elif iteration == 5:
                print()
                print("...")
                print()
            self.SSE.append(self.compute_SSE())
            iteration += 1
        return self.y
    
    def plot(self, y_ref):
        """Plot the result of regularisation.
        
           Parameter
           ---------
               y_ref: vector of float
                   Represents the y reference of X to plot it
        """
        fig = plt.figure(figsize=(21, 6))
        ax = fig.add_subplot(121)
        ax.set_title("Evolution of the SSE")
        plt.xlabel("epochs")
        plt.ylabel("SSE")
        ax.plot(np.arange(1,self.max_iter+1), self.SSE, c='green', label='SSE')
        leg = ax.legend()
        
        ax = fig.add_subplot(122)
        ax.set_title("")
        plt.xlabel("X")
        plt.ylabel("Y")
        ax.scatter(self.X[:,0], y_ref, c='red', marker='o', label='y reference')
        print(self.X.shape)
        if self.X.shape[1] == 1:
            ax.plot(self.X[:,0], self.y, c='green', label='y pred')
        else:
            ax.scatter(self.X[:,0], self.y, c='green', marker='x', label='y pred')
        leg = ax.legend()