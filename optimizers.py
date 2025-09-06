"""
Optimizers for the mini neural network framework
"""
from mymicrograd.engine import Value

class Optimizer:
    """Base optimizer class"""
    def __init__(self, parameters):
        self.parameters = list(parameters)
    
    def zero_grad(self):
        """Zero out gradients of all parameters"""
        for param in self.parameters:
            param.grad = 0.0
    
    def step(self):
        """Update parameters - to be implemented by subclasses"""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr
    
    def step(self):
        """Update parameters using SGD"""
        for param in self.parameters:
            param.data -= self.lr * param.grad

class SGDMomentum(Optimizer):
    """SGD with momentum"""
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0.0 for _ in self.parameters]
    
    def step(self):
        """Update parameters using SGD with momentum"""
        for i, param in enumerate(self.parameters):
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param.grad
            param.data -= self.velocities[i]

class Adam(Optimizer):
    """Adam optimizer (simplified version)"""
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0 for _ in self.parameters]  # First moment
        self.v = [0.0 for _ in self.parameters]  # Second moment
        self.t = 0  # Time step
    
    def step(self):
        """Update parameters using Adam"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)