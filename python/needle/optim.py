"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        
        for w in self.params:
            if w.grad is None:
                continue
            
            d_p = w.grad.detach() + self.weight_decay * w.detach()
            
            if hash(w) in self.u:
                self.u[hash(w)] = self.momentum * self.u[hash(w)] + (1 - self.momentum) * d_p
            else:
                self.u[hash(w)] = (1 - self.momentum) * d_p
            
            w.data = w.data - (self.lr * self.u[hash(w)])
        

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        
        self.t += 1
        for w in self.params:
            if w.grad is None:
                continue
            
            grad = w.grad.detach() + self.weight_decay * w.data
            if hash(w) in self.m:
                self.m[hash(w)] = self.beta1 * self.m[hash(w)] + (1 - self.beta1) * grad
            else:
                self.m[hash(w)] = (1 - self.beta1) * grad
            
            if hash(w) in self.v:
                self.v[hash(w)] = self.beta2 * self.v[hash(w)] + (1 - self.beta2) * grad ** 2
            else:
                self.v[hash(w)] = (1 - self.beta2) * grad ** 2
                
            m_hat = self.m[hash(w)] / (1 - self.beta1 ** (self.t))
            v_hat = self.v[hash(w)] / (1 - self.beta2 ** (self.t))
            
            w.data = w.data - self.lr * m_hat / (v_hat ** (0.5) + self.eps)
        
