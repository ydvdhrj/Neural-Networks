import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mymicrograd.engine import Value
from mymicrograd.neuralnet import MLP

class TestIntegration(unittest.TestCase):
    
    def test_simple_training_loop(self):
        """Test a simple training scenario"""
        # Create a simple MLP
        mlp = MLP(2, [3, 1])
        
        # Simple dataset: XOR-like problem
        dataset = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0)
        ]
        
        # Training loop
        learning_rate = 0.1
        initial_loss = None
        
        for epoch in range(10):
            total_loss = Value(0.0)
            
            for x_data, y_target in dataset:
                # Forward pass
                x = [Value(xi) for xi in x_data]
                y_pred = mlp(x)
                
                # Loss (MSE)
                loss = (y_pred - Value(y_target)) ** 2
                total_loss = total_loss + loss
            
            if epoch == 0:
                initial_loss = total_loss.data
            
            # Backward pass
            mlp.zero_grad()
            total_loss.backward()
            
            # Update parameters
            for param in mlp.parameters():
                param.data -= learning_rate * param.grad
        
        # Loss should decrease
        final_loss = total_loss.data
        self.assertLess(final_loss, initial_loss)
    
    def test_gradient_checking(self):
        """Test gradients using numerical differentiation"""
        mlp = MLP(1, [2, 1])
        
        x = [Value(0.5)]
        y = mlp(x)
        y.backward()
        
        # Numerical gradient checking
        h = 1e-5
        for param in mlp.parameters():
            # Store original values
            original_data = param.data
            original_grad = param.grad
            
            # f(x + h)
            param.data = original_data + h
            y_plus = mlp(x)
            
            # f(x - h)
            param.data = original_data - h
            y_minus = mlp(x)
            
            # Numerical gradient
            numerical_grad = (y_plus.data - y_minus.data) / (2 * h)
            
            # Restore original data
            param.data = original_data
            
            # Compare gradients (should be close)
            self.assertAlmostEqual(original_grad, numerical_grad, places=3)

if __name__ == '__main__':
    unittest.main()