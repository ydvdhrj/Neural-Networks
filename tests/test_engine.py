import unittest
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mymicrograd.engine import Value

class TestValue(unittest.TestCase):
    
    def test_basic_operations(self):
        """Test basic arithmetic operations"""
        a = Value(2.0)
        b = Value(3.0)
        
        # Addition
        c = a + b
        self.assertEqual(c.data, 5.0)
        
        # Multiplication
        d = a * b
        self.assertEqual(d.data, 6.0)
        
        # Power
        e = a ** 2
        self.assertEqual(e.data, 4.0)
    
    def test_backward_propagation(self):
        """Test gradient computation through backpropagation"""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        
        self.assertEqual(a.grad, 3.0)  # dc/da = b = 3
        self.assertEqual(b.grad, 2.0)  # dc/db = a = 2
    
    def test_complex_expression(self):
        """Test gradient computation for complex expressions"""
        x = Value(2.0)
        y = x ** 2 + 3 * x + 1
        y.backward()
        
        # dy/dx = 2*x + 3 = 2*2 + 3 = 7
        self.assertAlmostEqual(x.grad, 7.0, places=5)
    
    def test_activation_functions(self):
        """Test activation functions"""
        x = Value(0.5)
        
        # Test tanh
        tanh_out = x.tanh()
        expected_tanh = math.tanh(0.5)
        self.assertAlmostEqual(tanh_out.data, expected_tanh, places=5)
        
        # Test ReLU
        x_pos = Value(2.0)
        x_neg = Value(-2.0)
        
        relu_pos = x_pos.relu()
        relu_neg = x_neg.relu()
        
        self.assertEqual(relu_pos.data, 2.0)
        self.assertEqual(relu_neg.data, 0.0)
        
        # Test exp
        exp_out = x.exp()
        expected_exp = math.exp(0.5)
        self.assertAlmostEqual(exp_out.data, expected_exp, places=5)
    
    def test_activation_gradients(self):
        """Test gradients of activation functions"""
        x = Value(0.5)
        
        # Test tanh gradient
        y = x.tanh()
        y.backward()
        expected_grad = 1 - math.tanh(0.5) ** 2
        self.assertAlmostEqual(x.grad, expected_grad, places=5)
        
        # Reset gradient
        x.grad = 0.0
        
        # Test ReLU gradient
        y = x.relu()
        y.backward()
        self.assertEqual(x.grad, 1.0)  # x > 0, so gradient is 1
    
    def test_chain_rule(self):
        """Test chain rule with multiple operations"""
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x ** 2
        z.backward()
        
        # dz/dx = y + 2*x = 3 + 2*2 = 7
        # dz/dy = x = 2
        self.assertAlmostEqual(x.grad, 7.0, places=5)
        self.assertAlmostEqual(y.grad, 2.0, places=5)

if __name__ == '__main__':
    unittest.main()