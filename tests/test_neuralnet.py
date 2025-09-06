import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mymicrograd.engine import Value
from mymicrograd.neuralnet import Neuron, Layer, MLP, Module

class TestNeuralNet(unittest.TestCase):
    
    def test_neuron_creation(self):
        """Test neuron initialization"""
        neuron = Neuron(3)  # 3 inputs
        
        # Should have 3 weights + 1 bias
        params = neuron.parameters()
        self.assertEqual(len(params), 4)
        
        # All parameters should be Value objects
        for param in params:
            self.assertIsInstance(param, Value)
    
    def test_neuron_forward(self):
        """Test neuron forward pass"""
        neuron = Neuron(2, nonlin=False)  # Linear neuron
        
        # Set known weights for testing
        neuron.w[0].data = 0.5
        neuron.w[1].data = -0.3
        neuron.b.data = 0.1
        
        # Test input
        x = [Value(1.0), Value(2.0)]
        output = neuron(x)
        
        # Expected: 0.5*1 + (-0.3)*2 + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
        self.assertAlmostEqual(output.data, 0.0, places=5)
    
    def test_neuron_nonlinear(self):
        """Test neuron with nonlinearity"""
        neuron = Neuron(2, nonlin=True)  # Tanh neuron
        
        x = [Value(0.0), Value(0.0)]
        output = neuron(x)
        
        # Should apply tanh to the linear combination
        self.assertIsInstance(output, Value)
    
    def test_layer_creation(self):
        """Test layer initialization"""
        layer = Layer(3, 2)  # 3 inputs, 2 outputs
        
        self.assertEqual(len(layer.neurons), 2)
        
        # Each neuron should have 3 weights + 1 bias = 4 parameters
        # Total parameters: 2 * 4 = 8
        params = layer.parameters()
        self.assertEqual(len(params), 8)
    
    def test_layer_forward(self):
        """Test layer forward pass"""
        layer = Layer(2, 3, nonlin=False)  # 2 inputs, 3 outputs, linear
        
        x = [Value(1.0), Value(2.0)]
        outputs = layer(x)
        
        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertIsInstance(output, Value)
    
    def test_mlp_creation(self):
        """Test MLP initialization"""
        mlp = MLP(3, [4, 2])  # 3 inputs, hidden layer of 4, output layer of 2
        
        self.assertEqual(len(mlp.layers), 2)
        
        # First layer: 3 inputs, 4 outputs -> 4 * (3 + 1) = 16 parameters
        # Second layer: 4 inputs, 2 outputs -> 2 * (4 + 1) = 10 parameters
        # Total: 26 parameters
        params = mlp.parameters()
        self.assertEqual(len(params), 26)
    
    def test_mlp_forward(self):
        """Test MLP forward pass"""
        mlp = MLP(2, [3, 1])  # 2 inputs, 3 hidden, 1 output
        
        x = [Value(1.0), Value(2.0)]
        output = mlp(x)
        
        self.assertIsInstance(output, Value)
    
    def test_zero_grad(self):
        """Test gradient zeroing"""
        mlp = MLP(2, [2, 1])
        
        # Set some gradients
        for param in mlp.parameters():
            param.grad = 1.0
        
        # Zero gradients
        mlp.zero_grad()
        
        # All gradients should be zero
        for param in mlp.parameters():
            self.assertEqual(param.grad, 0.0)
    
    def test_gradient_flow(self):
        """Test gradient flow through network"""
        mlp = MLP(2, [2, 1])
        
        x = [Value(1.0), Value(2.0)]
        output = mlp(x)
        output.backward()
        
        # All parameters should have non-zero gradients
        for param in mlp.parameters():
            self.assertNotEqual(param.grad, 0.0)

if __name__ == '__main__':
    unittest.main()