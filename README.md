# MyMicrograd - Mini Neural Network Framework

A lightweight neural network framework inspired by PyTorch, built for educational purposes and understanding deep learning internals.

## 🎯 Features

- **Automatic Differentiation** - Complete backpropagation engine
- **Neural Networks** - Neurons, Layers, and Multi-Layer Perceptrons  
- **Optimizers** - SGD, SGD with Momentum, Adam
- **Loss Functions** - MSE, Cross Entropy, Hinge Loss
- **Pure Python** - No external dependencies
- **Educational** - Clean, readable code for learning

## 🚀 Quick Start

### Installation
```bash
pip install git+https://github.com/ydvdhrj/Neural-Networks.git
```

### Basic Usage
```python
from mymicrograd import Value, MLP
from optimizers import SGD
from losses import MSELoss

# Create a neural network
model = MLP(2, [4, 1])  # 2 inputs, 4 hidden, 1 output

# Create optimizer and loss
optimizer = SGD(model.parameters(), lr=0.1)
loss_fn = MSELoss()

# Training data (XOR problem)
data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

# Training loop
for epoch in range(100):
    total_loss = Value(0.0)
    
    for x, y in data:
        # Forward pass
        inputs = [Value(xi) for xi in x]
        pred = model(inputs)
        
        # Compute loss
        loss = loss_fn([pred], [y])
        total_loss = total_loss + loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.data:.4f}")
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python run_tests.py
```

Try the example:
```bash
python example_usage.py
```

## 📁 Project Structure

```
mymicrograd/
├── mymicrograd/          # Core framework
│   ├── engine.py        # Automatic differentiation
│   └── neuralnet.py     # Neural network components
├── tests/               # Test suite
├── optimizers.py        # SGD, Adam optimizers
├── losses.py           # Loss functions
├── example_usage.py    # Usage example
└── setup.py           # Package setup
```

## 🎓 What You'll Learn

- Automatic differentiation and backpropagation
- Neural network architecture design
- Optimization algorithms implementation
- Loss functions and gradients
- Training loop mechanics

Perfect for understanding the internals of deep learning frameworks like PyTorch and TensorFlow!

## 📄 License

MIT License - feel free to use for educational purposes.

## 🙏 Acknowledgments

Special thanks to **Andrej Karpathy** for his excellent educational content on neural networks and backpropagation, which inspired and guided this implementation.

## 👨‍💻 Author

**Dheeraj Yadav** - Educational neural network framework implementation