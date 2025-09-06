"""
Example usage of the mini neural network framework
"""

from mymicrograd.engine import Value
from mymicrograd.neuralnet import MLP
from optimizers import SGD, Adam
from losses import MSELoss

def example_training():
    """Example training loop"""
    # Create network
    mlp = MLP(2, [4, 1])
    
    # Create optimizer and loss
    optimizer = SGD(mlp.parameters(), lr=0.1)
    loss_fn = MSELoss()
    
    # Simple dataset
    dataset = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0)
    ]
    
    # Training loop
    for epoch in range(100):
        total_loss = Value(0.0)
        
        for x_data, y_target in dataset:
            # Forward pass
            x = [Value(xi) for xi in x_data]
            y_pred = mlp(x)
            
            # Compute loss
            loss = loss_fn([y_pred], [y_target])
            total_loss = total_loss + loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.data:.4f}")

if __name__ == "__main__":
    example_training()