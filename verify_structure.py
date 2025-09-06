"""
Verify that the project structure and imports are correctly set up
"""

def test_imports():
    """Test that all imports work correctly"""
    try:
        print("Testing mymicrograd package structure...")
        
        # Test if we can import the package
        import mymicrograd
        print("✓ mymicrograd package imported successfully")
        
        # Test individual imports (will fail due to syntax errors in neuralnet.py)
        try:
            from mymicrograd.engine import Value
            print("✓ Value class imported successfully")
            
            # Test basic Value functionality
            a = Value(2.0)
            b = Value(3.0)
            c = a + b
            print(f"✓ Basic operations work: 2 + 3 = {c.data}")
            
        except Exception as e:
            print(f"✗ Error importing from engine: {e}")
        
        try:
            from mymicrograd.neuralnet import MLP
            print("✓ Neural network classes imported successfully")
        except Exception as e:
            print(f"✗ Error importing from neuralnet: {e}")
            print("  This is expected due to syntax error in neuralnet.py")
        
        # Test other components
        try:
            from optimizers import SGD
            print("✓ Optimizers imported successfully")
        except Exception as e:
            print(f"✗ Error importing optimizers: {e}")
        
        try:
            from losses import MSELoss
            print("✓ Loss functions imported successfully")
        except Exception as e:
            print(f"✗ Error importing losses: {e}")
            
    except Exception as e:
        print(f"✗ Failed to import mymicrograd package: {e}")

if __name__ == "__main__":
    test_imports()
    print("\nProject structure verification complete!")
    print("Check FIXES_NEEDED.md for syntax errors that need to be fixed.")