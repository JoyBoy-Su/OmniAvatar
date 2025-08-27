#!/usr/bin/env python3

import subprocess
import time
import os
import sys
import signal
import requests

def test_server_client():
    """Test the server-client implementation"""
    print("Testing OmniAvatar server-client implementation...")
    
    # Check if Flask is available
    try:
        import flask
        import requests
        print("✓ Flask and requests are available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install: pip install flask requests")
        return False
    
    server_process = None
    try:
        # Start the server in background
        print("\n1. Starting server...")
        server_process = subprocess.Popen([
            sys.executable, "server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        print("Waiting for server to initialize...")
        max_wait = 60  # Wait up to 60 seconds
        server_ready = False
        
        for i in range(max_wait):
            try:
                response = requests.get("http://localhost:8080/health", timeout=2)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('model_loaded'):
                        server_ready = True
                        print("✓ Server is ready and model is loaded")
                        break
                    else:
                        print(f"Server starting... model not loaded yet ({i+1}/{max_wait})")
                except:
                    print(f"Server starting... ({i+1}/{max_wait})")
                time.sleep(1)
        
        if not server_ready:
            print("✗ Server failed to start or model failed to load")
            return False
        
        # Test client health check
        print("\n2. Testing client health check...")
        result = subprocess.run([
            sys.executable, "client.py", "--check-health"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Client health check passed")
            print(result.stdout)
        else:
            print("✗ Client health check failed")
            print(result.stderr)
            return False
        
        # Test basic text-to-video generation
        print("\n3. Testing basic text-to-video generation...")
        result = subprocess.run([
            sys.executable, "client.py",
            "--prompt", "A person speaking with natural facial expressions",
            "--output", "test_output.mp4",
            "--steps", "2"  # Use fewer steps for faster testing
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✓ Basic text-to-video generation successful")
            print(result.stdout)
            if os.path.exists("test_output.mp4"):
                print("✓ Output video file created")
            else:
                print("⚠ Output video file not found")
        else:
            print("✗ Text-to-video generation failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        print("\n✓ All tests passed!")
        return True
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        return False
    finally:
        # Clean up server process
        if server_process:
            print("\nShutting down server...")
            try:
                server_process.terminate()
                server_process.wait(timeout=10)
            except:
                server_process.kill()
        
        # Clean up test files
        for test_file in ["test_output.mp4"]:
            if os.path.exists(test_file):
                os.remove(test_file)

if __name__ == "__main__":
    success = test_server_client()
    sys.exit(0 if success else 1)