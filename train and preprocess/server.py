import subprocess
import sys
import os
import socket

def get_host_ip():
    """Get host IP address"""
    try:
        # Connect to an external address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == "__main__":
    # Model path (local directory)
    model_path = "Qwen/Qwen3-8B"

    # Get host information
    hostname = socket.gethostname()
    host_ip = get_host_ip()
    
    print(f"🖥️  Hostname: {hostname}")
    print(f"📡 Host IP: {host_ip}")
    print(f"🚀 Service will start at:")
    print(f"   - http://{host_ip}:8888")
    print(f"   - http://localhost:8888 (Local)")
    print(f"   - http://0.0.0.0:8888 (All interfaces)")

    # Set environment variables to disable compilation optimization
    env = os.environ.copy()
    env["VLLM_DISABLE_COMPILATION"] = "1"
    env["TORCH_COMPILE_DISABLE"] = "1"
    
    # Start vLLM OpenAI compatible API service
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8888",
        "--dtype", "auto",  # Explicitly specify data type parameter
        "--enforce-eager"  # Force eager mode to avoid compilation
    ]
    
    print(f"🎯 Startup command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)
