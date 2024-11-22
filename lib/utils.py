import subprocess

def is_cuda_available():
    try:
        # Run nvidia-smi command
        output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        
        # Check for CUDA-capable devices
        if "CUDA Version" in output:
            return True
        
    except Exception:
        pass

    return False