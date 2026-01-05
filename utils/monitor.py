import torch
import os
import time


class Monitor:
    """精简监视器类，只用于设置GPU设备和记录时间"""
    
    def __init__(self, device_id="0"):
        """
        初始化监视器
        
        Args:
            device_id: CUDA设备ID，默认为"0"
        """
        self.device_id = device_id
        self.start_time = None
        
    def setup(self):
        """设置CUDA环境"""
        print(f"原始CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
        
        # 设置CUDA可见设备
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device_id
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，请确保已分配GPU资源！")
        
        print(f'CUDA是否可用: {torch.cuda.is_available()}')
        print(f'当前CUDA设备: {torch.cuda.current_device()}')
        print(f'GPU名称: {torch.cuda.get_device_name(0)}')
        print(f'监视器初始化成功')
    
    def start(self, interval=None):
        """
        开始时间记录
        
        Args:
            interval: 兼容参数，不使用
        """
        self.start_time = time.time()
        print(f"时间记录已开始")
    
    def end(self):
        """
        结束时间记录
        
        Returns:
            float: 运行时间（秒）
        """
        if self.start_time is None:
            print("警告: 时间记录未开始")
            return 0.0
        
        end_time = time.time()
        run_time = end_time - self.start_time
        self.start_time = None
        
        print(f"时间记录已结束，运行时间: {run_time:.2f} 秒")
        
        return run_time
