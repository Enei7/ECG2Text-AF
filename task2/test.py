import sys
import os

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 task1 导入数据加载逻辑
from task1.train import load_and_split_data
from task1.dataset import ECGDataset
from task2_dataset import ECGInstructionDataset, custom_collate_fn
from task2_model import ECGLLM