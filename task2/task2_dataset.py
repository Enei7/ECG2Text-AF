import torch
from torch.utils.data import Dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task1.dataset import ECGDataset  # 从 task1 导入数据集类

class ECGInstructionDataset(Dataset):
    def __init__(self, ecg_dataset, tokenizer, max_length=512):
        """
        包装 Task 1 的 ECGDataset，生成用于 LLM 的指令数据
        """
        self.ecg_dataset = ecg_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 定义指令模板
        self.instruction_template = "<|ecg_feature|>,是否患有房颤？"
        
        # 定义标签映射文本
        self.answer_map = {
            0: "无房颤",
            1: "有房颤"
        }

    def __len__(self):
        return len(self.ecg_dataset)

    def __getitem__(self, index):
        # 1. 获取原始 ECG 信号和数字标签 (0/1)
        # 注意：task1 dataset 返回 (sig_tensor, label_tensor)
        ecg_signal, label_tensor = self.ecg_dataset[index]
        label_val = int(label_tensor.item())
        
        # 2. 构建指令文本 (Prompt) 和 答案文本 (Completion)
        instruction_text = self.instruction_template
        answer_text = self.answer_map.get(label_val, "无法判断。")
        
        # 3. 构建完整的对话文本用于 Tokenize
        # 格式: User: <instruction> \n Assistant: <answer>
        # 注意：这里我们手动构建 prompt，具体格式取决于 Qwen 的 chat template，
        # 这里使用一个通用的简化格式，在 Model 中处理 <|ecg_feature|> 的替换
        
        # Qwen-Chat 通常使用 ChatML 格式，但为了简化多模态拼接，我们手动处理 input_ids
        # 我们将文本分为两部分：Prompt 和 Answer
        
        prompt_ids = self.tokenizer.encode(instruction_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        
        # 4. 返回数据
        return {
            "ecg_signal": ecg_signal,       # [1, 2400]
            "prompt_ids": prompt_ids,       # List[int]
            "answer_ids": answer_ids,       # List[int]
            "label": label_val              # 用于评估计算 Acc/F1
        }

def custom_collate_fn(batch):
    """
    自定义整理函数，用于处理变长的文本序列并进行 Padding
    """
    ecg_signals = []
    input_ids_list = []
    labels_list = [] # 用于计算 LM loss
    raw_labels = []  # 用于评估 (0/1)
    
    for item in batch:
        ecg_signals.append(item['ecg_signal'])
        raw_labels.append(item['label'])
        
        p_ids = item['prompt_ids']
        a_ids = item['answer_ids']
        
        # 拼接 Input IDs: Prompt + Answer
        full_ids = p_ids + a_ids
        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        
        # 构建 Labels: Prompt 部分掩盖为 -100 (不计算 Loss)，Answer 部分保留
        # label_ids: [-100, ..., -100] + [answer_token_1, ..., eos]
        label_ids = [-100] * len(p_ids) + a_ids
        labels_list.append(torch.tensor(label_ids, dtype=torch.long))
        
    # Padding
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != 0).long()
    
    ecg_signals = torch.stack(ecg_signals)
    
    return {
        "ecg_signals": ecg_signals,
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask,
        "raw_labels": torch.tensor(raw_labels)
    }