import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task1.model import MSCNN # 复用 Task 1 的模型定义

class ECGProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 一个稳定的 MLP 将 ECG 特征维度 (1024) 映射到 LLM 维度 (e.g. 4096)
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ECGLLM(nn.Module):
    def __init__(self, llm_path, cnn_checkpoint_path, ecg_feature_dim=1024, k_size_stream2=7):
        super().__init__()
        
        print(f"正在加载 LLM: {llm_path}")
        # 加载 Qwen 模型 (fp16 或 bf16 以节省显存)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        
        # 获取 LLM 的 hidden size
        self.llm_hidden_size = self.llm.config.hidden_size
        
        print(f"正在加载 CNN Encoder: {cnn_checkpoint_path}")
        # 加载 Task 1 训练好的 MSCNN
        # 注意：k_size_stream2 必须与 Task 1 训练时一致
        self.ecg_encoder = MSCNN(in_channels=1, out_channels=1, use_single_stream=False, k_size_stream2=k_size_stream2)
        
        # 加载权重
        state_dict = torch.load(cnn_checkpoint_path, map_location='cpu')
        self.ecg_encoder.load_state_dict(state_dict)
        
        # 冻结 CNN Encoder 参数
        for param in self.ecg_encoder.parameters():
            param.requires_grad = False
            
        # 定义投影层 (Projector)
        self.projector = ECGProjector(input_dim=ecg_feature_dim, output_dim=self.llm_hidden_size)
        # 确保 Projector 与 LLM dtype 一致
        self.projector.to(dtype=self.llm.dtype)

    def encode_ecg(self, x):
        """
        使用 Task 1 的 MSCNN 提取特征。
        """
        # 确保input在CPU上(因为CNN encoder被移到CPU)
        device = next(self.ecg_encoder.parameters()).device
        x = x.to(device)
        
        # 模拟MSCNN.forward直到global_pool
        # Stream 1
        x1 = self.ecg_encoder.s1_pool1(self.ecg_encoder.s1_conv1(x))
        x1 = self.ecg_encoder.s1_pool2(self.ecg_encoder.s1_conv2(x1))
        x1 = self.ecg_encoder.s1_pool3(self.ecg_encoder.s1_conv3(x1))
        x1 = self.ecg_encoder.s1_pool4(self.ecg_encoder.s1_conv4(x1))
        x1 = self.ecg_encoder.s1_pool5(self.ecg_encoder.s1_conv5(x1))
        x1_pooled = self.ecg_encoder.global_pool(x1)

        # Stream 2
        x2 = self.ecg_encoder.s2_pool1(self.ecg_encoder.s2_conv1(x))
        x2 = self.ecg_encoder.s2_pool2(self.ecg_encoder.s2_conv2(x2))
        x2 = self.ecg_encoder.s2_pool3(self.ecg_encoder.s2_conv3(x2))
        x2 = self.ecg_encoder.s2_pool4(self.ecg_encoder.s2_conv4(x2))
        x2 = self.ecg_encoder.s2_pool5(self.ecg_encoder.s2_conv5(x2))
        x2_pooled = self.ecg_encoder.global_pool(x2)
        
        # 融合
        x_merged = torch.cat((x1_pooled, x2_pooled), dim=1) # [B, 1024, 1]
        x_flat = x_merged.view(x_merged.size(0), -1)        # [B, 1024]
        
        return x_flat

    def forward(self, ecg_signals, input_ids, attention_mask=None, labels=None):
        """
        input_ids 中包含了文本 token ID。
        """
        # 1. 获取 ECG 特征并投影
        with torch.no_grad():
            ecg_feats = self.encode_ecg(ecg_signals) # [B, 1024]
        
        # 2. 获取文本 Embeddings (先获取以确定目标dtype)
        text_embeds = self.llm.get_input_embeddings()(input_ids) # [B, Seq_Len, hidden_size]
        target_dtype = text_embeds.dtype
        target_device = text_embeds.device
        
        # 3. 将ecg_feats转换为目标dtype和device后投影
        ecg_feats = ecg_feats.to(device=target_device, dtype=target_dtype)
        ecg_embeds = self.projector(ecg_feats) # [B, llm_hidden_size]
        # 增加序列维度: [B, 1, llm_hidden_size]
        ecg_embeds = ecg_embeds.unsqueeze(1)
        
        # 4. 拼接 Embeddings: [ECG, Text]
        # 我们将 ECG 特征直接拼在文本最前面
        inputs_embeds = torch.cat([ecg_embeds, text_embeds], dim=1)
        
        # 5. 处理 Attention Mask 和 Labels
        # 因为在前面加了1个token长度的ECG特征，mask 和 labels 也要填充
        if attention_mask is not None:
            # [B, 1] 的 1
            prefix_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
        if labels is not None:
            # Labels 前面补 -100 (忽略 ECG 特征的预测)
            prefix_labels = torch.full((labels.shape[0], 1), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([prefix_labels, labels], dim=1)
            
        # 6. 输入 LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

    def generate(self, ecg_signals, tokenizer, max_new_tokens=50):
        """
        推理生成函数
        """
        # 1. 构建 Prompt (只有指令部分)
        prompt = "<|ecg_feature|>,是否患有房颤？"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # 复制 input_ids 以匹配 batch size
        batch_size = ecg_signals.shape[0]
        input_ids = input_ids.repeat(batch_size, 1)
        
        # 2. 获取文本embeddings以确定目标dtype/device
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        target_dtype = text_embeds.dtype
        target_device = text_embeds.device
        
        # 3. 编码ECG并转换dtype
        ecg_feats = self.encode_ecg(ecg_signals)
        ecg_feats = ecg_feats.to(device=target_device, dtype=target_dtype)
        ecg_embeds = self.projector(ecg_feats).unsqueeze(1)
        
        # 4. 拼接embeddings
        inputs_embeds = torch.cat([ecg_embeds, text_embeds], dim=1)
        
        # 3. 生成
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False # 贪婪搜索，保证结果确定性
        )
        
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)