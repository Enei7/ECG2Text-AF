import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import sys

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 task1 导入数据加载逻辑
from task1.train import load_and_split_data
from task1.dataset import ECGDataset
from task2_dataset import ECGInstructionDataset, custom_collate_fn
from task2_model import ECGLLM

# ================= 配置 =================
LLM_PATH = "/home/hzw/models/Qwen3-8B" 
CNN_PATH = "/home/hzw/workspace/ecg/task1/results/best.pth"
DATA_DIR = "/home/hzw/workspace/ecg/training2017"
CSV_PATH = os.path.join(DATA_DIR, "REFERENCE.csv")
RESULTS_DIR = "./task2_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 8       # LLM 显存占用大，Batch Size 调小，使用梯度累积
GRAD_ACCUM_STEPS = 4
EPOCHS = 3
LR = 1e-3

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备 Tokenizer
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. 准备数据 (复用 Task 1 逻辑)
    print("正在准备数据集...")
    train_data, val_data, test_data = load_and_split_data(CSV_PATH)
    
    # 创建ECGDataset实例 (训练集开启增强)
    train_ecg_ds = ECGDataset(
        DATA_DIR, 
        train_data[0],  # file_ids
        train_data[1],  # labels
        target_len=2400,
        is_train=True,   # 开启随机裁剪等
        aug_mode='random' # 开启信号增强
    )
    
    val_ecg_ds = ECGDataset(
        DATA_DIR,
        val_data[0],
        val_data[1],
        target_len=2400,
        is_train=False,
        aug_mode='none'
    )
    
    test_ecg_ds = ECGDataset(
        DATA_DIR,
        test_data[0],
        test_data[1],
        target_len=2400,
        is_train=False,
        aug_mode='none'
    )
    
    # 包装成 Instruction Dataset
    train_ds = ECGInstructionDataset(train_ecg_ds, tokenizer)
    val_ds = ECGInstructionDataset(val_ecg_ds, tokenizer)
    test_ds = ECGInstructionDataset(test_ecg_ds, tokenizer)
    
    # 创建 LLM 专用的 DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    # 3. 初始化模型
    print("正在初始化多模态模型...")
    model = ECGLLM(
        llm_path=LLM_PATH, 
        cnn_checkpoint_path=CNN_PATH, 
        ecg_feature_dim=1024,          # 双流模型输出 1024
        k_size_stream2=args.k_size_stream2
    )
    
    # 将CNN encoder移到CPU以节省GPU显存
    print("将CNN Encoder移到CPU...")
    model.ecg_encoder = model.ecg_encoder.cpu()
    
    # 清理不用的CNN分类头
    if hasattr(model.ecg_encoder, 'mlp'):
        del model.ecg_encoder.mlp
    
    # 4. 配置 LoRA
    print("配置 LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16,  # 增加到16以提高表达能力
        lora_alpha=32, 
        lora_dropout=0.05,  # 降低dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Qwen全部注意力和FFN模块
    )
    
    # 将 LoRA 应用于 LLM 部分
    model.llm = get_peft_model(model.llm, peft_config)
    model.llm.print_trainable_parameters()
    
    # 确保 Projector 是可训练的
    for param in model.projector.parameters():
        param.requires_grad = True
        
    # model.to(device) # 不要将整个模型移动到 device，因为 LLM 使用了 device_map="auto"
    # 只需将 projector 移动到 device (假设 LLM 输入层在 device 上)
    model.projector.to(device)
    
    # 5. 优化器
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    
    # 6. 训练循环
    if not args.eval_only:
        print("开始训练...")
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                # 移动数据到 GPU
                ecg_signals = batch['ecg_signals'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(ecg_signals, input_ids, attention_mask, labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()
                
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * GRAD_ACCUM_STEPS
                progress_bar.set_postfix({"loss": loss.item() * GRAD_ACCUM_STEPS})
            
            print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f}")
            
            # 保存 Checkpoint
            save_path = os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            # 我们只保存 Projector 和 LoRA 权重，不保存整个 LLM
            # 但为了简单起见，这里保存 state_dict，加载时需注意
            torch.save(model.state_dict(), save_path)
    else:
        print("跳过训练，直接进行评估...")
        if args.checkpoint_path:
            print(f"正在加载 Checkpoint: {args.checkpoint_path}")
            # 加载权重
            # 注意：model.state_dict() 在 PEFT 模型中只包含 LoRA 和 Projector 参数
            # 所以我们需要 strict=False 来加载，或者确保 key 匹配
            state_dict = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            print("警告: 未指定 checkpoint_path，将使用随机初始化的 Projector 和 LoRA 进行评估 (除非已在代码中硬编码加载)")

    # 7. 评估 (测试集)
    print("开始评估...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            ecg_signals = batch['ecg_signals'].to(device)
            raw_labels = batch['raw_labels'].numpy()
            
            # 生成文本
            # 注意：generate 函数里我们硬编码了 prompt，只测试生成能力
            generated_texts = model.generate(ecg_signals, tokenizer)
            
            for text, label in zip(generated_texts, raw_labels):
                # 解析输出
                # 简单的关键词匹配
                pred = 0
                if "有房颤" in text or "患有房颤" in text:
                    pred = 1
                
                all_preds.append(pred)
                all_labels.append(label)
                
                # 打印几个例子看看
                if len(all_preds) <= 3:
                    print(f"\n真实标签: {label} ('有' if 1 else '无')")
                    print(f"模型输出: {text}")
                    print(f"解析预测: {pred}")

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    precision = precision_score(all_labels, all_preds, pos_label=1)
    recall = recall_score(all_labels, all_preds, pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算特异度 (Specificity)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n=== Task 2 最终评估 ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (AF): {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score (AF): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'AF']))
    
    # 保存结果
    with open(os.path.join(RESULTS_DIR, "task2_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 确保这里的 default 值和你 Task 1 训练 best.pth 时用的一样！
    parser.add_argument('--k_size_stream2', type=int, default=7, help='CNN Stream 2 Kernel Size (Must match Task 1)')
    parser.add_argument('--eval_only', action='store_true', help='Skip training and run evaluation only')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for evaluation')
    args = parser.parse_args()
    
    train(args)