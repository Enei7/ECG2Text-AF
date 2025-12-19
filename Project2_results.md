# Project2

## task1

### task1.1：可视化原始数据

1. normal

   a. 时域

![img](https://cdn.nlark.com/yuque/0/2025/png/40371192/1763540186454-3772084c-abd6-40d4-b420-ea152f4e478b.png)

   b. 频域

![img](https://cdn.nlark.com/yuque/0/2025/png/40371192/1763540253969-1b73f5d7-0c98-4ffd-ba8a-9d78bba0622f.png)

2. Atrial Fibrillation

   a. 时域

![img](https://cdn.nlark.com/yuque/0/2025/png/40371192/1763540288594-88361b52-a1c0-478a-ba08-52aab111108f.png)

   b. 频域

![img](https://cdn.nlark.com/yuque/0/2025/png/40371192/1763540298529-b8040e12-69fc-4f41-bcfb-c9f670af3261.png)

### task1.2-3：量化不同stream2对模型性能的影响

**超参数【**epoch：50，batch：64，Learning Rate：0.01**】**

| Model         | Loss   | Accuracy   | Recall | Precision | F1         | AUC    |
| ------------- | ------ | ---------- | ------ | --------- | ---------- | ------ |
| single stream | 0.1390 | 0.9735     | 0.8378 | 0.9490    | 0.8900     | 0.9858 |
| MS-CNN(3,5)   | 0.1698 | 0.9666     | 0.8288 | 0.9020    | 0.8638     | 0.9863 |
| MS-CNN(3,7)   | 0.1660 | 0.9712     | 0.8919 | 0.8839    | 0.8879     | 0.9869 |
| MS-CNN(3,9)   | 0.1593 | **0.9770** | 0.9009 | 0.9174    | **0.9091** | 0.9840 |

ROC曲线如图所示。

<img src="https://cdn.nlark.com/yuque/0/2025/png/32796972/1766074478485-d5492be6-ab71-473e-a6c5-f46a79869d3b.png" alt="img" style="zoom: 17%;" /><img src="https://cdn.nlark.com/yuque/0/2025/png/32796972/1766074559167-b5166842-cd57-44c5-8c93-d38fa95b2998.png" alt="img" style="zoom:17%;" />

<img src="https://cdn.nlark.com/yuque/0/2025/png/32796972/1766074529991-af08f319-8032-4a92-a375-5186b8b99e85.png" alt="img" style="zoom:17%;" /><img src="https://cdn.nlark.com/yuque/0/2025/png/32796972/1766074508543-d14835ce-b391-46f9-a40b-f850d973298a.png" alt="img" style="zoom:17%;" />



### task1.4：超参数调优与优化

选择不同的学习率(LR)和batch size进行训练，其中选用MS-CNN(3,5)进行训练

| LR(batch=64)            | Loss     | Accuracy     | Recall     | Precision     | F1     | AUC     |
| ----------------------- | -------- | ------------ | ---------- | ------------- | ------ | ------- |
| 0.01                    | 0.1698   | 0.9666       | 0.8288     | 0.9020        | 0.8638 | 0.9863  |
| 0.001                   | 0.1432   | 0.9655       | 0.9009     | 0.8403        | 0.8696 | 0.9864  |
| 0.0001                  | 0.2870   | 0.9459       | 0.7748     | 0.7963        | 0.7854 | 0.9529  |
| 0.00001                 | 0.5536   | 0.9367       | 0.7568     | 0.7500        | 0.7534 | 0.9312  |
| **batch size(LR=0.01)** | **Loss** | **Accuracy** | **Recall** | **Precision** | **F1** | **AUC** |
| 32                      | 0.1561   | 0.9689       | 0.8378     | 0.9118        | 0.8732 | 0.9882  |
| 64                      | 0.1698   | 0.9666       | 0.8288     | 0.9020        | 0.8638 | 0.9863  |
| 128                     | 0.2298   | 0.9678       | 0.8649     | 0.8807        | 0.8727 | 0.9781  |



### task1.5：数据增强与训练

加入高斯噪声，随机尺度变换【epoch：50，batch：64，Learning Rate：0.01，MS-CNN(3,7)】

| augment | Loss   | Accuracy | Recall | Precision | F1     | AUC    |
| ------- | ------ | -------- | ------ | --------- | ------ | ------ |
| none    | 0.1660 | 0.9712   | 0.8919 | 0.8839    | 0.8879 | 0.9869 |
| noise   | 0.1610 | 0.9712   | 0.8559 | 0.9135    | 0.8837 | 0.9795 |
| scale   | 0.2136 | 0.9620   | 0.8468 | 0.8545    | 0.8507 | 0.9703 |
| all     | 0.1561 | 0.9758   | 0.9099 | 0.9018    | 0.9058 | 0.9813 |

1. 原始数据

<img src="https://cdn.nlark.com/yuque/0/2025/png/40371192/1763552626752-0a116106-1077-4179-821a-901ecb0d4397.png" alt="img" style="zoom:33%;" />

1. 高斯噪声

<img src="https://cdn.nlark.com/yuque/0/2025/png/40371192/1763552637344-45f1d2a3-781a-4ef0-8209-a51163b5b957.png" alt="img" style="zoom:33%;" />

1. 尺度变换

<img src="https://cdn.nlark.com/yuque/0/2025/png/40371192/1763552645087-40327287-67f4-4364-b31b-8e1a013d4c2b.png" alt="img" style="zoom:33%;" />

1. 高斯噪声+尺度变换

<img src="https://cdn.nlark.com/yuque/0/2025/png/40371192/1763552650814-3c71d690-4173-4511-89a1-917f2f481bab.png" alt="img" style="zoom: 33%;" />

## task2

使用Qwen3-8B进行LoRA，batch_size=8，epoch=3，learning rate=0.001

```yaml
prompt1:请作为一名心脏病专家，分析这段心电图信号特征。<|ecg_feature|>请判断该患者是否患有房颤（Atrial Fibrillation）？
```

| Model       | Accuracy | Recall | Precision | F1     |
| ----------- | -------- | ------ | --------- | ------ |
| CNN+LoRA    | 0.9678   | 0.8559 | 0.8879    | 0.8716 |
| MS-CNN(3,5) | 0.9666   | 0.8288 | 0.9020    | 0.8638 |
| MS-CNN(3,7) | 0.9712   | 0.8919 | 0.8839    | 0.8879 |
| MS-CNN(3,9) | 0.9770   | 0.9009 | 0.9174    | 0.9091 |

```matlab
prompt2:是否有房颤？
```

| Model   | Accuracy | Recall | Precision | F1     |
| ------- | -------- | ------ | --------- | ------ |
| prompt1 | 0.9678   | 0.8559 | 0.8879    | 0.8716 |
| prompt2 | 0.9597   | 0.7838 | 0.8878    | 0.8325 |
