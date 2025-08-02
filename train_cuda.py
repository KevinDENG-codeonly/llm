import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers.integrations import MLflowCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import mlflow
import mlflow.pytorch


def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            context = data["text"]
            catagory = data["category"]
            label = data["output"]
            message = {
   
                "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
                "input": f"文本:{context},类型选型:{catagory}",
                "output": label,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 确保模型在正确的设备上
    if hasattr(model, 'module'):
        model = model.module
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            use_cache=False,  # 显式禁用缓存以避免警告
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)
    return response

# 检查CUDA可用性
if torch.cuda.is_available():
    print(f"CUDA 可用，设备数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.get_device_name()}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA 不可用，将使用 CPU")

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)

# 确保 tokenizer 有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("设置 pad_token 为 eos_token")

# 检查设备可用性 - CUDA 优先
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 使用 device_map="auto" 让模型自动分配到最佳设备上
model = AutoModelForCausalLM.from_pretrained(
    "./qwen/Qwen2-1___5B-Instruct/", 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "test.jsonl"

train_jsonl_new_path = "new_train.jsonl"
test_jsonl_new_path = "new_test.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

# CUDA 优化的训练参数
args = TrainingArguments(
    output_dir="./output/Qwen1.5",
    per_device_train_batch_size=8,  # CUDA 可以使用更大的 batch size
    gradient_accumulation_steps=2,  # 减少梯度累积步数
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="mlflow",
    fp16=torch.cuda.is_available(),  # 在 CUDA 上启用混合精度训练
    dataloader_pin_memory=True,  # 加速数据传输
    dataloader_num_workers=4,  # 多线程数据加载
    remove_unused_columns=False,
    warmup_steps=100,  # 学习率预热
    logging_strategy="steps",
    save_strategy="steps",
)

# 设置 MLflow 实验
mlflow.set_experiment("Qwen2-finetune-cuda")

# 开始 MLflow run
with mlflow.start_run(run_name="Qwen2-1.5B-Instruct-CUDA") as run:
    # 记录参数
    mlflow.log_params({
        "model": "qwen/Qwen2-1.5B-Instruct",
        "dataset": "huangjintao/zh_cls_fudan-news",
        "device": device,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 2,
        "learning_rate": 1e-4,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "fp16": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    })

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    # 保存模型 - 需要解包混合精度包装的模型
    try:
        # 尝试从 trainer 中获取解包的模型
        unwrapped_model = trainer.model
        if hasattr(trainer.model, 'module'):
            # 如果是 DataParallel 包装的模型
            unwrapped_model = trainer.model.module
        
        # 保存模型权重而不是整个模型对象
        model_path = "./output/Qwen1.5/final_model"
        os.makedirs(model_path, exist_ok=True)
        unwrapped_model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # 记录模型路径到 MLflow
        mlflow.log_artifacts(model_path, "model")
        mlflow.log_text(f"Model saved to {model_path}", "model_info.txt")
        
    except Exception as e:
        print(f"模型保存出现问题: {e}")
        # 备用方案：只保存模型权重
        model_state_path = "./output/Qwen1.5/model_state.pth"
        torch.save(unwrapped_model.state_dict(), model_state_path)
        mlflow.log_artifact(model_state_path, "model")
        print(f"已保存模型权重到: {model_state_path}")

    # 用测试集的前10条，测试模型
    test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]

    print("开始测试模型...")
    test_results = []
    
    # 确保模型在测试前处于正确状态
    unwrapped_model.eval()
    
    for index, row in test_df.iterrows():
        instruction = row['instruction']
        input_value = row['input']

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        try:
            response = predict(messages, unwrapped_model, tokenizer)
            messages.append({"role": "assistant", "content": f"{response}"})
            result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
            test_results.append({
                "input": input_value,
                "prediction": response,
                "full_conversation": result_text
            })
            print(f"测试样本 {index + 1}/10 完成")
        except Exception as e:
            print(f"测试样本 {index + 1} 失败: {e}")
            test_results.append({
                "input": input_value,
                "prediction": f"Error: {str(e)}",
                "full_conversation": f"Error occurred: {str(e)}"
            })

    # 记录测试结果到 MLflow
    test_results_df = pd.DataFrame(test_results)
    mlflow.log_table(test_results_df, "test_predictions.json")
    
    # 记录一些样例预测结果作为文本
    for i, result in enumerate(test_results[:3]):  # 只记录前3个样例
        mlflow.log_text(result["full_conversation"], f"sample_prediction_{i}.txt")
    
    print("测试完成！")

# 清理 CUDA 缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA 缓存已清理")
