import pandas as pd
import os
import json
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm.auto import tqdm
import time
import numpy as np
import random

def init_translator():
    # 设置随机种子保持不变
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    # 改用英译中模型
    model_path = 'translate-model\en-zh'  # 修改为英译中模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return tokenizer, model, device

def batch_translate(texts, tokenizer, model, device, batch_size=16, desc="翻译中", timeout=30):
    translations = []
    
    # 添加进度条
    with tqdm(total=len(texts), desc=desc) as pbar:
        for i in range(0, len(texts), batch_size):
            retries = 0
            max_retries = 3
            
            while retries < max_retries:
                try:
                    batch = texts[i:i + batch_size]
                    # 清理GPU缓存
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        inputs = tokenizer(
                            batch,
                            return_tensors="pt", 
                            padding=True,
                            truncation=True,
                            max_length=256  # 限制长度加速
                        ).to(device)
                        
                        # 设置超时
                        torch.cuda.set_device(device)
                        start_time = time.time()
                        translated = model.generate(
                            **inputs,
                            max_length=256,
                            num_beams=2,  # 减小beam search宽度
                            length_penalty=0.6,
                            early_stopping=True  # 启用早停
                        )
                        
                        if time.time() - start_time > timeout:
                            raise TimeoutError("翻译超时")
                        
                        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
                        translations.extend(decoded)
                        pbar.update(len(batch))
                        break  # 成功跳出重试循环
                        
                except (RuntimeError, TimeoutError) as e:
                    retries += 1
                    print(f"\n批次 {i} 处理失败 (尝试 {retries}/{max_retries}): {str(e)}")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    if retries == max_retries:
                        print(f"批次 {i} 最终失败，使用原文本")
                        translations.extend(batch)
                    time.sleep(1)  # 失败后等待1秒重试
                    
    return translations

def process_single_file(file_path, tokenizer, model, device, batch_size=16):
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 读取JSON而不是Excel
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        texts = [item['text'] for item in data]
        
        # 批量翻译
        translated_texts = batch_translate(
            texts=texts,
            tokenizer=tokenizer,
            model=model, 
            device=device,
            batch_size=batch_size,
            desc=f"翻译 {file_name}"
        )
        
        # 更新数据
        for i, item in enumerate(data):
            item['text_zh'] = translated_texts[i]
            
        # 保存翻译结果
        output_path = file_path.replace('.json', '_zh.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"已保存翻译结果到: {output_path}")
        return len(texts)
        
    except Exception as e:
        print(f"处理文件出错 {file_path}: {str(e)}")
        return 0

def main():
    print("初始化翻译模型...")
    tokenizer, model, device = init_translator()
    print(f"使用设备: {device}")
    
    directory = "data/clickbait_detection_dataset"
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(directory) 
                 if f.endswith('.json') and not f.endswith('_zh.json')]
    
    total_processed = 0
    for filename in json_files:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path) and os.access(file_path, os.R_OK):
            print(f"\n处理文件: {filename}")
            processed_count = process_single_file(
                file_path,
                tokenizer,
                model, 
                device
            )
            total_processed += processed_count
        else:
            print(f"跳过文件 {filename}: 无法访问")
            
    print(f"\n处理完成! 共处理 {len(json_files)} 个文件, {total_processed} 条记录")

if __name__ == "__main__":
    main()