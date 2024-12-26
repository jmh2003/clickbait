import pandas as pd
import os
import json
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm.auto import tqdm
import time

def init_translator():
    model_path = 'translate-model/zh-en'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    translate_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    # 修正CUDA设备初始化
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    translate_model = translate_model.to(device)
    return tokenizer, translate_model, device

def batch_translate(texts, tokenizer, model, device, batch_size=16, desc="翻译进度", timeout=30, max_retries=3):
    translations = []
    
    with tqdm(total=len(texts), desc=desc) as pbar:
        for i in range(0, len(texts), batch_size):
            retries = 0
            while retries < max_retries:
                try:
                    batch = texts[i:i + batch_size]
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                    with torch.no_grad():
                        inputs = tokenizer(
                            batch,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=256  # 减小最大长度
                        ).to(device)
                        
                        # 设置超时
                        torch.cuda.set_device(device)
                        start_time = time.time()
                        translated = model.generate(
                            **inputs,
                            max_length=256,
                            num_beams=2,
                            length_penalty=0.6,
                            early_stopping=True
                        )
                        
                        if time.time() - start_time > timeout:
                            raise TimeoutError("翻译超时")
                            
                    decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
                    translations.extend(decoded)
                    pbar.update(len(batch))
                    break  # 成功则退出重试循环
                    
                except (RuntimeError, TimeoutError) as e:
                    retries += 1
                    print(f"\n批次 {i} 处理失败 (尝试 {retries}/{max_retries}): {str(e)}")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    if retries == max_retries:
                        print(f"批次 {i} 最终失败，使用原文本")
                        translations.extend(batch)
                    time.sleep(1)  # 等待一秒后重试
                    
    return translations

def process_single_file(file_path, tokenizer, model, device, batch_size=16, save_interval=100):
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_excel(file_path, usecols=[0, 1])
        
        titles = df.iloc[:, 0].fillna('').astype(str).tolist()
        urls = df.iloc[:, 1].fillna('').astype(str).tolist()
        
        output_path = os.path.join(os.path.dirname(file_path), f"{file_name}_translated.json")
        results = []
        total_processed = 0
        
        # 分批处理
        for i in range(0, len(titles), save_interval):
            batch_titles = titles[i:i + save_interval]
            batch_urls = urls[i:i + save_interval]
            
            # 翻译当前批次
            translated_titles = batch_translate(
                texts=batch_titles,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=batch_size,
                desc=f"翻译 {file_name} ({i}-{min(i+save_interval, len(titles))})"
            )
            
            # 构建当前批次结果
            batch_results = []
            for original, translated, url in zip(batch_titles, translated_titles, batch_urls):
                if original.strip():
                    batch_results.append({
                        "translated_title": translated,
                        "original_title": original,
                        "url": url
                    })
            
            # 保存当前批次
            if i == 0:
                # 第一批次，创建新文件
                mode = 'w'
            else:
                # 读取现有文件
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                mode = 'w'  # 重写模式
            
            # 更新并保存所有结果
            results.extend(batch_results)
            with open(output_path, mode, encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            total_processed += len(batch_results)
            print(f"\n已处理并保存 {total_processed}/{len(titles)} 条记录")
            
            # 清理GPU缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return total_processed
        
    except Exception as e:
        print(f"处理文件出错 {file_path}: {e}")
        return 0

def process_excel_files():
    print("初始化翻译模型...")
    tokenizer, model, device = init_translator()
    print(f"使用设备: {device}")
    
    # directory = "test_data"
    # directory = "data_backup"
    directory = "news"
    # 过滤临时文件
    excel_files = [f for f in os.listdir(directory) 
                  if f.endswith('.xlsx') and not f.startswith('~$')]
    
    total_processed = 0
    for filename in excel_files:
        try:
            # 处理文件名编码
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
        except Exception as e:
            print(f"处理文件出错 {filename}: {str(e)}")
            continue
    
    print(f"\n处理完成! 共处理 {len(excel_files)} 个文件, {total_processed} 条记录")

if __name__ == "__main__":
    process_excel_files()
