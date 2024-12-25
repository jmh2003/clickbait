import pandas as pd
import os
import json
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def init_translator():
    model_path = 'translate-model/zh-en'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    translate_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return transformers.pipeline("translation", model=translate_model, tokenizer=tokenizer)

def translate(text, pipeline):
    try:
        translate_text = pipeline(text)[0]['translation_text']
        return translate_text
    except Exception as e:
        print(f"翻译错误: {e}")
        return text

def process_excel_files():
    # 初始化翻译器
    translator = init_translator()
    
    # 指定目录路径
    directory = "test_data"
    
    # 存储所有数据的列表
    all_data = []
    
    # 遍历目录中的所有xlsx文件
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            try:
                # 读取Excel文件的标题和URL列
                df = pd.read_excel(file_path, usecols=[0, 1])
                print(f"正在处理 {filename}...")
                
                # 处理每一行
                for _, row in df.iterrows():
                    try:
                        title = str(row.iloc[0])  # 第一列是标题
                        url = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""  # 第二列是URL
                        
                        # 使用本地模型翻译标题
                        translated_title = translate(title, translator)
                        
                        # 创建数据字典
                        data_dict = {
                            "translated_title": translated_title,
                            "original_title": title,
                            "url": url
                        }
                        all_data.append(data_dict)
                        print(f"原标题: {title}")
                        print(f"翻译后: {translated_title}")
                        print(f"URL: {url}\n")
                        
                    except Exception as e:
                        print(f"处理标题时出错 '{title}': {e}")
                        continue
                        
            except Exception as e:
                print(f"读取文件出错 {filename}: {e}")
                continue
    
    # 将数据保存为JSON文件
    output_path = os.path.join(directory, "translated_titles.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"数据已保存到 {output_path}")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")

if __name__ == "__main__":
    process_excel_files()