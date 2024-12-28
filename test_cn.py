import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import jieba
import json
import os

def load_cn_data(base_path):
    # 加载训练、验证和测试数据
    train_data = []
    val_data = []
    test_data = []
    
    # 读取训练集
    with open(os.path.join(base_path, "train_zh.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    # 读取验证集
    with open(os.path.join(base_path, "val_zh.json"), "r", encoding="utf-8") as f:
        val_data = json.load(f)
        
    # 读取测试集
    with open(os.path.join(base_path, "test_zh.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    return train_data, val_data, test_data

def preprocess_text(text):
    # 使用jieba进行分词
    words = jieba.cut(text)
    # 将分词结果组合成字符串,用空格分隔
    return " ".join(words)

def train_naive_bayes():
    # 加载数据
    base_path = "D:/A大学各学期资料/大三上/信息内容安全/大作业/clickbait/data/clickbait_detection_dataset/data_cn"
    train_data, val_data, test_data = load_cn_data(base_path)
    
    # 预处理文本数据
    train_texts = [preprocess_text(item["text_zh"]) for item in train_data]
    train_labels = [item["label"] for item in train_data]
    
    val_texts = [preprocess_text(item["text_zh"]) for item in val_data]
    val_labels = [item["label"] for item in val_data]
    
    test_texts = [preprocess_text(item["text_zh"]) for item in test_data]
    test_labels = [item["label"] for item in test_data]

    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # 训练朴素贝叶斯模型
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, train_labels)

    # 在验证集上评估
    val_predictions = nb_classifier.predict(X_val)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print("\n验证集评估结果:")
    print(f"准确率: {val_accuracy:.4f}")
    print("\n详细评估报告:")
    print(classification_report(val_labels, val_predictions))

    # 在测试集上评估
    test_predictions = nb_classifier.predict(X_test)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print("\n测试集评估结果:")
    print(f"准确率: {test_accuracy:.4f}")
    print("\n详细评估报告:")
    print(classification_report(test_labels, test_predictions))

    # # 保存评估结果
    # evaluation_results = {
    #     "accuracy": float(test_accuracy),
    #     "classification_report": classification_report(test_labels, test_predictions, output_dict=True)
    # }
    
    # with open("朴素贝叶斯_evaluation.json", "w", encoding="utf-8") as f:
    #     json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

    # # 保存预测结果
    # test_predictions_list = []
    # for text, true_label, pred_label, prob in zip(
    #     test_texts, 
    #     test_labels,
    #     test_predictions, 
    #     nb_classifier.predict_proba(X_test)
    # ):
    #     test_predictions_list.append({
    #         "text": text,
    #         "true_label": int(true_label),
    #         "predicted_label": int(pred_label),
    #         "probability": float(max(prob))
    #     })

    # with open("朴素贝叶斯_predictions.json", "w", encoding="utf-8") as f:
    #     json.dump(test_predictions_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    train_naive_bayes()