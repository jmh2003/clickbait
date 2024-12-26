import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import glob, os, json
import torch
from tqdm import tqdm
import gc

def load_training_data():
    base_path = "data/clickbait_detection_dataset"
    train_data = pd.read_json(os.path.join(base_path, "train.json"))
    test_data = pd.read_json(os.path.join(base_path, "test.json"))
    val_data = pd.read_json(os.path.join(base_path, "val.json"))
    df = pd.concat([train_data, test_data, val_data], ignore_index=True)
    return df

def load_news_data():
    news_path = "news"
    news_files = glob.glob(os.path.join(news_path, "*_translated.json"))
    all_news = []
    
    for file in news_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            source = os.path.basename(file).replace('_translated.json', '')
            for item in data:
                item['source'] = source
            all_news.extend(data)
    return pd.DataFrame(all_news)

def train_and_predict_single_model(model_name, model_class, X_train, X_test, y_train, y_test, vectorizer, news_df, batch_size=1000):
    print(f"\n=== 开始训练 {model_name} 模型 ===")
    
    try:
        # 针对SVC特殊处理
        if model_name == "支持向量机":
            model = SVC(kernel='linear', probability=True, random_state=42)
        else:
            model = model_class()
        
        # 确保数据格式正确
        X_train = X_train.tocsr()
        X_test = X_test.tocsr()
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # 添加训练进度条
        print("开始训练...")
        n_samples = X_train.shape[0]
        batch_count = (n_samples + batch_size - 1) // batch_size
        
        if hasattr(model, "partial_fit"):
            with tqdm(total=n_samples, desc=f"{model_name} 训练进度") as pbar:
                classes = np.unique(y_train)
                for i in range(0, n_samples, batch_size):
                    end_idx = min(i + batch_size, n_samples)
                    batch_X = X_train[i:end_idx]
                    batch_y = y_train[i:end_idx]
                    model.partial_fit(batch_X, batch_y, classes=classes)
                    pbar.update(end_idx - i)
        else:
            print(f"模型不支持增量训练，使用全量训练...")
            model.fit(X_train, y_train)
        
        # 评估模型
        print("\n评估模型...")
        y_pred = model.predict(X_test)
        print(f"{model_name} 准确率:", accuracy_score(y_test, y_pred))
        print(f"{model_name} 分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 预测新闻标题
        results = []
        print("\n开始预测新闻标题...")
        
        for i in tqdm(range(0, len(news_df), batch_size), desc=f"{model_name} 预测进度"):
            batch_df = news_df.iloc[i:i + batch_size]
            titles = batch_df['translated_title'].fillna('').tolist()
            title_vectors = vectorizer.transform(titles)
            
            predictions = model.predict(title_vectors)
            
            # 处理概率预测
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(title_vectors)
                prob_values = [float(p[1]) for p in probabilities]
            else:
                # 对于不支持概率预测的模型，使用决策函数值
                decision_values = model.decision_function(title_vectors)
                prob_values = [float((v + 1) / 2) for v in decision_values]
            
            for j, (title, orig_title, source) in enumerate(zip(
                batch_df['translated_title'],
                batch_df['original_title'],
                batch_df['source']
            )):
                if pd.notna(title):
                    results.append({
                        "title": str(title),
                        "original_title": str(orig_title),
                        "source": str(source),
                        "prediction": int(predictions[j]),
                        "probability": prob_values[j]
                    })
        
        # 保存结果
        output_path = f"{model_name}_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"\n{model_name} 预测结果已保存至 {output_path}")
        
    except Exception as e:
        print(f"{model_name} 处理失败: {str(e)}")
        raise
    finally:
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # 加载数据
    print("加载训练数据...")
    train_df = load_training_data()
    print("加载新闻数据...")
    news_df = load_news_data()
    
    # 数据预处理
    print("数据向量化...")
    vectorizer = CountVectorizer(max_features=10000)  # 限制特征数量
    X = vectorizer.fit_transform(train_df["text"])
    y = train_df["label"]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 逐个训练和预测模型
    models = [
        ("朴素贝叶斯", MultinomialNB),
        ("随机森林", RandomForestClassifier),
        ("逻辑回归", LogisticRegression),
        ("支持向量机", SVC)
    ]
    
    for model_name, model_class in models:
        try:
            train_and_predict_single_model(
                model_name, 
                model_class, 
                X_train, 
                X_test, 
                y_train, 
                y_test,
                vectorizer, 
                news_df
            )
        except Exception as e:
            print(f"{model_name} 处理失败: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()