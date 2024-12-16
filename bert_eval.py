import re
import sqlite3
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import os
import random



test_data_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\test_labeled.db"  
model_paths = [f"C:\\Users\\happy\\Desktop\\학교\\고등학교\\2학년\\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\\model_{i}" for i in range(1, 6)]


def preprocess_text(text):
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\s\.,?!\(\)]', '', text) 
    return text.strip()  


def load_and_preprocess_data(db_path, limit):
    conn = sqlite3.connect(db_path)  
    cursor = conn.cursor()
    cursor.execute("SELECT comment, result FROM labeled_data")  
    rows = cursor.fetchall()  
    conn.close() 

    sampled_rows = random.sample(rows, limit)

    comments = [preprocess_text(row[0]) for row in sampled_rows]
    labels = [row[1] for row in sampled_rows]
    return comments, labels  


def predict_and_evaluate(model_path, test_data, true_labels, device, model_num):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)  
    model = model.to(device) 
    model.eval()  

    correct_predictions = 0

    for i, (sentence, true_label) in tqdm(enumerate(zip(test_data, true_labels)), total=len(test_data), desc=f"model_{model_num}"):
        sentence = preprocess_text(sentence) 
        inputs = tokenizer(
            sentence,
            return_tensors='pt',  
            max_length=95,  
            truncation=True  
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad(): 
            outputs = model(**inputs) 
            prediction = torch.argmax(outputs.logits, dim=1).item()  
            if int(prediction) == int(true_label):  
                correct_predictions += 1

    accuracy = correct_predictions / len(test_data) * 100
    return accuracy

if __name__ == "__main__":
    comments, labels = load_and_preprocess_data(test_data_path, limit=75000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_accuracies = []

    for i, model_path in enumerate(model_paths, 1):
        accuracy= predict_and_evaluate(model_path, comments, labels, device, i)
        model_accuracies.append((model_path, accuracy))

    model_accuracies.sort(key=lambda x: x[1], reverse=True)

    print("\n\n모델별 정확도 (높은 순):")
    for model_path, accuracy in model_accuracies:
        print(f"{model_path[-7:]}: {accuracy:.2f}%")


