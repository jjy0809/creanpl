import re
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from collections import defaultdict

# 경로 설정
train_db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\train_labeled.db"

# 데이터 전처리 함수
def preprocess_text(text):
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\s\.,?!\(\)]', '', text)
    return text.strip()

# 데이터 로드 함수
def load_and_preprocess_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT comment, result FROM labeled_data")
    rows = cursor.fetchall()
    conn.close()
    comments = [preprocess_text(row[0]) for row in rows]
    labels = [row[1] for row in rows]
    comments = comments[:30000]
    labels = labels[:30000]
    return comments, labels

# Vocabulary 생성
def build_vocab(comments):
    vocab = defaultdict(lambda: len(vocab))
    vocab["<pad>"] = 0  # 패딩 토큰 추가
    for comment in comments:
        for char in comment:
            if char not in vocab:
                vocab[char]
    return vocab

# PyTorch Dataset 클래스
class CustomDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_indices = [self.vocab[char] for char in text]
        text_tensor = torch.tensor(
            text_indices + [self.vocab["<pad>"]] * (self.max_length - len(text_indices)),
            dtype=torch.long
        )
        return text_tensor, label

# 데이터 로더 생성 함수
def create_dataloaders(comments, labels, vocab, batch_size=16, max_length=100):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
        labels = torch.tensor(labels, dtype=torch.long)
        return texts, labels

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        comments, labels, test_size=0.1, random_state=42
    )
    train_dataset = CustomDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = CustomDataset(val_texts, val_labels, vocab, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader

# BiLSTM 모델 정의
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

# 모델 생성
def create_model(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
    model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
    return model

# 학습 함수
def train_model(train_loader, val_loader, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
    model = create_model(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"{device}으로 학습 진행")
    
    initial_learning_rate = 1e-5
    num_epochs = 5
    max_grad_norm = 1.5
    logging_steps = 10
    optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=True)
        for batch in train_iterator:
            texts, labels = batch
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            if logging_steps and (epoch * len(train_loader) + len(batch)) % logging_steps == 0:
                tqdm.write(f"[Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}]")

        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch + 1} - Average Loss: {avg_loss:.4f}')
        
        # 검증
        model.eval()
        val_loss, correct_predictions = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch
                texts = texts.to(device)
                labels = labels.to(device)
                outputs = model(texts)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
        
        val_accuracy = correct_predictions / len(val_loader.dataset)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
        model.train()

# 메인 실행
if __name__ == "__main__":
    comments, labels = load_and_preprocess_data(train_db_path)
    vocab = build_vocab(comments)
    train_loader, val_loader = create_dataloaders(comments, labels, vocab)
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 2
    n_layers = 2
    bidirectional = True
    dropout = 0.5
    train_model(train_loader, val_loader, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
