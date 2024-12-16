import re  # 정규 표현식 처리
import sqlite3  # SQLite 데이터베이스 연동
import torch  # PyTorch 라이브러리
from torch.utils.data import Dataset, DataLoader  # 데이터셋과 데이터 로더
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments  # BERT 모델 관련 모듈
import torch.nn.utils as nn_utils
import torch.optim as optim
from sklearn.model_selection import train_test_split  # 학습/검증 데이터 분리
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # 평가 지표 계산
from transformers import AdamW  # AdamW 옵티마이저 가져오기
from tqdm import tqdm  # 학습 진행 상황을 표시하기 위해 tqdm 사용
import os  # 파일 및 경로 작업

# 경로 설정
train_db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\train_labeled.db"  # 학습 데이터 DB 경로
model_save_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\model"  # 모델 저장 경로
test_data_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\test_labeled.db"  # 테스트 데이터 DB 경로


# 데이터 전처리 함수
def preprocess_text(text):
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\s\.,?!\(\)]', '', text)  # 한글, 숫자, 문장부호만 남기고 나머지 제거
    return text.strip()  # 앞뒤 공백 제거 후 반환

# SQLite DB에서 데이터 로드 및 전처리 함수
def load_and_preprocess_data(db_path):
    conn = sqlite3.connect(db_path)  # 데이터베이스 연결
    cursor = conn.cursor()  # 커서 객체 생성
    cursor.execute("SELECT comment, result FROM labeled_data")  # 댓글과 결과를 쿼리
    rows = cursor.fetchall()  # 모든 행을 가져오기
    conn.close()  # 데이터베이스 연결 종료

    comments = [preprocess_text(row[0]) for row in rows]  # 전처리된 댓글 리스트 생성
    labels = [row[1] for row in rows]  # 라벨 리스트 생성
    #comments = comments[620000:]
    #labels = labels[620000:]
    return comments, labels  # 댓글과 라벨 반환

# 30000
# 200000

# 220000
# 400000
# 700500

# BERT 모델의 한글 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')  # 사전 학습된 한글 BERT 토크나이저 로드

# PyTorch Dataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts  # 입력 텍스트 리스트
        self.labels = labels  # 라벨 리스트

    def __len__(self):
        return len(self.texts)  # 데이터셋의 길이 반환

    def __getitem__(self, idx):
        text = self.texts[idx]  # 인덱스에 해당하는 텍스트 가져오기
        label = self.labels[idx]  # 인덱스에 해당하는 라벨 가져오기
        encoding = tokenizer(  # 텍스트를 토큰화하고 인코딩
            text,
            max_length=50,  # 최대 길이 50으로 설정
            padding='max_length',  # 길이가 부족할 경우 패딩 추가
            truncation=True,  # 길이가 넘칠 경우 자르기
            return_tensors='pt'  # PyTorch 텐서로 반환
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),  # 인풋 아이디 텐서
            'attention_mask': encoding['attention_mask'].flatten(),  # 어텐션 마스크 텐서
            'labels': torch.tensor(label, dtype=torch.long)  # 라벨 텐서
        }


# 학습/검증 데이터 분리 및 데이터 로더 생성 함수
def create_dataloaders(comments, labels, batch_size=16):
    train_texts, val_texts, train_labels, val_labels = train_test_split(  # 학습과 검증 데이터 분리
        comments, labels, test_size=0.05, random_state=42)  # 10% 검증 데이터
    train_dataset = CustomDataset(train_texts, train_labels)  # 학습용 데이터셋 생성
    val_dataset = CustomDataset(val_texts, val_labels)  # 검증용 데이터셋 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 학습 데이터 로더 생성
    val_loader = DataLoader(val_dataset, batch_size=batch_size)  # 검증 데이터 로더 생성
    return train_loader, val_loader  # 데이터 로더 반환

# 모델 로드 또는 새 모델 생성 함수
def load_or_create_model(model_save_path):
    if os.path.exists(model_save_path):  # 모델 경로가 존재하는지 확인
        print("기존 모델을 로드합니다.")  # 모델 로드 메시지 출력
        model = BertForSequenceClassification.from_pretrained(model_save_path)  # 기존 모델 로드
    else:
        print("새로운 모델을 생성합니다.")  # 새 모델 생성 메시지 출력
        model = BertForSequenceClassification.from_pretrained('klue/bert-base', num_labels=2)  # 새 모델 생성
    return model  # 모델 반환

# 정확도 및 평가지표 계산 함수
def compute_metrics(pred):
    labels = pred.label_ids  # 실제 라벨 가져오기
    preds = pred.predictions.argmax(-1)  # 예측된 라벨 가져오기
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=1)  # 평가지표 계산
    acc = accuracy_score(labels, preds)  # 정확도 계산
    return {
        'accuracy': acc,  # 정확도 반환
        'f1': f1,  # F1 스코어 반환
        'precision': precision,  # 정밀도 반환
        'recall': recall  # 재현율 반환
    }


# 학습 함수 (초기 학습률 감소 및 진행 상황, 예상 시간, 그래디언트 놈 출력)
def train_model(train_loader, val_loader, model_save_path):
    # 기존 모델 로드 또는 새 모델 생성
    model = load_or_create_model(model_save_path) # 기존 모델 로드 또는 새 모델 생성
    m = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(m)
    print(f"{m}으로 학습 진행")

    # 하이퍼파라미터 및 설정값
    initial_learning_rate = 2e-5 * 0.5  # 초기 학습률
    num_epochs = 1  # 에포크 수
    max_grad_norm = 1.5  # 그래디언트 클리핑 최대값
    logging_steps = 10  # 로그 출력 빈도

    # 옵티마이저 정의
    optimizer = AdamW(model.parameters(), lr=initial_learning_rate)

    # 모델 학습 모드 전환
    model.train()

    # 학습 루프
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0

        # tqdm을 사용하여 진행 상황 표시
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=True)
        
        for batch in train_iterator:
            global_step += 1  # 스텝 증가

            batch = {k: v.to(model.device) for k, v in batch.items()}

            optimizer.zero_grad()  # 옵티마이저 그래디언트 초기화

            # 모델 예측 및 손실 계산
            outputs = model(input_ids=batch['input_ids'], 
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'])
            
            loss = outputs.loss  # 손실 값 추출
            loss.backward()  # 역전파 수행

            # 그래디언트 클리핑 적용 및 그래디언트 놈 계산
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()  # 옵티마이저로 파라미터 업데이트

            total_loss += loss.item()  # 손실 누적

            # 10 스텝마다 로그 출력 (진행 상황, 손실, 그래디언트 놈)
            if global_step % logging_steps == 0:
                # tqdm 밖에서 로그 출력
                tqdm.write(f"[Step {global_step}: Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}]")


        # 에포크 끝날 때마다 평균 손실 출력
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch + 1}/{num_epochs} finished, Average Loss: {avg_loss:.4f}')
        
        # 검증 모드로 전환 후 검증 수행
        model.eval()
        val_loss, val_accuracy = 0, 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}

                # 검증 데이터로 모델 출력 계산
                outputs = model(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'], 
                                labels=batch['labels'])
                
                val_loss += outputs.loss.item()  # 검증 손실 계산
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == batch['labels']).sum().item()

        # 검증 결과 로그 출력
        val_accuracy = correct_predictions / len(val_loader.dataset)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

        model.train()

    # 모델 저장
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)



# 테스트 데이터 로드 및 예측 수행 함수
def predict(model_path, test_data, label):
    model = BertForSequenceClassification.from_pretrained(model_path)  # 저장된 모델 로드
    model.eval()  # 평가 모드 전환

    accuracy = []  # 정확도 리스트
    for i, sentence in enumerate(test_data):  # 테스트 데이터 순회
        sentence = preprocess_text(sentence)  # 텍스트 전처리
        inputs = tokenizer(
            sentence,
            return_tensors='pt',  # 텐서로 반환
            max_length=50,  # 최대 길이
            padding='max_length',  # 패딩 추가
            truncation=True  # 자르기
        )
        with torch.no_grad():  # 그라디언트 계산 비활성화
            outputs = model(**inputs)  # 모델 예측
            prediction = torch.argmax(outputs.logits, dim=1).item()  # 예측된 라벨
            if int(prediction) == int(label[i]):  # 예측이 맞으면
                accuracy.append(1)  # 정확도 리스트에 1 추가
            else:  # 예측이 틀리면
                accuracy.append(0)  # 정확도 리스트에 0 추가
                #print(f"{i+1}번 문장: {sentence} -> 예측 결과: {prediction}")  # 틀린 예측 출력

    print(f"\n정확도: {sum(accuracy)/len(accuracy)*100}%")  # 최종 정확도 출력



#################################

models = [
    r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\model_1", #79.59%
    r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\model_2", #86.72%
    r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\model_3",
    r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\model_4",
    r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\model_5"] #95.31%

def com_pred(model, txt):
    model_path = models[model-1]
    model = BertForSequenceClassification.from_pretrained(model_path)  # 저장된 모델 로드
    model.eval()  # 평가 모드 전환

    sentence = preprocess_text(txt)  # 텍스트 전처리
    inputs = tokenizer(
        sentence,
        return_tensors='pt',  # 텐서로 반환
        max_length=1000,  # 최대 길이
        truncation=True  # 자르기
    )
    with torch.no_grad():  # 그라디언트 계산 비활성화
        outputs = model(**inputs)  # 모델 예측
        prediction = torch.argmax(outputs.logits, dim=1).item()  # 예측된 라벨
        return prediction

################################


# 사용자 입력에 따라 학습 또는 예측 모드 선택
if __name__ == "__main__":
    mode = int(input("0: 학습, 1: 테스트 선택 -> "))  # 사용자 입력에 따라 모드 선택

    if mode == 0:
        comments, labels = load_and_preprocess_data(train_db_path)  # 학습 데이터 로드 및 전처리
        train_loader, val_loader = create_dataloaders(comments, labels)  # 데이터 로더 생성
        train_model(train_loader, val_loader, model_save_path=model_save_path)  # 모델 학습

    elif mode == 1:
        comments, labels = load_and_preprocess_data(test_data_path)  # 테스트 데이터 로드 및 전처리
        comments = comments[:500]  # 테스트 데이터 선택
        labels = labels[:500]  # 라벨 선택
        predict(model_path=model_save_path, test_data=comments, label=labels)  # 테스트 데이터 예측 수행
    else:
        while 1:
            ans = input("댓글 입력: ")
            if ans == "0": break
            print(com_pred(5, ans))

