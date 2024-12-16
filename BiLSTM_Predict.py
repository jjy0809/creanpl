import os
import numpy as np
import tensorflow as tf

# BiLSTM 모델 정의 (로드하기 위해 다시 정의)
@tf.keras.utils.register_keras_serializable()  # 커스텀 모델 클래스 등록
# 모델 복잡성을 증가시키는 예시
class BiLSTMModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, input_length):
        super(BiLSTMModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length)
        self.lstm_fw = tf.keras.layers.LSTM(128, return_sequences=True)  # 유닛 수 증가
        self.lstm_bw = tf.keras.layers.LSTM(128, return_sequences=True, go_backwards=True)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')  # 유닛 수 증가
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x_fw = self.lstm_fw(x)
        x_bw = self.lstm_bw(x)
        x = tf.concat([x_fw, x_bw], axis=-1)
        x = tf.reduce_mean(x, axis=1)
        x = self.dense1(x)
        return self.dense2(x)
# 모델 로드 경로 설정
load_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\model\bilstm_model.keras"

# 모델 로드
try:
    model = tf.keras.models.load_model(load_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 테스트 데이터 경로 설정
test_data_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\test_comments.txt"

# 테스트 텍스트 데이터 로드
with open(test_data_path, "r", encoding='utf-8') as f:
    test_coms = f.readlines()

# 테스트 데이터 전처리
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(test_coms)
sequences = tokenizer.texts_to_sequences(test_coms)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# 예측 수행
predictions = model.predict(padded_sequences)

# 예측 결과 출력
for i, pred in enumerate(predictions):
    print(f"Text: {test_coms[i].strip()}")
    print(f"Prediction: {pred[0]:.4f} (0: 정상, 1: 욕설/혐오 표현)\n")
