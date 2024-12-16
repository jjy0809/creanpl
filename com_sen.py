import re
import hgtk
import boto3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from hgtk import letter
import os
import re

# S3 클라이언트 설정
s3 = boto3.client('s3')
bucket_name = 'creanpl'

# 전역 변수로 필터 단어 리스트 초기화
swear_words = []
demean_words = []
aggres_words = []
sexual_words = []
politic_words = []
ilbe_words = []
gender_words = []
super_high_freq_words = []

# 전역 모델 리스트
models = [None, None, None]  # 모델을 저장할 전역 리스트
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

# 한글 초성 리스트
chosung = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 전역 변수 체크 및 S3에서 데이터 불러오기
def load_word_lists():
    global swear_words, demean_words, aggres_words, sexual_words, politic_words, ilbe_words, gender_words, super_high_freq_words
    if not swear_words:
        swear_words = load_s3_file('filters/swear_word.txt')
    if not demean_words:
        demean_words = load_s3_file('filters/demean_word.txt')
    if not aggres_words:
        aggres_words = load_s3_file('filters/aggres_word.txt')
    if not sexual_words:
        sexual_words = load_s3_file('filters/sexual_word.txt')
    if not politic_words:
        politic_words = load_s3_file('filters/politic_word.txt')
    if not ilbe_words:
        ilbe_words = load_s3_file('filters/ilbe_word.txt')
    if not gender_words:
        gender_words = load_s3_file('filters/gender_word.txt')
    if not super_high_freq_words:
        super_high_freq_words = load_s3_file('filters/superHigh_dc_word.txt')

def load_s3_file(key):
    response = s3.get_object(Bucket=bucket_name, Key=key)
    content = response['Body'].read().decode('utf-8').splitlines()
    return content

# 불필요한 문자 제거
def clean_word(word):
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣]', '', word)

# 한글 초성 추출
def get_chosung(word):
    return ''.join([letter.decompose(char)[0] if is_hangul(char) else char for char in word])

# 한글 여부 확인
def is_hangul(char):
    return '\uac00' <= char <= '\ud7a3'

# 초성 제거
def remove_chosung(com):
    return ''.join([c for c in com if c not in chosung])

# 한글 분해 및 재구성
def decompose_hangul(word):
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ]', '', hgtk.text.decompose(word))

def compose_hangul(word):
    return hgtk.letter.compose(word[0], word[1])

# 댓글 검열
def com_cen(com, lst):
    load_word_lists()  # 전역 필터 단어 리스트 로드
    paths = [swear_words, demean_words, aggres_words, sexual_words, politic_words, ilbe_words, gender_words]
    
    score = 0
    com = clean_word(com)
    
    # 고빈도 단어 필터링 및 점수 계산
    for word_list in paths:
        for w in word_list:
            if w in com:
                score += 6
    
    for w in super_high_freq_words:
        if w in com:
            score += 10
            continue
        if w in remove_chosung(com):
            score += 4
            continue
        if get_chosung(w) in com:
            score += 4
            continue
        if w[::-1] in com:
            score += 3
            continue
        if decompose_hangul(w) in com:
            score += 3
            continue
    
    return score

# BERT 모델 로드 및 예측
def com_pred(model_idx, txt):
    if models[model_idx - 1] is None:
        model_path = f'models/model_{model_idx}'
        models[model_idx - 1] = BertForSequenceClassification.from_pretrained(model_path)
        models[model_idx - 1].eval()
    
    model = models[model_idx - 1]
    sentence = preprocess_text(txt)
    inputs = tokenizer(sentence, return_tensors='pt', max_length=1000, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return prediction

def preprocess_text(text):
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\s\.,?!\(\)]', '', text)
    return text.strip()

# Lambda 함수 핸들러
def lambda_handler(event, context):
    com = event['headers'].get('com', '')
    lst = list(map(int, event['headers'].get('lst', '').split(',')))
    
    score = censor(com, lst)
    return {
        'statusCode': 200,
        'body': {'score': score}
    }

def censor(com, lst):
    score = 0
    score += com_cen(com, lst[:-1])
    if lst[-1] != 0:
        score += com_pred(lst[-1], com) * lst[-1] * 2
    return score
