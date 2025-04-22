import re
import creanpl.hgtk as hgtk
from creanpl.hgtk import letter
import time
import json
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone, timedelta


# 단어 필터 리스트 선언
swear_words = []                      # 욕설
demean_words = []                   # 비하 단어
aggres_words = []                    # 부정적 단어
sexual_words = []                     # 성 단어
politic_words = []                      # 정치 단어
ilbe_words = []                          # 커뮤니티 단어
gender_words = []                    # 젠더(성 차별) 단어
super_high_freq_words = []   # 고빈도 욕설 (필수)

# 단어 리스트 불러오기 함수
def load_word_lists():
    global swear_words, demean_words, aggres_words, sexual_words, politic_words, ilbe_words, gender_words, super_high_freq_words
    if not swear_words:
        swear_words = load_file('creanpl/filters/swear_word.txt')
    if not demean_words:
        demean_words = load_file('creanpl/filters/demean_word.txt')
    if not aggres_words:
        aggres_words = load_file('creanpl/filters/aggres_word.txt')
    if not sexual_words:
        sexual_words = load_file('creanpl/filters/sexual_word.txt')
    if not politic_words:
        politic_words = load_file('creanpl/filters/politic_word.txt')
    if not ilbe_words:
        ilbe_words = load_file('creanpl/filters/ilbe_word.txt')
    if not gender_words:
        gender_words = load_file('creanpl/filters/gender_word.txt')
    if not super_high_freq_words:
        super_high_freq_words = load_file('creanpl/filters/superHigh_dc_word.txt')

# 파일 읽기 함수
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

# DynamoDB 테이블 세팅
DYNAMODB_TABLE_NAME = 'creanpl_api_logs'
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

# 텍스트 전처리 함수
def clean_word(word):
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣]', '', word)

# 한글 초성 리스트
chosung = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 한글 여부 반환 함수
def is_hangul(char):
    return '\uac00' <= char <= '\ud7a3'

# 초성 추출 함수
def get_chosung(word):
    return ''.join([letter.decompose(char)[0] if is_hangul(char) else char for char in word])

# 초성 제거 함수
def remove_chosung(com):
    return ''.join([c for c in com if c not in chosung])

# 한글 분해 함수
def decompose_hangul(word):
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ]', '', hgtk.text.decompose(word))

# 함글 합성 함수
def compose_hangul(word):
    return hgtk.letter.compose(word[0], word[1])


# 텍스트 점수 측정 함수
def com_cen(com, lst):
    load_word_lists()

    high_freq_words = []
    if lst[0] == 1:
        high_freq_words.extend(swear_words)
    if lst[1] == 1:
        high_freq_words.extend(demean_words)
    if lst[2] == 1:
        high_freq_words.extend(aggres_words)
    if lst[3] == 1:
        high_freq_words.extend(sexual_words)
    if lst[4] == 1:
        high_freq_words.extend(politic_words)
    if lst[5] == 1:
        high_freq_words.extend(ilbe_words)
    if lst[6] == 1:
        high_freq_words.extend(gender_words)

    score = 0
    com = clean_word(com)

    # 필터링 단어 포함 여부 검사
    for w in high_freq_words:
        if w in com:
            score += 6

    # 고빈도 단어 필터링 검사
    for w in super_high_freq_words:
        
        # 고빈도 단어 포함 여부 검사
        if w in com:
            score += 10
            continue
        
        # 초성 제거 후 단어 포함 검사
        if w in remove_chosung(com):
            score += 4
            continue
        
        # 초성화 단어 포함 여부 검사
        if get_chosung(w) in com:
            if len(w) == 1 and com[com.find(get_chosung(w)) + 1] not in chosung or len(w) > 1:
                score += 4
                continue

        # 역순 단어 포함 여부 검사
        if w[::-1] in com:
            score += 3
            continue

        # 풀어쓰기 단어 포함 여부 검사
        if decompose_hangul(w) in com:
            score += 3
            continue

        # 모음을 이용한 발음 장음화 단어 포험 여부 검사
        for t in chosung[30:]:
            tmp = com
            tmp = tmp.replace(compose_hangul(["ㅇ", t]), "")
            if tmp == w:
                score += 2
                break
        tmp = com
        for t in chosung[30:]:
            tmp = tmp.replace(compose_hangul(["ㅇ", t]), "")
            if tmp == w:
                score += 2
                break

        # 일부 초성화 단어 포함 여부 검사
        if len(w) > 1 and get_chosung(w) in get_chosung(com):
            j = get_chosung(com).find(get_chosung(w))
            for l in w:
                if l in com[j:j+len(w)] and len(decompose_hangul(w)) >= len(decompose_hangul(com[j:j+len(w)])) + 1 and (len(decompose_hangul(com[j])) == 1 or len(decompose_hangul(com[j+1])) == 1):
                    score += 1
                    break

    return score


# 람다함수 시작점
def lambda_handler(event, context):
    start_time = time.time()  # 요청 시작 시각
    
    # 기본값 선언
    error_message = ''
    status_code = 200
    score = None

    try:
        # 요청 변수 읽기
        body = event.get('body')
        
        if body:
            body_data = json.loads(body)  
            com = body_data.get('com', '')  # com 값 저장
            lst = body_data.get('lst', [0, 0, 0, 0, 0, 0, 0])  # lst 값 저장
        else:
            com = ''
            lst = [0, 0, 0, 0, 0, 0, 0]

        # lst 빈 공간 채우기
        if len(lst) < 7:
            lst.extend([0] * (7 - len(lst)))

        # 점수 계산
        score = com_cen(com, lst)
        
    except Exception as e: 
        error_message = str(e)
        status_code = 500

    response_time = int((time.time() - start_time) * 1000)  # 응답 소요 시간

    ip_address = event.get('headers', {}).get('X-Forwarded-For', '')  # 요청자 IP

    # 요청 시각 저장
    kst = timezone(timedelta(hours=9)) 
    timestamp = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] 

    # 로깅용 고유 ID 생성
    unique_id = int(time.time() * 1000)  

    # 로그 생성
    log_entry = {
        'Version': 2,
        'ID': unique_id,
        'time': timestamp,
        'com': com,
        'lst': lst,
        'status_code': status_code,
        'score': score,
        'ip_address': ip_address,
        'response_time_ms': response_time,
        'error_message': error_message
    }

    # DynamoDB에 로그 저장
    try:
        table.put_item(Item=log_entry)
    except ClientError as e:
        error_message += f" | DynamoDB Error: {str(e)}"
        status_code = 500 

    # 클라이언트 응답
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Origin': '*',  # 모든 도메인 허용
            'Access-Control-Allow-Methods': 'OPTIONS,POST',  # 허용할 HTTP 메서드
            'Access-Control-Allow-Headers': 'Content-Type',  # 허용할 요청 헤더
            'Access-Control-Max-Age': '3600'  # Preflight 요청 캐싱 시간 (초 단위)
        },
        'body': json.dumps({
            'score': score,
            'error_message': error_message
        })
    }