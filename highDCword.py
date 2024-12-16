import sqlite3
import json

# 뉴스 댓글 DB와 디시인사이드 DB 경로
news_db_path = r'C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\news_freq_word.db'
dc_db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\freq_word.db"

# 뉴스 댓글과 디시인사이드 DB에서 단어 빈도를 가져오는 함수
def fetch_word_frequencies(db_path):
    # 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # word_counts 테이블에서 단어와 빈도를 가져옴
    cursor.execute("SELECT word, count FROM word_counts")
    word_freq = cursor.fetchall()
    
    conn.close()
    
    # 딕셔너리로 변환
    return {word: count for word, count in word_freq}

print("단어 빈도 리스트 불러오는 중")
# 두 데이터베이스에서 단어 빈도 데이터 가져오기
news_word_freq = fetch_word_frequencies(news_db_path)
dc_word_freq = fetch_word_frequencies(dc_db_path)

print("뉴스에 비해 디시에서 빈도 수 높은 단어 추출하는 중")
# 뉴스 댓글에 비해 디시인사이드에서 많이 사용된 단어만 추출하는 코드 (비교 기준)
relative_words = {
    word: count for word, count in dc_word_freq.items()
    if word not in news_word_freq or count > news_word_freq[word] * 30  # 두 배 이상 차이가 나는 단어
    if count > 30  # 디시인사이드에서 30번 이상 사용된 단어만 추가
}

# 30  27
#400 35

# 상대적으로 많이 사용된 단어들을 JSON 파일로 저장
output_json_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\high_freq_dc_word.json"
#output_json_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\superHigh_dc_word.json"
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(relative_words, json_file, ensure_ascii=False, indent=4)

print("작업 완료")
