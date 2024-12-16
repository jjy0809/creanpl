import sqlite3
from konlpy.tag import Okt
import time
import sys


# SQLite 연결
conn = sqlite3.connect(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\news_freq_word.db")
cursor = conn.cursor()

# 단어 빈도수 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS word_counts (
    word TEXT PRIMARY KEY,
    count INTEGER
)
''')

# 인덱스 저장 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY,
    last_index INTEGER
)
''')

# 마지막 인덱스 읽기
cursor.execute('SELECT last_index FROM progress WHERE id=1')
row = cursor.fetchone()
start = row[0] if row else 0

with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\naverNews_comments.txt", "r", encoding="utf-8") as f:
    coms = f.read()

comments = list(coms.splitlines())

okt = Okt()

# 불용어 리스트
stopwords = set([
    '은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한', '임', '거', '말', '더', '때', '그', '좀', '함', '안', '저', '니', '또', '남', '일', '걸', '게', '내'
])

print(f"전체 댓글 수: {len(comments)}")
print(f"{start+1}번째 댓글부터 시작\n\n")



for idx in range(start, len(comments)):
    comment = comments[idx]
    # Okt 명사 추출
    words = okt.nouns(comment)
    # 불용어 제거
    filtered_words = [word for word in words if word not in stopwords]

    # 빈도수 계산 및 데이터베이스 업데이트
    for word in filtered_words:
        cursor.execute('SELECT count FROM word_counts WHERE word=?', (word,))
        row = cursor.fetchone()
        if row:
            cursor.execute('UPDATE word_counts SET count=? WHERE word=?', (row[0] + 1, word))
        else:
            cursor.execute('INSERT INTO word_counts (word, count) VALUES (?, ?)', (word, 1))
    
    # 작업 커밋
    conn.commit()
    # 현재 인덱스 기록
    cursor.execute('INSERT OR REPLACE INTO progress (id, last_index) VALUES (1, ?)', (idx + 1,))
    conn.commit()
        
    sys.stdout.write("\033[F")  # 한 줄 위로 커서 이동
    sys.stdout.write("\033[K")  # 현재 줄을 지움
    sys.stdout.write(f"{idx+1}: {((idx+1)/len(comments)*100):.3f}% 완료 \n")
    sys.stdout.flush()  # 출력 버퍼를 비움
    
    
# 종료
print("\n모든 댓글 처리 완료")
conn.close()
