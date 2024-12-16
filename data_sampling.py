import sqlite3
import random

# DB 파일 경로
news_db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\news_sensor_res.db"
dc_db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\dc_sensor_res.db"
train_db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\test_labeled.db" 

# 뉴스 DB 연결 및 데이터 추출
news_conn = sqlite3.connect(news_db_path)
dc_conn = sqlite3.connect(dc_db_path)

# 새로운 DB 생성
train_conn = sqlite3.connect(train_db_path)
train_cursor = train_conn.cursor()

# 새 테이블 생성
train_cursor.execute('''CREATE TABLE IF NOT EXISTS labeled_data (
    comment TEXT,
    result INTEGER
)''')


# 데이터 추출 함수
def fetch_random_data(conn, label, count):
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT DISTINCT comment, result 
        FROM comments 
        WHERE result = {label} 
        AND LENGTH(comment) BETWEEN 25 AND 95
    """)
    rows = cursor.fetchall()
    return random.sample(rows, count)

print("샘플링 중")

# 5000개씩 1, 7500개씩 0인 데이터 추출 (뉴스)
news_pos_data = fetch_random_data(news_conn, 1, 50000)
news_neg_data = fetch_random_data(news_conn, 0, 50000)

# 5000개씩 1, 7500개씩 0인 데이터 추출 (디시)
dc_pos_data = fetch_random_data(dc_conn, 1, 50000)
dc_neg_data = fetch_random_data(dc_conn, 0, 50000)

# 데이터 합치기
combined_data = news_pos_data + news_neg_data + dc_pos_data + dc_neg_data
random.shuffle(combined_data)

# 합친 데이터 새 DB에 삽입
train_cursor.executemany("INSERT INTO labeled_data (comment, result) VALUES (?, ?)", combined_data)

# 커밋 및 연결 종료
train_conn.commit()
train_conn.close()
news_conn.close()
dc_conn.close()
print("완료")