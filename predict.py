import sqlite3
import sentence_sensor
from tqdm import tqdm

# DB 파일 경로
train_db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\test_labeled.db"

# SQLite 데이터베이스 연결
conn = sqlite3.connect(train_db_path)
cursor = conn.cursor()

# 라벨이 0인 댓글만 선택
cursor.execute("SELECT comment FROM labeled_data WHERE result = 0")
comments_to_relabel = cursor.fetchall()

# 총 댓글 개수 계산
total_comments = len(comments_to_relabel)

sum = 0

# tqdm으로 진행 상황 표시
with tqdm(total=total_comments, desc="Processing comments") as pbar:
        for comment_row in comments_to_relabel:
                comment = comment_row[0]  # 댓글 가져오기
                
                # sentence_sensor를 사용하여 댓글 재분석
                result = sentence_sensor.sentence_censor(comment)[0]
                
                # 라벨이 1로 예측되면 결과 업데이트
                if result == 1:
                        cursor.execute("UPDATE labeled_data SET result = ? WHERE comment = ?", (result, comment))
                        sum += 1
                
                pbar.update(1)  # 진행 상황 업데이트
                conn.commit()  # 변경 사항 커밋

# 처리 완료 후 연결 종료

print(f"데이터베이스에 저장되었습니다.")
print(f"새로 1로 라벨링된 댓글 수: {sum}개")

conn.close()
