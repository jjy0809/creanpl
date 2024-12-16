import sqlite3
import os

# DB 파일 경로
db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\test_labeled.db"
rec_idx_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\rec_idx.txt"

# 마지막 저장된 인덱스를 불러오는 함수
def load_last_index():
    if os.path.exists(rec_idx_path):
        with open(rec_idx_path, 'r') as f:
            return int(f.read().strip())
    return 0  # 파일이 없으면 처음부터 시작

# 현재 인덱스를 저장하는 함수
def save_current_index(index):
    with open(rec_idx_path, 'w') as f:
        f.write(str(index))

# DB 연결 및 데이터 로드
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 모든 데이터를 가져옴
cursor.execute("SELECT rowid, comment, result FROM labeled_data")
data = cursor.fetchall()

# 마지막 저장된 인덱스 로드
start_idx = load_last_index()

# 라벨링 루프
for i in range(start_idx, len(data)):
    rowid, comment, result = data[i]

    # 댓글과 기존 결과 출력
    print(f"\n[{i+1}/{len(data)}] 댓글: {comment}\n현재 라벨: {result}")
    new_label = input("라벨: ")

    # 입력 처리
    if new_label == "00":
        print("작업 종료.")
        save_current_index(i)  # 현재 인덱스 저장 후 종료
        exit()

    # 라벨 업데이트
    if new_label == "":
        # 기존 라벨 유지
        updated_label = result
    elif new_label == ".":
        # 기존 결과 반전
        updated_label = 1 if result == 0 else 0
    else:
        # 새로운 라벨로 업데이트
        updated_label = int(new_label)

    # DB 업데이트 (라벨이 변경된 경우에만)
    if updated_label != result:
        cursor.execute("UPDATE labeled_data SET result = ? WHERE rowid = ?", (updated_label, rowid))
        conn.commit()  # 실시간 저장

    # 진행 상황 저장
    save_current_index(i)

# 작업 완료 시 DB 및 파일 정리
conn.close()
print("모든 작업이 완료되었습니다.")
