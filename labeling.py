import json
import time


# 감성 분석 결과 파일과 댓글 파일 경로
comments_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\comments.txt"
labels_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\label.txt"
sentiment_results_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\senti.txt"


# 댓글과 감성 분석 결과 불러오기
with open(comments_file, "r", encoding="utf-8") as f:
    coms = f.readlines()

with open(sentiment_results_file, "r", encoding="utf-8") as f:
    senti = f.readlines()

# 완료된 라벨링 불러오기
with open(labels_file, "r", encoding="utf-8") as f:
    labs = f.readlines()

s = len(labs)  # 이미 라벨링된 데이터 수
coms = coms[s:len(senti)]  # 아직 처리되지 않은 댓글들
senti = senti[s:]  # 아직 처리되지 않은 감성 분석 결과들

print(f"{s+1}번째 댓글부터 라벨링을 시작합니다")
time.sleep(0.5)


# 라벨링 진행
for i, (comment, sentiment_result) in enumerate(zip(coms, senti), start=s):
    # 첫 번째로 등장하는 'sentiment' 값을 추출
    sentiment_start = sentiment_result.find("'sentiment': '") + len("'sentiment': '")
    sentiment_end = sentiment_result.find("'", sentiment_start)
    document_sentiment = sentiment_result[sentiment_start:sentiment_end]

    print(f"{i+1}번째 댓글: {comment.strip()}")
    
    # 자동 라벨링 (negative -> 1, positive -> 0)
    if document_sentiment == "negative":
        label = "1"
    elif document_sentiment == "positive":
        label = "0"
    else:
        # neutral의 경우 수동 라벨링
        label = input("라벨을 입력하세요 (1: 욕설, 0: 정상): ")
        if label == "00":
            print("라벨링 작업을 중단합니다.")
            exit()

    # 라벨을 파일에 저장
    with open(labels_file, "a", encoding="utf-8") as f:
        f.write(label + "\n")
        
    print(label)
    # 진행 상황 출력
    print(f"진행 상황: {i + 1}/{len(coms) + s} ({(i + 1) / (len(coms) + s) * 100:.2f}%) 완료")


print("모든 댓글 라벨링 완료")







