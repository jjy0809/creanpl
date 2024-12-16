from konlpy.tag import Okt  
import json
import time

with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\new_comments.txt", "r", encoding="utf-8") as f:
    coms = f.read()

try:
    with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\rec_idx.txt", "r", encoding='utf-8') as rf:
        start = int(rf.read().strip())
except:
    start = 0 

# 중복 댓글 제거
comments = list(coms.splitlines())


okt = Okt()

# 불용어 리스트 
stopwords = set([
    '은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한', '임', '거', '말', '더', '때', '그', '좀', '함', '안', '저', '니', '또', '남', '일', '걸', '게', '내'
])


word_counts = {}
try:
    with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\freq_word.json", "r", encoding='utf-8') as f:
        word_counts = json.load(f)
except:
    pass

print(f"전체 댓글 수: {len(comments)}")
print(f"중복 댓글 수: {len(coms)-len(comments)}\n")

print(f"{start+1}번째 댓글부터 시작\n")

time.sleep(1)

for idx in range(start, len(comments)):
    comment = comments[idx]
    
    print(f"{idx+1}: {comment}")
    # Okt 명사 추출
    words = okt.nouns(comment)
    print(f"{len(words)}개의 명사 추출 완료")

    # 불용어 제거
    filtered_words = [word for word in words if word not in stopwords]

    print(filtered_words)
    
    # 빈도수 계산
    for word in filtered_words:
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1
    
    

    with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\freq_word.json", "w", encoding='utf-8') as f:
        json.dump(word_counts, f, ensure_ascii=False, indent=4)


    with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\rec_idx.txt", "w", encoding='utf-8') as f:
        f.write(str(idx + 1))
        
    print(f"\n{(idx+1)/len(comments)*100}% 완료\n")
    
    if (idx + 1) % 750 == 0:
        print("\n---------- 휴식 중 ----------\n")
        time.sleep(1.5)
    

print("\n모든 댓글 처리 완료")
