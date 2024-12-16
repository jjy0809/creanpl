from tqdm import tqdm

# 댓글 파일 읽기
with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\naverNews_comments.txt", "r", encoding="utf-8", errors='ignore') as f:
    coms = f.read()

# 줄 단위로 댓글 분리
coms = list(coms.splitlines())

# 집합을 사용하여 중복 제거
set_coms = set()
comments = []
for c in tqdm(coms, desc="중복 댓글 제거 진행 중", unit="댓글"):
    if c not in set_coms:
        set_coms.add(c)
        comments.append(c)

# 결과를 새로운 파일에 저장
with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\naverNews_comments.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(comments))  # 리스트를 줄바꿈 문자로 구분하여 저장

print(f"전체 댓글 수: {len(comments)}")
print(f"중복 댓글 수: {len(coms)-len(comments)}")
