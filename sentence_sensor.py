import re
import hgtk
from hgtk import letter

# 한글 초성 리스트
chosung =  ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 불필요한 문자 제거
def clean_word(word):
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣]', '', word)

# 한글 여부 확인
def is_hangul(char):
    return '\uac00' <= char <= '\ud7a3'

# 초성 추출
def get_chosung(word):
    return ''.join([letter.decompose(char)[0] if is_hangul(char) else char for char in word])

# 초성 제거
def remove_chosung(com):
    return ''.join([c for c in com if c not in chosung])

# 한글 분해
def decompose_hangul(word):
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ]', '', hgtk.text.decompose(word))

def compose_hangul(word):
    return hgtk.letter.compose(word[0], word[1])



# 욕설 검열 함수
def sentence_censor(com):
    # 단어 리스트 파일 경로 설정
    super_high_freq_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\superHigh_dc_word.txt"
    
    swear_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\swear_word.txt"
    demean_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\demean_word.txt"
    aggres_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\aggres_word.txt"
    sexual_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\sexual_word.txt"
    politic_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\politic_word.txt"
    ilbe_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\ilbe_word.txt"
    gender_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\gender_word.txt"

    # 파일에서 단어 리스트 불러오기
    paths = [swear_path, demean_path, aggres_path, sexual_path, politic_path, ilbe_path, gender_path]
    high_freq_words = []

    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            high_freq_words.extend([word.strip() for word in f.readlines()])

    # super_high_freq_words는 기존과 동일하게 처리
    super_high_freq_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\superHigh_dc_word.txt"
    
    with open(super_high_freq_path, 'r', encoding='utf-8') as f:
        super_freq_words = [word.strip() for word in f.readlines()]

    res = 0
    reason = []
    score = 0
    
    com = clean_word(com)
    
    for w in high_freq_words:
        if w in com:
            res = 1
            reason.append(f"부정 단어: {w}")
            score += 6
    
    for w in super_freq_words:
        if w in com:
            res = 1
            reason.append(f"부정 단어: {w}")
            score += 10
            continue

        if w in remove_chosung(com):
            res = 1
            score += 4
            reason.append(f"초성으로 검열 회피 부정 단어: {w}")
            continue
        
        if get_chosung(w) in com:
            if len(w) == 1 and com[com.find(get_chosung(w))+1] not in chosung or len(w) > 1:
                res = 1
                reason.append(f"초성 부정 단어: {w}")
                score += 4
                continue
        
        if w[::-1] in com:
            res = 1
            reason.append(f"역순 부정 단어: {w}")
            score += 3
            continue
        
        if decompose_hangul(w) in com:
            res = 1
            score += 3
            reason.append(f"풀어쓰기로 검열 회피 부정 단어: {w}")
            continue
            
        for t in chosung[30:]:
            tmp = com
            tmp = tmp.replace(compose_hangul(["ㅇ",  t]), "")
            if tmp == w:
                res = 1
                reason.append(f"모음으로 발음 장음화 회피 부정 단어: {w}")
                score += 2
                break
        tmp = com
        for t in chosung[30:]:
            tmp = tmp.replace(compose_hangul(["ㅇ",  t]), "")
            if tmp == w:
                res = 1
                reason.append(f"모음으로 발음 장음화 회피 부정 단어: {w}")
                score += 2
                break
                        
        if len(w) > 1 and get_chosung(w) in get_chosung(com):
            j = get_chosung(com).find(get_chosung(w))
            for l in w:
                if l in com[j:j+len(w)] and len(decompose_hangul(w)) >= len(decompose_hangul(com[j:j+len(w)])) + 1 and (len(decompose_hangul(com[j])) == 1 or len(decompose_hangul(com[j+1])) == 1):
                    res = 1
                    reason.append(f"일부 초성 검열 회피 부정 단어: {w}")
                    score += 1
                    break
        
    return [res, reason, score]


def com_cen(com, lst):
    # 단어 리스트 파일 경로 설정
    super_high_freq_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\superHigh_dc_word.txt"
    
    swear_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\swear_word.txt"
    demean_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\demean_word.txt"
    aggres_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\aggres_word.txt"
    sexual_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\sexual_word.txt"
    politic_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\politic_word.txt"
    ilbe_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\ilbe_word.txt"
    gender_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\gender_word.txt"

    # 파일에서 단어 리스트 불러오기
    paths = [swear_path, demean_path, aggres_path, sexual_path, politic_path, ilbe_path, gender_path]
    high_freq_words = []

    for i, path in enumerate(paths):
        if lst[i] == 1:
            with open(path, 'r', encoding='utf-8') as f:
                high_freq_words.extend([word.strip() for word in f.readlines()])

    # super_high_freq_words는 기존과 동일하게 처리
    super_high_freq_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\superHigh_dc_word.txt"
    
    with open(super_high_freq_path, 'r', encoding='utf-8') as f:
        super_freq_words = [word.strip() for word in f.readlines()]

    res = 0
    score = 0
    
    com = clean_word(com)
    
    for w in high_freq_words:
        if w in com:
            score += 6
    
    for w in super_freq_words:
        if w in com:
            score += 10
            continue

        if w in remove_chosung(com):
            score += 4
            continue
        
        if get_chosung(w) in com:
            if len(w) == 1 and com[com.find(get_chosung(w))+1] not in chosung or len(w) > 1:
                score += 4
                continue
        
        if w[::-1] in com:
            score += 3
            continue
        
        if decompose_hangul(w) in com:
            score += 3
            continue
            
        for t in chosung[30:]:
            tmp = com
            tmp = tmp.replace(compose_hangul(["ㅇ",  t]), "")
            if tmp == w:
                score += 2
                break
        tmp = com
        for t in chosung[30:]:
            tmp = tmp.replace(compose_hangul(["ㅇ",  t]), "")
            if tmp == w:
                score += 2
                break
                        
        if len(w) > 1 and get_chosung(w) in get_chosung(com):
            j = get_chosung(com).find(get_chosung(w))
            for l in w:
                if l in com[j:j+len(w)] and len(decompose_hangul(w)) >= len(decompose_hangul(com[j:j+len(w)])) + 1 and (len(decompose_hangul(com[j])) == 1 or len(decompose_hangul(com[j+1])) == 1):
                    score += 1
                    break
        
    return score

if __name__ == '__main__':
    ans = ''
    while 1:
        ans = input("입력: ")
        if ans == '00': exit()
        print(sentence_censor(ans))
        