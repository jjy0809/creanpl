import json
import os

# 파일 경로 설정
high_freq_txt_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\high_freq_dc_word.txt"
high_freq_json_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\high_freq_dc_word.json"
super_high_freq_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\superHigh_dc_word.txt"
rec_idx_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\rec_idx.txt"
swear_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\swear_word.txt"
demean_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\demean_word.txt"
aggres_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\aggres_word.txt"
sexual_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\sexual_word.txt"
politic_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\politic_word.txt"
ilbe_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\ilbe_word.txt"
gender_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\gender_word.txt"


def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [w.strip() for w in f.readlines()]

def loat_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_word(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(data))  
        
def save_index(index):
    with open(rec_idx_path, 'w', encoding='utf-8') as f:
        f.write(str(index))
        
def load_index():
    with open(rec_idx_path, 'r') as f:
        return int(f.read().strip())

#words = load_txt(high_freq_txt_path)
words = loat_json(high_freq_json_path)
super_high_freq_words = load_txt(super_high_freq_path)
swear_word = load_txt(swear_path)
demean_word = load_txt(demean_path)
aggres_word = load_txt(aggres_path)
sexual_word = load_txt(sexual_path)
politic_word = load_txt(politic_path)
ilbe_word = load_txt(ilbe_path)
gender_word = load_txt(gender_path)

def add_to_txt(user_input, word, index):
    if user_input == '1':
        if word not in swear_word:
            swear_word.append(word)
            save_word(swear_path, swear_word)
            print(f"단어 '{word}'이 욕설 단어 목록에 추가되었습니다.")
        else: print("이미 존재하는 단어")
        index += 1
            
    elif user_input == '2':
        if word not in demean_word:
            demean_word.append(word)
            save_word(demean_path, demean_word)
            print(f"단어 '{word}'이 비하 단어 목록에 추가되었습니다.")
        else: print("이미 존재하는 단어")
        index += 1
        
    elif user_input == '3':
        if word not in aggres_word:
            aggres_word.append(word)
            save_word(aggres_path, aggres_word)
            print(f"단어 '{word}'이 공격적 단어 목록에 추가되었습니다.")
        else: print("이미 존재하는 단어")
        index += 1
        
    elif user_input == '4':
        if word not in sexual_word:
            sexual_word.append(word)  
            save_word(sexual_path, sexual_word)
            print(f"단어 '{word}'이 성 단어 목록에 추가되었습니다.")
        else: print("이미 존재하는 단어")
        index += 1
        
    elif user_input == '5':
        if word not in politic_word:
            politic_word.append(word)
            save_word(politic_path, politic_word)
            print(f"단어 '{word}'이 정치 단어 목록에 추가되었습니다.")
        else: print("이미 존재하는 단어")
        index += 1
        
    elif user_input == '6':
        if word not in ilbe_word:
            ilbe_word.append(word)
            save_word(ilbe_path, ilbe_word)
            print(f"단어 '{word}'이 일베/디시 단어 목록에 추가되었습니다.")
        else: print("이미 존재하는 단어")
        index += 1
        
    elif user_input == '7':
        if word not in gender_word:
            gender_word.append(word)
            save_word(gender_path, gender_word)
            print(f"단어 '{word}'이 젠더 단어 목록에 추가되었습니다.")
        else: print("이미 존재하는 단어")
        index += 1
    return index

def review_words():
    global words
    index = load_index()
    words = list(words.keys())
    save_word(high_freq_txt_path, words)
    print(f"전체 단어 수: {len(words)}\n\n")
    while index < len(words):
        word = words[index]
        print(f"{index}번째 단어: {word}")
            
        user_input = input("0: 건너뛰기, 1: 욕설, 2: 비하 단어, 3: 공격 단어, 4: 성 단어, 5: 정치 단어, 6: 일베/디시 단어, 7: 젠더 단어, 8: 이전 인덱스\n-> ")
    
            
        if user_input == "00": break
        elif user_input == "0": index += 1
        elif user_input == "8": index -= 1
        else: index = add_to_txt(user_input, word, index)
        
        print(f"{((index+1)/len(words)*100):.3f}% 완료\n")
        save_index(index)
        
def add_word():
    ans = input("추가 할 단어 카테고리(1:욕설, 2:비하, 3:공격, 4:성, 5:정치, 6:일베, 7:젠더): ")
    if ans == "00": exit()
    
    # 카테고리 선택에 따라 해당 리스트 참조
    if ans == '1': cate_words = swear_word
    elif ans == '2': cate_words = demean_word
    elif ans == '3': cate_words = aggres_word
    elif ans == '4': cate_words = sexual_word
    elif ans == '5': cate_words = politic_word
    elif ans == '6': cate_words = ilbe_word
    elif ans == '7': cate_words = gender_word
    
    while True:
        word = input("추가 할 단어: ")
        if word == "00": break
        if word not in cate_words:
            add_to_txt(ans, word, 1)
            print(f"추가 완료")
        else:
            print("이미 존재하는 단어")


if __name__ == "__main__":
    ans = input("단어 리뷰: 1, 단어 추가: 2 -> ")
    if ans == "1": review_words()
    else: add_word()
