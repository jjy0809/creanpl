import sqlite3

# 데이터베이스에서 빈도수가 50개 이상인 단어를 추출하여 파일로 저장
def extract_high_frequency_words(db_path, output_file, min_count=50):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 빈도수가 50개 이상인 단어 추출
    cursor.execute("SELECT word, count FROM word_counts WHERE count >= ? ORDER BY count DESC", (min_count,))
    high_freq_words = cursor.fetchall()

    # 추출한 단어를 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in high_freq_words:
            f.write(f"{word}: {count}\n")

    conn.close()
    print(f"빈도수 {min_count}개 이상의 단어를 {output_file}에 저장 완료.")

if __name__ == "__main__":
    db_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\freq_word.db"
    output_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\high_freq_words.txt"

    # 단어 빈도수가 50개 이상인 단어를 추출하여 파일로 저장
    extract_high_frequency_words(db_path, output_file, min_count=50)
