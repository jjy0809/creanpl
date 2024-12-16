import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#conn = sqlite3.connect(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\news_freq_word.db")
conn = sqlite3.connect(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\dc_sensor_res.db")
cursor = conn.cursor()

cursor.execute('SELECT word, count FROM word_counts')
word_freq = dict(cursor.fetchall())


conn.close()


wordcloud = WordCloud(
    font_path='C:/Windows/Fonts/malgun.ttf', 
    width=1600,
    height=1600,
    background_color='white',
    max_words=10000,    
    min_font_size=15
).generate_from_frequencies(word_freq)


plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
