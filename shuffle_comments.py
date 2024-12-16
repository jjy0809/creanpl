import sys
import os
from random import shuffle


with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\copcom.txt", "r", encoding='utf-8') as f:
        coms = f.readlines()
        
with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\label.txt", "r", encoding='utf-8') as f:
        labs = f.readlines()
        
s = 0 #len(labs)+1

print(sum(len(s.replace(" ", "")) for s in coms))

coms = coms[s:]
shuffle(coms)

for i, c in enumerate(coms, s):
        with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\comments3.txt", "a", encoding='utf-8') as f:
                f.write(c)
                print(f"{i}: {c}")
        

        
