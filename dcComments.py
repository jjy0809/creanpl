import re
import subprocess
import time
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException  # 시간 초과 예외 처리
from bs4 import BeautifulSoup

def get_comments_from_url(driver, url):
    driver.get(url)
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'p.usertxt'))
    )
    html = driver.page_source
    c_soup = BeautifulSoup(html, "html.parser")
    comments = c_soup.select('p.usertxt')
    return [comment.text for comment in comments]

def main():
    input_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\dcNo.txt"  
    output_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\comments2.txt"
    recent_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\recent_n.txt"  

    # Selenium 웹 드라이버 옵션 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--log-level=1')  # 로그 레벨 설정
    options.add_argument("--headless")  # 브라우저를 표시하지 않음
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)  # 수정된 옵션을 적용하여 웹 드라이버 생성

    url = "https://gall.dcinside.com/board/view/?id=dcbest&no="
    
    # 마지막으로 처리한 게시글 번호 읽기
    try:
        with open(recent_file, "r", encoding='utf-8') as rf:
            last_processed = rf.read().strip()
    except FileNotFoundError:
        last_processed = None  # 파일이 없을 경우, None으로 설정

    # input_file에서 게시글 번호 읽기
    with open(input_file, "r", encoding='utf-8') as f:
        nums = f.readlines()

    # output_file에서 기존 댓글 읽기
    with open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\comments.txt" , "r", encoding='utf-8') as f:
        coms = f.readlines()

    # 크롤링 시작 번호 설정
    start_index = 0
    if last_processed:
        # recent_n.txt에 저장된 번호 다음 번호를 찾아서 시작하도록 설정
        found_last = False
        for idx, num in enumerate(nums):
            if found_last:
                start_index = idx
                break
            if num.strip() == last_processed:
                found_last = True
        else:
            print("처음부터 시작")
            start_index = 0

    print(f"{nums[start_index]}부터 시작")
    # 크롤링 시작
    for i, num in enumerate(nums[start_index:], start_index + 1):
        with open(output_file, "a", encoding='utf-8') as f:
            urli = url + str(num).strip()
            print(urli)
            if not urli:
                continue
            try:
                comments = get_comments_from_url(driver, urli)
                for comment in comments:
                    if comment[-9:] == " - dc App":
                        comment = comment[:-9]
                    if comment == "여자 처녀막+몸무게+나이 = 남자 키" or comment == "중고 쿵쾅이 아줌마 = 난쟁이" or comment == "안녕하세요. GM사무실 입니다. 국내최초 신개념 재테크 스코어역베팅 적중률 99% 하루 2%때 복리이율로 수익을 가져가세요. 해외에선 유명한 재테크 대한민국 최초GM에서 시작합니다. 텔레:newyork424 텔레:newyork424":
                        print(f"pass comment: {comment}")
                        continue
                    f.write(comment + '\n')                    
                    print(comment)
                print(f"\nURL {i}: {num.strip()} - 댓글 {len(comments)}개 저장 완료\n")
                
                # 마지막으로 처리한 게시글 번호 업데이트
                with open(recent_file, "w", encoding='utf-8') as rf:
                    rf.write(num.strip())
            except TimeoutException:
                print(f"\nURL {i}: {num.strip()} - 시간 초과 오류\n")
                with open(recent_file, "w", encoding='utf-8') as rf:
                    rf.write(num.strip())
                driver.quit()
                sys.exit(1)  # 프로그램 종료
            except Exception as e:
                print(f"\nURL {i}: {num.strip()} - 오류 발생: {e}\n")
            
            time.sleep(0.75)

    # 웹 드라이버 종료
    driver.quit()

if __name__ == "__main__":
    while True:
        try:
            main()
            print("모든 댓글 크롤링 완료")
            break  # main 함수가 정상적으로 종료되면 루프를 빠져나옴
        except SystemExit:
            print("프로그램 재실행")
            # 프로그램이 종료될 때 재실행을 위해 5초 대기
            time.sleep(2)
            # 현재 스크립트를 재실행
            subprocess.Popen([sys.executable] + sys.argv)
            sys.exit()  # 현재 프로세스 종료
        except Exception as e:
            print(f"오류 발생: {e}")
            time.sleep(2)  # 오류 발생 시 5초 대기 후 재시도
