import concurrent.futures
import threading
import queue
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

class DriverPool:
    def __init__(self, max_size, options):
        self.max_size = max_size
        self.options = options
        self.pool = queue.Queue(max_size)
        self.lock = threading.Lock()
        
        # 드라이버 인스턴스 미리 생성
        for _ in range(max_size):
            driver = webdriver.Chrome(options=self.options)
            self.pool.put(driver)
    
    def get_driver(self):
        with self.lock:
            return self.pool.get()
    
    def return_driver(self, driver):
        with self.lock:
            self.pool.put(driver)
    
    def close_all(self):
        while not self.pool.empty():
            driver = self.pool.get()
            driver.quit()

def get_comments_from_url(driver, url):
    driver.get(url)
    
    # 최대 15회까지 더보기 버튼 클릭
    for _ in range(10):
        try:
            more_button = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.u_cbox_btn_more'))
            )
            more_button.click()
        except Exception:
            break
    
    # 댓글이 모두 로드된 후 페이지 소스 파싱
    WebDriverWait(driver, 2.5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'span.u_cbox_contents'))
    )
    html = driver.page_source
    c_soup = BeautifulSoup(html, "html.parser")
    comments = c_soup.select('span.u_cbox_contents')
    
    return [comment.text for comment in comments]

def load_last_index(file_path):
    """
    마지막으로 크롤링한 URL 인덱스를 불러오는 함수.
    파일이 없거나 오류가 발생하면 0을 반환.
    """
    try:
        with open(file_path, "r") as f:
            return int(f.read().strip())
    except:
        return 0

def save_last_index(file_path, index):
    """
    마지막으로 크롤링한 URL 인덱스를 파일에 저장하는 함수.
    """
    li = load_last_index(file_path)
    if li < index:
        with open(file_path, "w") as f:
            f.write(str(index))

def process_url(driver_pool, url, idx, output_file, index_file):
    """
    각 URL을 처리하고 댓글을 저장하는 함수 (병렬 실행).
    """
    driver = driver_pool.get_driver()
    try:
        url = url.strip()
        url = url[:33] + "comment/" + url[33:]
        
        if not url:
            return
        
        comments = get_comments_from_url(driver, url)
        
        # 댓글을 파일에 저장
        with open(output_file, "a", encoding='utf-8') as f:
            for comment in comments:
                f.write(comment + '\n')
            print(f"\nURL {idx}: {url} - 댓글 {len(comments)}개 저장 완료\n")
        
        # 현재 인덱스를 파일에 저장
        save_last_index(index_file, idx)

    except Exception as e:
        print(f"\nURL {idx}: {url} - 오류 발생: {e}\n")
    
    finally:
        driver_pool.return_driver(driver)

def main():
    input_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\NnewsRankLink.txt"  # URL이 포함된 텍스트 파일 경로
    output_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\naverNews_comments.txt"  # 댓글을 저장할 파일 경로
    index_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\rec_idx_news.txt"  # 마지막 인덱스를 저장할 파일 경로

    # Selenium 웹 드라이버 옵션 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--log-level=1')  # 로그 레벨 설정
    options.add_argument("--disable-webusb")
    options.add_argument("--headless")  # 브라우저를 표시하지 않음
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    # 드라이버 풀 초기화
    driver_pool = DriverPool(max_size=6, options=options)

    # 마지막으로 크롤링한 인덱스 불러오기
    last_index = load_last_index(index_file)

    with open(input_file, "r", encoding='utf-8') as f:
        urls = f.readlines()
    
    # ThreadPoolExecutor를 사용해 병렬로 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for idx, url in enumerate(urls[last_index:], last_index + 1):
            futures.append(
                executor.submit(process_url, driver_pool, url, idx, output_file, index_file)
            )
        
        # 모든 태스크가 완료될 때까지 대기
        concurrent.futures.wait(futures)

    driver_pool.close_all()

if __name__ == "__main__":
    main()
