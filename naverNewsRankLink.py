from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


start_date = datetime(2024, 7, 31)  # 시작 날짜
end_date = datetime(2024, 7, 1)  # 끝나는 날짜
output_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\NnewsRankLink.txt"  # 추출된 뉴스 링크 저장 경로

def get_ranking_news_urls(driver, ranking_url, max_rank=5):
    """
    네이버 뉴스 랭킹 페이지에서 언론사별 1~5위 기사 링크를 추출하는 함수.
    """
    # 해당 페이지 로드
    driver.get(ranking_url)
    
    # 페이지의 주요 요소가 로드될 때까지 대기
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'div.rankingnews_box'))
    )
    
    # 페이지 HTML을 가져와 BeautifulSoup으로 파싱
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    # 언론사별 랭킹 뉴스 박스 찾기
    ranking_boxes = soup.select('div.rankingnews_box')
    
    news_urls = []
    
    # 각 언론사의 랭킹 박스 내에서 상위 5개의 뉴스 기사 추출
    for box in ranking_boxes:
        news_links = box.select('ul li a')[:max_rank]  # 상위 5개의 뉴스만 추출
        for link in news_links:
            news_url = link.get('href')  # 상대 경로를 절대 경로로 변환
            news_urls.append(news_url)
            with open(output_file, "a", encoding="utf-8") as file:
                file.write(news_url + "\n")
                print(news_url)
    
    return news_urls



def main():
    # Selenium 웹 드라이버 옵션 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--log-level=1')  # 로그 레벨 설정
    options.add_argument("--headless")  # 브라우저를 표시하지 않음
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)  # 수정된 옵션을 적용하여 웹 드라이버 생성

    all_news_urls = []

    current_date = start_date
    while current_date >= end_date:
        # 날짜 형식: yyyymmdd로 변환 (예: 20240905)
        date_str = current_date.strftime("%Y%m%d")
        ranking_url = f"https://news.naver.com/main/ranking/popularMemo.naver?date={date_str}"

        print(ranking_url)
        # 언론사별 상위 5개 뉴스 링크를 가져옴
        news_urls = get_ranking_news_urls(driver, ranking_url)


        # 날짜를 하루 감소
        current_date -= timedelta(days=1)


    driver.quit()


if __name__ == "__main__":
    main()
