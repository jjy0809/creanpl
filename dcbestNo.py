from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def get_no_from_url(driver, url):
    driver.get(url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'td.gall_num'))
    )
    html = driver.page_source
    c_soup = BeautifulSoup(html, "html.parser")
    nums = c_soup.select('td.gall_num')
    return [n.text for n in nums]

def main():
    start = 343
    page = start + 100
    output_file = r"C:\Users\happy\Desktop\학교\고등학교\2학년\양방향 장단기 메모리 신경망을 이용한 인터넷 혐오표현 검열 확장 프로그램\dcNo.txt"  

    options = webdriver.ChromeOptions()
    options.add_argument('--log-level=1')  
    options.add_argument("--headless")  
    driver = webdriver.Chrome(options=options)
    
    

    url = "https://gall.dcinside.com/board/lists/?id=dcbest&page="

    for i in range(start, page):
        print(i)
        urli = url + str(1+i)
        with open(output_file, "a", encoding='utf-8') as f:
            nums = get_no_from_url(driver, urli)
            for num in nums:
                if num != "설문" and num != "공지":
                    f.write(num + '\n')
                    print(num)

    driver.quit()

if __name__ == "__main__":
    main()
