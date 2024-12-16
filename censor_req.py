import requests

def cen_req(com, lst):
    # Lambda 함수 URL
    url = "https://nlu35qe7zd.execute-api.ap-northeast-2.amazonaws.com/creanpl/creanpl"

    # 헤더 설정 (한글은 데이터에서 처리)
    headers = {
        'Content-Type': 'application/json'  # JSON 형식으로 데이터 전송
    }

    # JSON으로 데이터를 보내기 위한 딕셔너리
    data = {
        'com': com,  # 한글 포함 데이터
        'lst': lst  # lst 값
    }

    # POST 요청 보내기
    response = requests.post(url, headers=headers, json=data)

    # 응답 출력
    return response.status_code, response.text

if __name__ == "__main__":
    lst = [1, 1, 1, 1, 1, 1, 1]
    while 1:
        ans = input("문장 입력 -> ")
        if ans == "00": exit()
        print(cen_req(ans, lst))