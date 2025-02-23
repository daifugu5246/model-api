if __name__ == "__main__":
    import requests

    url = "https://qp6u3o0dakmdy0-8000.proxy.runpod.net/generate"

    data = {
        "instruction": "วิเคราะห์ผลประกอบการไตรมาสล่าสุดของ PTT",
        "data": "รายได้สุทธิของ PTT ในไตรมาสล่าสุดเพิ่มขึ้น 15%",
        "event": "ราคาน้ำมันดิบพุ่งสูงขึ้นในตลาดโลก"
    }

    params = {
        "symbol": "PTT",
        "quarter": "Q1_66"
    }

    response = requests.post(url, json=data, params=params)

    print("Status Code:", response.status_code)
    print("Response:", response.json())
