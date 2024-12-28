import requests
import random
import time
import threading

def download_file_with_retry(url, filename, retries=10, delay=5, backoff=2):
    """
    下载文件，失败后重试
    :param url: 文件 URL
    :param filename: 保存文件名
    :param retries: 最大重试次数
    :param delay: 初始重试等待时间（秒）
    :param backoff: 每次重试的等待时间倍数
    """
    attempt = 0
    while attempt < retries:
        try:
            print(f"尝试下载文件: {url} (第 {attempt + 1} 次)")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(filename, "wb") as f:
                    f.write(response.content)
            print("下载成功！")
            return True
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {e}")
            attempt += 1
            if attempt < retries:
                wait_time = delay * (backoff ** (attempt - 1))  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("已达到最大重试次数，下载失败。")
                return False

maxWidth = 1024
minWidth = 128
urls = []
filenames = []
for imgId in range(765, 766):
    width = random.randint(minWidth, maxWidth)
    height = random.randint(minWidth, maxWidth)
    url = f"https://picsum.photos/{width}/{height}.jpg?random={imgId}"
    urls.append(url)
    filename = f"../AnimalData/random/random.{imgId}.jpg"
    filenames.append(filename)

threads = []  # 存储线程
for url, filename in zip(urls, filenames):
    thread = threading.Thread(target=download_file_with_retry, args=(url, filename))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
