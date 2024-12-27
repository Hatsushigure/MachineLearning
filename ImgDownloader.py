import requests
import random

maxWidth = 1024
minWidth = 128
for num in range(1224, 1225):
    width = random.randint(minWidth, maxWidth)
    height = random.randint(minWidth, maxWidth)
    url = f"https://picsum.photos/{width}/{height}.jpg?random={num}"
    imgdata = requests.get(url, allow_redirects=True).content
    file = open(f"../AnimalData/random/random.{num}.jpg", "wb")
    file.write(imgdata)
    file.close()
