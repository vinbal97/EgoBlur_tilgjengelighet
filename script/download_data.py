
from bs4 import BeautifulSoup
import requests

class ImageScraper:
    def __init__(self):
        pass

    def grab_image_urls(self, side=1):
        """Function to grab all image urls from the first page. Return a list with all image urls."""
        url = f"http://kv-vm-00830.statkart.no/bilder/?side={side}"
        payload = {}
        headers = {}

        response = requests.request("GET", url, headers=headers, data=payload)

        soup = BeautifulSoup(response.text, "html.parser")
        urls = [a["href"] for a in soup.find_all("a", href=True)]
        return urls

    def grab_pages(self, urls):
        """Function to grab number of pages from url list. Return a number of pages and list with all image urls."""
        for link in urls:
            splits = link.split()
            for split in splits:
                if "side=" in split:
                    pages = int(split.split("=")[1])
        return urls[pages:], pages
    
    def download_image(self, url, filename):
        """Function to download an image from a given url and save it with the specified filename."""
        payload = {}
        headers = {}

        response = requests.request("GET", url, headers=headers, data=payload)

        with open(filename, "wb") as f:
            f.write(response.content)

if __name__ == "__main__":
    scraper = ImageScraper()
    urls = scraper.grab_image_urls(side=1)
    urls, pages = scraper.grab_pages(urls)

"""
url = "https://data.kartverket.no/tilgjengelighet/tilgjengelighet_thumb/3118_TettstedParkeringsområde_20250916145208_t.jpg"

payload = {}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
with open("image.jpg", "wb") as f:
    f.write(response.content)

"""