
from bs4 import BeautifulSoup
import requests
import os
import time

class ImageScraper:
    def __init__(self, download_folder=r"C:\Users\balvin\Documents\Github\tilgjengelighet\bilder"):
        if os.path.isdir(download_folder) != True:
            os.mkdir(download_folder)
        self.download_folder = download_folder

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
    
    def download_image(self, url):
        """Function to download an image from a given url and save it with the specified filename."""
        payload = {}
        headers = {}

        response = requests.request("GET", url, headers=headers, data=payload)
        filename = os.path.join(self.download_folder, url.split("/")[-1])
        with open(filename, "wb") as f:
            f.write(response.content)
            
    
    def main(self):
        """Run skript"""
        start = time.time()
        urls = []
        for i in range(1, 92):
            print("-- Starting on page: ", i)
            url = self.grab_image_urls(side=i)
            url, pages = self.grab_pages(url)
            urls.extend(url)
        uniq = list(set(urls))

        for link in uniq:
            self.download_image(link)
        slutt = time.time()

        print("Tidbruk: ", slutt-start)

if __name__ == "__main__":
    scraper = ImageScraper()
    scraper.main()
