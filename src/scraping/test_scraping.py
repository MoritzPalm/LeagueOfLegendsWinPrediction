
from scrapy.crawler import CrawlerProcess

from src.scraping.myspider import MySpider


def test_scraping():
    data = {
        "summonerName": "Solo Tiger",
        "region": "euw1",
    }

    process = CrawlerProcess()
    process.crawl(MySpider, [data])
    process.start()


if __name__ == "__main__":
    test_scraping()
