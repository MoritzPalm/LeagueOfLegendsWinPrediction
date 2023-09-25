from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

from src import utils

from scrapy import Spider
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
import csv
from io import StringIO
from twisted.internet.defer import inlineCallbacks, returnValue

class MySpider(Spider):
    name = 'my_spider'
    custom_settings = {'LOG_LEVEL': 'INFO'}

    def __init__(self, summoner_name, region, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_urls = [f"https://u.gg/lol/profile/{region}/{summoner_name}/champion-stats"]
        self.data = {}
        self.columns = ['Rank', 'Champion', 'Win Rate', 'Wins/Loses', 'Unnamed', 'Kills', 'Deaths', 'Assists', 'LP',
                        'Max Kills', 'Max Deaths', 'CS', 'Damage', 'Gold']

    def parse(self, response):
        try:
            row = {}
            selectors = [
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(1) > span:nth-child(1)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > span:nth-child(2)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(3) > div:nth-child(1) > strong:nth-child(1)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(3) > div:nth-child(1) > span:nth-child(3)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(4) > div:nth-child(1) > div:nth-child(1) > strong:nth-child(1)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(4) > div:nth-child(1) > span:nth-child(2) > strong:nth-child(1)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(4) > div:nth-child(1) > span:nth-child(2) > strong:nth-child(3)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(4) > div:nth-child(1) > span:nth-child(2) > strong:nth-child(5)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(5) > span:nth-child(1) > span:nth-child(2)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(6) > span:nth-child(1)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(7) > span:nth-child(1)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(8) > span:nth-child(1)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(9) > span:nth-child(1)::text",
                "div.rt-tr-group:nth-child(1) > div:nth-child(1) > div:nth-child(10) > span:nth-child(1)::text",
            ]
            for col, selector in zip(self.columns, selectors):
                item = response.css(selector).get()
                row[col] = item.strip() if item else 'N/A'
            self.data.update(row)

        except Exception as e:
            self.log(f"Error: {e}")

    def closed(self, reason):
        reactor.stop()

def run_spider(summoner_name, region):
    configure_logging({'LOG_LEVEL': 'INFO'})
    runner = CrawlerRunner()
    spider = MySpider(summoner_name=summoner_name, region=region)
    d = runner.crawl(spider)
    d.addBoth(lambda _: reactor.stop())
    reactor.run()

    return spider.data

def scrape_champion_metrics():
    options = Options()

    # Define the URL containing the metrics
    url = "https://u.gg/lol/tier-list"

    # Configure Chrome options for headless browsing
    options.add_argument("--headless")
    # Path to Chrome executable
    # TODO: change path of chrome driver to use path? venv?

    # Start Chrome WebDriver service
    driver = webdriver.Chrome(options=options)

    # Open the URL
    driver.get(url)

    # Initialize WebDriverWait and wait until the rows in the table are loaded
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.rt-tr-group")))

    # Initialize empty list to store data
    data = []

    # Scroll to the bottom and top of the page to load all rows
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        html_source = driver.page_source
        soup = BeautifulSoup(html_source, 'html.parser')
        rows = soup.select("div.rt-tr-group")

        # Break if there are no rows or if we've scraped all the rows
        if not rows or len(data) >= len(rows):
            break

        for i in range(len(data), len(rows)):
            row = rows[i]
            try:
                rank = row.select("div.rt-td:nth-of-type(1)")[0].text.strip()
                champion = row.select("div.rt-td:nth-of-type(3)")[0].text.strip()
                tier = row.select("div.rt-td:nth-of-type(4)")[0].text.strip()
                win_rate = row.select("div.rt-td:nth-of-type(5)")[0].text.strip()
                pick_rate = row.select("div.rt-td:nth-of-type(6) > span")[0].text.strip()
                ban_rate = row.select("div.rt-td:nth-of-type(6)")[0].text.strip()
                matches = row.select("div.rt-td:nth-of-type(8)")[0].text.strip().replace(',', '')
                data.append([rank, champion, tier, win_rate, pick_rate, ban_rate, matches])

            except Exception as e:
                print(f"Error in row {i}: {e}")

    # Define columns for the DataFrame
    columns = ['Rank', 'Champion Name', 'Tier', 'Win rate', 'Pick Rate', 'Ban Rate', 'Matches']

    # Create a DataFrame from the scraped data
    df_scraped = pd.DataFrame(data, columns=columns)
    # Close the browser
    driver.quit()
    df_scraped = utils.clean_champion_data(df_scraped)
    # Return the DataFrame converted to a dictionary, indexed by "Champion Name"
    return df_scraped.reset_index().to_dict('index')
