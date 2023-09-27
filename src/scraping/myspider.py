import scrapy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time

from src import utils

from scrapy import Spider, Request
from scrapy.item import Item, Field


class MySpider(Spider):
    name = "my_spider"
    custom_settings = {
        "ITEM_PIPELINES": {"src.scraping.pipeline.DataPipeline": 300},
    }

    def __init__(self, summoner_name, region, champion, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_urls = [
            f"https://u.gg/lol/profile/{region}/{summoner_name}/champion-stats"
        ]
        self.champion = champion
        self.data = {}

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url=url, callback=self.parse)

    def parse(self, response):
        columns = [
            "rank",
            "champion",
            "winRate",
            "winsLoses",
            "kda",
            "kills",
            "deaths",
            "assists",
            "lp",
            "maxKills",
            "maxKills",
            "cs",
            "damage",
            "gold",
        ]
        row_index = 1
        while True:
            try:
                row = {}
                base_selector = (
                    f"div.rt-tr-group:nth-child({row_index}) > div:nth-child(1)"
                )
                selectors = [
                    f"{base_selector} > div:nth-child(1) > span:nth-child(1)::text",
                    f"{base_selector} > div:nth-child(2) > div:nth-child(1) > span:nth-child(2)::text",
                    f"{base_selector} > div:nth-child(3) > div:nth-child(1) > strong:nth-child(1)::text",
                    f"{base_selector} > div:nth-child(3) > div:nth-child(1) > span:nth-child(3)::text",
                    f"{base_selector} > div:nth-child(4) > div:nth-child(1) > div:nth-child(1) > strong:nth-child(1)::text",
                    f"{base_selector} > div:nth-child(4) > div:nth-child(1) > span:nth-child(2) > strong:nth-child(1)::text",
                    f"{base_selector} > div:nth-child(4) > div:nth-child(1) > span:nth-child(2) > strong:nth-child(3)::text",
                    f"{base_selector} > div:nth-child(4) > div:nth-child(1) > span:nth-child(2) > strong:nth-child(5)::text",
                    f"{base_selector} > div:nth-child(5) > span:nth-child(1) > span:nth-child(2)::text",
                    f"{base_selector} > div:nth-child(6) > span:nth-child(1)::text",
                    f"{base_selector} > div:nth-child(7) > span:nth-child(1)::text",
                    f"{base_selector} > div:nth-child(8) > span:nth-child(1)::text",
                    f"{base_selector} > div:nth-child(9) > span:nth-child(1)::text",
                    f"{base_selector} > div:nth-child(10) > span:nth-child(1)::text",
                ]
                is_row_empty = True
                for column, selector in zip(columns, selectors):
                    item = response.css(selector).get()
                    if item:
                        is_row_empty = False
                        item = item.strip()
                        if column == "winRate":
                            item = item.rstrip("%")
                        elif column in ["damage", "gold"]:
                            item = item.replace(",", "")
                        row[column] = item
                    else:
                        row[column] = "N/A"
                if is_row_empty:
                    break
                if row.get("champion") == self.champion:
                    item = SummonerItem()
                    item['rank'] = row['rank']
                    item['champion'] = row['champion']
                    item['winRate'] = row['winRate']
                    item['winsLoses'] = row['winsLoses']
                    item['kda'] = row['kda']
                    item['kills'] = row['kills']
                    item['deaths'] = row['deaths']
                    item['assists'] = row['assists']
                    item['lp'] = row['lp']
                    item['maxKills'] = row['maxKills']
                    item['cs'] = row['cs']
                    item['damage'] = row['damage']
                    item['gold'] = row['gold']
                    yield item
                row_index += 1
            except Exception as e:
                self.log(f"Error: {e}")
                break
        return self.data


class SummonerItem(Item):
    rank = Field()
    champion = Field()
    winRate = Field()
    winsLoses = Field()
    kda = Field()
    kills = Field()
    deaths = Field()
    assists = Field()
    lp = Field()
    maxKills = Field()
    cs = Field()
    damage = Field()
    gold = Field()


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
    wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.rt-tr-group"))
    )

    # Initialize empty list to store data
    data = []

    # Scroll to the bottom and top of the page to load all rows
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        html_source = driver.page_source
        soup = BeautifulSoup(html_source, "html.parser")
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
                pick_rate = row.select("div.rt-td:nth-of-type(6) > span")[
                    0
                ].text.strip()
                ban_rate = row.select("div.rt-td:nth-of-type(6)")[0].text.strip()
                matches = (
                    row.select("div.rt-td:nth-of-type(8)")[0]
                    .text.strip()
                    .replace(",", "")
                )
                data.append(
                    [rank, champion, tier, win_rate, pick_rate, ban_rate, matches]
                )

            except Exception as e:
                print(f"Error in row {i}: {e}")

    # Define columns for the DataFrame
    columns = [
        "Rank",
        "Champion Name",
        "Tier",
        "Win rate",
        "Pick Rate",
        "Ban Rate",
        "Matches",
    ]

    # Create a DataFrame from the scraped data
    df_scraped = pd.DataFrame(data, columns=columns)
    # Close the browser
    driver.quit()
    df_scraped = utils.clean_champion_data(df_scraped)
    # Return the DataFrame converted to a dictionary, indexed by "Champion Name"
    return df_scraped.reset_index().to_dict("index")
