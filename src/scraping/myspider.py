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
from scrapy.utils.project import get_project_settings


class MySpider(Spider):
    name = "my_spider"
    custom_settings = {
        "ITEM_PIPELINES": {"src.scraping.pipeline.DataPipeline": 300},
    }

    def __init__(self, init_data, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_urls = []
        for data in init_data:
            self.start_urls.append({
                'url': f"https://u.gg/lol/profile/{data['region']}/{data['summonerName']}/champion-stats",
                'alternative_url': f"https://u.gg/lol/profile/{data['region']}/{data['summonerName']}/champion-stats?queueType=normal_draft_5x5",
                'tried_alternative': False  # To track whether the alternative URL has been tried
            })

    def start_requests(self):
        for url_data in self.start_urls:
            yield Request(url=url_data['url'], callback=self.parse, meta={'url_data': url_data})

    def parse(self, response):
        settings = get_project_settings()
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
        url_data = response.meta['url_data']
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
                    if not url_data['tried_alternative']:
                        # If no data and haven't tried the alternative URL, then try it.
                        url_data['tried_alternative'] = True
                        yield Request(url=url_data['alternative_url'], callback=self.parse, meta={'url_data': url_data})
                    break

                item = SummonerItem()
                item['url'] = response.meta['url_data']['url']
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


class SummonerItem(Item):
    url = Field()
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
