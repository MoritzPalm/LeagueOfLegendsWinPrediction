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


def scrape_summonerdata(name: str, region: str) -> pd.DataFrame:
    # URL
    url = f"https://u.gg/lol/profile/{region}/{name}/champion-stats"

    options = Options()
    options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    driver.get(url)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.rt-tr-group")))

    # Initialize list to store each row of data
    data = []

    # Get page source and create BeautifulSoup object
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Close the driver
    driver.quit()

    # Get all the rows using BeautifulSoup
    rows = soup.select("div.rt-tr-group")

    for i, row in enumerate(rows, 1):  # start from 1 because CSS nth-child starts from 1
        try:
            selectors = {
                'Champion': "div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > span:nth-child(2)",
                'WinsLoses': "div:nth-child(1) > div:nth-child(3) > div:nth-child(1) > span:nth-child(3)",
                'Winrate': "div:nth-child(1) > div:nth-child(3) > div:nth-child(1) > strong:nth-child(1)",
                'KDA': "div:nth-child(1) > div:nth-child(4) > div:nth-child(1) > div:nth-child(1) > strong:nth-child(1)",
                'KillsDeathsAssists': "div:nth-child(1) > div:nth-child(4) > div:nth-child(1) > span:nth-child(2)",
                # New field
                'LP': "div:nth-child(1) > div:nth-child(5) > span:nth-child(1)",
                'MaxKills': "div:nth-child(1) > div:nth-child(6) > span:nth-child(1)",
                'MaxDeaths': "div:nth-child(1) > div:nth-child(7)",
                'CS': "div:nth-child(1) > div:nth-child(8) > span:nth-child(1)",
                'Damage': "div:nth-child(1) > div:nth-child(9) > span:nth-child(1)",
                'Gold': "div:nth-child(1) > div:nth-child(10) > span:nth-child(1)"
            }

            row_data = []
            for key, selector in selectors.items():
                element = row.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    if key in ['Damage', 'Gold']:
                        text = text.replace(',', '')
                else:
                    text = 'N/A'
                row_data.append(text)
            data.append(row_data)
        except Exception as e:
            logging.warning(f"Error in row {i}: {e}")


    # Convert the data to a DataFrame
    columns = ['Champion', 'WinsLoses', 'Winrate', 'KDA', 'KillsDeathsAssists', 'LP', 'MaxKills', 'MaxDeaths', 'CS',
               'Damage', 'Gold']
    df_individual = pd.DataFrame(data, columns=columns)

    # Display the DataFrame
    return df_individual


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

