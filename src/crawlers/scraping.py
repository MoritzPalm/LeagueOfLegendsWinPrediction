from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time


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
                text = element.get_text(strip=True) if element else 'N/A'
                row_data.append(text)

            data.append(row_data)

        except Exception as e:
            print(f"Error in row {i}: {e}")

    # Convert the data to a DataFrame
    columns = ['Champion', 'WinsLoses', 'Winrate', 'KDA', 'KillsDeathsAssists', 'LP', 'MaxKills', 'MaxDeaths', 'CS',
               'Damage', 'Gold']
    df_individual = pd.DataFrame(data, columns=columns)

    # Display the DataFrame
    return df_individual