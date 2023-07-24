from riotwatcher import LolWatcher, ApiError
from zilean import TimelineCrawler, SnapShots, read_api_key
import pandas as pd
import numpy as np
import os
import time
import keys

api_key = read_api_key(keys.API_KEY_1)
new_folder = f'./{int(time.time())}'    # folder name is timestamp of script execution
try:
    if not (os.path.exists(new_folder)):    # make folder if it does not exists
        os.mkdir(new_folder)
    else:   # handle case if folder already exists
        if os.listdir(new_folder):  # folder exists and is not empty
            raise OSError(f"Directory {new_folder} already exists and is not empty. Aborting...")
except OSError as error:
    print(error)


crawler = TimelineCrawler(api_key, region="euw1",
                          tier="CHALLENGER", queue="RANKED_SOLO_5x5")

result = crawler.crawl(1, match_per_id=30, file=f"{new_folder}/results.json")




