import os
import time
import keys
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from riotwatcher import LolWatcher, ApiError

from src.zilean import TimelineCrawler
from src.zilean.core import read_api_key

parser = argparse.ArgumentParser(description='Downloading all match, player and champion data')
parser.add_argument('-f', '--folder', action='store', default='./data', type=str,
                    help='path to target folder in which the folder for this run will be created',
                    dest='folder')
parser.add_argument('-v', '--visited', action='store', default=None, type=str,
                    help='path to pickle file containing a set of visited and thus to be excluded match IDs',
                    dest='visitedPath')
parser.add_argument('-l', '--log', action='store', default='error', type=str.lower,
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='choosing the level of logging used', dest='logginglevel')
parser.add_argument('-r', '--region', actio='store', default='euw1', type=str,
                    choices=["br1", "eun1", "euw1", "jp1", "kr", "la1", "la2", "na1", "oc1", "ru", "tr1"],
                    help='region from which matches should be crawled', dest='region')
parser.add_argument('-t', '--tier', action='store', default='challenger', type=str.lower,
                    choices=["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"],
                    help='elo tier from which matches should be crawled', dest='tier')
args = parser.parse_args()
api_key = read_api_key(keys.API_KEY_1)
new_folder = f'./matches_{int(time.time())}'  # folder name is timestamp of script execution
try:
    if not (os.path.exists(new_folder)):  # make folder if it does not exist
        os.mkdir(new_folder)
    else:  # handle case if folder already exists
        if os.listdir(new_folder):  # folder exists and is not empty
            raise OSError(f"Directory {new_folder} already exists and is not empty. Aborting...")
except OSError as error:
    print(error)
    raise

logginglevel = getattr(logging, args.logginglevel.upper(), None)
if not isinstance(logginglevel, int):
    raise ValueError('Invalid log level: %s' % args.logginglevel)
logging.basicConfig(filename='logging.log', encoding='utf-8', level=logginglevel)

try:
    with open(args.visitedPath, 'rb') as f:
        visited_matchIDs = pickle.load(f)
except FileNotFoundError as error:
    print(error)
    visited_matchIDs = None

logging.info('Starting crawling...')

crawler = TimelineCrawler(api_key, region=args.region, tier=args.tier.upper(), queue="RANKED_SOLO_5x5")

timelines, matchIDs = crawler.crawl(n=1, match_per_id=30, file=f"{new_folder}/results.json", excludingIDs=visited_matchIDs)

