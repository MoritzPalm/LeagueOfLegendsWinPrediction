import os
import time
import sys
import keys
import pickle
import logging
import argparse

import pandas as pd
import numpy as np
import psycopg2

from riotwatcher import LolWatcher, ApiError
from configparser import ConfigParser, Error
from src.zilean import TimelineCrawler
from src.zilean.core import read_api_key

import os
print (os.getcwd())
print (__file__)

parser = argparse.ArgumentParser(description='Downloading all match, player and champion data')
parser.add_argument('-f', '--folder', action='store', default='./data', type=str,
                    help='path to target folder in which the folder for this run will be created',
                    dest='folder')
parser.add_argument('-v', '--visited', action='store', default='', type=str,
                    help='path to pickle file containing a set of visited and thus to be excluded match IDs',
                    dest='visitedPath')
parser.add_argument('-l', '--log', action='store', default='error', type=str.upper,
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='choosing the level of logging used', dest='logginglevel')
parser.add_argument('-r', '--region', action='store', default='euw1', type=str,
                    choices=["br1", "eun1", "euw1", "jp1", "kr", "la1", "la2", "na1", "oc1", "ru", "tr1"],
                    help='region from which matches should be crawled', dest='region')
parser.add_argument('-t', '--tier', action='store', default='challenger', type=str.upper,
                    choices=["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"],
                    help='elo tier from which matches should be crawled', dest='tier')
parser.add_argument('-n', action='store', default=0, type=int, help='number of matches to be crawled, 0 means that every available match will be crawled')
parser.add_argument('-m', '--matches_per_id', action='store', default=15, type=int,
                    help='number of matches to be crawled per id', dest='matches_per_id')
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
logging.basicConfig(filename=f'{new_folder}/logging.log', encoding='utf-8', level=logginglevel, format="%(asctime)s - %(levelname)s - %(funcName)s() - %(message)s")

try:
    with open(args.visitedPath, 'rb') as f:
        visited_matchIDs = pickle.load(f)
except FileNotFoundError as error:
    print("No file with matchIDs passed, saving all matches.")
    visited_matchIDs = set()



logging.info('Starting crawling...')

if args.n == 0:
    args.n = sys.maxsize

crawler = TimelineCrawler(api_key, region=args.region, tier=args.tier.upper(), queue="RANKED_SOLO_5x5")

timelines, matchIDs = crawler.crawl(n=args.n, match_per_id=args.matches_per_id,
                                    file=f"{new_folder}/results.json", excludingIDs=visited_matchIDs)



def config(filename='src/database.ini', section='postgresql'):
    # create a parser
    configparser = ConfigParser()
    # read config file
    configparser.read(filename)
    # get section, default to postgresql
    db = {}
    if configparser.has_section(section):
        params = configparser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db


def connect_to_db():
    """Connect to db and return the connection object"""
    params = config()
    return psycopg2.connect(**params)


try:
    conn = connect_to_db()

    cur = conn.cursor()
    cur.execute('INSERT INTO match_test(matchid) VALUES (%s) RETURNING matchid', (matchIDs.pop(),))
    matchID = cur.fetchone()[0]
    print(matchID)
    conn.commit()
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
        print(error)
finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


