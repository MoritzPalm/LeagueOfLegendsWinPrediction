


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
from crawlers.MatchIdCrawler import MatchIdCrawler


def db_config(filename='database.ini', section='postgresql'):
    # create a parser
    db_configparser = ConfigParser()
    # read config file
    db_configparser.read(filename)
    # get section, default to postgresql
    db = {}
    if db_configparser.has_section(section):
        params = db_configparser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db


def connect_to_db():
    """Connect to db and return the connection object"""
    params = db_config()
    return psycopg2.connect(**params)


def main():
    api_key = keys.API_KEY_1
    print(api_key)
    logginglevel = getattr(logging, args.logginglevel.upper(), None)
    if not isinstance(logginglevel, int):
        raise ValueError('Invalid log level: %s' % args.logginglevel)
    file_handler = logging.FileHandler(filename=f'{args.folder}/logging.log')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(encoding='utf-8', level=logginglevel,
                        format="%(asctime)s - %(levelname)s - %(funcName)s() - %(message)s", handlers=handlers)
    logger = logging.getLogger(__name__)

    logger.info(f'starting getData.py with arguments {sys.argv}')

    try:
        logger.info('opening file containing already visited matchIDs')
        with open(args.visitedPath, 'rb') as f:
            visited_matchIDs: set = pickle.load(f)
            logger.info('file found, pickle load succeeded')
    except FileNotFoundError as error:
        logger.warning('No file with matchIDs passed, saving all matches.')
        visited_matchIDs = set()

    if args.n == 0:
        args.n = sys.maxsize

    crawler = MatchIdCrawler(api_key=api_key, region=args.region, tier=args.tier)
    matchIDs = crawler.getMatchIDs(n=1)

    conn = None
    try:
        conn = connect_to_db()
        logger.info('Database connection established')
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
            logger.info('Database connection closed.')


parser = argparse.ArgumentParser(description='Downloading all match, player and champion data')
parser.add_argument('-f', '--folder', action='store', default='../data', type=str,
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

if __name__ == '__main__':
    main()





