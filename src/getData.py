import sys
import keys
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from riotwatcher import LolWatcher
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists
from sqlalchemy.exc import IntegrityError

from utils import get_season
from crawlers.MatchIdCrawler import MatchIdCrawler
from src.sqlstore.db import get_conn, get_session
from src.sqlstore.match import SQLmatch
from src.sqlstore.participant import SQLparticipantStats
from src.sqlstore.timeline import SQLTimeline, SQLTimelineEvent, SQLTimelineFrame, SQLTimelineParticipantFrame


# TODO: make logging actually useful
# TODO: implement champion stat parsing and update strategy

def main():
    api_key = keys.API_KEY_1
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
    matchIDs = crawler.getMatchIDs(n=args.n)
    watcher = LolWatcher(api_key)
    with get_session(cleanup=False) as session:
        for matchID in matchIDs:
            if session.query(exists().where(SQLmatch.matchId == matchID)).scalar():
                logger.info(f"matchID {matchID} already present in database")
                continue
            current_match_info = watcher.match.by_id(match_id=matchID, region=args.region)['info']
            current_match_timeline = watcher.match.timeline_by_match(region=args.region, match_id=matchID)['info']
            seasonId = get_season(current_match_info['gameVersion'])
            current_match = SQLmatch(matchId=matchID,
                                     platformId=current_match_info['platformId'],
                                     gameId=current_match_info['gameId'],
                                     seasonId=seasonId,
                                     queueId=current_match_info['queueId'],
                                     gameVersion=current_match_info['gameVersion'],
                                     mapId=current_match_info['mapId'],
                                     gameDuration=current_match_info['gameDuration'],
                                     gameCreation=current_match_info['gameCreation'],
                                     )
            session.add(current_match)  # if performance is an issue, we can still use the core api, see here:
            # https://towardsdatascience.com/how-to-perform-bulk-inserts-with-sqlalchemy-efficiently-in-python-23044656b97d
            for participant in current_match_info['participants']:
                participant['platformId'] = current_match_info['platformId']
                participant['gameId'] = current_match_info['gameId']
                # TODO: challenges table implementation
                # TODO: perks table implementation
                curr_participantStats = SQLparticipantStats(**participant)
                session.add(curr_participantStats)
            current_timeline = SQLTimeline(platformId=current_match_info['platformId'],
                                           gameId=current_match_info['gameId'],
                                           frameInterval=current_match_timeline['frameInterval'])
            session.add(current_timeline)
            for frameId, frame in enumerate(current_match_timeline['frames']):
                frame_obj = SQLTimelineFrame(platformId=current_match_info['platformId'],
                                             gameId=current_match_info['gameId'],
                                             frameId=frameId,
                                             timestamp=current_match_timeline['frames'][frameId]['timestamp'])
                session.add(frame_obj)
                for eventId, event in enumerate(current_match_timeline['frames'][frameId]['events']):
                    if event['type'] in {'CHAMPION_KILL', 'CHAMPION_SPECIAL_KILL', 'TURRET_PLATE_DESTROYED',
                                         'BUILDING_KILL'}:
                        continue
                    event['platformId'] = current_match_info['platformId']
                    event['gameId'] = current_match_info['gameId']
                    event['frameId'] = frameId
                    event['eventId'] = eventId
                    event_obj = SQLTimelineEvent(**event)
                    session.add(event_obj)
                for i, participantFrame in enumerate(
                        current_match_timeline['frames'][frameId]['participantFrames'].items(), start=1):
                    participantFrameData = participantFrame[1]
                    participantFrameData['platformId'] = current_match_info['platformId']
                    participantFrameData['gameId'] = current_match_info['gameId']
                    participantFrameData['frameId'] = frameId
                    participantFrameData['participantId'] = i
                    participantFrame_obj = SQLTimelineParticipantFrame(**participantFrameData)
                    session.add(participantFrame_obj)
            try:
                session.flush()
                session.commit()  # TODO: this should be handled differently, maybe with postgres ON INSERT.. DO NOTHING?
            except Exception as e:
                print(str(e))
                session.rollback()


if __name__ == '__main__':
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
                        choices=["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD",
                                 "SILVER",
                                 "BRONZE", "IRON"],
                        help='elo tier from which matches should be crawled', dest='tier')
    parser.add_argument('-n', action='store', default=0, type=int,
                        help='number of matches to be crawled, 0 means that every available match will be crawled')
    parser.add_argument('-m', '--matches_per_id', action='store', default=15, type=int,
                        help='number of matches to be crawled per id', dest='matches_per_id')

    args = parser.parse_args()

    main()
