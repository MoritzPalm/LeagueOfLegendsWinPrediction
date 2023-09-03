import sys
import keys
import pickle
import logging
import argparse
import pytest

from riotwatcher import LolWatcher
from sqlalchemy.sql import exists
from sqlalchemy.exc import IntegrityError
import sqlalchemy.orm.session

from src.utils import get_season, get_patch
from src.crawlers.MatchIdCrawler import MatchIdCrawler
from src.sqlstore.db import get_conn, get_session
from src.sqlstore.match import SQLmatch
from src.sqlstore.participant import SQLparticipantStats, SQLChallenges
from src.sqlstore.timeline import SQLTimeline, SQLTimelineEvent, SQLTimelineFrame, SQLTimelineParticipantFrame
from src.sqlstore.champion import SQLChampion, SQLChampionStats


# TODO: implement update strategy after new patch

def getData():
    if args.n == 0:
        args.n = sys.maxsize    # if no maximum number of matches or 0 passed, use maximum number of matches possible
    logger.info(f"initializing matchIdCrawler with api key {api_key}, region {args.region} and tier {args.tier}")
    crawler = MatchIdCrawler(api_key=api_key, region=args.region, tier=args.tier)
    logger.info(f"crawling {args.n} matchIDs")
    matchIDs = crawler.getMatchIDs(n=args.n)
    logger.info(f"{len(matchIDs)} non-unique matchIDs crawled")
    watcher = LolWatcher(api_key)
    new_patch = False

    # TODO: check if new patch is out, if yes, parse new champion stats from data dragon
    with get_session(cleanup=False) as session:
        if new_patch:
            parse_champion_data(session, watcher)
        for matchID in matchIDs:
            if session.query(exists().where(SQLmatch.matchId == matchID)).scalar():
                logger.warning(f"matchID {matchID} already present in database")
                continue
            current_match_info = watcher.match.by_id(match_id=matchID, region=args.region)['info']
            current_match_timeline = watcher.match.timeline_by_match(region=args.region, match_id=matchID)['info']
            seasonId = get_season(current_match_info['gameVersion'])
            patch = get_patch(current_match_info['gameVersion'])
            current_match = SQLmatch(matchId=matchID,
                                     platformId=current_match_info['platformId'],
                                     gameId=current_match_info['gameId'],
                                     seasonId=seasonId,
                                     patch=patch,
                                     queueId=current_match_info['queueId'],
                                     gameVersion=current_match_info['gameVersion'],
                                     mapId=current_match_info['mapId'],
                                     gameDuration=current_match_info['gameDuration'],
                                     gameCreation=current_match_info['gameCreation'],
                                     )
            session.add(current_match)  # if performance is an issue, we can still use the core api, see here:
            # https://towardsdatascience.com/how-to-perform-bulk-inserts-with-sqlalchemy-efficiently-in-python-23044656b97d
            for participant in current_match_info['participants']:
                print(participant['challenges'])

                participant['platformId'] = current_match_info['platformId']
                participant['gameId'] = current_match_info['gameId']
                # TODO: perks table implementation
                curr_participantStats = SQLparticipantStats(**participant)
                session.add(curr_participantStats)
                participant['challenges']['puuid'] = participant['puuid']
                participant['challenges']['platformId'] = current_match_info['platformId']
                participant['challenges']['gameId'] = current_match_info['gameId']
                participant['challenges']['Assist12StreakCount'] = participant['challenges']['12AssistStreakCount']
                curr_participantChallenges = SQLChallenges(**participant['challenges'])
                session.add(curr_participantChallenges)
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
                logger.info(f"session commit")
                session.commit()
                # TODO: this should be handled differently, maybe with postgres ON INSERT.. DO NOTHING?
            except Exception as e:
                logger.error(e)
                logger.error(f"session rollback because something went wrong with parsing matchId {matchID}")
                session.rollback()


def parse_champion_data(session: sqlalchemy.orm.Session, watcher: LolWatcher):
    """ parses champion information provided by datadragon and fill corresponding Champion and ChampionStats tables
      WARNING: parses only the brief summary of champion data, if additional data is needed this needs to be reworked
    :param session: sqlalchemy session
    :param watcher: riotwatcher LolWatcher
    :returns: None
    """
    data = watcher.data_dragon.champions(version="13.17.1", full=False)['data']  # TODO: patch number as argument
    for champion in data:
        championdata = data[champion]
        championstats = championdata['stats']
        tags = pickle.dumps(championdata['tags'], protocol=pickle.HIGHEST_PROTOCOL)
        champion_obj = SQLChampion(championId=championdata['key'], championName=championdata['name'],
                                   championTitle=championdata['title'], infoAttack=championdata['info']['attack'],
                                   infoDefense=championdata['info']['defense'], infoMagic=championdata['info']['magic'],
                                   infoDifficulty=championdata['info']['difficulty'], tags=tags)
        session.add(champion_obj)
        championStats_obj = SQLChampionStats(championId=championdata['key'], hp=championstats['hp'],
                                             hpperlevel=championstats['hpperlevel'], mp=championstats['mp'],
                                             mpperlevel=championstats['mpperlevel'], movespeed=championstats['movespeed'],
                                             armor=championstats['armor'], armorperlevel=championstats['armorperlevel'],
                                             spellblock=championstats['spellblock'],
                                             spellblockperlevel=championstats['spellblockperlevel'],
                                             attackrange=championstats['attackrange'], hpregen=championstats['hpregen'],
                                             hpregenperlevel=championstats['hpregenperlevel'],
                                             mpregen=championstats['mpregen'],
                                             mpregenperlevel=championstats['mpregenperlevel'], crit=championstats['crit'],
                                             critperlevel=championstats['critperlevel'],
                                             attackdamage=championstats['attackdamage'],
                                             attackdamageperlevel=championstats['attackdamage'],
                                             attackspeed=championstats['attackspeed'])
        session.add(championStats_obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloading all match, player and champion data')
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

    api_key = keys.API_KEY_1
    logginglevel = getattr(logging, args.logginglevel.upper(), None)
    if not isinstance(logginglevel, int):
        raise ValueError('Invalid log level: %s' % args.logginglevel)
    file_handler = logging.FileHandler(filename=f'logging.log')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(encoding='utf-8', level=logginglevel,
                        format="%(asctime)s - %(levelname)s - %(funcName)s() - %(message)s", handlers=handlers)
    logger = logging.getLogger(__name__)
    logging.getLogger("sqlalchemy.engine").setLevel(logginglevel)
    logging.getLogger("riotwatcher.LolWatcher").setLevel(logginglevel)
    logger.info(f'starting getData.py with arguments {sys.argv}')
    getData()
