import sys
import keys
import pickle
import logging
import argparse
import pytest

from riotwatcher import LolWatcher
from sqlalchemy import insert
from sqlalchemy.sql import exists
from sqlalchemy.exc import IntegrityError
import sqlalchemy.orm.session

from src.utils import get_patch, get_season
from src.crawlers.MatchIdCrawler import MatchIdCrawler
from src.sqlstore.db import get_session
from src.sqlstore.match import SQLmatch
from src.sqlstore.participant import SQLparticipantStats, SQLChallenges
from src.sqlstore.timeline import SQLTimeline, SQLEvent, SQLFrame, SQLParticipantFrame
from src.sqlstore.champion import SQLChampion, SQLChampionStats, SQLChampionRoles, SQLChampionTags
from src.sqlstore.utils import champ_patch_present


# TODO: abstract away more logic
def getData():
    if args.n == 0:
        args.n = sys.maxsize  # if no maximum number of matches or 0 passed, use maximum number of matches possible
    logger.info(f"initializing matchIdCrawler with api key {api_key}, region {args.region} and tier {args.tier}")
    crawler = MatchIdCrawler(api_key=api_key, region=args.region, tier=args.tier)
    logger.info(f"crawling {args.n} matchIDs")
    matchIDs: set[str] = crawler.getMatchIDs(n=args.n)
    logger.info(f"{len(matchIDs)} non-unique matchIDs crawled")
    watcher = LolWatcher(api_key)
    with get_session(cleanup=False) as session:

        for matchID in matchIDs:
            try:
                if session.query(exists().where(SQLmatch.matchId == matchID)).scalar():
                    logger.warning(f"matchID {matchID} already present in database")
                    continue
                logger.debug(f"getting match info for match {matchID}")
                current_match_info = watcher.match.by_id(match_id=matchID, region=args.region)['info']
                if not is_valid_match(current_match_info):
                    logger.warning(f"match {matchID} is not valid")
                    continue
                season = get_season(current_match_info['gameVersion'])
                patch = get_patch(current_match_info['gameVersion'])
                if not champ_patch_present(session=session, season=season, patch=patch):
                    parse_champion_data(session=session, watcher=watcher, season=season, patch=patch)
                current_match_timeline = watcher.match.timeline_by_match(region=args.region, match_id=matchID)['info']
                current_match = SQLmatch(matchId=matchID,
                                         platformId=current_match_info['platformId'],
                                         gameId=current_match_info['gameId'],
                                         queueId=current_match_info['queueId'],
                                         gameVersion=current_match_info['gameVersion'],
                                         mapId=current_match_info['mapId'],
                                         gameDuration=current_match_info['gameDuration'],
                                         gameCreation=current_match_info['gameCreation'],
                                         )
                session.add(current_match)  # if performance is an issue, we can still use the core api, see here:
                # https://towardsdatascience.com/how-to-perform-bulk-inserts-with-sqlalchemy-efficiently-in-python-23044656b97d
                parse_participant_data(session=session, platformId=current_match_info['platformId'],
                                       gameId=current_match_info['gameId'],
                                       participants=current_match_info['participants'])
                current_timeline = SQLTimeline(platformId=current_match_info['platformId'],
                                               gameId=current_match_info['gameId'],
                                               frameInterval=current_match_timeline['frameInterval'])
                session.add(current_timeline)
                for frameId, frame in enumerate(current_match_timeline['frames']):
                    frame_obj = SQLFrame(platformId=current_match_info['platformId'],
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
                        event_obj = SQLEvent(**event)
                        session.add(event_obj)
                    for i, participantFrame in enumerate(
                            current_match_timeline['frames'][frameId]['participantFrames'].items(), start=1):
                        participantFrameData = participantFrame[1]
                        participantFrameData['platformId'] = current_match_info['platformId']
                        participantFrameData['gameId'] = current_match_info['gameId']
                        participantFrameData['frameId'] = frameId
                        participantFrameData['participantId'] = i
                        participantFrame_obj = SQLParticipantFrame(**participantFrameData)
                        session.add(participantFrame_obj)
            except Exception as e:
                logger.warning(f"skipping match Id {matchID} because of the following error: ")
                logger.warning(str(e))
                continue
            try:
                session.flush()
                logger.info(f"session commit")
                session.commit()
                # TODO: this should be handled differently, maybe with postgres ON INSERT.. DO NOTHING?
            except Exception as e:  # TODO: catch narrow exception
                logger.error(str(e))
                logger.error(f"session rollback because something went wrong with parsing matchId {matchID}")
                session.rollback()
                raise


def parse_champion_data(session: sqlalchemy.orm.Session, watcher: LolWatcher, season: int, patch: int):
    """ parses champion information provided by datadragon and fill corresponding Champion and ChampionStats tables
      WARNING: parses only the brief summary of champion data, if additional data is needed this needs to be reworked
    :param session: sqlalchemy session
    :param watcher: riotwatcher LolWatcher
    :param season: season number
    :param patch: patch number
    :returns: None
    """
    data = watcher.data_dragon.champions(version=f"{season}.{patch}.1", full=False)['data']
    # the .1 is correct for modern patches, for very old patches (season 4 and older) another solution would be needed
    for champion in data:  # TODO: this can be vastly improved by using bulk inserts
        championdata = data[champion]
        print(championdata)
        tags = pickle.dumps(championdata['tags'], protocol=pickle.HIGHEST_PROTOCOL)
        champion_obj = SQLChampion(championNumber=int(championdata['key']), championName=championdata['name'],
                                   championTitle=championdata['title'], infoAttack=championdata['info']['attack'],
                                   infoDefense=championdata['info']['defense'], infoMagic=championdata['info']['magic'],
                                   infoDifficulty=championdata['info']['difficulty'], seasonNumber=season,
                                   patchNumber=patch)
        session.add(champion_obj)
        session.commit()  # this commit is needed to get the generated champion_obj id
        stats = data[champion]['stats']
        championStats_obj = SQLChampionStats(championId=champion_obj.id, hp=stats['hp'],
                                             hpperlevel=stats['hpperlevel'], mp=stats['mp'],
                                             mpperlevel=stats['mpperlevel'], movespeed=stats['movespeed'],
                                             armor=stats['armor'], armorperlevel=stats['armorperlevel'],
                                             spellblock=stats['spellblock'],
                                             spellblockperlevel=stats['spellblockperlevel'],
                                             attackrange=stats['attackrange'], hpregen=stats['hpregen'],
                                             hpregenperlevel=stats['hpregenperlevel'],
                                             mpregen=stats['mpregen'],
                                             mpregenperlevel=stats['mpregenperlevel'], crit=stats['crit'],
                                             critperlevel=stats['critperlevel'],
                                             attackdamage=stats['attackdamage'],
                                             attackdamageperlevel=stats['attackdamage'],
                                             attackspeed=stats['attackspeed'], patchNumber=patch,
                                             seasonNumber=season)
        session.add(championStats_obj)
        # TODO: add champion roles with data from webscraping
        tags = championdata['tags']
        championTags_obj = SQLChampionTags(champion_obj.id, tags)
        session.add(championTags_obj)
        session.commit()
    session.commit()


def parse_participant_data(session: sqlalchemy.orm.Session, platformId: str, gameId: int, participants: dict) -> None:
    """
    parses participant stats and adds it to sqlalchemy session
    :param session: sqlalchemy orm session
    :param platformId: platformId string (e.g. "EUW1")
    :param gameId: game Id int (e.g. 6572642807)
    :param participants: list of dicts containing participant stats
    :return: None
    """
    for participant in participants:
        participant['platformId'] = platformId
        participant['gameId'] = gameId
        # TODO: perks table implementation
        participantStats_obj = SQLparticipantStats(**participant)
        session.add(participantStats_obj)
        participant['challenges']['puuid'] = participant['puuid']
        participant['challenges']['platformId'] = platformId
        participant['challenges']['gameId'] = gameId
        participant['challenges']['Assist12StreakCount'] = participant['challenges']['12AssistStreakCount']
        curr_participantChallenges = SQLChallenges(**participant['challenges'])
        session.add(curr_participantChallenges)


def is_valid_match(match_info: dict) -> bool:
    logger.debug(f"validating match info")
    if match_info['gameDuration'] < 960:  # 16 min = 960 sec
        logger.warning(f"match is too short: match length was {match_info['gameDuration']}s, more than 960s expected")
        return False
    if match_info['queueId'] != 420:
        # queue ID for Ranked 5v5 solo, see: https://static.developer.riotgames.com/docs/lol/queues.json
        logger.warning(f"match has wrong queue: queue was {match_info['queue']}, 420 expected")
        return False
    if match_info['mapId'] not in [1, 2, 11]:
        # map ids for summoners rift, see https://static.developer.riotgames.com/docs/lol/maps.json
        logger.warning(f"match was played on wrong map: played on map {match_info['mapId']}, 1, 2 or 11 expected")
        return False
    return True


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
    parser.add_argument('-s', '--season', action='store', default=13, type=int, choices=[x for x in range(13)],
                        help='season from which matches get pulled', dest='season')
    parser.add_argument('-p', '--patch', action='store', default=17, type=int,
                        help='patch from which matches are pulled', dest='patch')

    args = parser.parse_args()

    api_key = keys.API_KEY_1
    logginglevel = getattr(logging, args.logginglevel.upper(), None)
    if not isinstance(logginglevel, int):
        raise ValueError('Invalid log level: %s' % args.logginglevel)
    file_handler = logging.FileHandler(filename=f'logging.log', mode='w')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(encoding='utf-8', level=logginglevel,
                        format="%(asctime)s - %(levelname)s - %(funcName)s() - %(message)s",
                        handlers=handlers)  # TODO: improve logging format
    logger = logging.getLogger(__name__)
    logging.getLogger("sqlalchemy.engine").setLevel(logginglevel)
    logging.getLogger("riotwatcher.LolWatcher").setLevel(logginglevel)
    logger.info(f'starting getData.py with arguments {sys.argv}')
    getData()
