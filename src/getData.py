import argparse
import logging
import pickle
import sys
import time

import requests.exceptions
import sqlalchemy.orm.session
from riotwatcher import LolWatcher
from sqlalchemy.sql import exists

import keys
from src.crawlers.MatchIdCrawler import MatchIdCrawler
from src.sqlstore.champion import SQLChampion, SQLChampionStats, SQLChampionTags
from src.sqlstore.db import get_session
from src.sqlstore.match import SQLmatch
from src.sqlstore.participant import SQLParticipant, SQLparticipantStats, SQLChallenges, SQLStyle, SQLStyleSelection, \
    SQLStatPerk
from src.sqlstore.timeline import SQLTimeline, SQLEvent, SQLFrame, SQLParticipantFrame, SQLKillEvent, \
    SQLTimelineDamageDealt, SQLTimelineDamageReceived
from src.sqlstore.utils import champ_patch_present
from src.utils import get_patch, get_season


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
            for attempt in range(50):
                try:
                    if check_matchId_present(session, matchID):
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
                        pass
                        parse_champion_data(session=session, watcher=watcher, season=season, patch=patch)
                    current_match_timeline = watcher.match.timeline_by_match(region=args.region, match_id=matchID)['info']
                    parse_data(session, matchID, season, patch, current_match_info, current_match_timeline)
                except requests.exceptions.ConnectionError as e:
                    logger.error(str(e))
                    time.sleep(10)
                except Exception as e:
                    logger.error(f"skipping match Id {matchID} because of the following error: ")
                    logger.error(str(e))
                    break
                else:
                    break

            try:
                session.flush()
                logger.info(f"session commit")
                session.commit()
                # TODO: this should be handled differently, maybe with postgres ON INSERT.. DO NOTHING?

            except Exception as e:  # TODO: catch narrow exception
                logger.error(str(e))
                logger.error(f"session rollback because something went wrong with parsing matchId {matchID}")
                session.rollback()
                continue


def check_matchId_present(session: sqlalchemy.orm.Session, matchID: str) -> bool:
    return session.query(exists().where(SQLmatch.matchId == matchID)).scalar()


def parse_data(session: sqlalchemy.orm.Session, matchID: str, season: int, patch: int, match_info: dict, match_timeline: dict) -> None:
    current_match = SQLmatch(matchId=matchID,
                             platformId=match_info['platformId'],
                             gameId=match_info['gameId'],
                             queueId=match_info['queueId'],
                             gameVersion=match_info['gameVersion'],
                             mapId=match_info['mapId'],
                             gameDuration=match_info['gameDuration'],
                             gameCreation=match_info['gameCreation'],
                             )
    session.add(current_match)  # if performance is an issue, we can still use the core api, see here:
    # https://towardsdatascience.com/how-to-perform-bulk-inserts-with-sqlalchemy-efficiently-in-python-23044656b97d
    parse_participant_data(session=session, match=current_match, participants=match_info['participants'])
    parse_timeline_data(session=session, platformId=match_info['platformId'],
                        gameId=match_info['gameId'], timeline=match_timeline)


def parse_timeline_data(session: sqlalchemy.orm.Session, platformId: str, gameId: int, timeline: dict):
    current_timeline = SQLTimeline(platformId=platformId,
                                   gameId=gameId,
                                   frameInterval=timeline['frameInterval'])
    session.add(current_timeline)
    for frameId, frame in enumerate(timeline['frames']):
        frame_obj = SQLFrame(platformId=platformId,
                             gameId=gameId,
                             frameId=frameId,
                             timestamp=timeline['frames'][frameId]['timestamp'])
        current_timeline.frames.append(frame_obj)
        session.add(frame_obj)
        for eventId, event in enumerate(timeline['frames'][frameId]['events']):
            if event['type'] in {'CHAMPION_KILL', 'CHAMPION_SPECIAL_KILL', 'TURRET_PLATE_DESTROYED',
                                 'BUILDING_KILL'}:
                assistingParticipantIds = pickle.dumps(event.get('assistingParticipantIds'),
                                                       protocol=pickle.HIGHEST_PROTOCOL)
                event_obj = SQLKillEvent(assistingParticipantIds=assistingParticipantIds,
                                         bounty=event.get('bounty'),
                                         killStreakLength=event.get('killStreakLength'),
                                         killerId=event.get('killerId'),
                                         laneType=event.get('laneType'),
                                         position=event.get('position'),
                                         shutdownBounty=event.get('shutdownBounty'),
                                         timestamp=event.get('timestamp'),
                                         type=event.get('type'),
                                         victimId=event.get('victimId')
                                         )
                dmgDealt = event.get('victimDamageDealt')
                if dmgDealt is not None:
                    for dmg in dmgDealt:
                        dmgDealt_obj = SQLTimelineDamageDealt(basic=dmg.get('basic'),
                                                              magicDamage=dmg.get('magicDamage'),
                                                              name=dmg.get('name'),
                                                              participantId=dmg.get('participantId'),
                                                              physicalDamage=dmg.get('physicalDamage'),
                                                              spellName=dmg.get('spellName'),
                                                              spellSlot=dmg.get('SpellSlot'),
                                                              trueDamage=dmg.get('trueDamage'),
                                                              type=dmg.get('type')
                                                              )
                        event_obj.dmgdealt.append(dmgDealt_obj)
                dmgReceived = event.get('victimDamageReceived')
                if dmgReceived is not None:
                    for dmg in dmgReceived:
                        dmgReceived_obj = SQLTimelineDamageReceived(
                            basic=dmg.get('basic'),
                            magicDamage=dmg.get('magicDamage'),
                            name=dmg.get('name'),
                            participantId=dmg.get('participantId'),
                            physicalDamage=dmg.get('physicalDamage'),
                            spellName=dmg.get('spellName'),
                            spellSlot=dmg.get('SpellSlot'),
                            trueDamage=dmg.get('trueDamage'),
                            type=dmg.get('type')
                        )
                        event_obj.dmgreceived.append(dmgReceived_obj)
                frame_obj.killevents.append(event_obj)
            else:
                event_obj = SQLEvent(eventId=eventId,
                                     timestamp=event.get('timestamp'),
                                     type=event.get('type'),
                                     participantId=event.get('participantId'),
                                     itemId=event.get('itemId'),
                                     skillSlot=event.get('skillSlot'),
                                     creatorId=event.get('creatorId'),
                                     teamId=event.get('teamId'),
                                     afterId=event.get('afterId'),
                                     beforeId=event.get('beforeId'),
                                     wardType=event.get('wardType')
                                     )
                frame_obj.events.append(event_obj)
            session.add(event_obj)
        for i, participantFrame in enumerate(
                timeline['frames'][frameId]['participantFrames'].items(), start=1):
            participantFrameData = participantFrame[1]
            participantFrameData['platformId'] = platformId
            participantFrameData['gameId'] = gameId
            participantFrameData['frameId'] = frameId
            participantFrameData['participantId'] = i
            participantFrame_obj = SQLParticipantFrame(**participantFrameData)
            frame_obj.participantframe.append(participantFrame_obj)
            session.add(participantFrame_obj)


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
        champion_obj.stats.append(championStats_obj)
        session.add(championStats_obj)
        # TODO: add champion roles with data from webscraping
        tags = championdata.get('tags')
        length = len(tags)
        tag1 = tags[0] if 0 < length else None
        tag2 = tags[1] if 1 < length else None
        championTags_obj = SQLChampionTags(champion_obj.id, tag1, tag2)
        champion_obj.tags.append(championTags_obj)
        session.add(championTags_obj)
        session.commit()
    session.commit()


def parse_participant_data(session: sqlalchemy.orm.Session, match: SQLmatch, participants: dict) -> None:
    """
    parses participant stats and adds it to sqlalchemy session
    :param session: sqlalchemy orm session
    :param match: SQLMatch to get id for foreign key
    :param participants: list of dicts containing participant stats
    :return: None
    """
    for participant in participants:
        participant_obj = SQLParticipant(puuid=participant['puuid'], participantId=participant['participantId'])
        session.add(participant_obj)
        participantStats_obj = SQLparticipantStats(**participant)
        match.participant.append(participant_obj)   # TODO: double check logic regarding adding match to session
        participant_obj.stats.append(participantStats_obj)
        session.add(participantStats_obj)
        statPerks = participant['perks']['statPerks']
        participantPerk_obj = SQLStatPerk(participant['puuid'], statPerks['defense'], statPerks['flex'],
                                          statPerks['offense'])
        participant_obj.statPerks.append(participantPerk_obj)
        session.add(participantPerk_obj)
        styles = participant['perks']['styles']
        participantStyle_obj = SQLStyle(styles['description'], styles['style'])
        participant_obj.styles.append(participantStyle_obj)
        for selection in styles['selections']:
            participantStyleSelection_obj = SQLStyleSelection(selection['perk'], selection['var1'], selection['var2'],
                                                              selection['var3'])
            participantStyle_obj.selection.append(participantStyleSelection_obj)
            session.add(participantStyleSelection_obj)
        participant['challenges']['Assist12StreakCount'] = participant['challenges']['12AssistStreakCount']     # rename
        participantChallenges_obj = SQLChallenges(**participant['challenges'])
        participant_obj.challenges.append(participantChallenges_obj)
        session.add(participantChallenges_obj)


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
