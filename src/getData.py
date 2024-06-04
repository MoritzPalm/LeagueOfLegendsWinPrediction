import argparse
import logging
import logging.handlers
import sys

import sqlalchemy
import sqlalchemy.orm.session
from riotwatcher import LolWatcher
from tqdm import tqdm

import keys
from src import utils
from src.crawlers.MatchIdCrawler import MatchIdCrawler
from src.parsers import champion, summoner, timeline, participant
from src.parsers.summoner import scrape_champion_masteries
from src.sqlstore import queries
from src.sqlstore.db import get_session
from src.sqlstore.match import SQLMatch
from src.sqlstore.queries import get_all_matchIds

logger = logging.getLogger(__name__)


def setup_logging(internal_log_level: str, external_log_level: str) -> None:
    internal_level = getattr(logging, internal_log_level.upper(), None)
    external_level = getattr(logging, external_log_level.upper(), None)
    if not isinstance(internal_level, int):
        raise ValueError(f"Invalid internal log level: {internal_log_level}")

    logging.getLogger("sqlalchemy.engine").setLevel(external_level)
    logging.getLogger("riotwatcher.LolWatcher").setLevel(external_level)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename="logs/logging.log", when="midnight", backupCount=7
    )
    stdout_handler = logging.StreamHandler(stream=sys.stdout)

    logging.basicConfig(
        encoding="utf-8",
        level=internal_level,
        format="%(asctime)s - %(levelname)s - %(funcName)s() - %(message)s",
        handlers=[file_handler, stdout_handler],
    )


def get_match_data(match_id: str, watcher: LolWatcher, region: str) -> dict | None:
    try:
        match_info = watcher.match.by_id(region=region, match_id=match_id)["info"]
        if utils.is_valid_match(match_info):
            return match_info
        else:
            logger.warning(f"Match {match_id} is not valid")
            return
    except Exception as e:
        logger.error(f"Error fetching match {match_id}: {e}")
        return


def getData(arguments: argparse.Namespace) -> None:
    if arguments.n == 0 or not arguments.n:
        arguments.n = sys.maxsize  # if no maximum number of matches or 0 passed,
        # use maximum number of matches possible

    logger.info("Pulling all match IDs already present in the database")
    with get_session() as session:
        present_matchIDs: set = get_all_matchIds(session=session, patch=arguments.patch,
                                                 season=arguments.season)
        logger.info(f"Present match IDs: {len(present_matchIDs)}")

    logger.info(
        f"Initializing MatchIdCrawler with API key {api_key}, "
        f"region {arguments.region} and tier {arguments.tier}")
    crawler = MatchIdCrawler(api_key=api_key,
                             region=arguments.region,
                             tier=arguments.tier,
                             patch=arguments.patch,
                             season=arguments.season,
                             known_matchIDs=present_matchIDs)
    logger.info(f"Crawling {arguments.n} match IDs")
    matchIDs: set[str] = crawler.getMatchIDs(n=arguments.n)
    logger.info(f"{len(matchIDs)} non-unique match IDs crawled")
    watcher = LolWatcher(api_key)

    with get_session() as session:
        for matchID in tqdm(matchIDs):
            try:
                if queries.check_matchId_present(session, matchID): # TODO: test if this is necessary, as the matchIDs are already filtered
                    logger.warning(f"Match ID {matchID} already present in database")
                    continue

                match_info = get_match_data(matchID, watcher, arguments.region)
                if not match_info:
                    continue

                season = utils.get_season(match_info["gameVersion"])
                patch = utils.get_patch(match_info["gameVersion"])

                if not queries.champ_patch_present(session=session,
                                                   season=season,
                                                   patch=patch):
                    logger.info(f"Fetching champion data as no "
                                f"data from patch {season}.{patch} in database")
                    champion.parse_champion_data(session, watcher, season, patch)

                current_match_timeline = \
                    watcher.match.timeline_by_match(region=args.region,
                                                    match_id=matchID)["info"]
                parse_data(session, watcher, matchID, season, patch, match_info,
                           current_match_timeline, args.region)
                session.flush()
                logger.info("Session commit")
                session.commit()
            except Exception as e:
                logger.error(
                    f"Skipping match ID {matchID} because of the following error:")
                logger.error(str(e))
                session.rollback()
                continue

        if not args.noScraping:
            scrape_champion_masteries(session)


def parse_data(
        session: sqlalchemy.orm.Session,
        watcher: LolWatcher,
        matchID: str,
        season: int,
        patch: int,
        match_info: dict,
        match_timeline: dict,
        region: str,
) -> None:
    try:
        current_match = SQLMatch(
            matchId=matchID,
            platformId=match_info["platformId"],
            gameId=match_info["gameId"],
            queueId=match_info["queueId"],
            gameVersion=match_info["gameVersion"],
            mapId=match_info["mapId"],
            gameDuration=match_info["gameDuration"],
            gameCreation=match_info["gameCreation"],
        )
        session.add(current_match)

        for participant_data in match_info["participants"]:
            summoner.parse_summoner_data(
                session=session,
                watcher=watcher,
                region=region,
                championId=participant_data["championId"],
                puuid=participant_data["puuid"],
                expiration=14,
            )
            participant.parse_participant_data(session=session, match=current_match,
                                               participant=participant_data)

        timeline.parse_timeline_data(
            session=session,
            platformId=match_info["platformId"],
            gameId=match_info["gameId"],
            timeline=match_timeline,
        )
    except Exception as e:
        logger.error(f"Error parsing data for match {matchID}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloading all match, player and champion data")

    parser.add_argument("-v", "--visited", action="store", default="", type=str,
                        help="Path to pickle file containing "
                             "a set of visited and thus to be excluded match IDs",
                        dest="visitedPath")

    parser.add_argument("-l", "--log", action="store", default="error",
                        type=lambda s: s.lower(),
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Choosing the level of logging used", dest="logginglevel")

    parser.add_argument("-r", "--region", action="store", default="euw1", type=str,
                        choices=["br1", "eun1", "euw1", "jp1", "kr", "la1", "la2",
                                 "na1", "oc1", "ru", "tr1"],
                        help="Region from which matches should be crawled",
                        dest="region")

    parser.add_argument("-t", "--tier", action="store", default="challenger",
                        type=lambda s: s.lower(),
                        choices=["challenger", "grandmaster", "master", "diamond",
                                 "emerald", "platinum", "gold",
                                 "silver", "bronze", "iron"],
                        help="Elo tier from which matches should be crawled",
                        dest="tier")

    parser.add_argument("-n", action="store", default=0, type=int,
                        help="Number of matches to be crawled, "
                             "0 means that every available match will be crawled")

    parser.add_argument("-m", "--matches_per_id", action="store", default=15, type=int,
                        help="Number of matches to be crawled per id",
                        dest="matches_per_id")

    parser.add_argument("-s", "--season", action="store", default=13, type=int,
                        choices=[x for x in range(13)],
                        help="Season from which matches get pulled", dest="season")

    parser.add_argument("-p", "--patch", action="store", default=17, type=int,
                        help="Patch from which matches are pulled", dest="patch")

    parser.add_argument("-ll", "--otherloglevel", action="store", default="warning",
                        type=lambda s: s.lower(),
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Choosing the level of logging used by imported code",
                        dest="otherlogginglevel")

    parser.add_argument("-b", "--buildOnly", action="store_true",
                        help="If flag is set, does not fetch new data",
                        dest="buildOnly")

    parser.add_argument("--noScraping", action="store_true",
                        help="If flag is set, does not scrape u.gg for champion data",
                        dest="noScraping")

    args = parser.parse_args()

    api_key = keys.API_KEY_1

    setup_logging(args.logginglevel, args.otherlogginglevel)
    logger.info(f"Starting getData.py with arguments {sys.argv}")

    getData(args)
