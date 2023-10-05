import argparse
import logging
import sys

import sqlalchemy.orm.session
from riotwatcher import LolWatcher

import keys
from src.crawlers.MatchIdCrawler import MatchIdCrawler
from src.sqlstore.db import get_session
from src.sqlstore.match import SQLMatch
from src import utils
from src.sqlstore import queries
from src.parsers import champion, summoner, timeline, participant
from src.buildDataset import build_static_dataset
from src.parsers.summoner import scrape_champion_masteries


# TODO: review logging


def getData():
    if args.n == 0:
        args.n = (
            sys.maxsize
        )  # if no maximum number of matches or 0 passed, use maximum number of matches possible
    logger.info(
        f"initializing matchIdCrawler with api key {api_key}, region {args.region} and tier {args.tier}"
    )
    crawler = MatchIdCrawler(api_key=api_key, region=args.region, tier=args.tier)
    logger.info(f"crawling {args.n} matchIDs")
    matchIDs: set[str] = crawler.getMatchIDs(n=args.n)
    logger.info(f"{len(matchIDs)} non-unique matchIDs crawled")
    watcher = LolWatcher(api_key)
    counter = 0
    with get_session(cleanup=False) as session:
        for matchID in matchIDs:
            try:
                if queries.check_matchId_present(session, matchID):
                    logger.warning(f"matchID {matchID} already present in database")
                    continue
                logger.info(f"getting match info for match {matchID}")
                current_match_info = watcher.match.by_id(
                    match_id=matchID, region=args.region
                )["info"]
                if not utils.is_valid_match(current_match_info):
                    logger.warning(f"match {matchID} is not valid")
                    continue
                season = utils.get_season(current_match_info["gameVersion"])
                patch = utils.get_patch(current_match_info["gameVersion"])
                if not queries.champ_patch_present(
                        session=session, season=season, patch=patch
                ):
                    logger.info(
                        f"fetching champion data as no data from patch {season}.{patch} in database"
                    )
                    champion.parse_champion_data(session, watcher, season, patch)
                current_match_timeline = watcher.match.timeline_by_match(
                    region=args.region, match_id=matchID
                )["info"]
                parse_data(
                    session,
                    watcher,
                    matchID,
                    season,
                    patch,
                    current_match_info,
                    current_match_timeline,
                    args.region,
                )
                session.flush()
                logger.info("session commit")
                session.commit()
                counter += 1
            except Exception as e:  # skip match id if other errors were thrown
                logger.error(
                    f"skipping match Id {matchID} because of the following error: "
                )
                logger.error(str(e))
                session.rollback()
                raise
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
        session.add(
            current_match
        )  # if performance is an issue, we can still use the core api, see here:
        # https://towardsdatascience.com/how-to-perform-bulk-inserts-with-sqlalchemy-efficiently-in-python-23044656b97d
        for participant_data in match_info["participants"]:
            summoner.parse_summoner_data(
                session=session,
                watcher=watcher,
                region=region,
                championId=participant_data["championId"],
                puuid=participant_data["puuid"],
                expiration=14,
            )
            participant.parse_participant_data(
                session=session, match=current_match, participant=participant_data
            )

        timeline.parse_timeline_data(
            session=session,
            platformId=match_info["platformId"],
            gameId=match_info["gameId"],
            timeline=match_timeline,
        )
    except Exception:
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloading all match, player and champion data"
    )
    parser.add_argument(
        "-v",
        "--visited",
        action="store",
        default="",
        type=str,
        help="path to pickle file containing a set of visited and thus to be excluded match IDs",
        dest="visitedPath",
    )
    parser.add_argument(
        "-l",
        "--log",
        action="store",
        default="error",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="choosing the level of logging used",
        dest="logginglevel",
    )
    parser.add_argument(
        "-r",
        "--region",
        action="store",
        default="euw1",
        type=str,
        choices=[
            "br1",
            "eun1",
            "euw1",
            "jp1",
            "kr",
            "la1",
            "la2",
            "na1",
            "oc1",
            "ru",
            "tr1",
        ],
        help="region from which matches should be crawled",
        dest="region",
    )
    parser.add_argument(
        "-t",
        "--tier",
        action="store",
        default="challenger",
        type=str.upper,
        choices=[
            "CHALLENGER",
            "GRANDMASTER",
            "MASTER",
            "DIAMOND",
            "EMERALD",
            "PLATINUM",
            "GOLD",
            "SILVER",
            "BRONZE",
            "IRON",
        ],
        help="elo tier from which matches should be crawled",
        dest="tier",
    )
    parser.add_argument(
        "-n",
        action="store",
        default=0,
        type=int,
        help="number of matches to be crawled, 0 means that every available match will be crawled",
    )
    parser.add_argument(
        "-m",
        "--matches_per_id",
        action="store",
        default=15,
        type=int,
        help="number of matches to be crawled per id",
        dest="matches_per_id",
    )
    parser.add_argument(
        "-s",
        "--season",
        action="store",
        default=13,
        type=int,
        choices=[x for x in range(13)],
        help="season from which matches get pulled",
        dest="season",
    )
    parser.add_argument(
        "-p",
        "--patch",
        action="store",
        default=17,
        type=int,
        help="patch from which matches are pulled",
        dest="patch",
    )
    parser.add_argument(
        "-ll",
        "-otherloglevel",
        action="store",
        default="warning",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="choosing the level of logging used by imported code",
        dest="otherlogginglevel",
    )
    parser.add_argument(
        "-b",
        "--buildOnly",
        action="store_true",
        help="if flag is set, does not fetch new data",
        dest="buildOnly",
    )

    args = parser.parse_args()

    api_key = keys.API_KEY_1
    logginglevel = getattr(logging, args.logginglevel.upper(), None)
    otherlogginglevel = getattr(logging, args.otherlogginglevel.upper(), None)
    if not isinstance(logginglevel, int):
        raise ValueError("Invalid log level: %s" % args.logginglevel)
    file_handler = logging.FileHandler(filename=f"logging.log", mode="w")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        encoding="utf-8",
        level=logginglevel,
        format="%(asctime)s - %(levelname)s - %(funcName)s() - %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("sqlalchemy.engine").setLevel(otherlogginglevel)
    logging.getLogger("riotwatcher.LolWatcher").setLevel(otherlogginglevel)
    logger.info(f"starting getData.py with arguments {sys.argv}")
    if not args.buildOnly:
        getData()
    build_static_dataset(size=1000)
