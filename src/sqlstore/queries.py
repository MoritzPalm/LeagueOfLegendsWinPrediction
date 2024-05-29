import datetime
import logging

import sqlalchemy.exc
from scrapy.item import Item
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import exists

from src import sqlstore
from src.sqlstore.champion import SQLChampion
from src.sqlstore.match import SQLMatch, SQLParticipantStats
from src.sqlstore.summoner import SQLChampionMastery, SQLSummoner
from src.utils import clean_champion_name


def check_matchId_present(session: Session, matchID: str) -> bool:
    return session.query(
        exists().where(sqlstore.match.SQLMatch.matchId == matchID)
    ).scalar()


def champ_patch_present(session: Session, season: int, patch: int) -> bool:
    """
    queries if the champion table has entries for the passed patch-season combination
    :param session: sqlalchemy session as connection to database, where a table "champion" needs to be present
    :param season: season number
    :param patch: patch number
    :return: True if champion data is already present in table, False otherwise or if no table has been found
    """
    try:
        present: bool = session.query(
            exists()
            .where(sqlstore.champion.SQLChampion.seasonNumber == season)
            .where(sqlstore.champion.SQLChampion.patchNumber == patch)
        ).scalar()
    except sqlalchemy.exc.DatabaseError as e:
        logging.warning(e)
        return False
    return present


def check_summoner_present(session: sqlalchemy.orm.Session, puuid: str) -> bool:
    """
    checks if a summoner with the specified puuid is already present in the database
    :param session:
    :param puuid:
    :return:
    """
    return session.query(
        session.query(SQLSummoner).filter(SQLSummoner.puuid == puuid).exists()
    ).scalar()


def check_summoner_data_recent(
        session: sqlalchemy.orm.Session, puuid: str, expiration_time: int
) -> bool:
    """
    checks if the data in the database for the specified summoner is older than expiration_time
    :param session: sqlalchemy session
    :param puuid: encrypted player puuid
    :param expiration_time: time in days before the data gets updated
    :return: True if the data has not yet expired, False otherwise or if no summoner with this puuid could be found
    """
    delta = datetime.timedelta(days=expiration_time)
    today = datetime.datetime.now(datetime.timezone.utc)
    query = session.query(SQLSummoner).filter(SQLSummoner.puuid == puuid)
    result = session.execute(query).first()
    if result is None:
        return False
    if (
            result[0].lastUpdate is None
    ):  # has never been updated, need to get first creation time
        lastUpdate: datetime.date = result[0].timeCreated
    else:
        lastUpdate: datetime.date = result[0].lastUpdate
    timedelta = today - lastUpdate
    if timedelta < delta:
        return True
    return False


def get_last_champion(session: sqlalchemy.orm.Session, championId: int) -> SQLChampion:
    """
    gets most recent data from champion with id championId
    :param session: sqlalchemy session with access to champion table with columns championNumber, lastUpdate
    :param championId: number of the champion defined by riot games
    :return: sqlstore.champion.SQLChampion object
    """
    query = (
        select(SQLChampion)
        .filter(SQLChampion.championNumber == championId)
        .order_by(SQLChampion.lastUpdate)
    )
    champion_obj = session.execute(query).scalar()
    return champion_obj


def get_champ_name(session: sqlalchemy.orm.Session, championId: int) -> str:
    query = select(SQLChampion.championName).filter(
        SQLChampion.championNumber == championId
    )
    return session.execute(query).scalar()


def get_champ_id(session: sqlalchemy.orm.Session, championName: str, season: int, patch: int) -> int:
    """
    gets the champion id for the specified champion name from the most recent patch
    :param patch:
    :param season:
    :param session:
    :param championName:
    :return:
    """
    championName = clean_champion_name(championName)
    query = select(SQLChampion.id).filter(
        SQLChampion.championName == championName, SQLChampion.seasonNumber == season, SQLChampion.patchNumber == patch
    )
    return session.execute(query).scalar()


def get_champ_number_from_name(session: sqlalchemy.orm.session, championName: str):
    championName = clean_champion_name(championName)
    return session.query(SQLChampion.championNumber).filter(SQLChampion.championName == championName).limit(1).scalar()


def get_missing_masteries(session: sqlalchemy.orm.Session) -> list:
    """
    gets all summoner championmastery objects that have not yet been updated with scraped champion mastery data
    :param session:
    :return: list of SQLChampionMastery objects
    """
    query = select(SQLChampionMastery).filter(or_(
        SQLChampionMastery.kda == None,
        SQLChampionMastery.cs == None,
        SQLChampionMastery.damage == None,
        SQLChampionMastery.gold == None,
        SQLChampionMastery.maxKills == None,
        SQLChampionMastery.lp == None,
        SQLChampionMastery.kills == None,
        SQLChampionMastery.deaths == None,
        SQLChampionMastery.assists == None,
    ))
    return session.scalars(query).all()


def get_all_champIds_for_number(session: sqlalchemy.orm.Session, championNumber: int) -> list:
    """
    gets all champion ids for the specified champion number
    :param session:
    :param championNumber:
    :return:
    """
    query = select(SQLChampion.id).filter(SQLChampion.championNumber == championNumber)
    return session.scalars(query).all()


def scraping_needed(session: sqlalchemy.orm.Session, region: str, summonerName: str, championName: str) -> bool:
    """
    checks if the champion-summoner combination has already been scraped
    and if there is at least one match played by the summoner playing the champion
    :param session: sqlalchemy session
    :param region: region the summoner is playing in
    :param summonerName:
    :param championName:
    :return: True if scraping is needed, False otherwise
    """
    championName = clean_champion_name(championName)
    summoner_query = select(SQLSummoner).filter(SQLSummoner.name == summonerName,
                                                SQLSummoner.platformId == region)
    # there should only be one summoner with this name in this region
    try:
        summoner = session.scalars(summoner_query).one_or_none()
    except sqlalchemy.orm.exc.MultipleResultsFound:
        logging.error(f"multiple summoners with name {summonerName} found in region {region}")
        return False
    if summoner is None:
        logging.error(f"no summoner with name {summonerName} found in region {region}")
        return False
    # check if there is at least one match played by the summoner with champion championName
    match_exists: bool = session.query(exists().where(SQLParticipantStats.summonerId == summoner.summonerId,
                                                      SQLParticipantStats.championName == championName)).scalar()
    if not match_exists:  # no match played by summoner with champion championName found
        logging.info(f"no match found for summoner {summonerName} in region {region} with champion {championName}")
        return False  # thus no scraping needed
    championNumber = get_champ_number_from_name(session, championName)  # the number of the champion played
    champIds = get_all_champIds_for_number(session, championNumber)  # all champion Ids matching the champion number
    mastery_query = select(SQLChampionMastery).filter(SQLChampionMastery.puuid == summoner.puuid,
                                                      SQLChampionMastery.championId.in_(champIds))
    mastery = session.scalars(mastery_query).one_or_none()
    if mastery is None:
        logging.error(
            f"no champion mastery found for summoner {summonerName} in region {region} with champion {championName}")
        return False  # no champion mastery object found, something went wrong
    # the only time multiple masteries should be found if we have multiple games by the summoner in the database
    if any(x is None for x in mastery.__dict__.values()):
        return True  # at least one value is None, so scraping is needed
    return False


def update_mastery(session: sqlalchemy.orm.Session, scraped: Item, region: str, summonerName: str, championName: str):
    """
    updates the champion mastery data for the specified summoner and champion
    :param session:
    :param scraped:
    :param region:
    :param summonerName:
    :param championName:
    :return:
    """
    championName = clean_champion_name(championName)
    championNumber = get_champ_number_from_name(session, championName)
    championIds = get_all_champIds_for_number(session, championNumber)
    puuid = session.scalars(
        select(SQLSummoner.puuid).filter(SQLSummoner.name == summonerName, SQLSummoner.platformId == region)).one()
    mastery_query = select(SQLChampionMastery).filter(SQLChampionMastery.puuid == puuid,
                                                      SQLChampionMastery.championId.in_(championIds))
    mastery = session.scalars(mastery_query).one()
    for key, value in scraped.items():
        if key == "url" or key == "champion":
            continue
        if key == "wins" or key == "championWinrate":
            continue  # TODO: as soon as the scraping of these is working, comment this and the columns in the table back in
        if value == "N/A":
            setattr(mastery, key, None)
            continue
        setattr(mastery, key, value)
    session.commit()


def champion_mastery_present(session: sqlalchemy.orm.Session, puuid: str, championNumber: int) -> bool:
    """
    checks if a champion mastery object for the specified summoner and champion is already present in the database
    :param session:
    :param puuid:
    :param championNumber:
    :return:
    """
    championIds = get_all_champIds_for_number(session, championNumber)
    return session.query(
        session.query(SQLChampionMastery).filter(SQLChampionMastery.puuid == puuid,
                                                 SQLChampionMastery.championId.in_(championIds)).exists()
    ).scalar()


def get_all_matchIds(session: sqlalchemy.orm.Session, season: int, patch: int) -> set:
    """
    gets all matchIds from the database for given season and patch
    :param session:
    :param season:
    :param patch:
    :return:
    """
    query = select(SQLMatch.matchId).filter(
        SQLMatch.seasonId == season, SQLMatch.patch == patch)
    return set(session.scalars(query).all())
