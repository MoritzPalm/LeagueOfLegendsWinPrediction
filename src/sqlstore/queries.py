import logging
import datetime

from sqlalchemy import select, or_
from sqlalchemy.sql import exists
from sqlalchemy.orm import Session
import sqlalchemy.exc
from scrapy.item import Item

from src import sqlstore
from src.sqlstore.champion import SQLChampion
from src.sqlstore.summoner import SQLSummoner, SQLChampionMastery


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


def get_missing_masteries(session: sqlalchemy.orm.Session) -> list:
    """
    gets all summoner championmastery objects that have not yet been updated with scraped champion mastery data
    :param session:
    :return: list of SQLChampionMastery objects
    """
    query = select(SQLChampionMastery).filter(or_(SQLChampionMastery.championWinrate == None,
                                                  SQLChampionMastery.kda == None,
                                                  SQLChampionMastery.cs == None,
                                                  SQLChampionMastery.damage == None,
                                                  SQLChampionMastery.gold == None,
                                                  SQLChampionMastery.maxKills == None,
                                                  SQLChampionMastery.lp == None,
                                                  SQLChampionMastery.wins == None,
                                                  SQLChampionMastery.kills == None,
                                                  SQLChampionMastery.deaths == None,
                                                  SQLChampionMastery.assists == None,
                                                  ))
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
    summoner_query = select(SQLSummoner).filter(SQLSummoner.name == summonerName,
                                                SQLSummoner.platformId == region)
    # there should only be one summoner with this name in this region
    summoner = session.scalars(summoner_query).one()
    mastery_query = select(SQLChampionMastery).filter(SQLChampionMastery.puuid == summoner.puuid)
    masteries = session.scalars(mastery_query).all()
    # the only time multiple masteries should be found if we have multiple games by the summoner in the database
    for mastery in masteries:
        if mastery.champion.championName == championName:
            if any(x is None for x in mastery.__dict__.values()):
                return True
            else:
                return False
    # if no champion mastery could be found, something went wrong, but scraping is needed
    logging.error(
        f"no champion mastery found for summoner {summonerName} in region {region} and champion {championName}")
    return True


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
    query = session.query(SQLSummoner).filter(SQLSummoner.name == summonerName,
                                              SQLSummoner.platformId == region).update(scraped.__dict__)
    session.execute(query)
    print("test")
