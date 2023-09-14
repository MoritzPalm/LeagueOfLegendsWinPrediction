import logging
import datetime

from sqlalchemy import select
from sqlalchemy.sql import exists
from sqlalchemy.orm import Session
import sqlalchemy.exc

from src import sqlstore
from src.sqlstore.champion import SQLChampion
from src.sqlstore.summoner import SQLSummoner


def check_matchId_present(session: Session, matchID: str) -> bool:
    return session.query(exists().where(sqlstore.match.SQLMatch.matchId == matchID)).scalar()


def champ_patch_present(session: Session, season: int, patch: int) -> bool:
    """
    queries if the champion table has entries for the passed patch-season combination
    :param session: sqlalchemy session as connection to database, where a table "champion" needs to be present
    :param season: season number
    :param patch: patch number
    :return: True if champion data is already present in table, False otherwise or if no table has been found
    """
    try:
        present: bool = session.query(exists()
                                      .where(sqlstore.champion.SQLChampion.seasonNumber == season)
                                      .where(sqlstore.champion.SQLChampion.patchNumber == patch)).scalar()
    except sqlalchemy.exc.DatabaseError as e:  # TODO: test if this is correct exception
        logging.warning(e)
        return False
    return present


def check_summoner_present(session: sqlalchemy.orm.Session, puuid: str) -> bool:
    return session.query(exists().where(SQLSummoner.puuid == puuid)).scalar()


def check_summoner_data_recent(session: sqlalchemy.orm.Session, puuid: str, expiration_time: int) -> bool:
    """
    checks if the data in the database for the specified summoner is older than expiration_time
    :param session: sqlalchemy session
    :param puuid: encrypted player puuid
    :param expiration_time: time in days before the data gets updated
    :return: True if the data has not yet expired, False otherwise
    """
    delta = datetime.timedelta(days=expiration_time)
    today = datetime.date.today()
    query = session.query(select(SQLSummoner.lastUpdate).where(SQLSummoner.puuid == puuid))
    result = session.execute(query).one_or_none()
    lastUpdate: datetime.date = datetime.date.fromtimestamp(result.lastUpdate)
    if lastUpdate is None:  # has never been updated, need to get first creation time
        query = session.query(select(SQLSummoner.timeCreated).where(SQLSummoner.puuid == puuid))
        result = session.execute(query).one()
        lastUpdate: datetime.date = datetime.date.fromtimestamp(result.lastUpdate)
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
    query = select(SQLChampion).filter(SQLChampion.championNumber == championId).order_by(
        SQLChampion.lastUpdate)
    champion_obj = session.execute(query).scalar()
    return champion_obj
