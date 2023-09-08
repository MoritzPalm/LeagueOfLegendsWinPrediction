import logging

import sqlalchemy.exc
from sqlalchemy.orm import Session
from sqlalchemy import exists

from src.sqlstore.champion import SQLChampion

logger = logging.getLogger(__file__)


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
                                      .where(SQLChampion.seasonNumber == season)
                                      .where(SQLChampion.patchNumber == patch)).scalar()
    except sqlalchemy.exc.DatabaseError as e:  # TODO: test if this is correct exception
        logger.warning(e)
        return False
    return present

