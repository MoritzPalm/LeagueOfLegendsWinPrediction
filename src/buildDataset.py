from typing import Tuple, Any

import sqlalchemy.orm
from sqlalchemy import select, Row
from sqlalchemy import func
from src.sqlstore.match import SQLMatch, SQLParticipant
from src.sqlstore.champion import SQLChampion
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague, SQLChampionMastery
from src.sqlstore.db import get_session
import pandas as pd
import numpy as np


def build_dataset(size: int):
    """

    :param session:
    :param size: number of matches
    :return:
    """
    with get_session() as session:
        matches = session.query(SQLMatch).order_by(func.random()).limit(size).all()
        for match in matches:
            participants = session.query(SQLParticipant).filter(SQLParticipant.matchId == match.matchId).all()
            assert len(participants) == 10
            summoners = []
            for participant in participants:
                summoner = session.query(SQLSummoner).filter(SQLSummoner.puuid == participant.puuid).one()
                summoners.append(summoner)



