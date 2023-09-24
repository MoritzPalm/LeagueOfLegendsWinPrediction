from typing import Tuple, Any

import sqlalchemy.orm
from sqlalchemy import select, Row
from sqlalchemy import func
from src.sqlstore.match import SQLMatch, SQLParticipant, SQLParticipantStats
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
            # wrapping the dict in a list to prevent missing index issues, this is hacky and may have unforeseen issues
            df_match = pd.DataFrame([match.__dict__])
            participants = session.query(SQLParticipant).filter(SQLParticipant.matchId == match.matchId).all()
            assert len(participants) == 10
            for i, participant in enumerate(participants):
                stat: SQLParticipantStats = session.query(SQLParticipantStats).filter(
                    SQLParticipantStats.participantId == participant.id).one()
                df_participant = pd.DataFrame([stat.__dict__])
                # renames all columns to have a participant and the number in front of the attribute
                df_participant.rename(columns=lambda x: f"participant{i}_" + x, inplace=True)
                summoner = session.query(SQLSummoner).filter(SQLSummoner.puuid == participant.puuid).one()
                df_summoner = pd.DataFrame([summoner.__dict__])
                df_summoner.rename(columns=lambda x: f"participant{i}_" + x, inplace=True)
                #mastery = session.query(SQLChampionMastery).filter(SQLChampionMastery.puuid == participant.puuid,
                                                                #SQLChampionMastery.championId == stat.championId).one()
                #df_mastery = pd.DataFrame([mastery.__dict__])
                #df_mastery.rename(columns=lambda x: f"participant{i}_" + x, inplace=True)
                #appending all dataframes to df_match
                df_match = pd.concat([df_match, df_participant, df_summoner], axis=1, copy=False)
                print(df_match.shape)
    print("test")

