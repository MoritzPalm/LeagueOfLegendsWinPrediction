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


def build_static_dataset(size: int) -> pd.DataFrame:
    """
    builds dataset with all static information (info available prior to match start) from database
    :param size: number of matches in dataset
    :return: Dataframe with large number of columns (features)
    """
    with get_session() as session:
        data = pd.DataFrame()
        matches = session.query(SQLMatch).order_by(func.random()).limit(size).all()
        for match in matches:
            match: SQLMatch
            # wrapping the dict in a list to prevent missing index issues, this is hacky and may have unforeseen issues
            df_match = pd.DataFrame([match.get_training_data()])
            participants = session.query(SQLParticipant).filter(SQLParticipant.matchId == match.matchId).all()
            assert len(participants) == 10
            for i, participant in enumerate(participants):
                summoner: SQLSummoner = session.query(SQLSummoner).filter(SQLSummoner.puuid == participant.puuid).one()
                df_summoner = pd.DataFrame([summoner.get_training_data()])
                # renames all columns to have a participant and the number in front of the attribute
                df_summoner.rename(columns=lambda x: f"participant{i}_" + x, inplace=True)
                summonerLeague: SQLSummonerLeague = session.query(SQLSummonerLeague).filter(
                    SQLSummonerLeague.puuid == participant.puuid).one()
                df_summonerLeague = pd.DataFrame([summonerLeague.get_training_data()])
                df_summonerLeague.rename(columns=lambda x: f"participant{i}_" + x, inplace=True)
                # mastery: SQLChampionMastery = session.query(SQLChampionMastery).filter(SQLChampionMastery.puuid == participant.puuid,
                # SQLChampionMastery.championId == stat.championId).one()
                # df_mastery = pd.DataFrame([mastery.get_training_data])
                # df_mastery.rename(columns=lambda x: f"participant{i}_" + x, inplace=True)
                # appending all dataframes to df_match
                df_match = pd.concat([df_match, df_summoner, df_summonerLeague], axis=1, copy=False)
            data = pd.concat([data, df_match], axis=0, copy=False)
    return data
