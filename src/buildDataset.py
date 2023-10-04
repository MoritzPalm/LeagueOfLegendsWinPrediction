from typing import Tuple, Any
import pickle
import os

import sqlalchemy.orm
from sqlalchemy import select, Row
from sqlalchemy import func
from src.sqlstore.match import SQLMatch, SQLParticipant, SQLParticipantStats
from src.sqlstore.champion import SQLChampion
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague, SQLChampionMastery
from src.sqlstore.db import get_session
import pandas as pd
import numpy as np

import logging


def build_static_dataset(size: int) -> pd.DataFrame:
    """
    Builds a dataset with all static information (info available prior to match start) from the database.
    :param size: Number of matches in the dataset
    :return: DataFrame with a large number of columns (features)
    """
    with get_session() as session:
        # Initialize an empty DataFrame
        data = pd.DataFrame()

        # Fetch a specific number of random matches from the SQLMatch table
        matches = session.query(SQLMatch).order_by(func.random()).limit(size).all()

        for match in matches:
            try:
                logging.info(f"Processing match with ID: {match.matchId}")

                # Create a DataFrame for each match's data
                match: SQLMatch
                df_match = pd.DataFrame([match.get_training_data()])
                logging.info("Match data fetched.")

                # Query participants for the current match
                participants = session.query(SQLParticipant).filter(SQLParticipant.matchId == match.matchId).all()

                # Ensure there are exactly 10 participants in the match
                assert len(participants) == 10

                for i, participant in enumerate(participants):
                    logging.info(f"Processing participant {i + 1} with puuid: {participant.puuid}")
                    j = i + 1
                    # Fetch Summoner data
                    summoner = session.query(SQLSummoner).filter(SQLSummoner.puuid == participant.puuid).one()
                    df_summoner = pd.DataFrame([summoner.get_training_data()])
                    df_summoner.rename(columns=lambda x: f"participant{j}_" + x, inplace=True)
                    logging.info("Summoner data fetched.")

                    # Fetch championId from ParticipantStats
                    champion_id = session.query(SQLParticipantStats.championId).filter(
                        SQLParticipantStats.participantId == participant.id).scalar()
                    logging.info(f"Fetched champion ID: {champion_id}")

                    # Fetch Summoner League data
                    summonerLeague = session.query(SQLSummonerLeague).filter(
                        SQLSummonerLeague.puuid == participant.puuid).one()
                    df_summonerLeague = pd.DataFrame([summonerLeague.get_training_data()])
                    df_summonerLeague.rename(columns=lambda x: f"participant{j}_" + x, inplace=True)
                    logging.info("Summoner League data fetched.")

                    # Fetch Mastery data
                    mastery = session.query(SQLChampionMastery).filter(
                        SQLChampionMastery.puuid == participant.puuid,
                        SQLChampionMastery.championId == champion_id).scalar()
                    if mastery is None:
                        df_mastery = pd.DataFrame([np.nan])
                    else:
                        df_mastery = pd.DataFrame([mastery.get_training_data()])
                        df_mastery.rename(columns=lambda x: f"participant{j}_" + x, inplace=True)

                    # Concatenate Summoner, Summoner League, and Mastery data to the match DataFrame
                    df_match = pd.concat([df_match, df_summoner, df_summonerLeague, df_mastery], axis=1, copy=False)

                # Append this match's DataFrame to the overall DataFrame
                data = pd.concat([data, df_match], axis=0, copy=False)
            except Exception as e:
                logging.error(f"An error occurred when processing match with ID {match.matchId}: {e}")
                continue  # Skip match and continue with next match
            logging.info(f"Successfully processed match with ID: {match.matchId}")
    path = os.getcwd()
    data.to_pickle("data/static_dataset.pkl")
    return data
