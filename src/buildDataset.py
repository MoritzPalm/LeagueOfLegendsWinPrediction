import logging
import os
import time

import pandas as pd
from joblib import Parallel, delayed, wrap_non_picklable_objects
from sqlalchemy import func

from src.sqlstore import queries
from src.sqlstore.champion import SQLChampion
from src.sqlstore.db import get_session
from src.sqlstore.match import SQLMatch, SQLParticipant, SQLParticipantStats
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague, SQLChampionMastery


def process_summoner(participant_puuid, session):
    """
    Processes a summoner and returns a DataFrame with the summoner's data
    :param participant_puuid: puuid of the participant
    :param session: sqlalchemy session
    :return: DataFrame with summoner data
    """
    summoner = session.query(SQLSummoner).filter(SQLSummoner.puuid == participant_puuid).one()
    df_summoner = pd.DataFrame([summoner.get_training_data()])
    return df_summoner


def process_champion(participant_id, session):
    """
    Processes a champion and returns a DataFrame with the champion's data and the participant's stats
    :param participant_id: id of the participant
    :param session: sqlalchemy session
    :return: DataFrame with champion data, DataFrame with participant stats
    """
    participant_stats = session.query(SQLParticipantStats).filter(
        SQLParticipantStats.participantId == participant_id).scalar()
    champion = session.query(SQLChampion).filter(SQLChampion.id == participant_stats.championId).scalar()
    df_champion = pd.DataFrame([champion.get_training_data()])
    return df_champion, participant_stats


def process_summoner_league(participant_puuid, session):
    """
    Processes a summoner league and returns a DataFrame with the summoner league's data
    :param participant_puuid:
    :param session:
    :return:
    """
    summonerLeague = session.query(SQLSummonerLeague).filter(
        SQLSummonerLeague.puuid == participant_puuid).one()
    df_summonerLeague = pd.DataFrame([summonerLeague.get_training_data()])
    return df_summonerLeague


def process_mastery(participant, participant_stats, session):
    """
    Processes a champion mastery and returns a DataFrame with the champion mastery's data
    :param participant:
    :param participant_stats:
    :param session: sqlalchemy session
    :return:
    """
    championNumber = queries.get_champ_number_from_name(session, participant_stats.championName)
    championIds = queries.get_all_champIds_for_number(session, championNumber)
    mastery = session.query(SQLChampionMastery).filter(
        SQLChampionMastery.puuid == participant.puuid,
        SQLChampionMastery.championId.in_(championIds)).limit(1).scalar()
    if mastery:
        df_mastery = pd.DataFrame([mastery.get_training_data()])
    else:
        df_mastery = pd.DataFrame([{}])  # Empty DataFrame with same columns
    return df_mastery


@wrap_non_picklable_objects
def process_match(match, save_path: str, save: bool = True):
    """
    Processes a match and returns a DataFrame with the match's data
    :param match:
    :param session:
    :param save_path: Path to save the match DataFrame to
    :param save: Whether to save the match DataFrame to a pickle file
    :return:
    """
    try:
        with get_session() as session:
            logging.info(f"Processing match with ID: {match.matchId}")

            # Create a DataFrame for each match's data
            df_match = pd.DataFrame([match.get_training_data()])

            participants = session.query(SQLParticipant).filter(SQLParticipant.matchId == match.matchId).all()
            assert len(participants) == 10, "Match must have exactly 10 participants"

            participant_data_frames = []

            for i, participant in enumerate(participants):
                j = i + 1

                df_summoner = process_summoner(participant.puuid, session)
                df_champion, participant_stats = process_champion(participant.id, session)
                df_summonerLeague = process_summoner_league(participant.puuid, session)
                df_mastery = process_mastery(participant, participant_stats, session)

                win = pd.Series([participant_stats.win], name=f"participant{j}_win")
                teamId = pd.Series([participant_stats.teamId], name=f"participant{j}_teamId")

                # Renaming columns to include participant index
                for df in [df_summoner, df_champion, df_summonerLeague, df_mastery]:
                    df.rename(columns=lambda x: f"participant{j}_" + x, inplace=True)

                participant_frame = pd.concat([df_summoner, df_champion,
                                               df_summonerLeague, df_mastery,
                                               teamId, win], axis=1)
                participant_data_frames.append(participant_frame)

            # Combine all participant data with the match data
            df_match = pd.concat([df_match] + participant_data_frames, axis=1)
            if save:
                match_filename = os.path.join(save_path, f"match_{match.matchId}.pkl")
                df_match.to_pickle(match_filename)
            logging.info(f"Successfully processed match with ID: {match.matchId}")
            return df_match

    except Exception as e:
        logging.error(f"An error occurred when processing match with ID {match.matchId}: {e}")
        return None


def build_static_dataset(size: int = None, save: bool = True) -> pd.DataFrame:
    """
    Builds a dataset with all static information (info available prior to match start) from the database.
    :param save: Whether to save the dataset to a pickle file in the data/raw folder
    :param size: Number of matches in the dataset
    :return: DataFrame with a large number of columns (features)
    """
    with get_session() as session:
        matches = session.query(SQLMatch).where(SQLMatch.patch == 20).order_by(func.random()).limit(size).all()
        logging.info(f"Fetched {len(matches)} matches from the database.")

    # Use joblib to parallelize match processing
    processed_data = Parallel(n_jobs=20, prefer='threads', verbose=10)(delayed(process_match)(match, 'data/raw/',
                                                                                              True) for
                                                                       match in
                                                                       matches)
    # Filter out None results due to errors and concatenate DataFrames
    # data = pd.concat([df for df in processed_data if df is not None], axis=0, ignore_index=True)

    # logging.info(f"Successfully processed all matches, length of dataframe: {len(data)}")

    # return data


if __name__ == "__main__":
    # cProfile.run('build_static_dataset(1, False)')
    start = time.time()
    build_static_dataset(None, True)
    end = time.time()
    print(f"Time elapsed: {end - start}")
