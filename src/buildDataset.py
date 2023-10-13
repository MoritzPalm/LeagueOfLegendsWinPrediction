import logging

import pandas as pd
from sqlalchemy import func
from tqdm import tqdm

from src.sqlstore import queries
from src.sqlstore.champion import SQLChampion
from src.sqlstore.db import get_session
from src.sqlstore.match import SQLMatch, SQLParticipant, SQLParticipantStats
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague, SQLChampionMastery


def build_static_dataset(size: int, save: bool) -> pd.DataFrame:
    """
    Builds a dataset with all static information (info available prior to match start) from the database.
    :param save:
    :param size: Number of matches in the dataset
    :return: DataFrame with a large number of columns (features)
    """
    with get_session() as session:
        # Initialize an empty DataFrame
        data = pd.DataFrame()

        # Fetch a specific number of random matches from the SQLMatch table
        matches = session.query(SQLMatch).order_by(func.random()).limit(size).all()
        logging.info(f"Fetched {len(matches)} matches from the database.")
        for i, match in enumerate(tqdm(matches)):  # TODO: parallelize with joblib
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
                    participant_stats = session.query(SQLParticipantStats).filter(
                        SQLParticipantStats.participantId == participant.id).scalar()
                    champion_id: int = participant_stats.championId
                    champion: SQLChampion = session.query(SQLChampion).filter(SQLChampion.id == champion_id).scalar()
                    df_champion = pd.DataFrame([champion.get_training_data()])
                    win: bool = participant_stats.win
                    se_win = pd.Series([win], name=f"participant{j}_win")
                    teamId = participant_stats.teamId
                    se_teamId = pd.Series([teamId], name=f"participant{j}_teamId")
                    logging.info(f"Fetched champion ID: {champion_id}")

                    # Fetch Summoner League data
                    summonerLeague = session.query(SQLSummonerLeague).filter(
                        SQLSummonerLeague.puuid == participant.puuid).one()
                    df_summonerLeague = pd.DataFrame([summonerLeague.get_training_data()])
                    df_summonerLeague.rename(columns=lambda x: f"participant{j}_" + x, inplace=True)
                    logging.info("Summoner League data fetched.")

                    # Fetch Mastery data
                    championNumber = queries.get_champ_number_from_name(session, participant_stats.championName)
                    championIds = queries.get_all_champIds_for_number(session, championNumber)
                    mastery = session.query(SQLChampionMastery).filter(
                        SQLChampionMastery.puuid == participant.puuid,
                        SQLChampionMastery.championId.in_(championIds)).limit(1).scalar()
                    if mastery is None:
                        logging.error(
                            f"No mastery data found for participant {j} with puuid: {participant.puuid} and championId: {champion_id}")
                        # df_mastery = pd.DataFrame([np.nan])
                        df_mastery = None
                    else:
                        df_mastery = pd.DataFrame([mastery.get_training_data()])
                        df_mastery.rename(columns=lambda x: f"participant{j}_champion_" + x, inplace=True)

                    # Concatenate Summoner, Summoner League, and Mastery data to the match DataFrame
                    df_match = pd.concat([df_match, df_summoner, df_summonerLeague, df_mastery, df_champion,
                                          se_teamId, se_win], axis=1, copy=False)

                # Append this match's DataFrame to the overall DataFrame
                data = pd.concat([data, df_match], axis=0, copy=False)
            except Exception as e:
                logging.error(f"An error occurred when processing match with ID {match.matchId}: {e}")
                raise  # Skip match and continue with next match
            logging.info(f"Successfully processed match with ID: {match.matchId}")
    logging.info(f"Successfully processed all matches, length of dataframe: {len(data)}")
    if save:
        data.to_pickle("data/static_dataset.pkl")
    return data


def build_timeline_dataset(n: int) -> pd.DataFrame:
    """
    Builds a dataset with all timeline information (info available after match start) from the database.
    :param n: number of matches in the dataset
    :return:
    """
    df_static = build_static_dataset(n, save=False)
    with get_session() as session:
        pass
