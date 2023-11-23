import logging

import pandas as pd
from sqlalchemy import func
from tqdm import tqdm
from joblib import Parallel, delayed

from src.dataHandling.cleanTimelineDataset import cleanTimelineDataset
from src.sqlstore import queries
from src.sqlstore.champion import SQLChampion
from src.sqlstore.db import get_session
from src.sqlstore.match import SQLMatch, SQLParticipant, SQLParticipantStats
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague, SQLChampionMastery
from src.sqlstore.timeline import SQLTimeline, SQLFrame, SQLParticipantFrame
from src.utils import separateMatchID


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


def process_match(match, session):
    try:
        logging.info(f"Processing match with ID: {match.matchId}")

        # Create a DataFrame for each match's data
        df_match = pd.DataFrame([match.get_training_data()])

        participants = session.query(SQLParticipant).filter(SQLParticipant.matchId == match.matchId).all()
        assert len(participants) == 10, "Match must have exactly 10 participants"

        participant_data_frames = []

        for i, participant in enumerate(participants):
            j = i + 1

            df_summoner = process_summoner(participant.puuid, session)
            df_champion, participant_stats = process_champion(participant.puuid, session)
            df_summonerLeague = process_summoner_league(participant.puuid, session)
            df_mastery = process_mastery(participant, participant_stats, session)

            # Renaming columns to include participant index
            for df in [df_summoner, df_champion, df_summonerLeague, df_mastery]:
                df.rename(columns=lambda x: f"participant{j}_" + x, inplace=True)

            participant_frame = pd.concat([df_summoner, df_champion, df_summonerLeague, df_mastery], axis=1)
            participant_data_frames.append(participant_frame)

        # Combine all participant data with the match data
        df_match = pd.concat([df_match] + participant_data_frames, axis=1)

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
        matches = session.query(SQLMatch).order_by(func.random()).limit(size).all()
        logging.info(f"Fetched {len(matches)} matches from the database.")

        # Use joblib to parallelize match processing
        #processed_data = Parallel(n_jobs=-1)(delayed(process_match)(match, session) for match in matches)
        # Filter out None results due to errors and concatenate DataFrames
        #data = pd.concat([df for df in processed_data if df is not None], axis=0, ignore_index=True)

        # List to hold DataFrame for each match
        match_dataframes = []

        for match in matches:
            match: SQLMatch
            # Process each match and append the result to the list
            df_match = process_match(match, session)
            if df_match is not None:
                match_dataframes.append(df_match)

        # Concatenate all match DataFrames into a single DataFrame
        data = pd.concat(match_dataframes, axis=0, ignore_index=True)

        logging.info(f"Successfully processed all matches, length of dataframe: {len(data)}")

        if save:
            data.to_pickle("data/raw/static_dataset.pkl")

    return data


def build_frame_dataset(size: int = None, save: bool = True):
    """
    Builds a dataset with all frame information (info available during match, but no events) from the database.
    :param size: Number of matches in the dataset, None if all matches should be used
    :param save: Whether to save the dataset to a pickle file in the data/raw folder
    :return: None
    """
    with get_session() as session:
        matches = session.query(SQLMatch).order_by(func.random()).limit(size).all()
        matchIds = []
        frameData = []
        # iterate over matches
        for match in tqdm(matches):
            platformId, gameId = separateMatchID(match.matchId)
            # calculate winning team
            participantId = session.query(SQLParticipant.id).filter(SQLParticipant.matchId == match.matchId,
                                                                    SQLParticipant.participantId == 1).scalar()
            winning_team = int(session.query(SQLParticipantStats.win
                                             ).filter(SQLParticipantStats.participantId == participantId
                                                      ).scalar())
            # get timeline and frames for that match
            timeline = session.query(SQLTimeline).filter_by(platformId=platformId, gameId=gameId).first()
            frames = session.query(SQLFrame).filter_by(timelineId=timeline.id).all()
            participantFrameData = []
            # iterate over frames
            for frame in frames:
                matchIds.append((match.matchId, frame.id))
                frameDict = {'timestamp': frame.timestamp, 'winning_team': winning_team}
                participantFrames = session.query(SQLParticipantFrame).filter_by(frameId=frame.id).all()
                for i, participantFrame in enumerate(participantFrames):
                    participantFrameTrainingData = participantFrame.get_training_data()
                    participantFrameData.append(participantFrameTrainingData)
                    participantFrameDict = {f'participant{i + 1}_{k}': v for k, v in
                                            participantFrameTrainingData.items()}
                    frameDict.update(participantFrameDict)
                frameData.append(frameDict)
        index = pd.MultiIndex.from_tuples(matchIds, names=['matchId', 'frameId'])
        print(f"len of framedata: {len(frameData)}")
        print(f"shape of index: {index.shape}")
        dfTimelines = pd.DataFrame(frameData, index=index)
        if save:
            dfTimelines.to_pickle('data/raw/timelines.pkl')


if __name__ == "__main__":
    build_static_dataset(None, True)
    build_frame_dataset(None, True)
    cleanTimelineDataset()
