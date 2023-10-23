import logging

import pandas as pd
from sqlalchemy import func
from tqdm import tqdm

from src.dataHandling.cleanTimelineDataset import cleanTimelineDataset
from src.sqlstore import queries
from src.sqlstore.champion import SQLChampion
from src.sqlstore.db import get_session
from src.sqlstore.match import SQLMatch, SQLParticipant, SQLParticipantStats
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague, SQLChampionMastery
from src.sqlstore.timeline import SQLTimeline, SQLFrame, SQLParticipantFrame
from src.utils import separateMatchID


def build_static_dataset(size: int = None, save: bool = True) -> pd.DataFrame:
    """
    Builds a dataset with all static information (info available prior to match start) from the database.
    :param save: Whether to save the dataset to a pickle file in the data/raw folder
    :param size: Number of matches in the dataset
    :return: DataFrame with a large number of columns (features)
    """
    with get_session() as session:
        # Initialize an empty DataFrame
        data = pd.DataFrame()

        # Fetch a specific number of random matches from the SQLMatch table
        matches = session.query(SQLMatch).order_by(func.random()).limit(size).all()
        logging.info(f"Fetched {len(matches)} matches from the database.")
        for match in tqdm(matches):  # TODO: parallelize with joblib
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
                    df_champion.rename(columns=lambda x: f"participant{j}_champion_" + x, inplace=True)
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
