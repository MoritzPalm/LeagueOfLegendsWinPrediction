import logging
import pickle
import time

import pandas as pd
from joblib import wrap_non_picklable_objects
from sqlalchemy import func

from src.sqlstore.db import get_session
from src.sqlstore.match import SQLMatch, SQLParticipant, SQLParticipantStats
from src.sqlstore.timeline import SQLTimeline, SQLFrame, SQLParticipantFrame, SQLKillEvent
from src.utils import separateMatchID, get_teamId_from_participantIds


@wrap_non_picklable_objects
def process_match(match):
    """
    Processes a match and returns a DataFrame with the match's data
    :param match:
    :param session:
    :return:
    """
    with get_session() as session:
        try:
            matchIds = []
            frameData = []

            platformId, gameId = separateMatchID(match.matchId)
            participantId = session.query(SQLParticipant.id).filter(
                SQLParticipant.matchId == match.matchId,
                SQLParticipant.participantId == 1
            ).scalar()
            winning_team = int(session.query(SQLParticipantStats.win).filter(
                SQLParticipantStats.participantId == participantId
            ).scalar())

            timeline = session.query(SQLTimeline).filter_by(platformId=platformId, gameId=gameId).first()
            frames = session.query(SQLFrame).filter_by(timelineId=timeline.id).all()
            team0BuildingsDestroyed = 0
            team1BuildingsDestroyed = 0
            for frame in frames:
                matchIds.append((match.matchId, frame.id))
                frameDict = {'timestamp': frame.timestamp, 'winning_team': winning_team}
                participantFrames = session.query(SQLParticipantFrame).filter_by(frameId=frame.id).all()
                for i, participantFrame in enumerate(participantFrames):
                    participantFrameTrainingData = participantFrame.get_training_data()
                    participantFrameDict = {f'participant{i + 1}_{k}': v for k, v in
                                            participantFrameTrainingData.items()}
                    frameDict.update(participantFrameDict)
                buildingKillEvents = session.query(SQLKillEvent).filter_by(frameId=frame.id, type="BUILDING_KILL").all()
                for buildingKillEvent in buildingKillEvents:
                    if buildingKillEvent.killerId == 0:
                        assistingParticipantIds = pickle.loads(buildingKillEvent.assistingParticipantIds)
                        if assistingParticipantIds is None:  # TODO: find out which team killed the turret,
                            # probably by using the position
                            continue
                        teamId = get_teamId_from_participantIds(assistingParticipantIds)
                    else:
                        teamId = get_teamId_from_participantIds([buildingKillEvent.killerId])  # this misses all
                    # cases where a building was destroyed by minions only
                    if teamId == 0:
                        team0BuildingsDestroyed += 1
                    elif teamId == 1:
                        team1BuildingsDestroyed += 1
                frameDict['team0_buildings_destroyed'] = team0BuildingsDestroyed
                frameDict['team1_buildings_destroyed'] = team1BuildingsDestroyed
                frameData.append(frameDict)
        except Exception as e:
            print(f"Error: {e}")
            raise
            return [], []
        finally:
            session.close()
    return matchIds, frameData


def build_frame_dataset(size: int = None, save: bool = True):
    """
    Builds a dataset of frames
    :param size: size of the dataset in number of matches
    :param save: if True, saves the dataset to data/raw/timelines.pkl
    :return: None
    """
    with get_session() as session:
        matches = session.query(SQLMatch).where(SQLMatch.patch == 20).order_by(func.random()).limit(size).all()
        logging.info(f"Processing {len(matches)} matches")

    matchData = []
    frameData = []

    for match in matches:
        matchIds, frames = process_match(match)
        matchData.extend(matchIds)
        frameData.extend(frames)

    index = pd.MultiIndex.from_tuples(matchData, names=['matchId', 'frameId'])
    print(f"len of framedata: {len(frameData)}")
    print(f"shape of index: {index.shape}")
    dfTimelines = pd.DataFrame(frameData, index=index)

    if save:
        dfTimelines.to_pickle('data/raw/timelines.pkl')


if __name__ == '__main__':
    start = time.time()
    build_frame_dataset(100, False)
    end = time.time()
    print(f"Time elapsed: {end - start}")
