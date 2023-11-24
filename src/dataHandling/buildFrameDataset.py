import pandas as pd
from sqlalchemy import func
from tqdm import tqdm

from src.sqlstore.db import get_session
from src.sqlstore.match import SQLMatch, SQLParticipant, SQLParticipantStats
from src.sqlstore.timeline import SQLTimeline, SQLFrame, SQLParticipantFrame
from src.utils import separateMatchID


def build_frame_dataset(size: int = None, save: bool = True):
    """
    Builds a dataset with all frame information (info available during match, but no events) from the database.
    :param size: Number of matches in the dataset, None if all matches should be used
    :param save: Whether to save the dataset to a pickle file in the data/raw folder
    :return: None
    """
    # TODO: every 500 or so matches save the dataset to a pickle file
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


if __name__ == '__main__':
    build_frame_dataset(100, False)
