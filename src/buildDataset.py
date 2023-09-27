from sqlalchemy import func
from src.sqlstore.match import SQLMatch, SQLParticipant
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague
from src.sqlstore.db import get_session
import pandas as pd


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
            # wrapping the dict in a list to prevent missing index issues, this is hacky and may have unforeseen issues
            df_match = pd.DataFrame([match.__dict__])
            participants = (
                session.query(SQLParticipant)
                .filter(SQLParticipant.matchId == match.matchId)
                .all()
            )
            assert len(participants) == 10
            for i, participant in enumerate(participants):
                summoner = (
                    session.query(SQLSummoner)
                    .filter(SQLSummoner.puuid == participant.puuid)
                    .one()
                )
                df_summoner = pd.DataFrame([summoner.__dict__])
                # renames all columns to have a participant and the number in front of the attribute
                df_summoner.rename(
                    columns=lambda x: f"participant{i}_" + x, inplace=True
                )
                summonerLeague = (
                    session.query(SQLSummonerLeague)
                    .filter(SQLSummonerLeague.puuid == participant.puuid)
                    .one()
                )
                df_summonerLeague = pd.DataFrame([summonerLeague.__dict__])
                df_summonerLeague.rename(
                    columns=lambda x: f"participant{i}_" + x, inplace=True
                )
                # mastery = session.query(SQLChampionMastery).filter(SQLChampionMastery.puuid == participant.puuid,
                # SQLChampionMastery.championId == stat.championId).one()
                # df_mastery = pd.DataFrame([mastery.__dict__])
                # df_mastery.rename(columns=lambda x: f"participant{i}_" + x, inplace=True)
                # appending all dataframes to df_match
                df_match = pd.concat(
                    [df_match, df_summoner, df_summonerLeague], axis=1, copy=False
                )
                print(df_match.shape)
            data = pd.concat([data, df_match], axis=0, copy=False)
    return data
