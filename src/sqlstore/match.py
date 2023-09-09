from sqlalchemy import Integer, String, BigInteger, DateTime
from sqlalchemy.orm import mapped_column
from sqlalchemy.sql import func
from src.sqlstore.db import Base

from src.utils import get_patch, get_season


class SQLmatch(Base):
    __tablename__ = "match"
    matchId = mapped_column("matchId", String(20), primary_key=True)
    platformId = mapped_column("platformId", String(7))
    gameId = mapped_column("gameId", BigInteger)
    seasonId = mapped_column("seasonId", Integer)
    patch = mapped_column(Integer)
    queueId = mapped_column("queueId", Integer)
    gameVersion = mapped_column("gameVersion", String(23))
    mapId = mapped_column("mapId", Integer)
    gameDuration = mapped_column("gameDuration", Integer)
    gameCreation = mapped_column("gameCreation", BigInteger)
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, matchId: str, platformId: str, gameId: int, queueId: int,
                 gameVersion: str, mapId: int, gameDuration: int, gameCreation: int):
        self.matchId = matchId
        self.platformId = platformId
        self.gameId = gameId
        self.seasonId = get_season(gameVersion)
        self.patch = get_patch(gameVersion)
        self.queueId = queueId
        self.gameVersion = gameVersion
        self.mapId = mapId
        self.gameDuration = gameDuration
        self.gameCreation = gameCreation

    def __repr__(self):
        return f'({self.platformId}) ({self.gameId}) ({self.gameCreation})'

