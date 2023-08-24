from sqlalchemy import Column, Integer, String, BigInteger, Boolean, ForeignKeyConstraint, DateTime
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLTimeline(Base):
    __tablename__ = "match_timeline"

    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    frameInterval = Column(Integer)
    # TODO: consider adding frame ids here
    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, platformId: str, gameId: int, frameInterval: int):
        self.platformId = platformId
        self.gameId = gameId
        self.frameInterval = frameInterval

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} with interval {self.frameInterval}ms"


class SQLTimelineFrame(Base):
    __tablename__ = "match_timeline_frame"

    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)

    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())


class SQLTimelineEvent(Base):
    pass


class SQLTimelineKill(Base):
    pass


class SQLTimelineParticipantFrame(Base):
    pass


