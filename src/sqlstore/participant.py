from sqlalchemy import Column, Integer, String, BigInteger, Boolean, ForeignKeyConstraint, DateTime
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLparticipantStats(Base):
    __tablename__ = "match_participant_stats"

    puuid = Column(String(40), primary_key=True)
    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    allInPings = Column(Integer)
    timeCreated = Column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, **kwargs):
        for attr in ('puuid', 'platformId', 'gameId', 'allInPings'):    # TODO: complete the list after testing
            setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"({self.platformId}) ({self.gameId}) {self.participantId} playing champion {self.championId}"
