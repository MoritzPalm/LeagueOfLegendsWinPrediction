from sqlalchemy import Column, Integer, String, BigInteger, Boolean, ForeignKey, DateTime, PickleType
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLTimeline(Base):
    __tablename__ = "match_timeline"

    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    frameInterval = Column(Integer)
    frameIds = Column(Integer, ForeignKey("match_timeline_frame.frameId"), nullable=False)
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
    frameId = Column(Integer, primary_key=True)     # this is not present in the data and needs to be calculated
    eventIds = Column(Integer, ForeignKey("match_timeline_event.eventId"))  # TODO: is this relevant here?
    timestamp = Column(Integer, nullable=False)
    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self):
        pass

    def __repr__(self):
        pass


class SQLTimelineEvent(Base):
    __tablename__ = "match_timeline_event"

    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    frameId = Column(Integer, primary_key=True)
    eventId = Column(Integer, primary_key=True)
    timestamp = Column(Integer, nullable=False)
    type = Column(String(100), nullable=False)
    participantId = Column(Integer)
    itemId = Column(Integer)
    skillSlot = Column(Integer)
    creatorId = Column(Integer)
    teamId = Column(Integer)
    killerId = Column(Integer)
    victimId = Column(Integer)
    afterId = Column(Integer)
    beforeId = Column(Integer)
    position_x = Column(Integer)
    position_y = Column(Integer)
    assistingParticipantIds = Column(PickleType)    # this holds a serialized list of ids
    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self):
        pass

    def __repr__(self):
        pass


class SQLTimelineParticipantFrame(Base):
    __tablename__ = "match_timeline_participant_frame"

    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    frameId = Column(Integer, primary_key=True)
    participantId = Column(Integer, primary_key=True)
    abilityHaste = Column(Integer)
    abilityPower = Column(Integer)
    armor = Column(Integer)
    armorPen = Column(Integer)
    armorPenPercent = Column(Integer)
    attackDamage = Column(Integer)
    attackSpeed = Column(Integer)
    bonusArmorPenPercent = Column(Integer)
    bonusMagicPenPercent = Column(Integer)
    ccReduction = Column(Integer)
    cooldownReduction = Column(Integer)
    health = Column(Integer)
    healthMax = Column(Integer)
    healthRegen = Column(Integer)
    lifesteal = Column(Integer)
    magicPen = Column(Integer)
    magicPenPercent = Column(Integer)
    magicResist = Column(Integer)
    movementSpeed = Column(Integer)
    omnivamp = Column(Integer)
    physicalVamp = Column(Integer)
    power = Column(Integer)
    powerRegen = Column(Integer)
    spellVamp = Column(Integer)
    currentGold = Column(Integer)
    magicDamageDone = Column(Integer)
    magicDamageDoneToChampions = Column(Integer)
    magicDamageTaken = Column(Integer)
    physicalDamageDone = Column(Integer)
    physicalDamageDoneToChampions = Column(Integer)
    physicalDamageTaken = Column(Integer)
    totalDamageDone = Column(Integer)
    totalDamageDoneToChampions = Column(Integer)
    totalDamageTaken = Column(Integer)
    trueDamageDone = Column(Integer)
    trueDamageDoneToChampions = Column(Integer)
    trueDamageTaken = Column(Integer)
    goldPerSecond = Column(Integer)
    jungleMinionsKilled = Column(Integer)
    level = Column(Integer)
    minionsKilled = Column(Integer)
    position_x = Column(Integer)
    position_y = Column(Integer)
    timeEnemySpentControlled = Column(Integer)
    totalGold = Column(Integer)
    xp = Column(Integer)

    def __init__(self):
        pass

    def __repr__(self):
        pass
