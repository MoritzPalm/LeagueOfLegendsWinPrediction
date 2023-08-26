from sqlalchemy import Column, Integer, String, BigInteger, Boolean, ForeignKey, DateTime, PickleType, Identity
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLTimeline(Base):
    __tablename__ = "match_timeline"

    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    frameInterval = Column(Integer)
    # frameIds = Column(Integer, ForeignKey("match_timeline_frame.frameId"), nullable=False)
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
    # eventIds = Column(Integer, ForeignKey("match_timeline_event.eventId"))  # TODO: is this relevant here?
    timestamp = Column(Integer, nullable=False)
    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, platformId: str, gameId: int, frameId: int, timestamp: int):
        self.platformId = platformId
        self.gameId = gameId
        self.frameId = frameId
        self.timestamp = timestamp

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} frame {self.frameId} at {self.timestamp}"


class SQLTimelineEvent(Base):
    __tablename__ = "match_timeline_event"

    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    frameId = Column(Integer, primary_key=True)
    eventId = Column(Integer, primary_key=True)
    timestamp = Column(Integer)
    type = Column(String(100))
    participantId = Column(Integer)
    itemId = Column(Integer)
    skillSlot = Column(Integer)
    creatorId = Column(Integer)
    teamId = Column(Integer)
    afterId = Column(Integer)
    beforeId = Column(Integer)
    wardType = Column(String(50))
    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, **kwargs):
        for attr in ('platformId', 'gameId', 'frameId', 'eventId', 'timestamp', 'type', 'participantId', 'itemId',
                     'skillSlot', 'creatorId', 'teamId', 'afterId', 'beforeId', 'wardType'):
            setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} frame {self.frameId} event {self.eventId} at {self.timestamp} of type {self.type}"


class SQLTimelineKillEvent(Base):
    __tablename = "match_timeline_event_kill"
    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    frameId = Column(Integer, primary_key=True)
    killId = Column(Integer, Identity(always=True), primary_key=True)
    # eventId = Column(Integer, ForeignKey("match_timeline_event.eventId"))   # TODO: is this correct?
    assistingParticipantIds = Column(PickleType)    # serialized list of participant ids
    bounty = Column(Integer)
    killStreakLength = Column(Integer)
    killerId = Column(Integer)
    position_x = Column(Integer)
    position_y = Column(Integer)
    shutdownBounty = Column(Integer)
    timestamp = Column(Integer)
    type = Column(String(30))
    victimId = Column(Integer)
    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self):
        pass

    def __repr__(self):
        pass


class SQLTimelineVictimDamageDealt(Base):
    __tablename__ = "match_timeline_kill_victimdmgdealt"
    platformId = Column(String(7), primary_key=True)    # TODO: these 4 rows should be foreign, not primary keys?
    gameId = Column(BigInteger, primary_key=True)
    frameId = Column(Integer, primary_key=True)
    damageId = Column(Integer, Identity(always=True), primary_key=True)
    killId = Column(Integer, ForeignKey("match_timeline_event_kill.killId"), nullable=False)
    basic = Column(Boolean)
    magicDamage = Column(Integer)
    name = Column(String(30))
    participantId = Column(Integer)
    physicalDamage = Column(Integer)
    spellName = Column(String(70))
    spellSlot = Column(Integer)
    trueDamage = Column(Integer)
    type = Column(String(40))
    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self):
        pass

    def __repr__(self):
        pass


class SQLTimelineVictimDamageReceived(Base):    # TODO: can/should this table be merged with victimdmgdealt?
    __tablename__ = "match_timeline_kill_victimdmgreceived"
    platformId = Column(String(7), primary_key=True)    # TODO: these 4 rows should be foreign, not primary keys?
    gameId = Column(BigInteger, primary_key=True)
    frameId = Column(Integer, primary_key=True)
    damageId = Column(Integer, Identity(always=True), primary_key=True)
    killId = Column(Integer, ForeignKey("match_timeline_event_kill.killId"), nullable=False)
    basic = Column(Boolean)
    magicDamage = Column(Integer)
    name = Column(String(30))
    participantId = Column(Integer)
    physicalDamage = Column(Integer)
    spellName = Column(String(70))
    spellSlot = Column(Integer)
    trueDamage = Column(Integer)
    type = Column(String(40))
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

    def __init__(self, **kwargs):
        for attr in ('platformId', 'gameId', 'frameId', 'participantId', 'currentGold', 'goldPerSecond',
                     'jungleMinionsKilled', 'level', 'minionsKilled', 'timeEnemySpentControlled', 'totalGold', 'xp'):
            setattr(self, attr, kwargs.get(attr))
        for attr in ('abilityHaste', 'abilityPower', 'armor', 'armorPen', 'armorPenPercent', 'attackDamage',
                     'attackSpeed', 'bonusArmorPenPercent', 'bonusMagicPenPercent', 'ccReduction', 'cooldownReduction',
                     'health', 'healthMax', 'healthRegen', 'lifesteal', 'magicPen', 'magicPenPercent', 'magicResist',
                     'movementSpeed', 'omnivamp', 'physicalVamp', 'power', 'powerMax', 'powerRegen', 'spellVamp'):
            setattr(self, attr, kwargs['championStats'].get(attr))
        for attr in ('magicDamageDone', 'magicDamageDoneToChampions', 'magicDamageTaken', 'physicalDamageDone',
                     'physicalDamageDoneToChampions', 'physicalDamageTaken', 'totalDamageDone', 'totalDamageTaken',
                     'totalDamageDoneToChampions', 'trueDamageDone', 'trueDamageDoneToChampions', 'trueDamageTaken'):
            setattr(self, attr, kwargs['damageStats'].get(attr))
        self.position_x = kwargs['position']['x']
        self.position_y = kwargs['position']['y']

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} frame {self.frameId} participant {self.participantId}"
