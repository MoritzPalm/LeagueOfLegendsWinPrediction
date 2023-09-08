from sqlalchemy import Integer, String, BigInteger, Boolean, ForeignKey, DateTime, PickleType, Identity, UniqueConstraint
from sqlalchemy.orm import mapped_column
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLTimeline(Base):
    __tablename__ = "match_timeline"

    platformId = mapped_column(String(7), primary_key=True)
    gameId = mapped_column(BigInteger, primary_key=True)
    frameInterval = mapped_column(Integer)
    frameIds = mapped_column(Integer, ForeignKey("match_timeline_frame.id"))
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, platformId: str, gameId: int, frameInterval: int):
        self.platformId = platformId
        self.gameId = gameId
        self.frameInterval = frameInterval

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} with interval {self.frameInterval}ms"


class SQLTimelineFrame(Base):
    __tablename__ = "match_timeline_frame"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    platformId = mapped_column(String(7), nullable=False)
    gameId = mapped_column(BigInteger, nullable=False)
    frameId = mapped_column(Integer, nullable=False)     # this is not present in the data and needs to be calculated
    eventIds = mapped_column(Integer, ForeignKey("match_timeline_event.id"))
    timestamp = mapped_column(Integer, nullable=False)
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, platformId: str, gameId: int, frameId: int, timestamp: int):
        self.platformId = platformId
        self.gameId = gameId
        self.frameId = frameId
        self.timestamp = timestamp

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} frame {self.frameId} at {self.timestamp}"


class SQLTimelineEvent(Base):
    __tablename__ = "match_timeline_event"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    platformId = mapped_column(String(7), nullable=False)
    gameId = mapped_column(BigInteger, nullable=False)
    frameId = mapped_column(Integer, nullable=False)
    eventId = mapped_column(Integer, nullable=False)
    timestamp = mapped_column(Integer, nullable=False)
    type = mapped_column(String(100), nullable=False)
    participantId = mapped_column(Integer)
    itemId = mapped_column(Integer)
    skillSlot = mapped_column(Integer)
    creatorId = mapped_column(Integer)
    teamId = mapped_column(Integer)
    afterId = mapped_column(Integer)
    beforeId = mapped_column(Integer)
    wardType = mapped_column(String(50))
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())
    UniqueConstraint("platformId", "gameId", "frameId", "eventId")

    def __init__(self, **kwargs):
        for attr in ('platformId', 'gameId', 'frameId', 'eventId', 'timestamp', 'type', 'participantId', 'itemId',
                     'skillSlot', 'creatorId', 'teamId', 'afterId', 'beforeId', 'wardType'):
            setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} frame {self.frameId} event {self.eventId} at {self.timestamp} of type {self.type}"


class SQLTimelineKillEvent(Base):
    __tablename__ = "match_timeline_event_kill"
    platformId = mapped_column(String(7), primary_key=True)
    gameId = mapped_column(BigInteger, primary_key=True)
    frameId = mapped_column(Integer, primary_key=True)
    killId = mapped_column(Integer, Identity(always=True), primary_key=True)
    # eventId = mapped_column(Integer, ForeignKey("match_timeline_event.eventId"))   # TODO: is this correct?
    assistingParticipantIds = mapped_column(PickleType)    # serialized list of participant ids
    bounty = mapped_column(Integer)
    killStreakLength = mapped_column(Integer)
    killerId = mapped_column(Integer)
    position_x = mapped_column(Integer)
    position_y = mapped_column(Integer)
    shutdownBounty = mapped_column(Integer)
    timestamp = mapped_column(Integer)
    type = mapped_column(String(30))
    victimId = mapped_column(Integer)
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, **kwargs):
        for attr in ('platformId', 'gameId', 'frameId', 'killId', 'eventId', 'assistingParticipantIds', 'bounty',
                     'killStreakLength', 'killerId', 'position_x', 'position_y', 'shutdownBounty', 'timestamp'):
            setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} at frame {self.frameId} (id: {self.killId}) {self.killerId} killed " \
               f"{self.victimId}"


class SQLTimelineVictimDamageDealt(Base):
    __tablename__ = "match_timeline_kill_victimdmgdealt"
    platformId = mapped_column(String(7), primary_key=True)    # TODO: these 4 rows should be foreign, not primary keys?
    gameId = mapped_column(BigInteger, primary_key=True)
    frameId = mapped_column(Integer, primary_key=True)
    damageId = mapped_column(Integer, Identity(always=True), primary_key=True)
    #killId = mapped_column(Integer, ForeignKey("match_timeline_event_kill.killId"), nullable=False)
    basic = mapped_column(Boolean)
    magicDamage = mapped_column(Integer)
    name = mapped_column(String(30))
    participantId = mapped_column(Integer)
    physicalDamage = mapped_column(Integer)
    spellName = mapped_column(String(70))
    spellSlot = mapped_column(Integer)
    trueDamage = mapped_column(Integer)
    type = mapped_column(String(40))
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, **kwargs):
        for attr in ('platformId', 'gameId', 'frameId', 'damageId', 'killId', 'magicDamage', 'name', 'participantId',
                     'physicalDamage', 'spellName', 'spellSlot', 'trueDamage', 'type'):
            setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} at frame {self.frameId} (id: {self.damageId}) {self.name} " \
               f"dealt damage before being killed"


class SQLTimelineVictimDamageReceived(Base):    # TODO: can/should this table be merged with victimdmgdealt?
    __tablename__ = "match_timeline_kill_victimdmgreceived"
    platformId = mapped_column(String(7), primary_key=True)    # TODO: these 4 rows should be foreign, not primary keys?
    gameId = mapped_column(BigInteger, primary_key=True)
    frameId = mapped_column(Integer, primary_key=True)
    damageId = mapped_column(Integer, Identity(always=True), primary_key=True)
    #killId = mapped_column(Integer, ForeignKey("match_timeline_event_kill.killId"), nullable=False)
    basic = mapped_column(Boolean)
    magicDamage = mapped_column(Integer)
    name = mapped_column(String(30))
    participantId = mapped_column(Integer)
    physicalDamage = mapped_column(Integer)
    spellName = mapped_column(String(70))
    spellSlot = mapped_column(Integer)
    trueDamage = mapped_column(Integer)
    type = mapped_column(String(40))
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, **kwargs):
        for attr in ('platformId', 'gameId', 'frameId', 'damageId', 'killId', 'magicDamage', 'name', 'participantId',
                     'physicalDamage', 'spellName', 'spellSlot', 'trueDamage', 'type'):
            setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} at frame {self.frameId} (id: {self.damageId}) {self.name} " \
               f"received fatal damage"


class SQLTimelineParticipantFrame(Base):
    __tablename__ = "match_timeline_participant_frame"

    platformId = mapped_column(String(7), primary_key=True)
    gameId = mapped_column(BigInteger, primary_key=True)
    frameId = mapped_column(Integer, primary_key=True)
    participantId = mapped_column(Integer, primary_key=True)
    abilityHaste = mapped_column(Integer)
    abilityPower = mapped_column(Integer)
    armor = mapped_column(Integer)
    armorPen = mapped_column(Integer)
    armorPenPercent = mapped_column(Integer)
    attackDamage = mapped_column(Integer)
    attackSpeed = mapped_column(Integer)
    bonusArmorPenPercent = mapped_column(Integer)
    bonusMagicPenPercent = mapped_column(Integer)
    ccReduction = mapped_column(Integer)
    cooldownReduction = mapped_column(Integer)
    health = mapped_column(Integer)
    healthMax = mapped_column(Integer)
    healthRegen = mapped_column(Integer)
    lifesteal = mapped_column(Integer)
    magicPen = mapped_column(Integer)
    magicPenPercent = mapped_column(Integer)
    magicResist = mapped_column(Integer)
    movementSpeed = mapped_column(Integer)
    omnivamp = mapped_column(Integer)
    physicalVamp = mapped_column(Integer)
    power = mapped_column(Integer)
    powerRegen = mapped_column(Integer)
    spellVamp = mapped_column(Integer)
    currentGold = mapped_column(Integer)
    magicDamageDone = mapped_column(Integer)
    magicDamageDoneToChampions = mapped_column(Integer)
    magicDamageTaken = mapped_column(Integer)
    physicalDamageDone = mapped_column(Integer)
    physicalDamageDoneToChampions = mapped_column(Integer)
    physicalDamageTaken = mapped_column(Integer)
    totalDamageDone = mapped_column(Integer)
    totalDamageDoneToChampions = mapped_column(Integer)
    totalDamageTaken = mapped_column(Integer)
    trueDamageDone = mapped_column(Integer)
    trueDamageDoneToChampions = mapped_column(Integer)
    trueDamageTaken = mapped_column(Integer)
    goldPerSecond = mapped_column(Integer)
    jungleMinionsKilled = mapped_column(Integer)
    level = mapped_column(Integer)
    minionsKilled = mapped_column(Integer)
    position_x = mapped_column(Integer)
    position_y = mapped_column(Integer)
    timeEnemySpentControlled = mapped_column(Integer)
    totalGold = mapped_column(Integer)
    xp = mapped_column(Integer)

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
