from sqlalchemy import Integer, String, BigInteger, Boolean, ForeignKey, DateTime, PickleType, Identity, \
    UniqueConstraint, Enum
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLTimeline(Base):
    __tablename__ = "timeline"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    platformId = mapped_column(String(7), nullable=False)
    gameId = mapped_column(BigInteger, nullable=False)
    frameInterval = mapped_column(Integer)
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())
    UniqueConstraint("platformId", "gameId")

    def __init__(self, platformId: str, gameId: int, frameInterval: int):
        self.platformId = platformId
        self.gameId = gameId
        self.frameInterval = frameInterval

    def __repr__(self):
        return f"timeline (id: {self.id} of match {self.platformId}_{self.gameId} with interval {self.frameInterval}ms"


class SQLFrame(Base):
    __tablename__ = "frame"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    platformId = mapped_column(String(7), nullable=False)
    gameId = mapped_column(BigInteger, nullable=False)
    timelineId = mapped_column(BigInteger, ForeignKey("timeline.id"), nullable=False)
    timeline = relationship("SQLTimeline", backref="frames")
    frameId = mapped_column(Integer)  # frame id is starting at 0 and counting up per game, encodes order of frames
    timestamp = mapped_column(Integer, nullable=False)  # in milliseconds?
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, platformId: str, gameId: int, frameId: int, timestamp: int):
        self.platformId = platformId
        self.gameId = gameId
        self.frameId = frameId
        self.timestamp = timestamp

    def __repr__(self):
        return f"timeline frame (id: {self.id} of timeline {self.timelineId} in game {self.platformId}_{self.gameId} " \
               f"at {self.timestamp}"


class SQLEvent(Base):
    __tablename__ = "event"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    frameId = mapped_column(BigInteger, ForeignKey("frame.id"), nullable=False)
    frame = relationship("SQLFrame", backref="events")
    eventId = mapped_column(Integer, nullable=False)  # starting at 0 and counting up per frame, encodes order of events
    timestamp = mapped_column(Integer, nullable=False)  # in milliseconds?
    type = mapped_column(String(100), nullable=False)  # e.g. SKILL_LEVEL_UP
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

    def __init__(self, eventId: int, timestamp: int, type: str, participantId: int, itemId: int, skillSlot: int,
                 creatorId: int, teamId: int, afterId: int, beforeId: int, wardType: str):
        self.eventId = eventId
        self.timestamp = timestamp
        self.type = type
        self.participantId = participantId
        self.itemId = itemId
        self.skillSlot = skillSlot
        self.creatorId = creatorId
        self.teamId = teamId
        self.afterId = afterId
        self.beforeId = beforeId
        self.wardType = wardType

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} frame {self.frameId} event {self.eventId} at {self.timestamp} of type {self.type}"


class SQLKillEvent(Base):
    __tablename__ = "killevent"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    frameId = mapped_column(BigInteger, ForeignKey("frame.id"), nullable=False)
    frame = relationship("SQLFrame", backref="killevents")
    assistingParticipantIds = mapped_column(PickleType)  # serialized list of participant ids
    bounty = mapped_column(Integer)
    killStreakLength = mapped_column(Integer)
    killerId = mapped_column(Integer)
    laneType = mapped_column((String(30)))
    position_x = mapped_column(Integer)
    position_y = mapped_column(Integer)
    shutdownBounty = mapped_column(Integer)
    timestamp = mapped_column(Integer)
    type = mapped_column(String(50))
    victimId = mapped_column(Integer)
    timeCreated = mapped_column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, assistingParticipantIds: PickleType, bounty: int, killStreakLength: int, killerId: int,
                 laneType: str, position: dict, shutdownBounty: int, timestamp: int, type: str,
                 victimId: int):
        self.assistingParticipantIds = assistingParticipantIds
        self.bounty = bounty
        self.killStreakLength = killStreakLength
        self.killerId = killerId
        self.laneType = laneType
        self.position_x = position['x']
        self.position_y = position['y']
        self.shutdownBounty = shutdownBounty
        self.timestamp = timestamp
        self.type = type
        self.victimId = victimId

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} at frame {self.frameId} (id: {self.killId}) {self.killerId} killed " \
               f"{self.victimId}"


class SQLTimelineDamageDealt(Base):
    """
    damage dealt by the victim of the kill to others
    """
    __tablename__ = "dmg_dealt"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    killId = mapped_column(BigInteger, ForeignKey("killevent.id"), nullable=False)
    kill = relationship("SQLKillEvent", backref="dmgdealt")
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

    def __init__(self, basic: bool, magicDamage: int, name: str, participantId: int,
                 physicalDamage: int, spellName: str, spellSlot: int, trueDamage: int, type: str):
        """
        name is victim name, participantId is the victims participantId

        :param basic: no idea what this is
        :param magicDamage: how much magic damage the victim dealt with the spell to the attacker(s)
        :param name: name of the victim
        :param participantId: id of the attacker
        :param physicalDamage: how much physical damage the victim dealt with the spell to the attacker(s)
        :param spellName: spell with which the victim dealt damage
        :param spellSlot: spell slot used by the victim
        :param trueDamage: how much true damage the victim dealt with the spell to the attacker(s)
        :param type: no idea, is always (?) "OTHER"
        """
        self.basic = basic
        self.magicDamage = magicDamage
        self.name = name
        self.participantId = participantId
        self.physicalDamage = physicalDamage
        self.spellName = spellName
        self.spellSlot = spellSlot
        self.trueDamage = trueDamage
        self.type = type

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} at frame {self.frameId} (id: {self.damageId}) {self.name} " \
               f"dealt damage before being killed"


class SQLTimelineDamageReceived(Base):
    """
    damage received by the victim from others
    """
    __tablename__ = "dmg_received"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    killId = mapped_column(BigInteger, ForeignKey("killevent.id"), nullable=False)
    kill = relationship("SQLKillEvent", backref="dmgreceived")
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

    def __init__(self, basic: bool, magicDamage: int, name: str, participantId: int,
                 physicalDamage: int, spellName: str, spellSlot: int, trueDamage: int, type: str):
        """
        name is attacker name, participantId is the victims participantId

        :param basic: no idea what this is
        :param magicDamage: how much magic damage the attacker dealt with the spell to the victim
        :param name: name of the attacker
        :param participantId: id of the victim (!!)
        :param physicalDamage: how much physical damage the attacker dealt with the spell to the victim
        :param spellName: spell with which the attacker dealt damage
        :param spellSlot: spell slot used by the attacker
        :param trueDamage: how much true damage the attacker dealt with the spell to the victim
        :param type: no idea, is always (?) "OTHER"
        """
        self.basic = basic
        self.magicDamage = magicDamage
        self.name = name
        self.participantId = participantId
        self.physicalDamage = physicalDamage
        self.spellName = spellName
        self.spellSlot = spellSlot
        self.trueDamage = trueDamage
        self.type = type

    def __repr__(self):
        return f"damage received by {self.participantId} dealt by {self.name}"


class SQLParticipantFrame(Base):
    __tablename__ = "participant_frame"
    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    frameId = mapped_column(BigInteger, ForeignKey("frame.id"), nullable=False)
    frame = relationship("SQLFrame", backref="participantframe")
    participantId = mapped_column(Integer, nullable=False)
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
        for attr in ('participantId', 'currentGold', 'goldPerSecond', 'jungleMinionsKilled', 'level', 'minionsKilled',
                     'timeEnemySpentControlled', 'totalGold', 'xp'):
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
