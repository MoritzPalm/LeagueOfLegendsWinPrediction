from sqlalchemy import Integer, String, Float, PickleType, DateTime, ForeignKey, Identity, BigInteger
from sqlalchemy.orm import mapped_column
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLChampion(Base):
    __tablename__ = "champion"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    championNumber = mapped_column(Integer, nullable=False, index=True)
    seasonNumber = mapped_column(Integer, nullable=False, index=True)
    patchNumber = mapped_column(Integer, nullable=False, index=True)
    championName = mapped_column(String(100), nullable=False, index=True)
    championTitle = mapped_column(String(100))
    infoAttack = mapped_column(Integer)
    infoDefense = mapped_column(Integer)
    infoMagic = mapped_column(Integer)
    infoDifficulty = mapped_column(Integer)
    # TODO: make tags not in binary format for easier querying
    tags = mapped_column(PickleType)   # serialized list of tags (e.g. [Marksman, Support] for Ashe)
    partype = mapped_column(String(150))   # type of mana or energy (e.g. "Blood Well" for Aatrox)
    patchWinRate = mapped_column(Float, nullable=True)  # Represented as a percent
    patchPlayRate = mapped_column(Float, nullable=True)  # Represented as a percent
    primaryRole = mapped_column(String(50), nullable=True)  # Top, Mid...
    # Maybe counters, abilities, Tier, maybe range, skill-shot-based, or not, cc-level.., trends in winrates,
    # role flexibility, new skin released (higher playrate)
    # -> this should not be saved in db, instead calculated server/analytics side imo
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def init(self, championId: int, seasonNumber: int, patchNumber: int, championName: str, championTitle: str,
             infoAttack: int, infoDefense: int, infoMagic: int, infoDifficulty: int, tags, partype: str,
             patchWinRate: float = None, patchPlayRate: float = None, role: str = None):
        """

        :param championId:
        :param patchNumber:
        :param seasonNumber
        :param championName:
        :param championTitle:
        :param infoAttack:
        :param infoDefense:
        :param infoMagic:
        :param infoDifficulty:
        :param tags:
        :param partype:
        :param patchWinRate:
        :param patchPlayRate:
        :param role:
        :return:
        """
        self.championNumber = championId
        self.championName = championName
        self.championTitle = championTitle
        self.infoAttack = infoAttack
        self.infoDefense = infoDefense
        self.infoMagic = infoMagic
        self.infoDifficulty = infoDifficulty
        self.tags = tags
        self.partype = partype
        self.patchWinRate = patchWinRate
        self.patchPlayRate = patchPlayRate
        self.patchNumber = patchNumber
        self.seasonNumber = seasonNumber
        self.primaryRole = role

    def repr(self):
        return f"<Champion {self.championName} ({self.key}) - {self.championTitle}>"


class SQLChampionStats(Base):

    __tablename__ = "champion_stats"
    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    championId = mapped_column(Integer, nullable=False, index=True)
    patchNumber = mapped_column(Integer, nullable=False, index=True)
    seasonNumber = mapped_column(Integer, nullable=False, index=True)
    hp = mapped_column(Integer)
    hpperlevel = mapped_column(Integer)
    mp = mapped_column(Integer)
    mpperlevel = mapped_column(Integer)
    movespeed = mapped_column(Integer)
    armor = mapped_column(Integer)
    armorperlevel = mapped_column(Float)
    spellblock = mapped_column(Integer)
    spellblockperlevel = mapped_column(Float)
    attackrange = mapped_column(Integer)
    hpregen = mapped_column(Float)
    hpregenperlevel = mapped_column(Float)
    mpregen = mapped_column(Float)
    mpregenperlevel = mapped_column(Float)
    crit = mapped_column(Integer)
    critperlevel = mapped_column(Integer)
    attackdamage = mapped_column(Integer)
    attackdamageperlevel = mapped_column(Float)
    attackspeed = mapped_column(Float)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, championId: int, patchNumber: int, seasonNumber: int, hp: int, hpperlevel: int, mp: int, mpperlevel: int, movespeed: int, armor: int,
                 armorperlevel: float, spellblock: int, spellblockperlevel: float, attackrange: int, hpregen: float,
                 hpregenperlevel: float, mpregen: float, mpregenperlevel: float, crit: int, critperlevel: int,
                 attackdamage: int, attackdamageperlevel: float, attackspeed: float):
        self.championId = championId
        self.patchNumber = patchNumber
        self.seasonNumber = seasonNumber
        self.hp = hp
        self.hpperlevel = hpperlevel
        self.mp = mp
        self.mpperlevel = mpperlevel
        self.movespeed = movespeed
        self.armor = armor
        self.armorperlevel = armorperlevel
        self.spellblock = spellblock
        self.spellblockperlevel = spellblockperlevel
        self.attackrange = attackrange
        self.hpregen = hpregen
        self.hpregenperlevel = hpregenperlevel
        self.mpregen = mpregen
        self.mpregenperlevel = mpregenperlevel
        self.crit = crit
        self.critperlevel = critperlevel
        self.attackdamage = attackdamage
        self.attackdamageperlevel = attackdamageperlevel
        self.attackspeed = attackspeed

    def __repr__(self):
        return f"Champion stats of champion {self.championId}"


class SQLChampionRoles(Base):
    __tablename__ = "champion_roles"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    championId = mapped_column(Integer, ForeignKey("champion.id"), nullable=False)
    championNumber = mapped_column(Integer, ForeignKey("champion.championNumber"), nullable=False)
    role = mapped_column()  # enum
    tags = mapped_column()  # enum
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self):
        pass

    def __repr__(self):
        pass
