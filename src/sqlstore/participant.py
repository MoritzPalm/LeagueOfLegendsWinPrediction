from sqlalchemy import Column, Integer, String, BigInteger, Boolean, ForeignKeyConstraint, DateTime
from sqlalchemy.sql import func
from src.sqlstore.db import Base

# TODO: should this be a dataclass?


class SQLparticipantStats(Base):
    __tablename__ = "match_participant_stats"

    puuid = Column(String(100), primary_key=True)
    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    allInPings = Column(Integer)
    assistMePings = Column(Integer)
    assists = Column(Integer)
    baitPings = Column(Integer)
    baronKills = Column(Integer)
    bountyLevel = Column(Integer)
    champExperience = Column(Integer)
    champLevel = Column(Integer)
    championId = Column(Integer)
    championName = Column(String(30))
    championTransform = Column(Integer)
    commandPings = Column(Integer)
    consumablesPurchased = Column(Integer)
    damageDealtToBuildings = Column(Integer)
    damageDealtToObjectives = Column(Integer)
    damageDealtToTurrets = Column(Integer)
    damageSelfMitigated = Column(Integer)
    dangerPings = Column(Integer)
    deaths = Column(Integer)
    detectorWardsPlaced = Column(Integer)
    doubleKills = Column(Integer)
    dragonKills = Column(Integer)
    eligibleForProgression = Column(Boolean)
    enemyMissingPings = Column(Integer)
    enemyVisionPings = Column(Integer)
    firstBloodAssist = Column(Boolean)
    firstBloodKill = Column(Boolean)
    firstTowerAssist = Column(Boolean)
    firstTowerKill = Column(Boolean)
    gameEndedInEarlySurrender = Column(Boolean)     # TODO: this info should be placed in a general team or match table
    gameEndedInSurrender = Column(Boolean)      # TODO: this info should be placed in a general team or match table
    getBackPings = Column(Integer)
    goldEarned = Column(Integer)
    goldSpent = Column(Integer)
    holdPings = Column(Integer)
    individualPosition = Column(String(10))
    inhibitorKills = Column(Integer)
    inhibitorTakedowns = Column(Integer)
    inhibitorsLost = Column(Integer)
    item0 = Column(Integer)
    item1 = Column(Integer)
    item2 = Column(Integer)
    item3 = Column(Integer)
    item4 = Column(Integer)
    item5 = Column(Integer)
    item6 = Column(Integer)
    itemsPurchased = Column(Integer)
    killingSprees = Column(Integer)
    kills = Column(Integer)
    lane = Column(String(20))
    largestCriticalStrike = Column(Integer)
    largestKillingSpree = Column(Integer)
    largestMultiKill = Column(Integer)
    longestTimeSpentLiving = Column(Integer)
    magicDamageDealt = Column(Integer)
    magicDamageDealtToChampions = Column(Integer)
    magicDamageTaken = Column(Integer)
    needVisionPings = Column(Integer)
    neutralMinionsKilled = Column(Integer)
    nexusKills = Column(Integer)    # This column is probably only important for special gamemodes, consider deleting it
    nexusLost = Column(Integer)     # This column is probably only important for special gamemodes, consider deleting it
    nexusTakedowns = Column(Integer)    # This column is probably only important for special gamemodes, consider deleting it
    objectivesStolen = Column(Integer)
    objectivesStolenAssists = Column(Integer)
    onMyWayPings = Column(Integer)
    participantId = Column(Integer)     # TODO: should this be part of the primary key?
    pentaKills = Column(Integer)
    # TODO: in matchDto are perks, which do not translate well into this table, consider putting those in separate table
    physicalDamageDealt = Column(Integer)
    physicalDamageDealtToChampions = Column(Integer)
    physicalDamageTaken = Column(Integer)
    placement = Column(Integer)
    playerAugment0 = Column(Integer)
    playerAugment1 = Column(Integer)
    playerAugment2 = Column(Integer)
    playerAugment3 = Column(Integer)
    playerAugment4 = Column(Integer)
    playerSubteamId = Column(Integer)
    profileIcon = Column(Integer)
    pushPings = Column(Integer)
    quadraKills = Column(Integer)
    riotIdName = Column(String(63))
    riotIdTagLine = Column(String(63))
    role = Column(String(20))
    sightWardsBoughtInGame = Column(Integer)
    spell1Casts = Column(Integer)
    spell2Casts = Column(Integer)
    spell3Casts = Column(Integer)
    spell4Casts = Column(Integer)
    subteamPlacement = Column(Integer)
    summoner1Casts = Column(Integer)
    summoner1Id = Column(Integer)
    summoner2Casts = Column(Integer)
    summoner2Id = Column(Integer)
    summonerId = Column(String(100))
    summonerLevel = Column(Integer)
    summonerName = Column(String(63))
    teamEarlySurrendered = Column(Boolean)
    teamId = Column(Integer)
    teamPosition = Column(String(20))
    timeCCingOthers = Column(Integer)
    timePlayed = Column(Integer)
    totalAllyJungleMinionsKilled = Column(Integer)
    totalDamageDealt = Column(Integer)
    totalDamageDealtToChampions = Column(Integer)
    totalDamageShieldedOnTeammates = Column(Integer)
    totalDamageTaken = Column(Integer)
    totalEnemyJungleMinionsKilled = Column(Integer)
    totalHeal = Column(Integer)
    totalHealOnTeammates = Column(Integer)
    totalMinionsKilled = Column(Integer)
    totalTimeCCDealt = Column(Integer)
    totalTimeSpentDead = Column(Integer)
    totalUnitsHealed = Column(Integer)
    tripleKills = Column(Integer)
    trueDamageDealt = Column(Integer)
    trueDamageDealtToChampions = Column(Integer)
    trueDamageTaken = Column(Integer)
    turretKills = Column(Integer)
    turretTakedowns = Column(Integer)
    turretsLost = Column(Integer)
    unrealKills = Column(Integer)
    visionClearedPings = Column(Integer)
    visionScore = Column(Integer)
    visionWardsBoughtInGame = Column(Integer)
    wardsKilled = Column(Integer)
    wardsPlaced = Column(Integer)
    win = Column(Boolean)
    timeCreated = Column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, **kwargs):   # TODO: include challenges and perks
        for attr in ('allInPings', 'assistMePings', 'assists', 'baitPings', 'baronKills', 'basicPings', 'bountyLevel',
                     'champExperience', 'champLevel', 'championId', 'championName', 'championTransform', 'commandPings',
                     'consumablesPurchased', 'damageDealtToBuildings', 'damageDealtToObjectives', 'damageDealtToTurrets',
                     'damageSelfMitigated', 'dangerPings', 'deaths', 'detectorWardsPlaced', 'doubleKills', 'dragonKills',
                     'eligibleForProgression', 'enemyMissingPings', 'enemyVisionPings', 'firstBloodAssist',
                     'firstBloodKill', 'firstTowerAssist', 'firstTowerKill', 'gameEndedInEarlySurrender',
                     'gameEndedInSurrender', 'getBackPings', 'goldEarned', 'goldSpent', 'holdPings',
                     'individualPosition', 'inhibitorKills', 'inhibitorTakedowns', 'inhibitorsLost', 'item0', 'item1',
                     'item2', 'item3', 'item4', 'item5', 'item6', 'itemsPurchased', 'killingSprees', 'kills', 'lane',
                     'largestCriticalStrike', 'largestKillingSpree', 'largestMultiKill', 'longestTimeSpentLiving',
                     'magicDamageDealt', 'magicDamageDealtToChampions', 'magicDamageTaken', 'needVisionPings',
                     'neutralMinionsKilled', 'nexusKills', 'nexusLost', 'nexusTakedowns', 'objectivesStolen',
                     'objectivesStolenAssists', 'onMyWayPings', 'participantId', 'pentaKills', 'physicalDamageDealt',
                     'physicalDamageDealtToChampions', 'physicalDamageTaken', 'placement', 'playerAugment1',
                     'playerAugment2', 'playerAugment3', 'playerAugment4', 'playerSubteamId', 'profileIcon', 'pushPings',
                     'puuid', 'quadraKills', 'riotIdName', 'riotIdTagline', 'role', 'sightWardsBoughtInGame',
                     'spell1Casts', 'spell2Casts', 'spell3Casts', 'spell4Casts', 'subteamPlacement', 'summoner1Casts',
                     'summoner1Id', 'summoner2Casts', 'summoner2Id', 'summonerId', 'summonerLevel', 'summonerName',
                     'teamEarlySurrendered', 'teamId', 'teamPosition', 'timeCCingOthers', 'timePlayed',
                     'totalAllyJungleMinionsKilled', 'totalDamageDealt', 'totalDamageDealtToChampions',
                     'totalDamageShieldedOnTeammates', 'totalDamageTaken', 'totalEnemyJungleMinionsKilled',
                     'totalHeal', 'totalHealsOnTeammates', 'totalMinionsKilled', 'totalTimeCCDealt',
                     'totalTimeSpentDead', 'totalUnitsHealed', 'tripleKills', 'trueDamageDealt',
                     'trueDamageDealtToChampions', 'trueDamageTaken', 'turretKills', 'turretTakedowns',
                     'turretsLost', 'unrealKills', 'visionClearedPings', 'visionScore', 'visionWardsBoughtInGame',
                     'wardsKilled', 'wardsPlaced', 'win', 'platformId', 'gameId'):
            setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} player {self.puuid} with number {self.participantId}"


class SQLStatPerks(Base):
    __tablename__ = "match_participant_stat_perks"

    puuid = Column(String(100), primary_key=True)
    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    defense = Column(Integer)
    flex = Column(Integer)
    offense = Column(Integer)
    timeCreated = Column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column(DateTime(timezone=True), onupdate=func.now())

    def __init(self, puuid: str, platformId: str, gameId: str, defense: int, flex: int, offense: int):
        self.puuid = puuid
        self.platformId = platformId
        self.gameId = gameId
        self.defense = defense
        self.flex = flex
        self.offense = offense

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} player {self.puuid} stats"


class SQLStyles(Base):
    __tablename__ = "match_participant_styles"

    puuid = Column(String(100), primary_key=True)
    platformId = Column(String(7), primary_key=True)
    gameId = Column(BigInteger, primary_key=True)
    description = Column(String(80))
    style = Column(Integer)
    timeCreated = Column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self):
        pass

    def __repr__(self):
        pass
