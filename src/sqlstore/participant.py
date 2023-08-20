from sqlalchemy import Column, Integer, String, BigInteger, Boolean, ForeignKeyConstraint, DateTime
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLparticipantStats(Base):
    __tablename__ = "match_participant_stats"

    puuid = Column(String(40), primary_key=True)
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
    # riotIdName = Column(String(63))
    # riotIdTagLine = Column(String(63))
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
    summonerId = Column(String(63))
    summonerLevel = Column(Integer)
    summonerName = Column(String(63))
    teamEarlySurrendered = Column(Boolean)
    teamId = Column(Integer)
    teamPosition = Column(String(10))
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

    def __init__(self,
                 puuid: str,
                 platformId: str,
                 gameId: int,
                 allInPings: int,
                 assistMePings: int,
                 assists: int,
                 baitPings: int,
                 baronKills: int,
                 bountyLevel: int,
                 champExperience: int,
                 champLevel: int,
                 championId: int,
                 championName: str,
                 championTransform: int,
                 commandPings: int,
                 consumablesPurchased: int,
                 damageDealtToBuildings: int,
                 damageDealtToObjectives: int,
                 damageDealtToTurrets: int,
                 damageSelfMitigated: int,
                 dangerPings: int,
                 deaths: int,
                 detectorWardsPlaced: int,
                 doubleKills: int,
                 dragonKills: int,
                 eligibleForProgression: int,
                 enemyMissingPings: int,
                 enemyVisionPings: int,
                 firstBloodAssist: bool,
                 firstBloodKill: bool,
                 firstTowerAssist: bool,
                 firstTowerKill: bool,
                 gameEndedInEarlySurrender: bool,
                 gameEndedInSurrender: bool,
                 getBackPings: int,
                 goldEarned: int,
                 goldSpent: int,
                 holdPings: int,
                 individualPosition: str,
                 inhibitorKills: int,
                 inhibitorTakedowns: int,
                 inhibitorsLost: int,
                 item0: int,
                 item1: int,
                 item2: int,
                 item3: int,
                 item4: int,
                 item5: int,
                 item6: int,
                 itemsPurchased: int,
                 killingSprees: int,
                 kills: int,
                 lane: str,
                 largestCriticalStrike: int,
                 largestKillingSpree: int,
                 largestMultiKill: int,
                 longestTimeSpentLiving: int,
                 magicDamageDealt: int,
                 magicDamageDealtToChampions: int,
                 magicDamageTaken: int,
                 needVisionPings: int,
                 neutralMinionsKilled: int,
                 nexusKills: int,
                 nexusLost: int,
                 nexusTakedowns: int,
                 objectivesStolen: int,
                 objectivesStolenAssists: int,
                 onMyWayPings: int,
                 participantId: int,
                 pentaKills: int,
                 physicalDamageDealt: int,
                 physicalDamageDealtToChampions: int,
                 physicalDamageTaken: int,
                 placement: int,
                 playerAugment0: int,
                 playerAugment1: int,
                 playerAugment2: int,
                 playerAugment3: int,
                 playerAugment4: int,
                 playerSubteamId: int,
                 profileIcon: int,
                 pushPings: int,
                 quadraKills: int,
                 role: str,
                 sightWardsBoughtInGame: int,
                 spell1Casts: int,
                 spell2Casts: int,
                 spell3Casts: int,
                 spell4Casts: int,
                 subteamPlacement: int,
                 summoner1Casts: int,
                 summoner1Id: int,
                 summoner2Casts: int,
                 summoner2Id: int,
                 summonerLevel: int,
                 summonerName: str,
                 teamEarlySurrendered: bool,
                 teamId: int,
                 teamPosition: str,
                 timeCCingOthers: int,
                 timePlayed: int,
                 totalAllyJungleMonstersKilled: int,
                 totalDamageDealt: int,
                 totalDamageDealtToChampions: int,
                 totalDamageShieldedOnTeammates: int,
                 totalDamageTaken: int,
                 totalEnemyJungleMinionsKilled: int,
                 totalHeal: int,
                 totalHealOnTeammates: int,
                 totalMinionsKilled: int,
                 totalTimeCCDealt: int,
                 totalTimeSpentDead: int,
                 totalUnitsHealed: int,
                 tripleKills: int,
                 trueDamageDealt: int,
                 trueDamageDealtToChampions: int,
                 trueDamageTaken: int,
                 turretKills: int,
                 turretTakedowns: int,
                 turretsLost: int,
                 unrealKills: int,
                 visionClearedPings: int,
                 visionScore: int,
                 visionWardsBoughtInGame: int,
                 wardsKilled: int,
                 wardsPlaced: int,
                 win: bool,
                 ):
        self.puuid = puuid
        self.platformId = platformId
        self.gameId = gameId
        self.allInPings = allInPings
        self.assistMePings = assistMePings
        self.assists = assists
        self.baitPings = baitPings
        self.baronKills = baronKills
        self.bountyLevel = bountyLevel
        self.champExperience = champExperience
        self.champLevel = champLevel
        self.championId = championId
        self.championName = championName
        self.championTransform = championTransform
        self.commandPings = commandPings
        self.consumablesPurchased = consumablesPurchased
        self.damageDealtToBuildings = damageDealtToBuildings
        self.damageDealtToObjectives = damageDealtToObjectives
        self.damageDealtToTurrets = damageDealtToTurrets
        self.damageSelfMitigated = damageSelfMitigated
        self.dangerPings = dangerPings
        self.deaths = deaths
        self.detectorWardsPlaced = detectorWardsPlaced
        self.doubleKills = doubleKills
        self.dragonKills = dragonKills
        self.eligibleForProgression = eligibleForProgression
        self.enemyMissingPings = enemyMissingPings
        self.enemyVisionPings = enemyVisionPings
        self.firstBloodAssist = firstBloodAssist
        self.firstBloodKill = firstBloodKill
        self.firstTowerAssist = firstTowerAssist
        self.firstTowerKill = firstTowerKill
        self.gameEndedInEarlySurrender = gameEndedInEarlySurrender
        self.gameEndedInSurrender = gameEndedInSurrender
        self.getBackPings = getBackPings
        self.goldEarned = goldEarned
        self.goldSpent = goldSpent
        self.holdPings = holdPings
        self.individualPosition = individualPosition
        self.inhibitorKills = inhibitorKills
        self.inhibitorTakedowns = inhibitorTakedowns
        self.inhibitorsLost = inhibitorsLost
        self.item0 = item0
        self.item1 = item1
        self.item2 = item2
        self.item3 = item3
        self.item4 = item4
        self.item5 = item5
        self.item6 = item6
        self.itemsPurchased = itemsPurchased
        self.killingSprees = killingSprees
        self.kills = kills
        self.lane = lane
        self.largestCriticalStrike = largestCriticalStrike
        self.largestKillingSpree = largestKillingSpree
        self.largestMultiKill = largestMultiKill
        self.longestTimeSpentLiving = longestTimeSpentLiving
        self.magicDamageDealt = magicDamageDealt
        self.magicDamageDealtToChampions = magicDamageDealtToChampions
        self.magicDamageTaken = magicDamageTaken
        self.needVisionPings = needVisionPings
        self.neutralMinionsKilled = neutralMinionsKilled
        self.nexusKills = nexusKills
        self.nexusLost = nexusLost
        self.nexusTakedowns = nexusTakedowns
        self.objectivesStolen = objectivesStolen
        self.objectivesStolenAssists = objectivesStolenAssists
        self.onMyWayPings = onMyWayPings
        self.participantId = participantId
        self.pentaKills = pentaKills
        self.physicalDamageDealt = physicalDamageDealt
        self.physicalDamageDealtToChampions = physicalDamageDealtToChampions
        self.physicalDamageTaken = physicalDamageTaken
        self.placement = placement
        self.playerAugment0 = playerAugment0
        self.playerAugment1 = playerAugment1
        self.playerAugment2 = playerAugment2
        self.playerAugment3 = playerAugment3
        self.playerAugment4 = playerAugment4
        self.playerSubteamId = playerSubteamId
        self.profileIcon = profileIcon
        self.pushPings = pushPings
        self.quadraKills = quadraKills
        self.role = role
        self.sightWardsBoughtInGame = sightWardsBoughtInGame
        self.spell1Casts = spell1Casts
        self.spell2Casts = spell2Casts
        self.spell3Casts = spell3Casts
        self.spell4Casts = spell4Casts
        self.subteamPlacement = subteamPlacement
        self.summoner1Casts = summoner1Casts
        self.summoner1Id = summoner1Id
        self.summoner2Casts = summoner2Casts
        self.summoner2Id = summoner2Id
        self.summonerId = summoner1Id
        self.summonerLevel = summonerLevel
        self.summonerName = summonerName
        self.teamEarlySurrendered = teamEarlySurrendered
        self.teamId = teamId
        self.teamPosition = teamPosition
        self.timeCCingOthers = timeCCingOthers
        self.timePlayed = timePlayed
        self.totalAllyJungleMinionsKilled = totalAllyJungleMonstersKilled
        self.totalDamageDealt = totalDamageDealt
        self.totalDamageDealtToChampions = totalDamageDealtToChampions
        self.totalDamageShieldedOnTeammates = totalDamageShieldedOnTeammates
        self.totalDamageTaken = totalDamageTaken
        self.totalEnemyJungleMinionsKilled = totalEnemyJungleMinionsKilled
        self.totalHeal = totalHeal
        self.totalHealOnTeammates = totalHealOnTeammates
        self.totalMinionsKilled = totalMinionsKilled
        self.totalTimeCCDealt = totalTimeCCDealt
        self.totalTimeSpentDead = totalTimeSpentDead
        self.totalUnitsHealed = totalUnitsHealed
        self.tripleKills = tripleKills
        self.trueDamageDealt = trueDamageDealt
        self.trueDamageDealtToChampions = trueDamageDealtToChampions
        self.trueDamageTaken = trueDamageTaken
        self.turretKills = turretKills
        self.turretTakedowns = turretTakedowns
        self.turretsLost = turretsLost
        self.unrealKills = unrealKills
        self.visionClearedPings = visionClearedPings
        self.visionScore = visionScore
        self.visionWardsBoughtInGame = visionWardsBoughtInGame
        self.wardsKilled = wardsKilled
        self.wardsPlaced = wardsPlaced
        self.win = win

    def __repr__(self):
        return f"({self.platformId}) ({self.gameId}) {self.participantId} playing champion {self.championId}"
