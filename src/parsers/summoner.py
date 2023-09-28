import logging

import sqlalchemy.orm
from riotwatcher import LolWatcher
from scrapy.crawler import CrawlerProcess
import pandas as pd

import src.utils
from src.sqlstore import queries
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague, SQLChampionMastery
from src.scraping.myspider import MySpider


def parse_summoner_data(
        session: sqlalchemy.orm.Session,
        watcher: LolWatcher,
        region: str,
        puuid: str,
        championId: int,
        expiration: int,
) -> bool:
    """
    checks if summoner data (different Ids, win rates, rank, etc.) is more recent than expiration date
    if no, updates/ inserts data
    :param expiration: number of days until data should be updated
    :param session: sqlalchemy session
    :param watcher: riotwatcher
    :param region:
    :param puuid: encrypted puuid of the summoner
    :return: True if all data has been updated, False otherwise
    """
    if queries.check_summoner_present(
            session, puuid
    ) and queries.check_summoner_data_recent(session, puuid, expiration):
        return False
    summoner_data = watcher.summoner.by_puuid(region=region, encrypted_puuid=puuid)
    summoner_obj = SQLSummoner(
        summoner_data["puuid"],
        region,
        summoner_data["id"],
        summoner_data["accountId"],
        summoner_data["name"],
        summoner_data["summonerLevel"],
    )
    session.add(summoner_obj)
    summoner_league_data = watcher.league.by_summoner(
        region=region, encrypted_summoner_id=summoner_obj.summonerId
    )
    summoner_league_obj = None
    for data in summoner_league_data:
        if data["queueType"] == "RANKED_SOLO_5x5":
            summoner_league_obj = SQLSummonerLeague(
                leagueId=data["leagueId"],
                queueType=data["queueType"],
                tier=data["tier"],
                rank=data["rank"],
                summonerName=data["summonerName"],
                leaguePoints=data["leaguePoints"],
                wins=data["wins"],
                losses=data["losses"],
                veteran=data["veteran"],
                inactive=data["inactive"],
                freshBlood=data["freshBlood"],
                hotStreak=data["hotStreak"],
            )
    if summoner_league_obj is None:
        raise Exception(f"no ranked summoner data found!")
    summoner_obj.leagues.append(summoner_league_obj)
    session.add(summoner_league_obj)

    summoner_champion_data = watcher.champion_mastery.by_summoner_by_champion(
        region, summoner_obj.summonerId, championId
    )
    with session.no_autoflush:
        summoner_championmastery_obj = SQLChampionMastery(
            championPointsUntilNextlevel=summoner_champion_data["championPointsUntilNextLevel"],
            chestGranted=summoner_champion_data["chestGranted"],
            lastPlayTime=summoner_champion_data["lastPlayTime"],
            championLevel=summoner_champion_data["championLevel"],
            summonerId=summoner_champion_data["summonerId"],
            championPoints=summoner_champion_data["championPoints"],
            championPointsSinceLastLevel=summoner_champion_data["championPointsSinceLastLevel"],
            tokensEarned=summoner_champion_data["tokensEarned"],
        )
        summoner_obj.mastery.append(summoner_championmastery_obj)
        champion_obj = queries.get_last_champion(session, championId)
        champion_obj.mastery.append(summoner_championmastery_obj)
    session.add(summoner_championmastery_obj)
    return True


def scrape_champion_masteries(session: sqlalchemy.orm.Session):
    """
    scrapes champion masteries for all summoners in data
    :param session: sqlalchemy session
    :return:
    """
    objs = queries.get_missing_masteries(session)
    init_data = []
    for obj in objs:
        championName = queries.get_champ_name(session, obj.championId)
        init_data.append(
            {
                "champion": championName,
                "summonerName": obj.summoner.name,
                "region": obj.summoner.platformId,
            }
        )
    process = CrawlerProcess()
    process.crawl(MySpider, init_data)
    process.start()
