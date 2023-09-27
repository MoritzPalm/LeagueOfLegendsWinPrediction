import logging

import sqlalchemy.orm
from riotwatcher import LolWatcher
from scrapy.crawler import CrawlerProcess
import pandas as pd

import src.utils
from src.sqlstore import queries
from src.sqlstore.summoner import SQLSummoner, SQLSummonerLeague, SQLChampionMastery
from src.scraping.spider import MySpider


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
    championName = queries.get_champ_name(session, championId)
    process = CrawlerProcess()
    process.crawl(MySpider, summoner_obj.name, region, championName)
    scraped = next(iter(process.crawlers)).spider.data
    process.start()
    df_scraped = pd.DataFrame(scraped).T
    if df_scraped.empty:
        return False
    df_scraped = src.utils.clean_summoner_data(df_scraped)
    data = summoner_champion_data
    with session.no_autoflush:
        try:  # TODO: change logic to prevent writing all none when one scraping field fails
            summoner_championmastery_obj = SQLChampionMastery(
                championPointsUntilNextlevel=data["championPointsUntilNextLevel"],
                chestGranted=data["chestGranted"],
                lastPlayTime=data["lastPlayTime"],
                championLevel=data["championLevel"],
                summonerId=data["summonerId"],
                championPoints=data["championPoints"],
                championPointsSinceLastLevel=data["championPointsSinceLastLevel"],
                tokensEarned=data["tokensEarned"],
                wins=df_scraped["wins"].item(),
                loses=df_scraped["loses"].item(),
                championWinrate=df_scraped["winRate"].item(),
                kda=df_scraped["kda"].item(),
                kills=df_scraped["kills"].item(),
                deaths=df_scraped["deaths"].item(),
                assists=df_scraped["assists"].item(),
                lp=df_scraped["lp"].item(),
                maxKills=df_scraped["maxKills"].item(),
                cs=df_scraped["cs"].item(),
                damage=df_scraped["damage"].item(),
                gold=df_scraped["gold"].item(),
            )
        except KeyError:  # TODO: try to scrape normal game data
            logging.warning(
                f"for champion {championId} no scraped data has been found"
            )
            summoner_championmastery_obj = SQLChampionMastery(
                championPointsUntilNextlevel=data["championPointsUntilNextLevel"],
                chestGranted=data["chestGranted"],
                lastPlayTime=data["lastPlayTime"],
                championLevel=data["championLevel"],
                summonerId=data["summonerId"],
                championPoints=data["championPoints"],
                championPointsSinceLastLevel=data["championPointsSinceLastLevel"],
                tokensEarned=data["tokensEarned"],
            )
        summoner_obj.mastery.append(summoner_championmastery_obj)
        champion_obj = queries.get_last_champion(session, championId)
        champion_obj.mastery.append(summoner_championmastery_obj)
        session.add(summoner_championmastery_obj)
    return True
