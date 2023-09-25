import logging

import sqlalchemy.orm
from riotwatcher import LolWatcher

from src.crawlers.scraping import scrape_champion_metrics
from src.sqlstore.champion import SQLChampionTags, SQLChampion, SQLChampionStats

import re


def parse_champion_data(
    session: sqlalchemy.orm.Session, watcher: LolWatcher, season: int, patch: int
):
    """parses champion information provided by datadragon and fill corresponding Champion and ChampionStats tables
      WARNING: parses only the brief summary of champion data, if additional data is needed this needs to be reworked
    :param session: sqlalchemy session
    :param watcher: riotwatcher LolWatcher
    :param season: season number
    :param patch: patch number
    :returns: None
    """
    data = watcher.data_dragon.champions(version=f"{season}.{patch}.1", full=False)[
        "data"
    ]
    # the .1 is correct for modern patches, for very old patches (season 4 and older) another solution would be needed

    # Scrape additional metrics from u.gg
    scraped_data = scrape_champion_metrics()

    for champion in data:  # TODO: this can be vastly improved by using bulk inserts
        championdata = data[champion]

        metrics = {
            "Tier": None,
            "Win rate": None,
            "Pick Rate": None,
            "Ban Rate": None,
            "Matches": None,
        }
        try:
            for item in scraped_data.items():
                # Remove special characters and spaces from existing champion name
                clean_existing_champion_name = (
                    re.sub(r"[^\w\s]", "", item[1]["Champion Name"])
                    .replace(" ", "")
                    .lower()
                )
                if clean_existing_champion_name == "wukong":
                    clean_existing_champion_name = "monkeyking"
                if clean_existing_champion_name == "nunuwillump":
                    clean_existing_champion_name = "nunu"
                if clean_existing_champion_name == "renataglasc":
                    clean_existing_champion_name = "renata"
                if clean_existing_champion_name == champion.lower():
                    metrics = item[1]
                    break
        except KeyError as e:
            logging.warning(str(e))
        champion_obj = SQLChampion(
            championNumber=int(championdata["key"]),
            championName=championdata["name"],
            championTitle=championdata["title"],
            infoAttack=championdata["info"]["attack"],
            infoDefense=championdata["info"]["defense"],
            infoMagic=championdata["info"]["magic"],
            infoDifficulty=championdata["info"]["difficulty"],
            seasonNumber=season,
            patchNumber=patch,
            tier=metrics["Tier"],
            win_rate=metrics["Win rate"],
            pick_rate=metrics["Pick Rate"],
            ban_rate=metrics["Ban Rate"],
            matches=metrics["Matches"],
        )

        # Use scraped_data to populate fields in SQLChampion
        session.add(champion_obj)

        session.commit()  # this commit is needed to get the generated champion_obj id
        stats = data[champion]["stats"]
        championStats_obj = SQLChampionStats(
            championId=champion_obj.id,
            hp=stats["hp"],
            hpperlevel=stats["hpperlevel"],
            mp=stats["mp"],
            mpperlevel=stats["mpperlevel"],
            movespeed=stats["movespeed"],
            armor=stats["armor"],
            armorperlevel=stats["armorperlevel"],
            spellblock=stats["spellblock"],
            spellblockperlevel=stats["spellblockperlevel"],
            attackrange=stats["attackrange"],
            hpregen=stats["hpregen"],
            hpregenperlevel=stats["hpregenperlevel"],
            mpregen=stats["mpregen"],
            mpregenperlevel=stats["mpregenperlevel"],
            crit=stats["crit"],
            critperlevel=stats["critperlevel"],
            attackdamage=stats["attackdamage"],
            attackdamageperlevel=stats["attackdamage"],
            attackspeed=stats["attackspeed"],
            patchNumber=patch,
            seasonNumber=season,
        )
        champion_obj.stats.append(championStats_obj)
        session.add(championStats_obj)
        # TODO: add champion roles with data from webscraping
        tags = championdata.get("tags")
        length = len(tags)
        tag1 = tags[0] if 0 < length else None
        tag2 = tags[1] if 1 < length else None
        championTags_obj = SQLChampionTags(champion_obj.id, tag1, tag2)
        champion_obj.tags.append(championTags_obj)
        session.add(championTags_obj)
        session.commit()
    session.commit()
