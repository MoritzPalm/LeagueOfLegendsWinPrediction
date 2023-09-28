import re

from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

from src.sqlstore.queries import scraping_needed, update_mastery
from src.sqlstore.db import get_session


class DataPipeline:
    def process_item(self, item, spider):
        """
        checks if champion-summoner combintion is needed and if so, adds it to the database
        :param item:
        :param spider:
        :return:
        """
        adapter = ItemAdapter(item)
        url = adapter.get("url")
        match = re.match(r"https:\/\/u\.gg\/lol\/profile\/(.+)\/(.+)\/champion-stats", url)
        region = match.group(1)
        summonerName = match.group(2)
        champion = adapter.get("champion")
        with get_session() as session:
            if scraping_needed(session, region, summonerName, champion):
                # update database entry with data from item
                update_mastery(session, item, region, summonerName, champion)
                return item
            else:
                raise DropItem(f"Scraping not needed for {region} {summonerName} {champion}")
