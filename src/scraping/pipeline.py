from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class DataPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        if adapter.get("champion") == spider.champion:
            spider.data[adapter.get("rank")] = adapter.asdict()
            return item
        else:
            raise DropItem(f"Champion {spider.champion} not found.")
