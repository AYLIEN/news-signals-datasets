from typing import Any


class SemanticFilter:

    def __call__(self, item: Any) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StoryKeywordMatchFilter(SemanticFilter):

    def __init__(self, keywords: list[str]):
        self.keywords = keywords

    def __call__(self, item: dict) -> bool:
        return any(kw in item['title'] for kw in self.keywords)
