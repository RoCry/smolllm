import random
from typing import Dict, Tuple


class SimpleBalancer:
    """Balances API key and URL pairs usage."""

    def __init__(self):
        self.pair_usage: Dict[Tuple[str, str], int] = {}

    def _parse_items(self, items: str) -> list[str]:
        return [item.strip() for item in items.split(",")]

    def choose_pair(self, keys: str, urls: str) -> Tuple[str, str]:
        """Choose a key-URL pair based on usage balancing.

        Cases:
        1. 1 key, 1 url -> single pair
        2. 1 key, n urls -> key paired with each url
        3. n keys, n urls -> matching pairs
        4. n keys, 1 url -> each key paired with url
        """
        key_list = self._parse_items(keys)
        url_list = self._parse_items(urls)

        # Generate valid pairs based on the cases
        if len(url_list) == 1:
            pairs = [(key, url_list[0]) for key in key_list]
        elif len(key_list) == 1:
            pairs = [(key_list[0], url) for url in url_list]
        else:
            if len(key_list) != len(url_list):
                raise ValueError("When using multiple keys and URLs, their counts must match")
            pairs = list(zip(key_list, url_list))

        # Initialize usage count for new pairs
        for pair in pairs:
            if pair not in self.pair_usage:
                self.pair_usage[pair] = 0

        # Find and choose least used pair
        min_usage = min(self.pair_usage[pair] for pair in pairs)
        least_used_pairs = [pair for pair in pairs if self.pair_usage[pair] == min_usage]
        chosen_pair = random.choice(least_used_pairs)
        self.pair_usage[chosen_pair] += 1

        return chosen_pair


# Create a global instance
balancer = SimpleBalancer()
