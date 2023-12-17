from fuzzywuzzy import fuzz

class isCache:
    def __init__(self, status, closest_match) -> None:
        self.status = status
        self.closest_match = closest_match

def in_cache (message: str, keysList: list):
    highest_score = 0
    for s in keysList:
        score = fuzz.partial_token_sort_ratio(message, s)
        if score > highest_score:
            highest_score = score
    if highest_score > 90:
        return isCache(True, s)
    return isCache(False, None)
        