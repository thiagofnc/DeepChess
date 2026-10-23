"""Print one downloaded game with all its fields."""

import json


with open("data/regressor_games.jsonl", encoding="utf-8") as games:
    record = json.loads(next(games))

print(json.dumps(record, indent=2, ensure_ascii=False))
