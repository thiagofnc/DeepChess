"""A constant regressor: predict the average eval for every position."""

import json
from statistics import mean


scores = []
with open("data/regressor_games.jsonl", encoding="utf-8") as games:
    for line in games:
        moves = json.loads(line)["movetext"]
        for annotation in moves.split("[%eval ")[1:]:
            score = annotation.split("]")[0].split(",")[0].strip()
            if not score.startswith("#"):
                scores.append(float(score))

average_eval = mean(scores)


def predict(position):
    return average_eval


if __name__ == "__main__":
    print(round(predict(None), 4))
