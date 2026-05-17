from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT_DIR / "data" / "student_dropout.csv"

GENDERS = ["Male", "Female", "Other"]
COURSE_TYPES = ["Programming", "Math", "English", "Data Science", "Business"]
PREVIOUS_GRADES = ["A", "B", "C", "D", "F"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def choose_previous_grade(risk_factor: float) -> str:
    if risk_factor < 0.25:
        weights = [0.35, 0.35, 0.20, 0.08, 0.02]
    elif risk_factor < 0.55:
        weights = [0.15, 0.32, 0.32, 0.16, 0.05]
    else:
        weights = [0.05, 0.18, 0.34, 0.28, 0.15]
    return random.choices(PREVIOUS_GRADES, weights=weights, k=1)[0]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def make_student(index: int) -> dict[str, object]:
    risk_factor = random.betavariate(2.1, 2.4)
    age = int(clamp(round(random.gauss(22 + risk_factor * 2.5, 2.4)), 18, 35))
    gender = random.choices(GENDERS, weights=[0.48, 0.48, 0.04], k=1)[0]
    course_type = random.choice(COURSE_TYPES)
    previous_grade = choose_previous_grade(risk_factor)

    attendance_rate = clamp(random.gauss(91 - risk_factor * 48, 7), 20, 100)
    assignment_submit_rate = clamp(random.gauss(88 - risk_factor * 50, 9), 10, 100)
    quiz_average = clamp(random.gauss(84 - risk_factor * 43, 10), 5, 100)
    login_count = clamp(random.gauss(33 - risk_factor * 25, 5), 0, 60)
    video_watch_time = clamp(random.gauss(420 - risk_factor * 290, 60), 20, 620)
    forum_activity = clamp(random.gauss(9 - risk_factor * 7, 2), 0, 18)

    grade_risk = {"A": -0.55, "B": -0.25, "C": 0.15, "D": 0.50, "F": 0.85}[previous_grade]
    score = (
        -3.2
        + risk_factor * 2.5
        + (70 - attendance_rate) / 18
        + (70 - assignment_submit_rate) / 20
        + (60 - quiz_average) / 22
        + (12 - login_count) / 8
        + (180 - video_watch_time) / 130
        + (2 - forum_activity) / 3
        + grade_risk
    )
    dropout_probability = sigmoid(score)
    dropout = 1 if random.random() < dropout_probability else 0

    return {
        "student_id": f"S{index:04d}",
        "gender": gender,
        "age": age,
        "course_type": course_type,
        "attendance_rate": round(attendance_rate, 1),
        "assignment_submit_rate": round(assignment_submit_rate, 1),
        "quiz_average": round(quiz_average, 1),
        "login_count": int(round(login_count)),
        "video_watch_time": int(round(video_watch_time)),
        "forum_activity": int(round(forum_activity)),
        "previous_grade": previous_grade,
        "dropout": dropout,
    }


def generate_dataset(row_count: int, seed: int) -> list[dict[str, object]]:
    random.seed(seed)
    return [make_student(index) for index in range(1, row_count + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EduRiskAI student data.")
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    rows = generate_dataset(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dropout_count = sum(int(row["dropout"]) for row in rows)
    print(f"saved: {args.output}")
    print(f"rows: {len(rows)}")
    print(f"dropout=1: {dropout_count} ({dropout_count / len(rows):.1%})")


if __name__ == "__main__":
    main()
