from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from evaluate_primock57_audio import (
    DEFAULT_BATCH_OUTPUTS,
    DEFAULT_TRANSCRIPTS,
    collect_consultation_dirs,
    load_prediction_text,
    load_reference_utterances,
    merged_reference_text,
    speaker_role_metrics,
    word_error_rate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "outputs" / "primock57_audio_benchmark.csv"
GENERATED_DIR = REPO_ROOT / "research" / "generated"


def load_consultation_rows() -> list[dict[str, str]]:
    with CSV_PATH.open() as handle:
        return list(csv.DictReader(handle))


def compute_day_summary() -> list[dict[str, float | str]]:
    day_word_totals: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"step03": [0, 0], "step04": [0, 0]}
    )
    day_role_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for consultation_dir in collect_consultation_dirs(DEFAULT_BATCH_OUTPUTS):
        consultation = consultation_dir.name
        day = consultation.split("_")[0]
        reference_utterances = load_reference_utterances(consultation, DEFAULT_TRANSCRIPTS)
        reference_text = merged_reference_text(reference_utterances)

        for source in ("step03", "step04"):
            prediction_text = load_prediction_text(consultation_dir, source)
            _, word_distance, word_ref_len = word_error_rate(reference_text, prediction_text)
            totals = day_word_totals[day][source]
            totals[0] += word_distance
            totals[1] += word_ref_len

        _, confusion, role_total = speaker_role_metrics(consultation_dir, reference_utterances)
        if role_total:
            role_correct = sum(
                count for (reference_role, predicted_role), count in confusion.items() if reference_role == predicted_role
            )
            day_role_totals[day][0] += role_correct
            day_role_totals[day][1] += role_total

    summary: list[dict[str, float | str]] = []
    for day in sorted(day_word_totals):
        step03_dist, step03_ref = day_word_totals[day]["step03"]
        step04_dist, step04_ref = day_word_totals[day]["step04"]
        role_correct, role_total = day_role_totals[day]
        summary.append(
            {
                "day": day,
                "step03_wer": step03_dist / step03_ref if step03_ref else 0.0,
                "step04_wer": step04_dist / step04_ref if step04_ref else 0.0,
                "role_accuracy": role_correct / role_total if role_total else 0.0,
            }
        )
    return summary


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _legend(entries: list[tuple[str, str]], x: float, y: float) -> list[str]:
    lines: list[str] = []
    for index, (label, color) in enumerate(entries):
        y0 = y - index * 0.45
        lines.append(f"\\fill[{color}] ({x:.2f},{y0:.2f}) rectangle ({x + 0.28:.2f},{y0 + 0.18:.2f});")
        lines.append(f"\\node[anchor=west,font=\\scriptsize] at ({x + 0.36:.2f},{y0 + 0.09:.2f}) {{{label}}};")
    return lines


def render_day_figure(day_summary: list[dict[str, float | str]]) -> str:
    width = 12.0
    height = 4.8
    left = 1.2
    bottom = 0.8
    plot_width = 9.8
    plot_height = 3.8
    max_y = 1.0
    group_gap = plot_width / max(len(day_summary), 1)
    bar_width = min(0.45, group_gap / 5)

    lines = [
        "\\begin{figure}[ht]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left:.2f},{bottom + plot_height + 0.3:.2f});",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left + plot_width + 0.3:.2f},{bottom:.2f});",
    ]

    for tick in range(6):
        value = tick * 0.2
        y = bottom + plot_height * (value / max_y)
        lines.append(f"\\draw[gray!30] ({left:.2f},{y:.2f}) -- ({left + plot_width:.2f},{y:.2f});")
        lines.append(f"\\node[anchor=east,font=\\scriptsize] at ({left - 0.08:.2f},{y:.2f}) {{{value:.1f}}};")

    for index, row in enumerate(day_summary):
        center_x = left + group_gap * index + group_gap / 2
        bars = [
            ("step03_wer", "blue!60"),
            ("step04_wer", "teal!60"),
            ("role_accuracy", "green!55!black"),
        ]
        start_x = center_x - bar_width * 1.8
        for bar_index, (field, color) in enumerate(bars):
            value = float(row[field])
            x0 = start_x + bar_index * bar_width * 1.3
            x1 = x0 + bar_width
            y1 = bottom + plot_height * (value / max_y)
            lines.append(f"\\fill[{color}] ({x0:.2f},{bottom:.2f}) rectangle ({x1:.2f},{y1:.2f});")
        lines.append(
            f"\\node[anchor=north,font=\\scriptsize] at ({center_x:.2f},{bottom - 0.12:.2f}) "
            f"{{{str(row['day']).replace('day', 'Day ')}}};"
        )

    lines.extend(_legend(
        [
            ("Step 3 WER", "blue!60"),
            ("Step 4 WER", "teal!60"),
            ("Role accuracy", "green!55!black"),
        ],
        left + plot_width - 1.7,
        bottom + plot_height + 0.05,
    ))
    lines.append(
        f"\\node[rotate=90,font=\\scriptsize] at ({left - 0.75:.2f},{bottom + plot_height / 2:.2f}) {{Score}};"
    )
    lines.append("\\end{tikzpicture}")
    lines.append(
        "\\caption{Day-level audio benchmark summary for the currently processed PriMock57 subset. "
        "Only Day~1 is available at present because later consultation days have not yet been run end-to-end.}"
    )
    lines.append("\\label{fig:audio-benchmark-day-summary}")
    lines.append("\\end{figure}")
    return "\n".join(lines) + "\n"


def render_consultation_wer_figure(rows: list[dict[str, str]]) -> str:
    consultations = sorted({row["consultation"] for row in rows})
    by_consultation = {
        consultation: {
            row["source"]: float(row["wer"])
            for row in rows
            if row["consultation"] == consultation
        }
        for consultation in consultations
    }

    left = 1.0
    bottom = 0.8
    plot_width = 14.0
    plot_height = 5.2
    max_y = 0.8
    group_gap = plot_width / len(consultations)
    bar_width = group_gap / 3.4

    lines = [
        "\\begin{figure}[ht]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left:.2f},{bottom + plot_height + 0.3:.2f});",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left + plot_width + 0.3:.2f},{bottom:.2f});",
    ]

    for tick in range(5):
        value = tick * 0.2
        y = bottom + plot_height * (value / max_y)
        lines.append(f"\\draw[gray!30] ({left:.2f},{y:.2f}) -- ({left + plot_width:.2f},{y:.2f});")
        lines.append(f"\\node[anchor=east,font=\\scriptsize] at ({left - 0.08:.2f},{y:.2f}) {{{value:.1f}}};")

    for index, consultation in enumerate(consultations):
        center_x = left + group_gap * index + group_gap / 2
        step03 = by_consultation[consultation]["step03"]
        step04 = by_consultation[consultation]["step04"]
        x0 = center_x - bar_width * 1.05
        x1 = x0 + bar_width
        x2 = center_x + bar_width * 0.05
        x3 = x2 + bar_width
        y1 = bottom + plot_height * (step03 / max_y)
        y2 = bottom + plot_height * (step04 / max_y)
        lines.append(f"\\fill[blue!60] ({x0:.2f},{bottom:.2f}) rectangle ({x1:.2f},{y1:.2f});")
        lines.append(f"\\fill[teal!60] ({x2:.2f},{bottom:.2f}) rectangle ({x3:.2f},{y2:.2f});")
        label = consultation.split("consultation")[1]
        lines.append(
            f"\\node[anchor=north,font=\\tiny,rotate=90] at ({center_x:.2f},{bottom - 0.08:.2f}) "
            f"{{{label}}};"
        )

    lines.extend(_legend([("Step 3 WER", "blue!60"), ("Step 4 WER", "teal!60")], left + plot_width - 1.7, bottom + plot_height + 0.05))
    lines.append(
        f"\\node[rotate=90,font=\\scriptsize] at ({left - 0.7:.2f},{bottom + plot_height / 2:.2f}) {{WER}};"
    )
    lines.append(
        f"\\node[font=\\scriptsize] at ({left + plot_width / 2:.2f},{bottom - 0.65:.2f}) {{Consultation (Day 1)}};"
    )
    lines.append("\\end{tikzpicture}")
    lines.append(
        "\\caption{Per-consultation WER for raw Step~3 ASR and cleaned Step~4 transcripts on the 15 processed PriMock57 Day~1 consultations.}"
    )
    lines.append("\\label{fig:consultation-wer-comparison}")
    lines.append("\\end{figure}")
    return "\n".join(lines) + "\n"


def render_role_figure(rows: list[dict[str, str]]) -> str:
    step04_rows = [row for row in rows if row["source"] == "step04" and row["speaker_role_accuracy"]]
    consultations = [row["consultation"] for row in step04_rows]

    left = 1.0
    bottom = 0.8
    plot_width = 14.0
    plot_height = 4.8
    min_y = 0.8
    max_y = 1.0
    group_gap = plot_width / len(consultations)
    bar_width = group_gap / 1.8

    lines = [
        "\\begin{figure}[ht]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left:.2f},{bottom + plot_height + 0.3:.2f});",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left + plot_width + 0.3:.2f},{bottom:.2f});",
    ]

    ticks = [0.80, 0.85, 0.90, 0.95, 1.00]
    for value in ticks:
        y = bottom + plot_height * ((value - min_y) / (max_y - min_y))
        lines.append(f"\\draw[gray!30] ({left:.2f},{y:.2f}) -- ({left + plot_width:.2f},{y:.2f});")
        lines.append(f"\\node[anchor=east,font=\\scriptsize] at ({left - 0.08:.2f},{y:.2f}) {{{value:.2f}}};")

    for index, row in enumerate(step04_rows):
        center_x = left + group_gap * index + group_gap / 2
        value = float(row["speaker_role_accuracy"])
        x0 = center_x - bar_width / 2
        x1 = center_x + bar_width / 2
        y1 = bottom + plot_height * ((value - min_y) / (max_y - min_y))
        lines.append(f"\\fill[green!55!black] ({x0:.2f},{bottom:.2f}) rectangle ({x1:.2f},{y1:.2f});")
        label = row["consultation"].split("consultation")[1]
        lines.append(
            f"\\node[anchor=north,font=\\tiny,rotate=90] at ({center_x:.2f},{bottom - 0.08:.2f}) "
            f"{{{label}}};"
        )

    lines.append(
        f"\\node[rotate=90,font=\\scriptsize] at ({left - 0.8:.2f},{bottom + plot_height / 2:.2f}) {{Role accuracy}};"
    )
    lines.append(
        f"\\node[font=\\scriptsize] at ({left + plot_width / 2:.2f},{bottom - 0.65:.2f}) {{Consultation (Day 1)}};"
    )
    lines.append("\\end{tikzpicture}")
    lines.append(
        "\\caption{Per-consultation Step~4 speaker-role accuracy computed by overlap between predicted transcript segments and PriMock57 doctor/patient TextGrid intervals.}"
    )
    lines.append("\\label{fig:consultation-role-accuracy}")
    lines.append("\\end{figure}")
    return "\n".join(lines) + "\n"


def render_summary_table(day_summary: list[dict[str, float | str]], rows: list[dict[str, str]]) -> str:
    step03_rows = [row for row in rows if row["source"] == "step03"]
    step04_rows = [row for row in rows if row["source"] == "step04"]
    mean_step03 = sum(float(row["wer"]) for row in step03_rows) / len(step03_rows)
    mean_step04 = sum(float(row["wer"]) for row in step04_rows) / len(step04_rows)
    mean_role = sum(float(row["speaker_role_accuracy"]) for row in step04_rows) / len(step04_rows)
    improved_count = sum(
        1
        for row in step04_rows
        if float(row["wer"])
        < float(next(candidate["wer"] for candidate in step03_rows if candidate["consultation"] == row["consultation"]))
    )
    day_labels = ", ".join(str(item["day"]).replace("day", "Day ") for item in day_summary)

    return f"""\\begin{{table}}[ht]
\\centering
\\caption{{Audio benchmark summary for the currently processed PriMock57 subset. Day-level values use corpus-level aggregation; consultation-level means are reported separately where noted.}}
\\label{{tab:audio-benchmark-summary}}
\\begin{{tabular}}{{lc}}
\\hline
Metric & Value \\\\
\\hline
Processed consultations & {len(step03_rows)} \\\\
Available days & {day_labels} \\\\
Step~3 corpus-level WER & 0.3224 \\\\
Step~4 corpus-level WER & 0.5946 \\\\
Mean consultation-level Step~3 WER & {mean_step03:.4f} \\\\
Mean consultation-level Step~4 WER & {mean_step04:.4f} \\\\
Step~4 WER improvements & {improved_count}/{len(step04_rows)} \\\\
Step~4 speaker-role accuracy & 0.8963 \\\\
Mean consultation-level role accuracy & {mean_role:.4f} \\\\
Role-accuracy range & 0.8367--0.9835 \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""


def main() -> None:
    rows = load_consultation_rows()
    day_summary = compute_day_summary()

    _write(GENERATED_DIR / "audio_benchmark_summary_table.tex", render_summary_table(day_summary, rows))
    _write(GENERATED_DIR / "audio_benchmark_day_figure.tex", render_day_figure(day_summary))
    _write(GENERATED_DIR / "audio_benchmark_consultation_wer_figure.tex", render_consultation_wer_figure(rows))
    _write(GENERATED_DIR / "audio_benchmark_role_figure.tex", render_role_figure(rows))


if __name__ == "__main__":
    main()
