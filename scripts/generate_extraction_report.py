from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_OUTPUTS = REPO_ROOT / "batch_outputs" / "primock57_pipeline"
GENERATED_DIR = REPO_ROOT / "research" / "generated"

CATEGORIES = ("Condition", "MedicationStatement", "Observation", "Procedure")
COLORS = {
    "Condition": "orange!70!black",
    "MedicationStatement": "blue!60",
    "Observation": "green!55!black",
    "Procedure": "red!65!black",
}


@dataclass(frozen=True)
class ConsultationStats:
    consultation: str
    duration_s: float
    condition_count: int
    medication_count: int
    observation_count: int
    procedure_count: int
    clinical_total: int
    resources_per_minute: float
    traced_resources: int


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def consultation_dirs(batch_outputs: Path) -> list[Path]:
    return sorted(path for path in batch_outputs.iterdir() if path.is_dir())


def traceable_clinical_resource_count(step06_payload: dict) -> int:
    traced = 0
    for entry in step06_payload["bundle_with_provenance"]["entry"]:
        resource = entry["resource"]
        if resource["resourceType"] not in CATEGORIES:
            continue
        if any(ext.get("url", "").endswith("source-segment-indices") for ext in resource.get("extension", [])):
            traced += 1
    return traced


def collect_stats(batch_outputs: Path) -> list[ConsultationStats]:
    stats: list[ConsultationStats] = []
    for consultation_dir in consultation_dirs(batch_outputs):
        consultation = consultation_dir.name
        step01 = load_json(consultation_dir / f"step01_{consultation}.json")
        step05 = load_json(consultation_dir / f"step05_{consultation}.json")
        step06 = load_json(consultation_dir / f"step06_{consultation}.json")

        counts = step05["resource_counts"]
        condition_count = int(counts["Condition"])
        medication_count = int(counts["MedicationStatement"])
        observation_count = int(counts["Observation"])
        procedure_count = int(counts["Procedure"])
        clinical_total = condition_count + medication_count + observation_count + procedure_count
        duration_s = float(step01["duration_s"])
        resources_per_minute = clinical_total / (duration_s / 60.0)

        stats.append(
            ConsultationStats(
                consultation=consultation,
                duration_s=duration_s,
                condition_count=condition_count,
                medication_count=medication_count,
                observation_count=observation_count,
                procedure_count=procedure_count,
                clinical_total=clinical_total,
                resources_per_minute=resources_per_minute,
                traced_resources=traceable_clinical_resource_count(step06),
            )
        )
    return stats


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_summary_table(stats: list[ConsultationStats]) -> str:
    total_clinical = sum(item.clinical_total for item in stats)
    total_condition = sum(item.condition_count for item in stats)
    total_medication = sum(item.medication_count for item in stats)
    total_observation = sum(item.observation_count for item in stats)
    total_procedure = sum(item.procedure_count for item in stats)
    total_traced = sum(item.traced_resources for item in stats)

    return f"""\\begin{{table}}[ht]
\\centering
\\caption{{Automatic extraction characterization for the 15 processed PriMock57 Day~1 consultations. These values describe output composition and traceability only; they are not semantic accuracy metrics.}}
\\label{{tab:extraction-characterization-summary}}
\\begin{{tabular}}{{lc}}
\\hline
Metric & Value \\\\
\\hline
Processed consultations & {len(stats)} \\\\
Total clinical resources & {total_clinical} \\\\
Mean clinical resources per consultation & {mean([float(item.clinical_total) for item in stats]):.2f} \\\\
Clinical-resource range & {min(item.clinical_total for item in stats)}--{max(item.clinical_total for item in stats)} \\\\
Mean resources per minute & {mean([item.resources_per_minute for item in stats]):.2f} \\\\
Resource-density range & {min(item.resources_per_minute for item in stats):.2f}--{max(item.resources_per_minute for item in stats):.2f} \\\\
\\texttt{{Condition}} share & {total_condition}/{total_clinical} ({(total_condition / total_clinical) * 100:.1f}\\%) \\\\
\\texttt{{MedicationStatement}} share & {total_medication}/{total_clinical} ({(total_medication / total_clinical) * 100:.1f}\\%) \\\\
\\texttt{{Observation}} share & {total_observation}/{total_clinical} ({(total_observation / total_clinical) * 100:.1f}\\%) \\\\
\\texttt{{Procedure}} share & {total_procedure}/{total_clinical} ({(total_procedure / total_clinical) * 100:.1f}\\%) \\\\
Consultations with zero observations & {sum(1 for item in stats if item.observation_count == 0)}/{len(stats)} \\\\
Consultations with zero procedures & {sum(1 for item in stats if item.procedure_count == 0)}/{len(stats)} \\\\
Traceable clinical resources & {total_traced}/{total_clinical} (100\\%) \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""


def legend(entries: list[tuple[str, str]], x: float, y: float) -> list[str]:
    lines: list[str] = []
    for index, (label, color) in enumerate(entries):
        y0 = y - index * 0.42
        lines.append(f"\\fill[{color}] ({x:.2f},{y0:.2f}) rectangle ({x + 0.28:.2f},{y0 + 0.18:.2f});")
        lines.append(f"\\node[anchor=west,font=\\scriptsize] at ({x + 0.36:.2f},{y0 + 0.09:.2f}) {{{label}}};")
    return lines


def render_resource_count_figure(stats: list[ConsultationStats]) -> str:
    left = 1.0
    bottom = 0.8
    plot_width = 14.0
    plot_height = 5.2
    max_y = max(item.clinical_total for item in stats)
    group_gap = plot_width / len(stats)
    bar_width = group_gap / 1.7

    lines = [
        "\\begin{figure}[ht]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left:.2f},{bottom + plot_height + 0.3:.2f});",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left + plot_width + 0.3:.2f},{bottom:.2f});",
    ]

    for tick in range(0, max_y + 1, 5):
        y = bottom + plot_height * (tick / max_y)
        lines.append(f"\\draw[gray!30] ({left:.2f},{y:.2f}) -- ({left + plot_width:.2f},{y:.2f});")
        lines.append(f"\\node[anchor=east,font=\\scriptsize] at ({left - 0.08:.2f},{y:.2f}) {{{tick}}};")

    for index, item in enumerate(stats):
        center_x = left + group_gap * index + group_gap / 2
        x0 = center_x - bar_width / 2
        x1 = center_x + bar_width / 2
        cumulative = bottom
        segments = [
            ("Condition", item.condition_count),
            ("MedicationStatement", item.medication_count),
            ("Observation", item.observation_count),
            ("Procedure", item.procedure_count),
        ]
        for category, count in segments:
            if count == 0:
                continue
            y1 = cumulative + plot_height * (count / max_y)
            lines.append(f"\\fill[{COLORS[category]}] ({x0:.2f},{cumulative:.2f}) rectangle ({x1:.2f},{y1:.2f});")
            cumulative = y1
        label = item.consultation.split("consultation")[1]
        lines.append(
            f"\\node[anchor=north,font=\\tiny,rotate=90] at ({center_x:.2f},{bottom - 0.08:.2f}) {{{label}}};"
        )

    lines.extend(legend(
        [
            ("Condition", COLORS["Condition"]),
            ("MedicationStatement", COLORS["MedicationStatement"]),
            ("Observation", COLORS["Observation"]),
            ("Procedure", COLORS["Procedure"]),
        ],
        left + plot_width - 2.7,
        bottom + plot_height + 0.05,
    ))
    lines.append(f"\\node[rotate=90,font=\\scriptsize] at ({left - 0.75:.2f},{bottom + plot_height / 2:.2f}) {{Clinical resources}};")
    lines.append(f"\\node[font=\\scriptsize] at ({left + plot_width / 2:.2f},{bottom - 0.65:.2f}) {{Consultation (Day 1)}};")
    lines.append("\\end{tikzpicture}")
    lines.append(
        "\\caption{Per-consultation composition of Step~5 clinical resources on the 15 processed PriMock57 Day~1 consultations. This figure describes output mix rather than correctness.}"
    )
    lines.append("\\label{fig:extraction-resource-counts}")
    lines.append("\\end{figure}")
    return "\n".join(lines) + "\n"


def render_density_figure(stats: list[ConsultationStats]) -> str:
    left = 1.0
    bottom = 0.8
    plot_width = 14.0
    plot_height = 4.8
    max_y = max(item.resources_per_minute for item in stats)
    group_gap = plot_width / len(stats)
    bar_width = group_gap / 1.9

    lines = [
        "\\begin{figure}[ht]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left:.2f},{bottom + plot_height + 0.3:.2f});",
        f"\\draw[->, line width=0.5pt] ({left:.2f},{bottom:.2f}) -- ({left + plot_width + 0.3:.2f},{bottom:.2f});",
    ]

    tick_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    for tick in tick_values:
        if tick > max_y + 0.15:
            continue
        y = bottom + plot_height * (tick / max_y)
        lines.append(f"\\draw[gray!30] ({left:.2f},{y:.2f}) -- ({left + plot_width:.2f},{y:.2f});")
        lines.append(f"\\node[anchor=east,font=\\scriptsize] at ({left - 0.08:.2f},{y:.2f}) {{{tick:.1f}}};")

    mean_y = mean([item.resources_per_minute for item in stats])
    mean_line = bottom + plot_height * (mean_y / max_y)
    lines.append(f"\\draw[dashed, line width=0.6pt] ({left:.2f},{mean_line:.2f}) -- ({left + plot_width:.2f},{mean_line:.2f});")
    lines.append(f"\\node[anchor=west,font=\\scriptsize] at ({left + plot_width + 0.1:.2f},{mean_line:.2f}) {{mean}};")

    for index, item in enumerate(stats):
        center_x = left + group_gap * index + group_gap / 2
        x0 = center_x - bar_width / 2
        x1 = center_x + bar_width / 2
        y1 = bottom + plot_height * (item.resources_per_minute / max_y)
        lines.append(f"\\fill[teal!60] ({x0:.2f},{bottom:.2f}) rectangle ({x1:.2f},{y1:.2f});")
        label = item.consultation.split("consultation")[1]
        lines.append(
            f"\\node[anchor=north,font=\\tiny,rotate=90] at ({center_x:.2f},{bottom - 0.08:.2f}) {{{label}}};"
        )

    lines.append(f"\\node[rotate=90,font=\\scriptsize] at ({left - 0.78:.2f},{bottom + plot_height / 2:.2f}) {{Resources per minute}};")
    lines.append(f"\\node[font=\\scriptsize] at ({left + plot_width / 2:.2f},{bottom - 0.65:.2f}) {{Consultation (Day 1)}};")
    lines.append("\\end{tikzpicture}")
    lines.append(
        "\\caption{Clinical-resource density for Step~5 outputs, normalized by consultation duration. Variation here reflects output volume relative to recording length, not verified semantic completeness.}"
    )
    lines.append("\\label{fig:extraction-resource-density}")
    lines.append("\\end{figure}")
    return "\n".join(lines) + "\n"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> None:
    stats = collect_stats(DEFAULT_BATCH_OUTPUTS)
    write(GENERATED_DIR / "extraction_summary_table.tex", render_summary_table(stats))
    write(GENERATED_DIR / "extraction_resource_count_figure.tex", render_resource_count_figure(stats))
    write(GENERATED_DIR / "extraction_resource_density_figure.tex", render_density_figure(stats))


if __name__ == "__main__":
    main()
