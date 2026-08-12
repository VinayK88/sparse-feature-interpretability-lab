from __future__ import annotations

from html import escape
from pathlib import Path


def write_recovery_svg(result: dict, destination: Path) -> None:
    values = result["runs"][0]["recovery"]["best_similarity_by_feature"]
    width, height = 940, 420
    left, top, plot_width, plot_height = 70, 70, 820, 270
    bar_width = plot_width / len(values)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" font-family="system-ui" font-size="22" font-weight="700" fill="#172033">Ground-truth feature recovery</text>',
        '<text x="40" y="56" font-family="system-ui" font-size="12" fill="#64748B">Best absolute cosine similarity to any learned decoder direction · seed 3</text>',
    ]
    for tick in (0.0, 0.5, 0.8, 1.0):
        y = top + plot_height * (1 - tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#E2E8F0"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="system-ui" font-size="11" fill="#64748B">{tick:.1f}</text>')
    for index, value in enumerate(values):
        x = left + index * bar_width + 2
        y = top + plot_height * (1 - value)
        color = "#2563EB" if value >= 0.8 else "#D4A72C"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 4:.1f}" height="{top + plot_height - y:.1f}" fill="{color}"/>')
        if index % 2 == 0:
            parts.append(f'<text x="{x + bar_width / 2 - 2:.1f}" y="{top + plot_height + 18}" text-anchor="middle" font-family="system-ui" font-size="9" fill="#64748B">{index}</text>')
    parts.extend([
        f'<text x="{left + plot_width / 2}" y="{height - 30}" text-anchor="middle" font-family="system-ui" font-size="12" fill="#172033">Ground-truth feature index</text>',
        '</svg>',
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")
