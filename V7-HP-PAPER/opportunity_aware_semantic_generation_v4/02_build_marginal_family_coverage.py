#!/usr/bin/env python3
"""Materialize publication-ready v3 marginal family coverage tables."""

from __future__ import annotations

from v4_common import OUTPUTS, ensure_layout, markdown_table, read_json


def main() -> None:
    ensure_layout()
    audit = read_json(OUTPUTS / "audits/v3_family_marginal_coverage.json")
    rows = []
    for family, value in sorted(audit["families"].items()):
        rows.append([
            family,
            value["positive_actions"],
            value["unique_positive_queries"],
            value["positive_queries_already_covered_by_v2"],
            value["new_positive_queries_not_covered_by_v2"],
            f"{value['coverage_after_adding_family_to_v2']:.1%}",
            value["leave_one_family_out_unique_query_loss"],
        ])
    content = "# V3 Marginal Family Coverage\n\n" + markdown_table(
        ["Family", "Positive actions", "Unique queries", "Already v2", "New vs v2", "V2 + family coverage", "LOFO loss"],
        rows,
    ) + "\n"
    path = OUTPUTS / "tables/v3_family_marginal_coverage.md"
    path.write_text(content, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
