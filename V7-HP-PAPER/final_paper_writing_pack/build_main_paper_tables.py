#!/usr/bin/env python3
from final_common import build_main_paper_tables


if __name__ == "__main__":
    result = build_main_paper_tables()
    if result is not None:
        print(result)
