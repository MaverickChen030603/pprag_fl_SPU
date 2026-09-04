#!/usr/bin/env python3
from final_common import build_appendix_tables


if __name__ == "__main__":
    result = build_appendix_tables()
    if result is not None:
        print(result)
