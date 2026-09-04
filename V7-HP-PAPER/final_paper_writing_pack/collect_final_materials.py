#!/usr/bin/env python3
from final_common import collect_final_materials


if __name__ == "__main__":
    result = collect_final_materials()
    if result is not None:
        print(result)
