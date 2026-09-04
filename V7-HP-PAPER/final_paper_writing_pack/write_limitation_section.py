#!/usr/bin/env python3
from final_common import write_limitation_section


if __name__ == "__main__":
    result = write_limitation_section()
    if result is not None:
        print(result)
