#!/usr/bin/env python3
from final_common import write_conclusion_section


if __name__ == "__main__":
    result = write_conclusion_section()
    if result is not None:
        print(result)
