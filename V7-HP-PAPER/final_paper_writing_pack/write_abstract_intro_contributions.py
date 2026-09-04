#!/usr/bin/env python3
from final_common import write_abstract_intro_contributions


if __name__ == "__main__":
    result = write_abstract_intro_contributions()
    if result is not None:
        print(result)
