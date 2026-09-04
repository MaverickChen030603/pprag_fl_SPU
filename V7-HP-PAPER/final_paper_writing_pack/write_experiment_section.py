#!/usr/bin/env python3
from final_common import write_experiment_section


if __name__ == "__main__":
    result = write_experiment_section()
    if result is not None:
        print(result)
