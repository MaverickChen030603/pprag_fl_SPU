#!/usr/bin/env python3
from final_common import build_reviewer_response_brief


if __name__ == "__main__":
    result = build_reviewer_response_brief()
    if result is not None:
        print(result)
