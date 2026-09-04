#!/usr/bin/env python3
from final_common import audit_claim_consistency


if __name__ == "__main__":
    result = audit_claim_consistency()
    if result is not None:
        print(result)
