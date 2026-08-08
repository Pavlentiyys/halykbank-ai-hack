"""Pure covenant comparison rules."""

from decimal import Decimal
from typing import Literal

Direction = Literal["at_least", "at_most"]
Status = Literal["COMPLIANT", "BREACH"]


def compare(actual: Decimal, threshold: Decimal, direction: Direction) -> Status:
    if direction == "at_least":
        return "COMPLIANT" if actual >= threshold else "BREACH"
    return "COMPLIANT" if actual <= threshold else "BREACH"

