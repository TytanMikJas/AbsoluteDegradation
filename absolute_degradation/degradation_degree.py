from enum import Enum


class DegradationDegree(Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"

    @classmethod
    def from_string(cls, s: str) -> "DegradationDegree":
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(f"Unknown degradation degree: {s}")

    @classmethod
    def degradation_degrees(cls) -> list[str]:
        return [degree.value for degree in cls]

    @classmethod
    def from_REDS_index(cls, index: int) -> "DegradationDegree":
        """
        Legacy method to map REDS video index to degradation degree.
        0-9:   Low (LOW)
        10-19: Medium (MEDIUM)
        20+:   High (HIGH)
        """
        if index < 10:
            return cls.LIGHT
        elif index < 20:
            return cls.MEDIUM
        else:
            return cls.HEAVY
