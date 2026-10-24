from enum import Enum


class BlendMode(Enum):
    ADDITION = 0
    SUBTRACT = 1
    MULTIPLY = 2

    @classmethod
    def from_int(cls, value: int) -> "BlendMode":
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"No BlendMode with value {value}")

    @classmethod
    def random_mode(cls) -> "BlendMode":
        import random

        return random.choice(list(cls))
