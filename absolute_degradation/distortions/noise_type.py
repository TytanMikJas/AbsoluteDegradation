from enum import Enum


class NoiseType(Enum):
    """
    Enum for different types of noise applied to images.
    """

    HETEROSCEDASTIC_FILM_GRAIN = "heteroscedastic_film_grain"
    HOMOSCEDASTIC_FILM_GRAIN = "homoscedastic_film_grain"
    GAUSSIAN = "gaussian"
    SPECKLE = "speckle"

    @classmethod
    def from_string(cls, s: str) -> "NoiseType":
        """
        Convert a string to a NoiseType enum value.

        Args:
            s: String representation of the noise type.

        Returns:
            Corresponding NoiseType enum value.

        Raises:
            ValueError: If the string doesn't match any noise type.
        """
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(f"Unknown noise type: {s}")

    @classmethod
    def all_types(cls) -> list["NoiseType"]:
        """
        Get a list of all noise types.

        Returns:
            List of all NoiseType enum values.
        """
        return list(cls)

    @classmethod
    def all_values(cls) -> list[str]:
        """
        Get a list of all noise type string values.

        Returns:
            List of all noise type string values.
        """
        return [noise_type.value for noise_type in cls]
