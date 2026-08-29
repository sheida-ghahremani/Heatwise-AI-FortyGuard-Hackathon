from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgeGroup(str, Enum):
    YOUNG = "18–34"
    MIDDLE = "35–64"
    OLDER = "65+"


class Activity(str, Enum):
    WALKING = "Walking"
    CYCLING = "Cycling"
    JOGGING = "Jogging"


class Clothing(str, Enum):
    VERY_LIGHT = "Very light summer clothing (0.3 clo)"
    LIGHT_SUMMER = "Light summer clothing (0.5 clo)"
    LONG_SLEEVES = "Long sleeves and trousers (0.7 clo)"
    PROTECTIVE = "Work or protective clothing (1.0 clo)"


class RoutePreference(str, Enum):
    FASTEST = "Fastest"
    LOWEST_RISK = "Lowest Heat Risk"
    BALANCED = "Balanced"


@dataclass(frozen=True)
class UserProfile:
    age_group: AgeGroup = AgeGroup.YOUNG
    activity: Activity = Activity.WALKING
    clothing: Clothing = Clothing.LIGHT_SUMMER

    @property
    def speed_kph(self) -> float:
        return {
            Activity.WALKING: 4.8,
            Activity.CYCLING: 14.0,
            Activity.JOGGING: 8.0,
        }[self.activity]

    @property
    def metabolic_rate_met(self) -> float:
        """2024 Adult Compendium representative metabolic intensity."""
        return {
            Activity.WALKING: 3.3,
            Activity.CYCLING: 6.8,
            Activity.JOGGING: 7.0,
        }[self.activity]

    @property
    def representative_age(self) -> int:
        return {AgeGroup.YOUNG: 26, AgeGroup.MIDDLE: 50, AgeGroup.OLDER: 70}[self.age_group]

    @property
    def summer_clothing_clo(self) -> float:
        return {
            Clothing.VERY_LIGHT: 0.3,
            Clothing.LIGHT_SUMMER: 0.5,
            Clothing.LONG_SLEEVES: 0.7,
            Clothing.PROTECTIVE: 1.0,
        }[self.clothing]
