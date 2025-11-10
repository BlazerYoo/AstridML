"""Synthetic Data Generator for wearable and menstrual cycle data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


class SyntheticDataGenerator:
    """
    Generates synthetic data for wearable training metrics and menstrual cycle symptoms.

    The generator creates realistic correlations between menstrual cycle phases and
    athletic performance metrics, as well as symptom patterns.
    """

    # Menstrual cycle phases (typical 28-day cycle)
    CYCLE_PHASES = {
        'menstrual': (1, 5),      # Days 1-5
        'follicular': (6, 13),     # Days 6-13
        'ovulatory': (14, 16),     # Days 14-16
        'luteal': (17, 28)         # Days 17-28
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the synthetic data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def _get_cycle_phase(self, day: int) -> str:
        """Determine menstrual cycle phase for a given cycle day."""
        cycle_day = ((day - 1) % 28) + 1
        for phase, (start, end) in self.CYCLE_PHASES.items():
            if start <= cycle_day <= end:
                return phase
        return 'luteal'

    def _generate_cycle_modifiers(self, phase: str) -> dict:
        """
        Generate performance and symptom modifiers based on cycle phase.

        Research shows correlations between cycle phases and various metrics:
        - Follicular/Ovulatory: Higher energy, better performance potential
        - Luteal: More fatigue, potential for mood changes
        - Menstrual: Variable energy, potential for cramping/pain
        """
        modifiers = {
            'menstrual': {
                'energy': -0.15,
                'pain': 0.25,
                'mood': -0.10,
                'sleep_quality': -0.10,
                'resting_hr': 0.02,
                'performance': -0.08
            },
            'follicular': {
                'energy': 0.10,
                'pain': -0.20,
                'mood': 0.15,
                'sleep_quality': 0.05,
                'resting_hr': -0.03,
                'performance': 0.12
            },
            'ovulatory': {
                'energy': 0.15,
                'pain': -0.15,
                'mood': 0.10,
                'sleep_quality': 0.02,
                'resting_hr': 0.01,
                'performance': 0.10
            },
            'luteal': {
                'energy': -0.10,
                'pain': 0.10,
                'mood': -0.15,
                'sleep_quality': -0.08,
                'resting_hr': 0.04,
                'performance': -0.05
            }
        }
        return modifiers[phase]

    def generate_wearable_data(
        self,
        n_days: int = 90,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic wearable device data.

        Args:
            n_days: Number of days of data to generate
            start_date: Starting date for the data (defaults to today)

        Returns:
            DataFrame with wearable metrics
        """
        if start_date is None:
            start_date = datetime.now()

        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        data = []

        for day_idx, date in enumerate(dates):
            phase = self._get_cycle_phase(day_idx)
            mods = self._generate_cycle_modifiers(phase)

            # Base values with noise
            base_resting_hr = 60 + self.rng.normal(0, 3)
            base_hrv = 65 + self.rng.normal(0, 8)
            base_sleep_hours = 7.5 + self.rng.normal(0, 0.8)
            base_steps = 8000 + self.rng.normal(0, 1500)
            base_calories = 2000 + self.rng.normal(0, 200)

            # Apply cycle phase modifiers
            resting_hr = base_resting_hr * (1 + mods['resting_hr'])
            hrv = base_hrv * (1 + mods['performance'] * 0.5)
            sleep_hours = base_sleep_hours * (1 + mods['sleep_quality'])
            sleep_quality = max(0, min(100, 75 + mods['sleep_quality'] * 100 + self.rng.normal(0, 10)))
            steps = int(base_steps * (1 + mods['energy'] * 0.5))
            active_minutes = int(45 + mods['energy'] * 30 + self.rng.normal(0, 10))
            calories_burned = int(base_calories * (1 + mods['performance'] * 0.3))

            # Training load (varies by day of week and energy)
            is_weekend = date.weekday() >= 5
            training_load = (40 if is_weekend else 60) * (1 + mods['performance']) + self.rng.normal(0, 10)
            training_load = max(0, training_load)

            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'resting_heart_rate': round(resting_hr, 1),
                'heart_rate_variability': round(hrv, 1),
                'sleep_hours': round(sleep_hours, 2),
                'sleep_quality_score': round(sleep_quality, 1),
                'steps': steps,
                'active_minutes': active_minutes,
                'calories_burned': calories_burned,
                'training_load': round(training_load, 1),
                'cycle_day': ((day_idx) % 28) + 1,
                'cycle_phase': phase
            })

        return pd.DataFrame(data)

    def generate_symptom_data(
        self,
        n_days: int = 90,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic menstrual cycle symptom data.

        Args:
            n_days: Number of days of data to generate
            start_date: Starting date for the data (defaults to today)

        Returns:
            DataFrame with symptom data
        """
        if start_date is None:
            start_date = datetime.now()

        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        data = []

        for day_idx, date in enumerate(dates):
            phase = self._get_cycle_phase(day_idx)
            mods = self._generate_cycle_modifiers(phase)

            # Scale modifiers to 1-10 scale
            energy_level = max(1, min(10, 6 + mods['energy'] * 10 + self.rng.normal(0, 1)))
            mood_score = max(1, min(10, 7 + mods['mood'] * 10 + self.rng.normal(0, 1)))
            pain_level = max(0, min(10, 2 + mods['pain'] * 10 + self.rng.normal(0, 1.5)))

            # Other symptoms influenced by cycle phase
            bloating = max(0, min(10, 3 + (mods['pain'] * 8) + self.rng.normal(0, 1.5)))
            breast_tenderness = max(0, min(10, 2 + (mods['pain'] * 7) + self.rng.normal(0, 1.5)))

            # Track if currently menstruating
            cycle_day = ((day_idx) % 28) + 1
            is_menstruating = 1 <= cycle_day <= 5
            flow_level = 0
            if is_menstruating:
                # Flow typically peaks around day 2-3
                if cycle_day <= 2:
                    flow_level = self.rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
                elif cycle_day <= 4:
                    flow_level = self.rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                else:
                    flow_level = self.rng.choice([1, 2], p=[0.7, 0.3])

            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'cycle_day': cycle_day,
                'cycle_phase': phase,
                'is_menstruating': is_menstruating,
                'flow_level': flow_level,
                'energy_level': round(energy_level, 1),
                'mood_score': round(mood_score, 1),
                'pain_level': round(pain_level, 1),
                'bloating': round(bloating, 1),
                'breast_tenderness': round(breast_tenderness, 1),
            })

        return pd.DataFrame(data)

    def generate_combined_data(
        self,
        n_days: int = 90,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate combined wearable and symptom data.

        Args:
            n_days: Number of days of data to generate
            start_date: Starting date for the data (defaults to today)

        Returns:
            DataFrame with both wearable and symptom data merged
        """
        wearable_df = self.generate_wearable_data(n_days, start_date)
        symptom_df = self.generate_symptom_data(n_days, start_date)

        # Merge on date
        combined_df = pd.merge(wearable_df, symptom_df, on=['date', 'cycle_day', 'cycle_phase'])

        return combined_df
