"""Generatore mappa di densità spaziale.

Genera una mappa 2D con kernel gaussiano centrato su ogni persona tracciata,
distinguendo visivamente tra persone rilevate direttamente e persone
in stato occluso (stimate).

Requirements: 7.2
"""

from dataclasses import dataclass

import numpy as np

from .homography import SceneBounds
from .models import GroundPoint, PersonState, TrackedPerson


@dataclass
class DensityMap:
    """Risultato della generazione della mappa di densità.

    Attributes:
        grid: Mappa di densità totale (somma detected + occluded).
        detected_grid: Densità contribuita da persone rilevate direttamente
            (VISIBILE, PARZIALMENTE_OCCLUSO).
        occluded_grid: Densità contribuita da persone in stato OCCLUSO.
        scene_bounds: Confini della scena utilizzati (x_min, y_min, x_max, y_max).
        resolution: Risoluzione della griglia in celle per metro.
    """

    grid: np.ndarray
    detected_grid: np.ndarray
    occluded_grid: np.ndarray
    scene_bounds: SceneBounds
    resolution: float


class DensityMapGenerator:
    """Genera mappe di densità 2D dal piano del suolo.

    Utilizza un kernel gaussiano centrato sulla posizione di ogni persona
    tracciata. La densità è pesata per la confidenza della persona.
    Distingue tra contributi da persone rilevate direttamente e persone
    in stato occluso per la visualizzazione differenziata.

    Args:
        resolution: Celle per metro della griglia (default: 10.0).
        sigma: Deviazione standard del kernel gaussiano in metri
            (default: 0.5, approssimazione del raggio di una persona).
    """

    # Stati considerati come "rilevamento diretto"
    _DETECTED_STATES = {PersonState.VISIBILE, PersonState.PARZIALMENTE_OCCLUSO}

    def __init__(
        self,
        resolution: float = 10.0,
        sigma: float = 0.5,
    ) -> None:
        if resolution <= 0:
            raise ValueError(
                f"resolution deve essere positiva, ricevuto: {resolution}"
            )
        if sigma <= 0:
            raise ValueError(
                f"sigma deve essere positivo, ricevuto: {sigma}"
            )
        self._resolution = resolution
        self._sigma = sigma

    @property
    def resolution(self) -> float:
        """Risoluzione della griglia in celle per metro."""
        return self._resolution

    @property
    def sigma(self) -> float:
        """Deviazione standard del kernel gaussiano in metri."""
        return self._sigma

    def generate(
        self,
        persons: list[TrackedPerson],
        scene_bounds: SceneBounds,
    ) -> DensityMap:
        """Genera la mappa di densità a partire dalle persone tracciate.

        Per ogni persona con ground_position definita, applica un kernel
        gaussiano centrato sulla posizione, pesato per la confidenza.
        La mappa risultante separa i contributi di persone rilevate
        direttamente da quelli di persone occluse.

        Args:
            persons: Lista delle persone tracciate nel frame corrente.
            scene_bounds: Confini della scena (x_min, y_min, x_max, y_max)
                in coordinate metriche del piano del suolo.

        Returns:
            DensityMap con griglia totale, detected e occluded separate.
        """
        x_min, y_min, x_max, y_max = scene_bounds

        # Calcolare dimensioni della griglia
        width_m = x_max - x_min
        height_m = y_max - y_min

        cols = max(1, int(np.ceil(width_m * self._resolution)))
        rows = max(1, int(np.ceil(height_m * self._resolution)))

        detected_grid = np.zeros((rows, cols), dtype=np.float64)
        occluded_grid = np.zeros((rows, cols), dtype=np.float64)

        # Sigma in celle
        sigma_cells = self._sigma * self._resolution

        # Raggio del kernel: 3 sigma è sufficiente per catturare ~99.7%
        kernel_radius = int(np.ceil(3.0 * sigma_cells))

        for person in persons:
            if person.ground_position is None:
                continue

            # Convertire posizione mondo → indici griglia
            col = (person.ground_position.x - x_min) * self._resolution
            row = (person.ground_position.y - y_min) * self._resolution

            # Limiti della regione del kernel (clipping ai bordi)
            row_start = max(0, int(row) - kernel_radius)
            row_end = min(rows, int(row) + kernel_radius + 1)
            col_start = max(0, int(col) - kernel_radius)
            col_end = min(cols, int(col) + kernel_radius + 1)

            if row_start >= row_end or col_start >= col_end:
                continue

            # Generare coordinate della sotto-griglia
            r_indices = np.arange(row_start, row_end)
            c_indices = np.arange(col_start, col_end)
            rr, cc = np.meshgrid(r_indices, c_indices, indexing="ij")

            # Distanza al quadrato dal centro (in celle)
            dist_sq = (rr - row) ** 2 + (cc - col) ** 2

            # Kernel gaussiano pesato per confidenza
            kernel = person.confidence * np.exp(
                -dist_sq / (2.0 * sigma_cells**2)
            )

            # Accumulare nel layer corretto
            if person.state in self._DETECTED_STATES:
                detected_grid[row_start:row_end, col_start:col_end] += kernel
            else:
                # OCCLUSO (e SCOMPARSO se ancora presente nella lista)
                occluded_grid[row_start:row_end, col_start:col_end] += kernel

        # Griglia totale
        grid = detected_grid + occluded_grid

        return DensityMap(
            grid=grid,
            detected_grid=detected_grid,
            occluded_grid=occluded_grid,
            scene_bounds=scene_bounds,
            resolution=self._resolution,
        )
