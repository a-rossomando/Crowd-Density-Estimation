"""Test per il modulo density.py — DensityMapGenerator.

Verifica la generazione della mappa di densità con kernel gaussiano,
la separazione tra persone rilevate e occluse, e i casi limite.
"""

import numpy as np
import pytest

from crowd_density.density import DensityMap, DensityMapGenerator
from crowd_density.models import GroundPoint, PersonState, TrackedPerson


def _make_person(
    person_id: int,
    x: float,
    y: float,
    state: PersonState = PersonState.VISIBILE,
    confidence: float = 1.0,
) -> TrackedPerson:
    """Crea una TrackedPerson con parametri minimi per i test."""
    return TrackedPerson(
        person_id=person_id,
        state=state,
        confidence=confidence,
        ground_position=GroundPoint(x=x, y=y),
    )


class TestDensityMapGeneratorInit:
    """Test inizializzazione DensityMapGenerator."""

    def test_default_params(self):
        gen = DensityMapGenerator()
        assert gen.resolution == 10.0
        assert gen.sigma == 0.5

    def test_custom_params(self):
        gen = DensityMapGenerator(resolution=5.0, sigma=1.0)
        assert gen.resolution == 5.0
        assert gen.sigma == 1.0

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError):
            DensityMapGenerator(resolution=0.0)
        with pytest.raises(ValueError):
            DensityMapGenerator(resolution=-1.0)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            DensityMapGenerator(sigma=0.0)
        with pytest.raises(ValueError):
            DensityMapGenerator(sigma=-0.5)


class TestDensityMapGeneratorGenerate:
    """Test generazione mappa di densità."""

    def setup_method(self):
        self.gen = DensityMapGenerator(resolution=10.0, sigma=0.5)
        self.bounds = (0.0, 0.0, 5.0, 5.0)  # 5m x 5m

    def test_empty_persons_returns_zero_grid(self):
        result = self.gen.generate([], self.bounds)
        assert isinstance(result, DensityMap)
        assert np.all(result.grid == 0.0)
        assert np.all(result.detected_grid == 0.0)
        assert np.all(result.occluded_grid == 0.0)

    def test_grid_shape(self):
        result = self.gen.generate([], self.bounds)
        # 5m * 10 celle/m = 50 celle per lato
        assert result.grid.shape == (50, 50)
        assert result.detected_grid.shape == (50, 50)
        assert result.occluded_grid.shape == (50, 50)

    def test_single_detected_person_contributes_to_detected_grid(self):
        person = _make_person(1, 2.5, 2.5, PersonState.VISIBILE)
        result = self.gen.generate([person], self.bounds)

        assert np.sum(result.detected_grid) > 0
        assert np.sum(result.occluded_grid) == 0.0
        # Il massimo è al centro (posizione della persona)
        max_row, max_col = np.unravel_index(
            np.argmax(result.detected_grid), result.detected_grid.shape
        )
        # Centro atteso: (2.5 - 0) * 10 = 25
        assert abs(max_row - 25) <= 1
        assert abs(max_col - 25) <= 1

    def test_single_occluded_person_contributes_to_occluded_grid(self):
        person = _make_person(1, 2.5, 2.5, PersonState.OCCLUSO, confidence=0.7)
        result = self.gen.generate([person], self.bounds)

        assert np.sum(result.occluded_grid) > 0
        assert np.sum(result.detected_grid) == 0.0

    def test_partially_occluded_contributes_to_detected(self):
        """PARZIALMENTE_OCCLUSO è considerato 'rilevato direttamente'."""
        person = _make_person(
            1, 1.0, 1.0, PersonState.PARZIALMENTE_OCCLUSO, confidence=0.8
        )
        result = self.gen.generate([person], self.bounds)

        assert np.sum(result.detected_grid) > 0
        assert np.sum(result.occluded_grid) == 0.0

    def test_total_grid_is_sum(self):
        persons = [
            _make_person(1, 1.0, 1.0, PersonState.VISIBILE),
            _make_person(2, 3.0, 3.0, PersonState.OCCLUSO, confidence=0.5),
        ]
        result = self.gen.generate(persons, self.bounds)

        np.testing.assert_allclose(
            result.grid,
            result.detected_grid + result.occluded_grid,
        )

    def test_confidence_weights_density(self):
        """Confidenza più bassa → densità massima più bassa."""
        person_high = _make_person(1, 2.5, 2.5, confidence=1.0)
        person_low = _make_person(2, 2.5, 2.5, confidence=0.5)

        result_high = self.gen.generate([person_high], self.bounds)
        result_low = self.gen.generate([person_low], self.bounds)

        assert np.max(result_high.grid) > np.max(result_low.grid)
        np.testing.assert_allclose(
            result_low.grid, result_high.grid * 0.5, atol=1e-10
        )

    def test_person_without_ground_position_ignored(self):
        person = TrackedPerson(
            person_id=1,
            state=PersonState.VISIBILE,
            confidence=1.0,
            ground_position=None,
        )
        result = self.gen.generate([person], self.bounds)
        assert np.all(result.grid == 0.0)

    def test_scene_bounds_stored(self):
        result = self.gen.generate([], self.bounds)
        assert result.scene_bounds == self.bounds

    def test_resolution_stored(self):
        result = self.gen.generate([], self.bounds)
        assert result.resolution == 10.0

    def test_person_at_edge_does_not_crash(self):
        """Persona al bordo della scena non causa errore."""
        person = _make_person(1, 0.0, 0.0)
        result = self.gen.generate([person], self.bounds)
        assert np.sum(result.grid) > 0

    def test_person_outside_bounds_contributes_nothing(self):
        """Persona fuori dai confini non contribuisce alla mappa."""
        person = _make_person(1, -5.0, -5.0)
        result = self.gen.generate([person], self.bounds)
        # Il kernel è centrato molto lontano dalla griglia,
        # non dovrebbe contribuire significativamente
        assert np.sum(result.grid) == pytest.approx(0.0, abs=1e-10)

    def test_multiple_persons_accumulate(self):
        """Due persone nella stessa area sommano le densità."""
        persons = [
            _make_person(1, 2.5, 2.5),
            _make_person(2, 2.5, 2.5),
        ]
        result = self.gen.generate(persons, self.bounds)
        single_result = self.gen.generate([persons[0]], self.bounds)

        np.testing.assert_allclose(
            result.grid, single_result.grid * 2.0, atol=1e-10
        )

    def test_non_square_bounds(self):
        """Bounds non quadrati producono griglia rettangolare."""
        bounds = (0.0, 0.0, 10.0, 5.0)  # 10m x 5m
        result = self.gen.generate([], bounds)
        # rows = 5 * 10 = 50, cols = 10 * 10 = 100
        assert result.grid.shape == (50, 100)
