"""Test per il modulo counting - CountingModule."""

import pytest

from crowd_density.counting import CountingModule
from crowd_density.models import CountResult, PersonState, TrackedPerson


@pytest.fixture
def module():
    """Fixture per istanza CountingModule."""
    return CountingModule()


def _make_person(state: PersonState, confidence: float) -> TrackedPerson:
    """Helper per creare una TrackedPerson con stato e confidenza."""
    return TrackedPerson(
        person_id=1,
        state=state,
        confidence=confidence,
        ground_position=None,
    )


class TestEmptyInput:
    """Req 6.6: lista vuota restituisce tutti zeri senza errore."""

    def test_empty_list_returns_zeros(self, module):
        result = module.compute_count([])
        assert result.frame_count == 0.0
        assert result.occluded_count == 0.0
        assert result.total_count == 0.0
        assert result.confidence_interval == (0.0, 0.0)
        assert result.persons_visible == 0
        assert result.persons_partial == 0
        assert result.persons_occluded == 0


class TestFrameCount:
    """Req 6.1: frame_count = somma confidenze visibili e parzialmente occluse."""

    def test_single_visible_full_confidence(self, module):
        persons = [_make_person(PersonState.VISIBILE, 1.0)]
        result = module.compute_count(persons)
        assert result.frame_count == pytest.approx(1.0)

    def test_single_partially_occluded(self, module):
        persons = [_make_person(PersonState.PARZIALMENTE_OCCLUSO, 0.6)]
        result = module.compute_count(persons)
        assert result.frame_count == pytest.approx(0.6)

    def test_multiple_direct_detections(self, module):
        persons = [
            _make_person(PersonState.VISIBILE, 1.0),
            _make_person(PersonState.VISIBILE, 0.8),
            _make_person(PersonState.PARZIALMENTE_OCCLUSO, 0.5),
        ]
        result = module.compute_count(persons)
        assert result.frame_count == pytest.approx(2.3)


class TestOccludedCount:
    """Req 6.2: occluded_count = somma confidenze persone occluse."""

    def test_single_occluded(self, module):
        persons = [_make_person(PersonState.OCCLUSO, 0.7)]
        result = module.compute_count(persons)
        assert result.occluded_count == pytest.approx(0.7)

    def test_multiple_occluded(self, module):
        persons = [
            _make_person(PersonState.OCCLUSO, 0.4),
            _make_person(PersonState.OCCLUSO, 0.6),
        ]
        result = module.compute_count(persons)
        assert result.occluded_count == pytest.approx(1.0)


class TestTotalCount:
    """Req 6.5: total_count = frame_count + occluded_count."""

    def test_total_is_sum(self, module):
        persons = [
            _make_person(PersonState.VISIBILE, 1.0),
            _make_person(PersonState.PARZIALMENTE_OCCLUSO, 0.5),
            _make_person(PersonState.OCCLUSO, 0.3),
        ]
        result = module.compute_count(persons)
        assert result.total_count == pytest.approx(1.8)
        assert result.total_count == pytest.approx(
            result.frame_count + result.occluded_count
        )


class TestConfidenceInterval:
    """Req 6.3: min = solo conf==1.0, max = frame_count + occluse come 1.0."""

    def test_all_full_confidence(self, module):
        persons = [
            _make_person(PersonState.VISIBILE, 1.0),
            _make_person(PersonState.VISIBILE, 1.0),
        ]
        result = module.compute_count(persons)
        # min = 2 persone con conf==1.0, max = 2.0 + 0 occluse
        assert result.confidence_interval == (2.0, 2.0)

    def test_mixed_with_occluded(self, module):
        persons = [
            _make_person(PersonState.VISIBILE, 1.0),
            _make_person(PersonState.PARZIALMENTE_OCCLUSO, 0.5),
            _make_person(PersonState.OCCLUSO, 0.4),
            _make_person(PersonState.OCCLUSO, 0.6),
        ]
        result = module.compute_count(persons)
        # min = 1 (solo conf==1.0)
        # max = frame_count (1.5) + 2 occluse (contate come 1.0 ciascuna) = 3.5
        assert result.confidence_interval == (1.0, 3.5)

    def test_no_full_confidence(self, module):
        persons = [
            _make_person(PersonState.VISIBILE, 0.7),
            _make_person(PersonState.PARZIALMENTE_OCCLUSO, 0.5),
        ]
        result = module.compute_count(persons)
        # min = 0, max = 1.2 + 0 = 1.2
        assert result.confidence_interval[0] == pytest.approx(0.0)
        assert result.confidence_interval[1] == pytest.approx(1.2)


class TestProportionalContribution:
    """Req 6.4: persona con confidenza ridotta contribuisce proporzionalmente."""

    def test_partial_confidence_contributes_proportionally(self, module):
        persons = [_make_person(PersonState.VISIBILE, 0.6)]
        result = module.compute_count(persons)
        assert result.frame_count == pytest.approx(0.6)

    def test_occluded_partial_confidence(self, module):
        persons = [_make_person(PersonState.OCCLUSO, 0.3)]
        result = module.compute_count(persons)
        assert result.occluded_count == pytest.approx(0.3)


class TestPersonCategories:
    """Req 6.5: riportare separatamente le categorie di persone."""

    def test_categories_counted_correctly(self, module):
        persons = [
            _make_person(PersonState.VISIBILE, 1.0),
            _make_person(PersonState.VISIBILE, 0.8),
            _make_person(PersonState.PARZIALMENTE_OCCLUSO, 0.5),
            _make_person(PersonState.OCCLUSO, 0.4),
            _make_person(PersonState.OCCLUSO, 0.6),
        ]
        result = module.compute_count(persons)
        assert result.persons_visible == 1
        assert result.persons_partial == 2  # 0.8 visibile + 0.5 parziale
        assert result.persons_occluded == 2


class TestScomparsoIgnored:
    """Persone in stato SCOMPARSO non contribuiscono al conteggio."""

    def test_scomparso_not_counted(self, module):
        persons = [
            _make_person(PersonState.VISIBILE, 1.0),
            _make_person(PersonState.SCOMPARSO, 0.0),
        ]
        result = module.compute_count(persons)
        assert result.frame_count == pytest.approx(1.0)
        assert result.occluded_count == pytest.approx(0.0)
        assert result.total_count == pytest.approx(1.0)
        assert result.persons_visible == 1
        assert result.persons_partial == 0
        assert result.persons_occluded == 0
