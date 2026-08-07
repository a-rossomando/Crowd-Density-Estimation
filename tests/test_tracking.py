"""Test per il modulo tracking (StateMachine).

Verifica le transizioni di stato, il matching e il decadimento confidenza.
"""

import pytest

from crowd_density.config import TrackingConfig
from crowd_density.models import (
    BoundingBox,
    FusedDetection,
    GroundPoint,
    HeadDetection,
    OcclusionType,
    PersonState,
    PixelPoint,
    TrackedPerson,
)
from crowd_density.tracking import StateMachine


def _make_detection(
    x: float, y: float, confidence: float = 1.0
) -> FusedDetection:
    """Helper per creare una FusedDetection con posizione ground."""
    return FusedDetection(
        head=HeadDetection(
            bbox=BoundingBox(x=0, y=0, width=50, height=50),
            confidence=confidence,
            center=PixelPoint(x=100.0, y=100.0),
            visible_ratio=1.0,
        ),
        pose=None,
        confidence=confidence,
        occlusion_type=OcclusionType.NESSUNA,
        ground_position=GroundPoint(x=x, y=y),
        position_source="head",
    )


class TestStateMachineInit:
    """Test inizializzazione StateMachine."""

    def test_default_config(self):
        sm = StateMachine()
        assert sm.config.temporal_window_seconds == 5.0
        assert sm.config.tolerance_radius_meters == 1.5
        assert sm.config.frames_to_occluded == 3
        assert sm.config.frames_to_partially_occluded == 2

    def test_custom_config(self):
        config = TrackingConfig(temporal_window_seconds=10.0)
        sm = StateMachine(config=config)
        assert sm.config.temporal_window_seconds == 10.0

    def test_empty_initial_state(self):
        sm = StateMachine()
        assert sm.get_active_persons() == []


class TestNewPersonCreation:
    """Test creazione nuove persone per detection non matchate."""

    def test_single_detection_creates_person(self):
        sm = StateMachine()
        det = _make_detection(1.0, 2.0)
        result = sm.update(1, 0.0, [det])

        assert len(result) == 1
        person = result[0]
        assert person.person_id == 1
        assert person.state == PersonState.VISIBILE
        assert person.confidence == 1.0
        assert person.ground_position == GroundPoint(x=1.0, y=2.0)
        assert person.frames_since_last_seen == 0

    def test_multiple_detections_create_multiple_persons(self):
        sm = StateMachine()
        dets = [_make_detection(1.0, 1.0), _make_detection(5.0, 5.0)]
        result = sm.update(1, 0.0, dets)

        assert len(result) == 2
        assert result[0].person_id == 1
        assert result[1].person_id == 2

    def test_person_ids_increment(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])
        sm.update(2, 0.1, [_make_detection(10.0, 10.0)])

        persons = sm.get_active_persons()
        ids = {p.person_id for p in persons}
        assert ids == {1, 2}


class TestMatching:
    """Test del matching detection ↔ persona tracciata."""

    def test_detection_within_tolerance_matches(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        # Stessa posizione (entro tolleranza 1.5m)
        result = sm.update(2, 0.1, [_make_detection(1.2, 1.2)])
        assert len(result) == 1
        assert result[0].person_id == 1

    def test_detection_outside_tolerance_creates_new(self):
        config = TrackingConfig(tolerance_radius_meters=1.0)
        sm = StateMachine(config=config)
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        # Posizione lontana (oltre tolleranza 1.0m)
        result = sm.update(2, 0.1, [_make_detection(5.0, 5.0)])
        assert len(result) == 2

    def test_greedy_matching_closest_first(self):
        sm = StateMachine()
        # Due persone
        sm.update(1, 0.0, [_make_detection(0.0, 0.0), _make_detection(3.0, 0.0)])

        # Una detection più vicina alla prima persona
        result = sm.update(2, 0.1, [_make_detection(0.1, 0.0)])
        # La prima persona viene matchata, la seconda resta non-matchata
        persons_at_origin = [
            p for p in result if p.ground_position and p.ground_position.x < 1.0
        ]
        assert len(persons_at_origin) == 1
        assert persons_at_origin[0].person_id == 1


class TestTransitionVisibileToParzialeOccluso:
    """Test Req 2.7: visibile con conf < 1.0 per ≥2 frame → parzialmente_occluso."""

    def test_two_frames_reduced_conf_transitions(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0, confidence=1.0)])

        # Frame con confidenza ridotta (matchato)
        sm.update(2, 0.1, [_make_detection(1.0, 1.0, confidence=0.7)])
        sm.update(3, 0.2, [_make_detection(1.0, 1.0, confidence=0.7)])

        persons = sm.get_active_persons()
        assert persons[0].state == PersonState.PARZIALMENTE_OCCLUSO

    def test_one_frame_reduced_stays_visible(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0, confidence=1.0)])

        # Solo 1 frame con confidenza ridotta
        sm.update(2, 0.1, [_make_detection(1.0, 1.0, confidence=0.7)])

        persons = sm.get_active_persons()
        assert persons[0].state == PersonState.VISIBILE

    def test_conf_reset_on_full_confidence(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0, confidence=1.0)])

        # 1 frame ridotto, poi 1 frame pieno, poi 1 ridotto
        sm.update(2, 0.1, [_make_detection(1.0, 1.0, confidence=0.7)])
        sm.update(3, 0.2, [_make_detection(1.0, 1.0, confidence=1.0)])
        sm.update(4, 0.3, [_make_detection(1.0, 1.0, confidence=0.7)])

        persons = sm.get_active_persons()
        assert persons[0].state == PersonState.VISIBILE


class TestTransitionToOccluso:
    """Test Req 2.2/2.8: non rilevato per ≥3 frame → occluso."""

    def test_visible_not_detected_3_frames_becomes_occluded(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        # 3 frame senza detection (detection lontana)
        sm.update(2, 0.1, [])
        sm.update(3, 0.2, [])
        sm.update(4, 0.3, [])

        persons = sm.get_active_persons()
        assert len(persons) == 1
        assert persons[0].state == PersonState.OCCLUSO
        assert persons[0].occlusion_start_time == 0.3

    def test_partially_occluded_not_detected_3_frames_becomes_occluded(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0, confidence=1.0)])

        # Prima porta a parzialmente_occluso
        sm.update(2, 0.1, [_make_detection(1.0, 1.0, confidence=0.7)])
        sm.update(3, 0.2, [_make_detection(1.0, 1.0, confidence=0.7)])
        assert sm.get_active_persons()[0].state == PersonState.PARZIALMENTE_OCCLUSO

        # Poi 3 frame senza detection
        sm.update(4, 0.3, [])
        sm.update(5, 0.4, [])
        sm.update(6, 0.5, [])

        persons = sm.get_active_persons()
        assert persons[0].state == PersonState.OCCLUSO

    def test_two_frames_not_detected_stays_visible(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        sm.update(2, 0.1, [])
        sm.update(3, 0.2, [])

        persons = sm.get_active_persons()
        # Con 2 frame mancanti + regola 2.7 → parzialmente_occluso (non ancora occluso)
        assert persons[0].state != PersonState.OCCLUSO


class TestTransitionOcclusoToVisibile:
    """Test Req 2.3: persona occlusa ri-rilevata → visibile, stessa identità."""

    def test_occluded_redetected_becomes_visible(self):
        config = TrackingConfig(tolerance_radius_meters=2.0)
        sm = StateMachine(config=config)
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        # Porta a occluso
        sm.update(2, 0.1, [])
        sm.update(3, 0.2, [])
        sm.update(4, 0.3, [])

        persons = sm.get_active_persons()
        assert persons[0].state == PersonState.OCCLUSO
        original_id = persons[0].person_id

        # Ri-rilevata entro tolleranza
        result = sm.update(5, 0.4, [_make_detection(1.5, 1.5)])

        assert len(result) == 1
        assert result[0].person_id == original_id
        assert result[0].state == PersonState.VISIBILE
        assert result[0].confidence >= 0.8

    def test_occluded_redetected_same_identity(self):
        config = TrackingConfig(tolerance_radius_meters=2.0)
        sm = StateMachine(config=config)
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        sm.update(2, 0.1, [])
        sm.update(3, 0.2, [])
        sm.update(4, 0.3, [])

        # Ri-rilevata
        result = sm.update(5, 0.4, [_make_detection(1.0, 1.0)])
        assert result[0].person_id == 1


class TestTransitionOcclusoToScomparso:
    """Test Req 2.5: persona occlusa che supera finestra temporale → scomparso."""

    def test_occluded_exceeds_window_becomes_disappeared(self):
        config = TrackingConfig(temporal_window_seconds=2.0)
        sm = StateMachine(config=config)
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        # Porta a occluso al frame 4 (timestamp 0.3)
        sm.update(2, 0.1, [])
        sm.update(3, 0.2, [])
        sm.update(4, 0.3, [])

        assert sm.get_active_persons()[0].state == PersonState.OCCLUSO

        # Avanza il tempo oltre la finestra (2 secondi da t_occ=0.3)
        sm.update(5, 2.5, [])

        # La persona dovrebbe essere rimossa (scomparsa)
        assert len(sm.get_active_persons()) == 0


class TestConfidenceDecay:
    """Test Req 2.4: decadimento lineare confidenza per persone occluse."""

    def test_linear_decay_formula(self):
        config = TrackingConfig(temporal_window_seconds=5.0)
        sm = StateMachine(config=config)
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        # Porta a occluso
        sm.update(2, 0.1, [])
        sm.update(3, 0.2, [])
        sm.update(4, 1.0, [])  # occlusion_start_time = 1.0

        persons = sm.get_active_persons()
        assert persons[0].state == PersonState.OCCLUSO

        # Dopo 2.5 secondi dall'inizio occlusione: conf = 1.0 - 2.5/5.0 = 0.5
        sm.update(5, 3.5, [])
        persons = sm.get_active_persons()
        assert persons[0].confidence == pytest.approx(0.5, abs=0.01)

    def test_confidence_reaches_zero_at_window_end(self):
        config = TrackingConfig(temporal_window_seconds=4.0)
        sm = StateMachine(config=config)
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])

        # Porta a occluso al timestamp 1.0
        sm.update(2, 0.1, [])
        sm.update(3, 0.2, [])
        sm.update(4, 1.0, [])

        # Esattamente alla fine della finestra: conf = 1.0 - 4.0/4.0 = 0.0 → scomparso
        sm.update(5, 5.0, [])
        assert len(sm.get_active_persons()) == 0


class TestTransitionParzialeToVisibile:
    """Test Req 2.9: parzialmente_occlusa rilevata con conf 1.0 → visibile."""

    def test_partial_detected_full_conf_becomes_visible(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0, confidence=1.0)])

        # Porta a parzialmente_occluso
        sm.update(2, 0.1, [_make_detection(1.0, 1.0, confidence=0.7)])
        sm.update(3, 0.2, [_make_detection(1.0, 1.0, confidence=0.7)])
        assert sm.get_active_persons()[0].state == PersonState.PARZIALMENTE_OCCLUSO

        # Rilevata con confidenza 1.0
        sm.update(4, 0.3, [_make_detection(1.0, 1.0, confidence=1.0)])
        persons = sm.get_active_persons()
        assert persons[0].state == PersonState.VISIBILE
        assert persons[0].confidence == 1.0

    def test_partial_detected_low_conf_stays_partial(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0, confidence=1.0)])

        # Porta a parzialmente_occluso
        sm.update(2, 0.1, [_make_detection(1.0, 1.0, confidence=0.7)])
        sm.update(3, 0.2, [_make_detection(1.0, 1.0, confidence=0.7)])

        # Rilevata con confidenza < 1.0 → resta parzialmente occluso
        sm.update(4, 0.3, [_make_detection(1.0, 1.0, confidence=0.8)])
        persons = sm.get_active_persons()
        assert persons[0].state == PersonState.PARZIALMENTE_OCCLUSO


class TestEdgeCases:
    """Test casi limite."""

    def test_empty_detections(self):
        sm = StateMachine()
        result = sm.update(1, 0.0, [])
        assert result == []

    def test_detection_without_ground_position(self):
        sm = StateMachine()
        det = FusedDetection(
            head=HeadDetection(
                bbox=BoundingBox(x=0, y=0, width=50, height=50),
                confidence=1.0,
                center=PixelPoint(x=100.0, y=100.0),
                visible_ratio=1.0,
            ),
            pose=None,
            confidence=1.0,
            occlusion_type=OcclusionType.NESSUNA,
            ground_position=None,
            position_source="head",
        )
        result = sm.update(1, 0.0, [det])
        # Persona creata ma senza ground position (non matchabile)
        assert len(result) == 1
        assert result[0].ground_position is None

    def test_trajectory_grows_with_matches(self):
        sm = StateMachine()
        sm.update(1, 0.0, [_make_detection(1.0, 1.0)])
        sm.update(2, 0.1, [_make_detection(1.1, 1.1)])
        sm.update(3, 0.2, [_make_detection(1.2, 1.2)])

        persons = sm.get_active_persons()
        assert len(persons[0].trajectory) == 3
