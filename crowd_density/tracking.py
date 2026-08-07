"""Macchina a stati per il tracking temporale delle persone.

Implementa il modulo StateMachine che gestisce le transizioni di stato
per ogni persona tracciata: visibile ↔ parzialmente_occluso ↔ occluso → scomparso.

Usa matching greedy basato su distanza euclidea con soglia configurabile.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .config import TrackingConfig
from .models import (
    FusedDetection,
    GroundPoint,
    PersonState,
    TrackedPerson,
)


@dataclass
class Match:
    """Risultato di un'associazione detection ↔ persona tracciata.

    Attributes:
        person_idx: Indice della persona nella lista persone attive.
        detection_idx: Indice della detection nella lista rilevamenti.
        distance: Distanza euclidea tra posizione prevista e detection.
    """

    person_idx: int
    detection_idx: int
    distance: float


class StateMachine:
    """Macchina a stati per il tracking temporale delle persone.

    Gestisce la creazione, aggiornamento e rimozione di persone tracciate.
    Implementa le transizioni di stato basate su rilevamento/non-rilevamento
    e applica il decadimento lineare della confidenza per persone occluse.

    Attributes:
        config: Configurazione del modulo di tracking.
    """

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        """Inizializza la macchina a stati.

        Args:
            config: Configurazione di tracking. Se None, usa valori di default.
        """
        self.config = config or TrackingConfig()
        self._tracked_persons: list[TrackedPerson] = []
        self._next_person_id: int = 1
        # Contatore frame consecutivi con confidenza ridotta per ciascun person_id
        self._consecutive_reduced_conf_frames: dict[int, int] = {}

    def update(
        self,
        frame_id: int,
        timestamp: float,
        detections: list[FusedDetection],
    ) -> list[TrackedPerson]:
        """Aggiorna tutti gli stati persona con i rilevamenti del frame corrente.

        Algoritmo:
        1. Matching detection ↔ persone tracciate (distanza euclidea, soglia)
        2. Aggiorna persone con match trovato (transizioni positive)
        3. Aggiorna persone senza match (transizioni negative, decadimento)
        4. Crea nuove TrackedPerson per detection non matchate
        5. Rimuovi persone in stato SCOMPARSO

        Args:
            frame_id: Identificativo del frame corrente.
            timestamp: Timestamp del frame corrente in secondi.
            detections: Lista dei rilevamenti fusi del frame corrente.

        Returns:
            Lista delle persone attive (non in stato scomparso).
        """
        # Step 1: Filtra persone attive (non scomparse) per il matching
        active_persons = [
            p for p in self._tracked_persons
            if p.state != PersonState.SCOMPARSO
        ]

        # Step 2: Matching detection ↔ persone tracciate
        matches = self._match_detections(detections, active_persons)

        matched_person_indices: set[int] = set()
        matched_detection_indices: set[int] = set()

        for match in matches:
            matched_person_indices.add(match.person_idx)
            matched_detection_indices.add(match.detection_idx)

        # Step 3: Aggiorna persone CON match
        for match in matches:
            person = active_persons[match.person_idx]
            detection = detections[match.detection_idx]
            self._update_matched_person(person, detection, timestamp)

        # Step 4: Aggiorna persone SENZA match
        for idx, person in enumerate(active_persons):
            if idx not in matched_person_indices:
                self._update_unmatched_person(person, timestamp)

        # Step 5: Crea nuove TrackedPerson per detection non matchate
        for idx, detection in enumerate(detections):
            if idx not in matched_detection_indices:
                self._create_new_person(detection)

        # Step 6: Rimuovi persone scomparse dalla lista attiva
        self._tracked_persons = [
            p for p in self._tracked_persons
            if p.state != PersonState.SCOMPARSO
        ]

        # Cleanup del dizionario frame ridotti per persone rimosse
        active_ids = {p.person_id for p in self._tracked_persons}
        self._consecutive_reduced_conf_frames = {
            pid: count
            for pid, count in self._consecutive_reduced_conf_frames.items()
            if pid in active_ids
        }

        return list(self._tracked_persons)

    def get_active_persons(self) -> list[TrackedPerson]:
        """Ritorna tutte le persone attive (non in stato scomparso).

        Returns:
            Lista delle persone con stato diverso da SCOMPARSO.
        """
        return [
            p for p in self._tracked_persons
            if p.state != PersonState.SCOMPARSO
        ]

    def match_detections(
        self,
        detections: list[FusedDetection],
        predicted: list[TrackedPerson],
    ) -> list[Match]:
        """Interfaccia pubblica per il matching (usata anche esternamente).

        Args:
            detections: Lista dei rilevamenti fusi.
            predicted: Lista delle persone tracciate per il matching.

        Returns:
            Lista di Match con le associazioni trovate.
        """
        return self._match_detections(detections, predicted)

    def _match_detections(
        self,
        detections: list[FusedDetection],
        persons: list[TrackedPerson],
    ) -> list[Match]:
        """Associa rilevamenti a persone tracciate con matching greedy.

        Usa distanza euclidea nel piano del suolo con soglia
        tolerance_radius_meters. Ordina le coppie per distanza crescente
        e assegna in modo greedy (ogni detection e persona al massimo una volta).

        Args:
            detections: Lista dei rilevamenti fusi del frame.
            persons: Lista delle persone tracciate attive.

        Returns:
            Lista di Match ordinata per distanza.
        """
        if not detections or not persons:
            return []

        tolerance = self.config.tolerance_radius_meters

        # Calcola tutte le distanze candidate
        candidates: list[Match] = []
        for p_idx, person in enumerate(persons):
            if person.ground_position is None:
                continue
            for d_idx, detection in enumerate(detections):
                if detection.ground_position is None:
                    continue
                dist = self._euclidean_distance(
                    person.ground_position, detection.ground_position
                )
                if dist <= tolerance:
                    candidates.append(Match(
                        person_idx=p_idx,
                        detection_idx=d_idx,
                        distance=dist,
                    ))

        # Ordina per distanza crescente (greedy)
        candidates.sort(key=lambda m: m.distance)

        # Assegnazione greedy: ogni persona e detection al massimo una volta
        used_persons: set[int] = set()
        used_detections: set[int] = set()
        result: list[Match] = []

        for candidate in candidates:
            if (candidate.person_idx not in used_persons
                    and candidate.detection_idx not in used_detections):
                result.append(candidate)
                used_persons.add(candidate.person_idx)
                used_detections.add(candidate.detection_idx)

        return result

    def _update_matched_person(
        self,
        person: TrackedPerson,
        detection: FusedDetection,
        timestamp: float,
    ) -> None:
        """Aggiorna una persona con match trovato.

        Gestisce transizioni:
        - OCCLUSO → VISIBILE (ri-rilevato entro finestra, conf >= 0.8)
        - PARZIALMENTE_OCCLUSO → VISIBILE (detection.confidence == 1.0)
        - Aggiorna posizione, traiettoria, confidenza

        Args:
            person: La persona tracciata da aggiornare.
            detection: Il rilevamento fuso associato.
            timestamp: Timestamp corrente.
        """
        # Transizione OCCLUSO → VISIBILE
        if person.state == PersonState.OCCLUSO:
            person.state = PersonState.VISIBILE
            person.confidence = max(0.8, detection.confidence)
            person.occlusion_start_time = None
            # Reset contatore confidenza ridotta
            self._consecutive_reduced_conf_frames[person.person_id] = 0

        # Transizione PARZIALMENTE_OCCLUSO → VISIBILE
        elif person.state == PersonState.PARZIALMENTE_OCCLUSO:
            if detection.confidence >= 1.0:
                person.state = PersonState.VISIBILE
                person.confidence = 1.0
                self._consecutive_reduced_conf_frames[person.person_id] = 0
            else:
                # Resta parzialmente occluso, aggiorna confidenza
                person.confidence = detection.confidence

        # Persona VISIBILE: aggiorna confidenza
        elif person.state == PersonState.VISIBILE:
            person.confidence = detection.confidence
            # Traccia frame con confidenza ridotta per regola 2.7
            if detection.confidence < 1.0:
                self._consecutive_reduced_conf_frames[person.person_id] = (
                    self._consecutive_reduced_conf_frames.get(person.person_id, 0) + 1
                )
                # Transizione VISIBILE → PARZIALMENTE_OCCLUSO
                if (self._consecutive_reduced_conf_frames[person.person_id]
                        >= self.config.frames_to_partially_occluded):
                    person.state = PersonState.PARZIALMENTE_OCCLUSO
            else:
                self._consecutive_reduced_conf_frames[person.person_id] = 0

        # Aggiorna campi comuni
        person.frames_since_last_seen = 0
        person.last_detection = detection
        if detection.ground_position is not None:
            person.ground_position = detection.ground_position
            person.trajectory.append(detection.ground_position)

    def _update_unmatched_person(
        self, person: TrackedPerson, timestamp: float
    ) -> None:
        """Aggiorna una persona senza match nel frame corrente.

        Gestisce transizioni:
        - VISIBILE con conf < 1.0 per ≥2 frame → PARZIALMENTE_OCCLUSO
        - VISIBILE/PARZIALMENTE_OCCLUSO non rilevato ≥3 frame → OCCLUSO
        - OCCLUSO: decadimento lineare confidenza, se ≤ 0 → SCOMPARSO

        Args:
            person: La persona tracciata non matchata.
            timestamp: Timestamp corrente.
        """
        person.frames_since_last_seen += 1

        if person.state == PersonState.VISIBILE:
            # Incrementa contatore confidenza ridotta (non rilevata = conf < 1.0)
            self._consecutive_reduced_conf_frames[person.person_id] = (
                self._consecutive_reduced_conf_frames.get(person.person_id, 0) + 1
            )

            # Req 2.7: visibile con conf < 1.0 per ≥2 frame → parzialmente_occluso
            if (self._consecutive_reduced_conf_frames[person.person_id]
                    >= self.config.frames_to_partially_occluded):
                person.state = PersonState.PARZIALMENTE_OCCLUSO

            # Req 2.2: non rilevato per ≥3 frame → occluso
            if person.frames_since_last_seen >= self.config.frames_to_occluded:
                person.state = PersonState.OCCLUSO
                person.occlusion_start_time = timestamp
                self._consecutive_reduced_conf_frames[person.person_id] = 0

        elif person.state == PersonState.PARZIALMENTE_OCCLUSO:
            # Req 2.8: parzialmente occlusa non rilevata per ≥3 frame → occluso
            if person.frames_since_last_seen >= self.config.frames_to_occluded:
                person.state = PersonState.OCCLUSO
                person.occlusion_start_time = timestamp
                self._consecutive_reduced_conf_frames[person.person_id] = 0

        elif person.state == PersonState.OCCLUSO:
            # Req 2.4: decadimento lineare confidenza
            if person.occlusion_start_time is not None:
                elapsed = timestamp - person.occlusion_start_time
                window = self.config.temporal_window_seconds
                person.confidence = 1.0 - (elapsed / window)

                # Req 2.5: confidenza ≤ 0 → scomparso
                if person.confidence <= 0.0:
                    person.confidence = 0.0
                    person.state = PersonState.SCOMPARSO

    def _create_new_person(self, detection: FusedDetection) -> None:
        """Crea una nuova TrackedPerson per una detection non matchata.

        Args:
            detection: Il rilevamento fuso senza associazione.
        """
        trajectory: list[GroundPoint] = []
        if detection.ground_position is not None:
            trajectory.append(detection.ground_position)

        person = TrackedPerson(
            person_id=self._next_person_id,
            state=PersonState.VISIBILE,
            confidence=detection.confidence,
            ground_position=detection.ground_position,
            trajectory=trajectory,
            frames_since_last_seen=0,
            occlusion_start_time=None,
            predicted_reappearance=None,
            last_detection=detection,
        )
        self._tracked_persons.append(person)
        self._consecutive_reduced_conf_frames[person.person_id] = 0
        self._next_person_id += 1

    @property
    def temporal_window(self) -> float:
        """Ritorna il valore corrente della finestra temporale in secondi.

        Returns:
            Valore corrente di temporal_window_seconds dalla configurazione.
        """
        return self.config.temporal_window_seconds

    def set_temporal_window(self, value: float) -> None:
        """Imposta un nuovo valore per la finestra temporale.

        Valida il range [1.0, 60.0] secondi. Se il valore è fuori range,
        solleva ValueError e mantiene il valore precedente (Req 3.3).
        Se il valore è valido, aggiorna la configurazione e controlla tutte
        le persone in stato occluso: se il loro tempo trascorso supera il
        nuovo valore ridotto, le transita immediatamente a scomparso (Req 3.5).

        Args:
            value: Nuovo valore della finestra temporale in secondi.

        Raises:
            ValueError: Se il valore è fuori dall'intervallo [1.0, 60.0].
        """
        if not (1.0 <= value <= 60.0):
            raise ValueError(
                f"temporal_window_seconds deve essere tra 1.0 e 60.0 secondi, "
                f"ricevuto: {value}"
            )

        old_window = self.config.temporal_window_seconds
        self.config.temporal_window_seconds = value

        # Req 3.5: Se finestra ridotta, transitare immediatamente a scomparso
        # le persone occluse il cui tempo trascorso supera il nuovo valore.
        for person in self._tracked_persons:
            if (person.state == PersonState.OCCLUSO
                    and person.occlusion_start_time is not None):
                # Calcolare il tempo trascorso dall'inizio dell'occlusione.
                # La confidenza corrente è: conf = 1.0 - elapsed / old_window
                # Quindi: elapsed = (1.0 - conf) * old_window
                elapsed = (1.0 - person.confidence) * old_window

                if elapsed >= value:
                    # Il tempo trascorso supera la nuova finestra: scomparso
                    person.confidence = 0.0
                    person.state = PersonState.SCOMPARSO
                else:
                    # Ricalcolare la confidenza con la nuova finestra
                    person.confidence = 1.0 - (elapsed / value)

    @staticmethod
    def _euclidean_distance(p1: GroundPoint, p2: GroundPoint) -> float:
        """Calcola la distanza euclidea tra due punti nel piano del suolo.

        Args:
            p1: Primo punto.
            p2: Secondo punto.

        Returns:
            Distanza euclidea in metri.
        """
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
