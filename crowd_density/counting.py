"""Modulo di conteggio persone con intervallo di confidenza.

Implementa il CountingModule che calcola il conteggio delle persone
a partire dalla lista di TrackedPerson fornita dalla macchina a stati.
Il modulo è stateless: opera esclusivamente sui dati del frame corrente.
"""

from crowd_density.models import CountResult, PersonState, TrackedPerson


class CountingModule:
    """Calcola il conteggio persone con intervallo di confidenza.

    Il modulo suddivide le persone tracciate in tre categorie:
    - Visibili (confidenza == 1.0): contribuiscono interamente al conteggio
    - Parziali (visibili/parzialmente_occluse con confidenza < 1.0):
      contribuiscono proporzionalmente alla propria confidenza
    - Occluse: contribuiscono con la propria confidenza residua

    L'intervallo di confidenza è calcolato come:
    - min: numero di persone con confidenza esattamente 1.0
    - max: frame_count + numero di persone occluse (contate ciascuna come 1.0)
    """

    def compute_count(self, persons: list[TrackedPerson]) -> CountResult:
        """Calcola il risultato di conteggio per il frame corrente.

        Args:
            persons: Lista delle persone tracciate attive dalla macchina a stati.

        Returns:
            CountResult con frame_count, occluded_count, total_count,
            confidence_interval, persons_visible, persons_partial, persons_occluded.
        """
        if not persons:
            return CountResult(
                frame_count=0.0,
                occluded_count=0.0,
                total_count=0.0,
                confidence_interval=(0.0, 0.0),
                persons_visible=0,
                persons_partial=0,
                persons_occluded=0,
            )

        frame_count = 0.0
        occluded_count = 0.0
        persons_visible = 0
        persons_partial = 0
        persons_occluded = 0

        for person in persons:
            if person.state in (
                PersonState.VISIBILE,
                PersonState.PARZIALMENTE_OCCLUSO,
            ):
                # Rilevamenti diretti: contribuiscono al frame_count
                frame_count += person.confidence
                if person.confidence == 1.0:
                    persons_visible += 1
                else:
                    persons_partial += 1
            elif person.state == PersonState.OCCLUSO:
                # Persone occluse: contribuiscono all'occluded_count
                occluded_count += person.confidence
                persons_occluded += 1

        total_count = frame_count + occluded_count

        # Intervallo di confidenza:
        # min = solo persone con confidenza == 1.0 (conteggio certo)
        # max = frame_count + numero occluse contate come 1.0
        conf_min = float(persons_visible)
        conf_max = frame_count + float(persons_occluded)

        return CountResult(
            frame_count=frame_count,
            occluded_count=occluded_count,
            total_count=total_count,
            confidence_interval=(conf_min, conf_max),
            persons_visible=persons_visible,
            persons_partial=persons_partial,
            persons_occluded=persons_occluded,
        )
