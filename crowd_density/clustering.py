"""Rilevamento assembramenti con DBSCAN.

Implementa l'Algoritmo 4 del design: clustering spaziale delle persone
tracciate per identificare assembramenti e generare allarmi di densità critica.

Requisiti coperti: 8.1, 8.2, 8.4, 8.5
"""

import math
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from .config import CrowdConfig
from .models import (
    CrowdAlarm,
    CrowdCluster,
    GroundPoint,
    PersonState,
    TrackedPerson,
)


class CrowdClusterer:
    """Rileva assembramenti di persone usando DBSCAN e genera allarmi.

    Mantiene lo stato interno per tracciare l'evoluzione dei cluster
    tra frame successivi e gestire la regola di dispersione (Req 8.5):
    se la densità scende sotto soglia per >= dispersal_frames aggiornamenti
    consecutivi, il cluster è dichiarato disperso.
    """

    def __init__(self) -> None:
        self._previous_clusters: list[CrowdCluster] = []
        self._tracked_clusters: list[CrowdCluster] = []  # Include disappeared ones
        self._frames_below_threshold: dict[int, int] = {}
        self._next_cluster_id: int = 0

    def detect_crowds(
        self, persons: list[TrackedPerson], config: CrowdConfig
    ) -> tuple[list[CrowdCluster], list[CrowdAlarm]]:
        """Rileva cluster di persone usando DBSCAN.

        Args:
            persons: Lista delle persone tracciate attive.
            config: Configurazione per il clustering.

        Returns:
            Tupla (clusters, alarms) con i cluster rilevati e gli allarmi generati.
        """
        # 1. Estrarre coordinate (x, y) delle persone attive con posizione ground
        active_persons = [
            p
            for p in persons
            if p.ground_position is not None
            and p.state != PersonState.SCOMPARSO
        ]

        if len(active_persons) < config.min_cluster_size:
            # Non abbastanza persone per formare un cluster
            self._update_dispersal_tracking([], config)
            self._previous_clusters = []
            return [], []

        coordinates = np.array(
            [[p.ground_position.x, p.ground_position.y] for p in active_persons]
        )

        # 2. Applicare DBSCAN: eps=cluster_radius_meters, min_samples=min_cluster_size
        dbscan = DBSCAN(
            eps=config.cluster_radius_meters,
            min_samples=config.min_cluster_size,
        )
        labels = dbscan.fit_predict(coordinates)

        # 3. Per ogni cluster trovato, calcolare proprietà
        clusters: list[CrowdCluster] = []
        alarms: list[CrowdAlarm] = []

        unique_labels = set(labels)
        unique_labels.discard(-1)  # Rimuovere noise label

        for label in sorted(unique_labels):
            cluster_mask = labels == label
            cluster_coords = coordinates[cluster_mask]
            cluster_persons = [
                p for p, m in zip(active_persons, cluster_mask) if m
            ]

            num_persons = len(cluster_persons)
            person_ids = [p.person_id for p in cluster_persons]

            # 3a. Calcolare centro (centroide)
            centroid_x = float(np.mean(cluster_coords[:, 0]))
            centroid_y = float(np.mean(cluster_coords[:, 1]))
            center = GroundPoint(x=centroid_x, y=centroid_y)

            # 3b. Calcolare area (convex hull)
            area_m2 = self._compute_cluster_area(
                cluster_coords, config.cluster_radius_meters
            )

            # 3c. Calcolare densità = num_persone / area_m2
            density = num_persons / area_m2 if area_m2 > 0 else 0.0

            # 3d. Creare CrowdCluster se densità >= density_threshold
            if density >= config.density_threshold:
                cluster_id = self._next_cluster_id
                self._next_cluster_id += 1

                cluster = CrowdCluster(
                    cluster_id=cluster_id,
                    center=center,
                    area_m2=area_m2,
                    person_count=num_persons,
                    density=density,
                    persons=person_ids,
                    active_frames=1,
                )

                # Aggiornare active_frames se il cluster corrisponde a uno precedente
                matched_prev = self._match_previous_cluster(cluster)
                if matched_prev is not None:
                    cluster.active_frames = matched_prev.active_frames + 1

                clusters.append(cluster)

                # Verificare soglia critica e generare allarme
                alarm = self.check_critical_density(
                    cluster, config.critical_threshold
                )
                if alarm is not None:
                    alarms.append(alarm)

        # 4. Confrontare con cluster frame precedente (dispersal tracking)
        self._update_dispersal_tracking(clusters, config)

        # Aggiornare tracked_clusters: cluster attivi + quelli ancora in dispersal tracking
        new_tracked: list[CrowdCluster] = list(clusters)
        for tracked in self._tracked_clusters:
            if tracked.cluster_id in self._frames_below_threshold:
                if not any(c.cluster_id == tracked.cluster_id for c in new_tracked):
                    new_tracked.append(tracked)
        self._tracked_clusters = new_tracked
        self._previous_clusters = clusters

        return clusters, alarms

    def track_crowd_evolution(
        self,
        current: list[CrowdCluster],
        previous: list[CrowdCluster],
    ) -> dict[str, list[CrowdCluster]]:
        """Monitora crescita, riduzione, dispersione dei cluster.

        Args:
            current: Cluster del frame corrente.
            previous: Cluster del frame precedente.

        Returns:
            Dizionario con chiavi 'growing', 'shrinking', 'dispersed', 'new'
            contenenti i rispettivi cluster.
        """
        evolution: dict[str, list[CrowdCluster]] = {
            "growing": [],
            "shrinking": [],
            "dispersed": [],
            "new": [],
        }

        matched_previous_ids: set[int] = set()

        for cluster in current:
            best_match = self._find_closest_cluster(cluster, previous)
            if best_match is not None:
                matched_previous_ids.add(best_match.cluster_id)
                if cluster.person_count > best_match.person_count:
                    evolution["growing"].append(cluster)
                elif cluster.person_count < best_match.person_count:
                    evolution["shrinking"].append(cluster)
            else:
                evolution["new"].append(cluster)

        # Cluster precedenti non matchati → potenzialmente dispersi
        for prev_cluster in previous:
            if prev_cluster.cluster_id not in matched_previous_ids:
                frames_below = self._frames_below_threshold.get(
                    prev_cluster.cluster_id, 0
                )
                if frames_below >= 3:
                    evolution["dispersed"].append(prev_cluster)

        return evolution

    def check_critical_density(
        self, cluster: CrowdCluster, critical_threshold: float
    ) -> Optional[CrowdAlarm]:
        """Genera allarme se la densità del cluster supera la soglia critica.

        Args:
            cluster: Cluster da verificare.
            critical_threshold: Soglia di densità critica (persone/m²).

        Returns:
            CrowdAlarm se la densità supera la soglia, None altrimenti.
        """
        if cluster.density >= critical_threshold:
            return CrowdAlarm(
                cluster=cluster,
                alarm_level="critical",
                message=(
                    f"Densità critica rilevata: {cluster.density:.2f} persone/m² "
                    f"nel cluster {cluster.cluster_id} "
                    f"({cluster.person_count} persone, "
                    f"area {cluster.area_m2:.2f} m²)"
                ),
            )
        return None

    def _compute_cluster_area(
        self, coords: np.ndarray, eps: float
    ) -> float:
        """Calcola l'area del cluster usando il convex hull.

        Se il convex hull non può essere calcolato (punti collineari o
        meno di 3 posizioni uniche), usa una stima minima basata su pi*eps².

        Args:
            coords: Array (N, 2) delle coordinate del cluster.
            eps: Raggio eps di DBSCAN (per la stima fallback).

        Returns:
            Area in m².
        """
        min_area = math.pi * eps * eps

        # Serve almeno 3 punti unici per un convex hull 2D
        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < 3:
            return min_area

        try:
            hull = ConvexHull(unique_coords)
            area = float(hull.volume)  # In 2D, volume = area
            # Se l'area è troppo piccola (punti quasi collineari), usa il minimo
            return max(area, min_area)
        except Exception:
            # QhullError per punti collineari o degeneri
            return min_area

    def _match_previous_cluster(
        self, cluster: CrowdCluster
    ) -> Optional[CrowdCluster]:
        """Trova il cluster del frame precedente più vicino a quello dato.

        Args:
            cluster: Cluster corrente.

        Returns:
            Il cluster precedente più vicino, o None se non trovato.
        """
        return self._find_closest_cluster(cluster, self._previous_clusters)

    def _find_closest_cluster(
        self,
        cluster: CrowdCluster,
        candidates: list[CrowdCluster],
    ) -> Optional[CrowdCluster]:
        """Trova il cluster candidato più vicino al cluster dato.

        La vicinanza è calcolata come distanza euclidea tra i centroidi.
        Un match è valido solo se la distanza è inferiore alla metà
        della somma delle radici quadrate delle aree dei due cluster.

        Args:
            cluster: Cluster di riferimento.
            candidates: Lista di cluster candidati.

        Returns:
            Il candidato più vicino, o None se nessuno è abbastanza vicino.
        """
        if not candidates:
            return None

        best_match: Optional[CrowdCluster] = None
        best_distance = float("inf")

        for candidate in candidates:
            dx = cluster.center.x - candidate.center.x
            dy = cluster.center.y - candidate.center.y
            distance = math.sqrt(dx * dx + dy * dy)

            # Soglia di matching basata sulle dimensioni dei cluster
            threshold = 0.5 * (
                math.sqrt(cluster.area_m2) + math.sqrt(candidate.area_m2)
            )

            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match = candidate

        return best_match

    def _update_dispersal_tracking(
        self, current_clusters: list[CrowdCluster], config: CrowdConfig
    ) -> None:
        """Aggiorna il contatore di frame sotto soglia per il dispersal.

        Req 8.5: Se la densità scende sotto soglia per >= dispersal_frames
        aggiornamenti consecutivi, il cluster è dichiarato disperso.

        Args:
            current_clusters: Cluster attivi nel frame corrente.
            config: Configurazione crowd.
        """
        # Confrontare con tutti i cluster tracciati (attivi + in tracking)
        for tracked_cluster in self._tracked_clusters:
            matched = self._find_closest_cluster(tracked_cluster, current_clusters)
            if matched is not None:
                # Il cluster è ancora presente, reset del contatore
                if tracked_cluster.cluster_id in self._frames_below_threshold:
                    del self._frames_below_threshold[tracked_cluster.cluster_id]
            else:
                # Il cluster non è più presente → incrementare contatore
                current_count = self._frames_below_threshold.get(
                    tracked_cluster.cluster_id, 0
                )
                self._frames_below_threshold[tracked_cluster.cluster_id] = (
                    current_count + 1
                )

        # Rimuovere contatori per cluster che hanno superato il dispersal_frames
        ids_to_remove = [
            cid
            for cid, count in self._frames_below_threshold.items()
            if count >= config.dispersal_frames
        ]
        for cid in ids_to_remove:
            del self._frames_below_threshold[cid]
