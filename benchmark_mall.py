"""Benchmark della pipeline su Mall Dataset.

Carica i frame e il ground truth del Mall Dataset, processa ogni frame
con la pipeline CrowdDensityPipeline e confronta il conteggio rilevato
con quello annotato. Genera metriche (MAE, RMSE, MAPE) e grafici.

Uso:
    python benchmark_mall.py --mall-dir <percorso_mall_dataset>
    python benchmark_mall.py --mall-dir ../../../mall_dataset/mall_dataset
    python benchmark_mall.py --mall-dir ../../../mall_dataset/mall_dataset --max-frames 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from crowd_density.config import PipelineConfig
from crowd_density.pipeline import CrowdDensityPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caricamento Ground Truth
# ---------------------------------------------------------------------------

def load_mall_ground_truth(mall_dir: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Carica il ground truth dal file mall_gt.mat.

    Args:
        mall_dir: Cartella radice del mall dataset (contiene mall_gt.mat e frames/).

    Returns:
        Tupla (counts, locations):
            counts: array (N,) con il numero di persone per frame.
            locations: lista di array (K, 2) con le coordinate (x, y) delle
                teste annotate per ciascun frame.

    Raises:
        FileNotFoundError: Se il file mall_gt.mat non esiste.
    """
    gt_path = mall_dir / "mall_gt.mat"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth non trovato: {gt_path}")

    gt = scipy.io.loadmat(str(gt_path))

    # count: (N, 1) uint8 → (N,)
    counts = gt["count"].flatten().astype(int)

    # frame: (1, N) object array, ciascun elemento è una struct con campo 'loc'
    frame_data = gt["frame"]
    num_frames = frame_data.shape[1]
    locations: list[np.ndarray] = []

    for i in range(num_frames):
        # frame[0, i] è un array (1,1) con dtype [('loc', 'O')]
        struct = frame_data[0, i]
        loc_array = struct["loc"][0, 0]  # array (K, 2)
        locations.append(loc_array)

    return counts, locations


def get_frame_paths(mall_dir: Path, num_frames: int) -> list[Path]:
    """Genera i percorsi dei frame ordinati.

    Args:
        mall_dir: Cartella radice del mall dataset.
        num_frames: Numero totale di frame attesi.

    Returns:
        Lista ordinata di Path ai file JPG dei frame.
    """
    frames_dir = mall_dir / "frames"
    paths = []
    for i in range(1, num_frames + 1):
        path = frames_dir / f"seq_{i:06d}.jpg"
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------

def compute_metrics(
    gt_counts: np.ndarray, pred_counts: np.ndarray
) -> dict[str, float]:
    """Calcola metriche di valutazione del conteggio.

    Args:
        gt_counts: Array con i conteggi ground truth.
        pred_counts: Array con i conteggi predetti dalla pipeline.

    Returns:
        Dizionario con MAE, RMSE, MAPE e conteggi medi.
    """
    errors = pred_counts - gt_counts
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: evita divisione per zero
    nonzero_mask = gt_counts > 0
    if np.any(nonzero_mask):
        mape = float(
            np.mean(abs_errors[nonzero_mask] / gt_counts[nonzero_mask]) * 100
        )
    else:
        mape = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Media GT": float(np.mean(gt_counts)),
        "Media predetto": float(np.mean(pred_counts)),
        "Min errore": float(np.min(errors)),
        "Max errore": float(np.max(errors)),
    }


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------

def plot_results(
    gt_counts: np.ndarray,
    pred_counts: np.ndarray,
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Genera e salva i grafici di confronto.

    Crea tre grafici:
    1. Conteggio GT vs Predetto nel tempo
    2. Distribuzione degli errori (istogramma)
    3. Scatter plot GT vs Predetto

    Args:
        gt_counts: Conteggi ground truth.
        pred_counts: Conteggi predetti.
        metrics: Dizionario delle metriche calcolate.
        output_dir: Cartella dove salvare i grafici.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_range = np.arange(1, len(gt_counts) + 1)
    errors = pred_counts - gt_counts

    # --- 1. Andamento temporale ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frames_range, gt_counts, label="Ground Truth", alpha=0.8, linewidth=1)
    ax.plot(frames_range, pred_counts, label="Predetto (pipeline)", alpha=0.8, linewidth=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Numero persone")
    ax.set_title(
        f"Conteggio persone: GT vs Pipeline\n"
        f"MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  MAPE={metrics['MAPE (%)']:.1f}%"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "conteggio_temporale.png", dpi=150)
    plt.close(fig)

    # --- 2. Distribuzione errori ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", label="Errore zero")
    ax.axvline(x=np.mean(errors), color="orange", linestyle="--",
               label=f"Media errore: {np.mean(errors):.2f}")
    ax.set_xlabel("Errore (predetto - GT)")
    ax.set_ylabel("Frequenza")
    ax.set_title("Distribuzione degli errori di conteggio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "distribuzione_errori.png", dpi=150)
    plt.close(fig)

    # --- 3. Scatter plot ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gt_counts, pred_counts, alpha=0.3, s=10)
    max_val = max(gt_counts.max(), pred_counts.max()) + 2
    ax.plot([0, max_val], [0, max_val], "r--", label="Predizione perfetta")
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Predetto")
    ax.set_title("Scatter: GT vs Predetto")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_gt_vs_pred.png", dpi=150)
    plt.close(fig)

    logger.info("Grafici salvati in: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark della pipeline su Mall Dataset.",
    )
    parser.add_argument(
        "--mall-dir",
        type=str,
        required=True,
        help="Percorso alla cartella mall_dataset (contiene mall_gt.mat e frames/).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Numero massimo di frame da processare (default: tutti).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Cartella per i grafici di output (default: benchmark_results/ nella dir del progetto).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Non generare i grafici (solo metriche testuali).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Soglia di confidenza YOLO (default: 0.25).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    mall_dir = Path(args.mall_dir)
    if not mall_dir.exists():
        logger.error("Cartella mall dataset non trovata: %s", mall_dir)
        sys.exit(1)

    # Cartella output
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "benchmark_results"

    # 1. Caricare ground truth
    logger.info("Caricamento ground truth da: %s", mall_dir)
    gt_counts, gt_locations = load_mall_ground_truth(mall_dir)
    total_frames = len(gt_counts)
    logger.info("Ground truth caricato: %d frame, media %.1f persone/frame",
                total_frames, np.mean(gt_counts))

    # Limite frame
    num_frames = total_frames
    if args.max_frames is not None:
        num_frames = min(args.max_frames, total_frames)
        logger.info("Limitato a %d frame", num_frames)

    # 2. Preparare percorsi frame
    frame_paths = get_frame_paths(mall_dir, total_frames)

    # 3. Creare pipeline (senza calibrazione omografica — solo detection)
    #    Senza omografia il tracking non può fare matching (ground_position=None),
    #    quindi usiamo direttamente il numero di head_detections per frame
    #    come conteggio. Questo è il modo corretto di misurare l'accuratezza
    #    del detector su frame indipendenti.
    config = PipelineConfig(
        display_output=False,
    )
    pipeline = CrowdDensityPipeline(config=config)

    # 4. Processare frame
    pred_counts_list: list[float] = []
    gt_counts_used: list[int] = []

    logger.info("Avvio benchmark su %d frame...", num_frames)
    start_time = time.time()

    for i in range(num_frames):
        frame_path = frame_paths[i]
        if not frame_path.exists():
            logger.warning("Frame non trovato, skip: %s", frame_path)
            continue

        # Caricare frame (OpenCV legge BGR)
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            logger.warning("Impossibile leggere frame: %s", frame_path)
            continue

        # Convertire a RGB come si aspetta la pipeline
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Usare direttamente il detector per ottenere il conteggio per frame.
        # process_frame esegue l'intera catena, ma senza omografia il tracking
        # accumula persone spurie. Usiamo len(head_detections) come conteggio
        # diretto: ogni detection è una persona rilevata nel frame.
        result = pipeline.process_frame(
            frame=frame_rgb,
            frame_id=i,
            timestamp=i / 2.0,  # Mall dataset: frame rate < 2 Hz
        )

        pred_count = float(len(result.head_detections))
        pred_counts_list.append(pred_count)
        gt_counts_used.append(gt_counts[i])

        # Progress ogni 100 frame
        if (i + 1) % 100 == 0 or i == num_frames - 1:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            logger.info(
                "Frame %d/%d (%.1f fps) — GT: %d, Pred: %.1f",
                i + 1, num_frames, fps, gt_counts[i], pred_count,
            )

    elapsed_total = time.time() - start_time
    logger.info(
        "Benchmark completato: %d frame in %.1f secondi (%.2f fps)",
        len(pred_counts_list), elapsed_total, len(pred_counts_list) / elapsed_total,
    )

    # 5. Calcolare metriche
    gt_arr = np.array(gt_counts_used, dtype=float)
    pred_arr = np.array(pred_counts_list, dtype=float)

    metrics = compute_metrics(gt_arr, pred_arr)

    # Stampare risultati
    print()
    print("=" * 50)
    print("  RISULTATI BENCHMARK — Mall Dataset")
    print("=" * 50)
    print(f"  Frame elaborati:   {len(pred_counts_list)}")
    print(f"  Tempo totale:      {elapsed_total:.1f} s")
    print(f"  FPS medio:         {len(pred_counts_list) / elapsed_total:.2f}")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"  {key:<20s} {value:.2f}")
    print("=" * 50)

    # 6. Generare grafici
    if not args.no_plots:
        logger.info("Generazione grafici...")
        plot_results(gt_arr, pred_arr, metrics, output_dir)
        print(f"\n  Grafici salvati in: {output_dir}")

    # 7. Salvare metriche in CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "risultati_per_frame.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,gt_count,pred_count,errore\n")
        for i, (gt_val, pred_val) in enumerate(zip(gt_counts_used, pred_counts_list)):
            f.write(f"{i + 1},{gt_val},{pred_val:.1f},{pred_val - gt_val:.1f}\n")
    print(f"  CSV dettagliato:   {csv_path}")


if __name__ == "__main__":
    main()
