"""Benchmark comparativo tra varianti YOLOv8 sul Mall Dataset.

Testa modelli generici (person → head crop) e modelli specializzati
(head detection diretto), producendo tabella comparativa e grafici.

Uso:
    python benchmark_modelli.py --mall-dir ../../mall_dataset/mall_dataset
    python benchmark_modelli.py --mall-dir ../../mall_dataset/mall_dataset --max-frames 500
    python benchmark_modelli.py --mall-dir ../../mall_dataset/mall_dataset --gruppo tutti
    python benchmark_modelli.py --mall-dir ../../mall_dataset/mall_dataset --gruppo generici
    python benchmark_modelli.py --mall-dir ../../mall_dataset/mall_dataset --gruppo head
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

from crowd_density.detection import HeadDetector

logger = logging.getLogger(__name__)


# (nome, path_pesi, is_head_model)
# is_head_model=True → il modello rileva teste direttamente (classe 0 = head)
# is_head_model=False → modello generico COCO (classe 0 = person, serve head crop)
MODELLI_GENERICI = [
    ("YOLOv8x", "yolov8x.pt", False),
    ("YOLOv8l", "yolov8l.pt", False),
    ("YOLOv8m", "yolov8m.pt", False),
    ("YOLOv8s", "yolov8s.pt", False),
    ("YOLOv8n", "yolov8n.pt", False),
]

MODELLI_HEAD = [
    ("YOLOv8m_SCUT-HEAD", "yolov8m_scut_head.pt", True),
    ("YOLOv8n_iRail-HEAD", "yolov8n_irail_head.pt", True),
]

GRUPPI = {
    "generici": MODELLI_GENERICI,
    "head": MODELLI_HEAD,
    "tutti": MODELLI_GENERICI + MODELLI_HEAD,
}


def load_mall_ground_truth(mall_dir: Path) -> np.ndarray:
    gt_path = mall_dir / "mall_gt.mat"
    gt = scipy.io.loadmat(str(gt_path))
    return gt["count"].flatten().astype(int)


def get_frame_paths(mall_dir: Path, num_frames: int) -> list[Path]:
    frames_dir = mall_dir / "frames"
    return [frames_dir / f"seq_{i:06d}.jpg" for i in range(1, num_frames + 1)]


def benchmark_model(
    model_name: str,
    model_path: str,
    is_head_model: bool,
    frame_paths: list[Path],
    gt_counts: np.ndarray,
    num_frames: int,
) -> dict:
    """Esegue il benchmark per un singolo modello."""
    print(f"\n{'='*50}")
    tipo = "head detection diretto" if is_head_model else "person → head crop"
    print(f"  Benchmark: {model_name} ({model_path}) [{tipo}]")
    print(f"{'='*50}")

    from ultralytics import YOLO

    if is_head_model:
        # Modello head: rileva teste direttamente, conta tutte le detection
        model = YOLO(model_path)
    else:
        # Modello generico COCO: usa HeadDetector (person → head crop)
        detector = HeadDetector(model_path=model_path)

    pred_counts = []
    gt_used = []
    start_time = time.time()

    for i in range(num_frames):
        frame_bgr = cv2.imread(str(frame_paths[i]))
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if is_head_model:
            # Inferenza diretta: conta tutte le detection (classe 0 = head)
            results = model(frame_rgb, conf=0.25, verbose=False)
            count = 0
            for result in results:
                if result.boxes is not None:
                    count += len(result.boxes)
            pred_counts.append(count)
        else:
            detections = detector.detect(frame_rgb)
            pred_counts.append(len(detections))

        gt_used.append(gt_counts[i])

        if (i + 1) % 500 == 0 or i == num_frames - 1:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            print(f"  Frame {i+1}/{num_frames} ({fps:.1f} fps)")

    elapsed_total = time.time() - start_time
    fps_avg = len(pred_counts) / elapsed_total

    gt_arr = np.array(gt_used, dtype=float)
    pred_arr = np.array(pred_counts, dtype=float)
    errors = pred_arr - gt_arr
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    nonzero = gt_arr > 0
    mape = float(np.mean(abs_errors[nonzero] / gt_arr[nonzero]) * 100) if np.any(nonzero) else 0.0

    result = {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mean_gt": float(np.mean(gt_arr)),
        "mean_pred": float(np.mean(pred_arr)),
        "fps": fps_avg,
        "pred_counts": pred_arr,
        "gt_counts": gt_arr,
    }

    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  Media predetto: {np.mean(pred_arr):.1f} (GT: {np.mean(gt_arr):.1f})")
    print(f"  FPS: {fps_avg:.2f}")

    return result


def plot_comparison(results: list[dict], gt_counts: np.ndarray, output_dir: Path):
    """Genera grafici comparativi."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tabella riepilogativa ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    headers = ["Modello", "MAE", "RMSE", "MAPE (%)", "Media pred.", "FPS"]
    rows = []
    for r in results:
        rows.append([
            r["model"],
            f"{r['mae']:.2f}",
            f"{r['rmse']:.2f}",
            f"{r['mape']:.1f}",
            f"{r['mean_pred']:.1f}",
            f"{r['fps']:.1f}",
        ])
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    # Header in grassetto
    for j in range(len(headers)):
        table[0, j].set_text_props(weight="bold")
    ax.set_title(f"Confronto modelli YOLOv8 — Mall Dataset (GT media: {results[0]['mean_gt']:.1f})",
                 fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(output_dir / "tabella_confronto_modelli.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Andamento temporale sovrapposto ---
    num_frames = len(results[0]["pred_counts"])
    frames_range = np.arange(1, num_frames + 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frames_range, results[0]["gt_counts"], label="Ground Truth",
            color="black", alpha=0.6, linewidth=1)
    colors = ["#9C27B0", "#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for i, r in enumerate(results):
        ax.plot(frames_range, r["pred_counts"], label=f"{r['model']} (MAE={r['mae']:.1f})",
                alpha=0.7, linewidth=1, color=colors[i % len(colors)])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Numero persone")
    ax.set_title("Conteggio: GT vs modelli YOLOv8")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "confronto_temporale_modelli.png", dpi=150)
    plt.close(fig)

    # --- Bar chart MAE e FPS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = [r["model"] for r in results]
    maes = [r["mae"] for r in results]
    fpss = [r["fps"] for r in results]

    bars1 = ax1.bar(names, maes, color=colors[:len(results)], alpha=0.8)
    ax1.set_ylabel("MAE (errore medio assoluto)")
    ax1.set_title("Precisione: MAE per modello")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", fontsize=10)

    bars2 = ax2.bar(names, fpss, color=colors[:len(results)], alpha=0.8)
    ax2.set_ylabel("FPS (frame al secondo)")
    ax2.set_title("Velocità: FPS per modello")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, fpss):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "confronto_mae_fps_modelli.png", dpi=150)
    plt.close(fig)

    print(f"\nGrafici salvati in: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark comparativo modelli YOLOv8.")
    parser.add_argument("--mall-dir", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gruppo", type=str, default="head",
                        choices=["generici", "head", "tutti"],
                        help="Gruppo di modelli da testare (default: head)")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    mall_dir = Path(args.mall_dir)
    modelli = GRUPPI[args.gruppo]
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "benchmark_results" / f"confronto_{args.gruppo}"

    gt_counts = load_mall_ground_truth(mall_dir)
    total_frames = len(gt_counts)
    num_frames = min(args.max_frames, total_frames) if args.max_frames else total_frames
    frame_paths = get_frame_paths(mall_dir, total_frames)

    print(f"Mall Dataset: {total_frames} frame, media {np.mean(gt_counts):.1f} persone/frame")
    print(f"Frame da processare: {num_frames}")
    print(f"Gruppo: {args.gruppo} ({len(modelli)} modelli)")
    print(f"Modelli da testare: {', '.join(m[0] for m in modelli)}")

    results = []
    for model_name, model_path, is_head_model in modelli:
        r = benchmark_model(model_name, model_path, is_head_model, frame_paths, gt_counts, num_frames)
        results.append(r)

    # Stampa tabella finale
    print(f"\n{'='*70}")
    print(f"  RISULTATI COMPARATIVI — Mall Dataset ({num_frames} frame)")
    print(f"{'='*70}")
    print(f"  {'Modello':<22} {'MAE':>8} {'RMSE':>8} {'MAPE %':>8} {'Pred.':>8} {'FPS':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['model']:<22} {r['mae']:>8.2f} {r['rmse']:>8.2f} {r['mape']:>7.1f}% {r['mean_pred']:>8.1f} {r['fps']:>8.1f}")
    print(f"  {'GT media':<22} {'':>8} {'':>8} {'':>8} {results[0]['mean_gt']:>8.1f}")
    print(f"{'='*70}")

    # Grafici
    plot_comparison(results, gt_counts, output_dir)


if __name__ == "__main__":
    main()
