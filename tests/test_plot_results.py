import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("sigma", [0.0])
def test_plot_results_generates_png(tmp_path, sigma):
    """
    Smoke-test for plot_results:
    - Prefer the repository's sample results.csv; if missing, write a minimal synthetic CSV.
    - Use a non-interactive Matplotlib backend.
    - Skip gracefully if SciPy/Matplotlib are unavailable.
    - Assert that a results.png is produced alongside the CSV.
    """

    # Skip if required libs are not available in this environment
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")  # plot_results unconditionally imports gaussian_filter1d

    # Use a safe backend for headless environments
    import matplotlib

    matplotlib.use("Agg", force=True)

    # Ensure vendored 'ultralytics' is importable even without installing the project.
    # Add '<repo>/mga_yolo/external/ultralytics' to sys.path so 'import ultralytics' resolves.
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "mga_yolo" / "external" / "ultralytics"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    # If scienceplots is available but LaTeX isn't, skip to avoid tex rendering failures.
    has_scienceplots = importlib.util.find_spec("scienceplots") is not None
    if has_scienceplots and shutil.which("latex") is None:
        pytest.skip("scienceplots detected but LaTeX is not installed; plotting would fail with text.usetex=True")

    # Lazy import after sys.path adjustment
    from ultralytics.utils.plotting import plot_results

    default_csv = (
        repo_root
        / "mga_yolo"
        / "external"
        / "ultralytics"
        / "tests"
        / "tmp"
        / "runs"
        / "mga"
        / "test_mga_train_v8_segloss6"
        / "results.csv"
    )

    dest_dir = tmp_path / "run"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = dest_dir / "results.csv"

    if default_csv.exists():
        shutil.copy2(default_csv, dest_csv)
    else:
        # Fallback: write a compact synthetic CSV compatible with the alias resolver for metrics.
        # The columns mirror the attached sample, ensuring metrics are present so at least some plots render.
        sample = (
            "epoch,train/det/total,train/det/box,train/det/dfl,train/det/cls,"
            "train/seg/total,train/seg/p3_bce,train/seg/p3_dice,train/seg/p4_bce,train/seg/p4_dice,"
            "train/seg/p5_bce,train/seg/p5_dice,val/det/total,val/det/box,val/det/dfl,val/det/cls,"
            "val/seg/total,val/seg/p3_bce,val/seg/p3_dice,val/seg/p4_bce,val/seg/p4_dice,val/seg/p5_bce,"
            "val/seg/p5_dice,lr/pg0,lr/pg1,lr/pg2,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),"
            "metrics/recall(B)\n"
            "1,14.04,4.27,4.10,5.66,4.78,0.63,0.93,1.25,0.93,0.27,0.77,13.45,4.37,4.48,4.60,3.21,0.25,0.94,0.21,0.95,0.09,0.77,0.00066,0.00066,0.00066,0.0003,0.0001,0.00056,0.06667\n"
            "2,12.96,4.19,3.86,4.91,2.82,0.18,0.85,0.17,0.88,0.13,0.61,12.87,3.86,3.86,5.14,3.21,0.25,0.86,0.34,0.86,0.26,0.63,0.00089,0.00089,0.00089,0.00078,0.00022,0.0009,0.1625\n"
            "3,12.16,3.86,3.73,4.56,2.53,0.16,0.73,0.16,0.76,0.11,0.60,11.43,3.59,3.68,4.15,2.64,0.20,0.75,0.18,0.75,0.14,0.62,0.00068,0.00068,0.00068,0.00166,0.00044,0.0019,0.34167\n"
        )
        dest_csv.write_text(sample)

    # Run plotting and verify output
    plot_results(file=str(dest_csv), smooth_sigma=sigma)

    out_png = dest_dir / "results.png"
    assert out_png.exists(), "results.png was not created"
    assert out_png.stat().st_size > 0, "results.png is empty"
