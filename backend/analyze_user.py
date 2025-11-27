#!/usr/bin/env python3
"""
analyze_user.py — FINAL VERSION (Option A)

Added flags:
  --output-dir <DIR>
  --cache
  --skip-stats

These now integrate fully into extractor, ML pipeline, and Loveble JSON output.
"""

from __future__ import annotations
import argparse
import subprocess
import json
import os
import sys
import time
import traceback
import csv
from pathlib import Path
from typing import Dict, Any, List
import argparse
import subprocess
import shlex
import time

import numpy as np
import pandas as pd

# safe load joblib
try:
    import joblib
except Exception:
    try:
        from sklearn.externals import joblib
    except Exception:
        import pickle as _pickle
        class _JoblibFallback:
            @staticmethod
            def load(path):
                with open(str(path), "rb") as f:
                    return _pickle.load(f)
        joblib = _JoblibFallback()

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -----------------------
# Config
# -----------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "outputs"
DEFAULT_FALLBACK_DATASET = BASE_DIR / "ml_ready_labeled.csv"

EXTRACTOR_SCRIPT = BASE_DIR / "github_dual_output.py"
LOVEABLE_SCRIPT = BASE_DIR / "transform_to_loveble.py"

# -----------------------
# Utility
# -----------------------

def safe_log(*args):
    """Avoid UnicodeEncodeError in Windows CMD."""
    try:
        print(*args)
    except Exception:
        print(*(str(a).encode("utf-8", errors="ignore").decode("utf-8") for a in args))


def now_s():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def elapsed(t0):
    return f"{(time.time()-t0):.2f}s"


def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------
# Extractor Runner
# -----------------------

def run_extractor(
    username: str,
    outdir: Path,
    top_k: int,
    use_cache: bool,
    skip_stats: bool
):
    mkdir(outdir)

    cmd = [
        sys.executable, str(EXTRACTOR_SCRIPT),
        "--username", username,
        "--top-k-detail", str(top_k),
        "--output-dir", str(outdir),
    ]
    if use_cache:
        cmd.append("--cache")
    if skip_stats:
        cmd.append("--skip-stats")

    safe_log(f"[{now_s()}] Running extractor:", " ".join(cmd))

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
    except Exception as e:
        return dict(ok=False, stdout="", stderr=str(e))

    safe_log(f"[{now_s()}] extractor rc={proc.returncode} time={elapsed(t0)}")

    # Print last 20 lines for debugging
    if proc.stdout:
        safe_log("---- extractor stdout ----")
        for l in proc.stdout.splitlines()[-20:]:
            safe_log(l)

    if proc.stderr:
        safe_log("---- extractor stderr ----")
        for l in proc.stderr.splitlines()[-20:]:
            safe_log(l)

    if proc.returncode != 0:
        return dict(ok=False, stdout=proc.stdout, stderr=proc.stderr)

    return dict(ok=True, stdout=proc.stdout, stderr=proc.stderr)
# -----------------------
# Load extractor outputs
# -----------------------

def load_extractor_outputs(outdir: Path):
    summary_path = outdir / "user_summary.json"
    repos_csv = outdir / "repos.csv"
    skills_csv = outdir / "skills.csv"

    out = {}

    if summary_path.exists():
        out["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        out["summary"] = {}

    out["repos_df"] = pd.read_csv(repos_csv) if repos_csv.exists() else pd.DataFrame()
    out["skills_df"] = pd.read_csv(skills_csv) if skills_csv.exists() else pd.DataFrame()

    return out

# -----------------------
# Build ML features
# -----------------------

def build_feature_row_from_summary(summary, repos_df, skills_df):
    row = {}

    # Basic metrics (from extractor summary)
    metrics = summary.get("metrics", {})
    row["followers"]   = metrics.get("followers", 0)
    row["following"]   = metrics.get("following", 0)
    row["public_repos"] = metrics.get("public_repos", 0)
    row["public_gists"] = metrics.get("public_gists", 0)

    # Account age
    exp = summary.get("experience", {})
    row["account_age_years"] = float(exp.get("account_age_years", 0.0))
    row["repos_per_year"]    = float(exp.get("repos_per_year", 0.0))
    row["avg_commits_per_year"] = float(exp.get("avg_commits_per_year", 0.0))
    row["avg_stars_per_year"]   = float(exp.get("avg_stars_per_year", 0.0))

    # Repo counts
    row["repo_count"] = summary.get("num_repos_fetched", len(repos_df))

    # Repo statistics
    if not repos_df.empty:
        def safe_mean(col):
            try:
                return float(repos_df[col].replace("", np.nan).dropna().astype(float).mean())
            except:
                return 0.0

        row["repo_avg_stars"]     = safe_mean("stars")
        row["repo_avg_forks"]     = safe_mean("forks")
        row["repo_avg_watchers"]  = safe_mean("watchers")
        row["repo_avg_size_kb"]   = safe_mean("size_kb")
        row["repo_avg_commits_past_12m"] = safe_mean("commits_past_12m")
        row["repo_avg_complexity_score"] = safe_mean("complexity_score") if "complexity_score" in repos_df else 0.0

        # Number of active repos
        try:
            row["active_repos"] = int((repos_df.get("commits_past_12m", 0) > 0).sum())
        except:
            row["active_repos"] = 0
    else:
        row.update({
            "repo_avg_stars":0, "repo_avg_forks":0, "repo_avg_watchers":0,
            "repo_avg_size_kb":0, "repo_avg_commits_past_12m":0,
            "repo_avg_complexity_score":0, "active_repos":0
        })

    # Top languages → lang_ columns
    langs = summary.get("top_languages", [])
    for i, item in enumerate(langs[:10], start=1):
        lang = item.get("name") or f"lang_{i}"
        pct  = float(item.get("percentage", 0.0))
        row[f"lang_{lang.replace(' ', '_')}"] = pct

    # Readme skills
    if not skills_df.empty:
        for _, r in skills_df.sort_values(by=["score", "count"], ascending=False).head(40).iterrows():
            s = str(r.get("skill", "")).strip()
            if s:
                col = "skill_" + s.replace(" ", "_").replace("-", "_").replace(".", "")
                row[col] = 1

    # Missing_* placeholders = 50
    missing_base = [
        "Python","JavaScript","TypeScript","Go","C","C++","Rust","Java","HTML","CSS",
        "SCSS","Sass","PHP","Ruby","Swift","Kotlin","Shell","Docker","Dockerfile",
        "Kubernetes","CI_CD","React","Vue","NodeJS","Express",
        "TensorFlow","PyTorch","Jupyter","Pandas","NumPy","Sklearn",
        "MySQL","PostgreSQL","MongoDB","Redis","DevOps","Terraform",
        "AWS","GCP","Azure","Linux","Bash","Makefile","CMake","OpenCV",
        "FastAPI","Flask","Django"
    ]
    for name in missing_base:
        row[f"missing_{name}"] = 0

    # Add extra placeholders to reach ≥50
    count_missing = len([k for k in row if k.startswith("missing_")])
    for i in range(50 - count_missing):
        row[f"missing_extra_{i}"] = 0

    # Convert to DF
    df = pd.DataFrame([row])
    numeric_cols = df.columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df

# -----------------------
# Align to model feature list
# -----------------------

def align_features(df, feature_cols):
    aligned = pd.DataFrame(columns=feature_cols)
    for c in feature_cols:
        aligned[c] = df[c] if c in df else 0.0
    return aligned.fillna(0).astype(float)

# -----------------------
# Load ML models
# -----------------------

def load_models_and_metadata(models_dir: Path):
    safe_log(f"[{now_s()}] Loading models from {models_dir}")

    models = {
        "expertise": joblib.load(models_dir/"expertise_model.pkl"),
        "domain": joblib.load(models_dir/"domain_model.pkl"),
        "ranking": joblib.load(models_dir/"dev_ranking_model.pkl"),
        "scaler": joblib.load(models_dir/"scaler.pkl"),
        "nn_scaler": joblib.load(models_dir/"nn_scaler.pkl"),
        "skill_nn": joblib.load(models_dir/"skill_nn_model.pkl"),
    }

    feature_cols = []
    skill_cols = []
    missing_cols = []

    if (models_dir/"feature_cols.txt").exists():
        feature_cols = [x.strip() for x in (models_dir/"feature_cols.txt").read_text().splitlines()]

    if (models_dir/"skill_cols.txt").exists():
        skill_cols = [x.strip() for x in (models_dir/"skill_cols.txt").read_text().splitlines()]

    if (models_dir/"missing_cols.txt").exists():
        missing_cols = [x.strip() for x in (models_dir/"missing_cols.txt").read_text().splitlines()]

    return models, feature_cols, skill_cols, missing_cols

# -----------------------
# Skill Recommendation Core
# -----------------------

def compute_skill_recommendation(
    models,
    feature_row,
    aligned,
    skill_cols,
    missing_cols,
    fallback_dataset: Path
):
    try:
        fb = pd.read_csv(fallback_dataset)
    except Exception:
        return []

    if "username" in fb.columns:
        fb = fb.drop(columns=["username"])

    # Build fallback skill matrix
    fb_skill_cols = [c for c in skill_cols if c in fb.columns]
    fb_skill_matrix = (fb[fb_skill_cols].fillna(0).astype(float) > 0.01).astype(int).values

    fb_features = fb[[c for c in fb.columns if c in aligned.columns]].fillna(0)
    fb_non_skill_cols = [c for c in fb_features.columns if c not in skill_cols + missing_cols]

    fb_non_skill_matrix = fb_features[fb_non_skill_cols].astype(float).values

    try:
        nn_model = NearestNeighbors(n_neighbors=6).fit(fb_non_skill_matrix)
        cur = aligned[fb_non_skill_cols].astype(float).values
        dist, nbrs = nn_model.kneighbors(cur, n_neighbors=6)
    except Exception:
        return []

    nbrs = nbrs[0].tolist()[1:]  # remove itself

    if not fb_skill_cols:
        return []

    agg = fb_skill_matrix[nbrs].sum(axis=0)

    user_skill_vec = np.array([
        1 if (c in feature_row.columns and feature_row[c].iloc[0] > 0.01) else 0
        for c in fb_skill_cols
    ])

    recs = []
    for idx in np.argsort(-agg):
        if agg[idx] <= 0:
            continue
        if user_skill_vec[idx] == 0:
            recs.append((fb_skill_cols[idx], int(agg[idx])))
        if len(recs) >= 3:
            break

    return recs
# -----------------------
# Part 3 — Main + CLI + Orchestration
# -----------------------

# safe_print avoids unicode characters that break cp1252 on Windows consoles
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # fallback: replace non-encodable characters
        out = " ".join(str(a) for a in args)
        print(out.encode("ascii", errors="replace").decode("ascii"), **kwargs)

def run_extractor_subprocess(username: str, outdir: Path, top_k: int, use_cache: bool, skip_stats: bool, timeout: int=300):
    """
    Spawn github_dual_output.py as a subprocess. Accepts both --output-dir and --output-root styles.
    Returns a dict with keys: ok, returncode, stdout, stderr
    """
    EXTRACTOR = BASE_DIR / "github_dual_output.py"
    if not EXTRACTOR.exists():
        return {"ok": False, "error": "extractor_missing", "stdout": "", "stderr": f"{EXTRACTOR} not found"}

    cmd = [sys.executable, str(EXTRACTOR), "--username", username, "--top-k-detail", str(top_k)]
    # support both option names used in various versions
    cmd += ["--output-dir", str(outdir)]
    if use_cache:
        cmd += ["--cache"]
    if skip_stats:
        cmd += ["--skip-stats"]

    safe_print("Running extractor:", " ".join(shlex.quote(x) for x in cmd), f"(timeout={timeout}s)")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(BASE_DIR))
    except subprocess.TimeoutExpired as e:
        safe_print("Extractor timed out:", e)
        return {"ok": False, "error": "timeout", "stdout": getattr(e, "output", "") or "", "stderr": getattr(e, "stderr", "") or ""}
    except Exception as e:
        safe_print("Failed to run extractor:", e)
        return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}

    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout or "", "stderr": proc.stderr or ""}

def safe_write_json(path: Path, obj):
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        safe_print("Failed to write JSON to", path, ":", e)
        return False

def main():
    parser = argparse.ArgumentParser(description="Analyze GitHub user and produce Loveble JSON + predictions")
    parser.add_argument("--username", "-u", required=True, help="GitHub username")
    parser.add_argument("--top-k", type=int, default=8, help="Top repos to fetch in detail")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Models directory")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory (alias supported)")
    parser.add_argument("--output-dir", default=None, help="(alias) output directory for single run")
    parser.add_argument("--use-cache", action="store_true", help="Enable caching for detailed repo calls (alias: --cache)")
    parser.add_argument("--cache", action="store_true", help="Alias for --use-cache")
    parser.add_argument("--fallback-dataset", default=str(DEFAULT_FALLBACK_DATASET), help="Fallback dataset CSV for recommender")
    parser.add_argument("--skip-stats", action="store_true", help="Skip expensive /stats calls")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for extractor subprocess in seconds")
    args = parser.parse_args()

    username = args.username
    top_k = args.top_k
    use_cache = args.use_cache or args.cache
    skip_stats = args.skip_stats
    models_dir = Path(args.models_dir)
    output_root = Path(args.output_root)
    if args.output_dir:
        outdir = Path(args.output_dir)
    else:
        outdir = output_root / username

    outdir.mkdir(parents=True, exist_ok=True)
    safe_print(f"[{now_s()}] Starting analysis for:", username, "→ output:", outdir)

    # Step A: run extractor as subprocess
    t0 = time.time()
    ext_res = run_extractor_subprocess(username, outdir, top_k, use_cache, skip_stats, timeout=args.timeout)
    safe_print(f"[{now_s()}] Extractor finished (elapsed {time.time()-t0:.2f}s) return_ok={ext_res.get('ok')}")
    if ext_res.get("stdout"):
        safe_print("---- extractor stdout (tail) ----")
        for line in ext_res["stdout"].splitlines()[-30:]:
            safe_print(line)
    if ext_res.get("stderr"):
        safe_print("---- extractor stderr (tail) ----")
        for line in ext_res["stderr"].splitlines()[-30:]:
            safe_print(line)

    # Step B: load outputs (if extractor produced them)
    outputs = load_extractor_outputs(outdir)
    summary = outputs.get("summary", {})
    repos_df = outputs.get("repos_df", pd.DataFrame())
    skills_df = outputs.get("skills_df", pd.DataFrame())

    # Step C: build feature row
    try:
        feature_row = build_feature_row_from_summary(summary, repos_df, skills_df)
    except Exception as e:
        safe_print("Feature build failed:", e)
        feature_row = pd.DataFrame()

    # Step D: load models (best-effort)
    models, feature_cols, skill_cols, missing_cols = {}, [], [], []
    try:
        models, feature_cols, skill_cols, missing_cols = load_models_and_metadata(models_dir)
    except Exception as e:
        safe_print("Model load failed:", e)

    # Step E: align & scale
    aligned = None
    if feature_cols:
        try:
            aligned = align_features(feature_row, feature_cols)
        except Exception as e:
            safe_print("Align features failed:", e)
            aligned = feature_row.select_dtypes(include=[float, int])
    else:
        aligned = feature_row.select_dtypes(include=[float, int])

    # scale
    X_scaled = None
    if models.get("scaler") is not None and aligned.shape[1] > 0:
        try:
            X_scaled = models["scaler"].transform(aligned.values)
        except Exception as e:
            safe_print("Scaler transform error:", e)
            # try fit-transform fallback
            try:
                fallback_scaler = StandardScaler().fit(aligned.values)
                X_scaled = fallback_scaler.transform(aligned.values)
            except Exception as e2:
                safe_print("Fallback scaler error:", e2)
                X_scaled = aligned.values
    else:
        X_scaled = aligned.values if aligned is not None else None

    # Step F: predictions (best-effort)
    exp_pred = None; dom_pred = None; rank_pred = None
    try:
        if models.get("expertise") is not None and X_scaled is not None:
            exp_pred = int(models["expertise"].predict(X_scaled)[0])
    except Exception as e:
        safe_print("Expertise predict failed:", e)
    try:
        if models.get("domain") is not None and X_scaled is not None:
            dom_pred = int(models["domain"].predict(X_scaled)[0])
    except Exception as e:
        safe_print("Domain predict failed:", e)
    try:
        if models.get("ranking") is not None and X_scaled is not None:
            rank_pred = float(models["ranking"].predict(X_scaled)[0])
            rank_pred = float(np.clip(rank_pred, 0.0, 10.0))
    except Exception as e:
        safe_print("Ranking predict failed:", e)

    # Step G: recommendations
    recommended = []
    try:
        recommended = compute_skill_recommendation(models, feature_row, aligned, skill_cols, missing_cols, Path(args.fallback_dataset))
    except Exception as e:
        safe_print("Recommendation step failed:", e)

    # Format recommended list to (skill,priority) if not already
    rec_formatted = []
    for r in recommended:
        if isinstance(r, tuple) and len(r) >= 2:
            name = r[0]
            pr = int(r[1]) if isinstance(r[1], (int, float)) else 1
            # normalize key names (drop skill_ prefix if present)
            if name.startswith("skill_"):
                name = name[len("skill_"):]
            rec_formatted.append({"skill": name, "priority": pr})
        elif isinstance(r, str):
            rec_formatted.append({"skill": r, "priority": 1})

    # Step H: write predictions
    preds = {
        "expertise_level": exp_pred,
        "domain_label": dom_pred,
        "developer_rank": round(float(rank_pred) if rank_pred is not None else 0.0, 3),
        "recommended_skills": rec_formatted
    }
    try:
        safe_write_json(outdir / "predictions.json", preds)
        # Also CSV single-row
        with open(outdir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["expertise_level", "domain_label", "developer_rank", "recommended_skills"])
            w.writerow([preds["expertise_level"], preds["domain_label"], preds["developer_rank"], json.dumps(preds["recommended_skills"], ensure_ascii=False)])
        safe_print(f"[{now_s()}] Saved predictions to: {outdir/'predictions.json'} , {outdir/'predictions.csv'}")
    except Exception as e:
        safe_print("Failed to save predictions:", e)

    # Step I: generate Loveble JSON by running transform_to_loveble.py (compatibility)
    loveble_path = outdir / "loveble_schema.json"
    TRANSFORM = BASE_DIR / "transform_to_loveble.py"
    if TRANSFORM.exists():
        try:
            safe_print(f"[{now_s()}] ▶ Generating Loveble JSON via transform_to_loveble.py")
            proc = subprocess.run([sys.executable, str(TRANSFORM), str(outdir)], capture_output=True, text=True, timeout=120)
            if proc.returncode == 0 and loveble_path.exists():
                safe_print(f"[{now_s()}] ✔ Loveble JSON created: {loveble_path}")
                # read and print tail of generated JSON for logs
                try:
                    loveble = json.loads(loveble_path.read_text(encoding="utf-8"))
                except Exception:
                    loveble = None
            else:
                safe_print("[!] transform_to_loveble.py did not produce loveble_schema.json")
                loveble = None
                safe_print("transform stdout (tail):")
                safe_print(proc.stdout.splitlines()[-20:])
                safe_print("transform stderr (tail):")
                safe_print(proc.stderr.splitlines()[-20:])
        except subprocess.TimeoutExpired:
            safe_print("[!] transform_to_loveble.py timed out")
            loveble = None
        except Exception as e:
            safe_print("Error while running transform_to_loveble.py:", e)
            loveble = None
    else:
        safe_print("[!] transform_to_loveble.py not found; skipping Loveble generation")
        loveble = None

    # Final status and exit codes for server compatibility:
    # - success: return 0
    # - loveble missing: return code 2
    # - other failures: return code 1
    if loveble and outdir.exists():
        safe_print(f"[{now_s()}] All done. loveble present at {loveble_path}")
        # print summary minimal JSON for server usage
        result = {"ok": True, "summary": str(outdir / "user_summary.json"), "loveble": loveble}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)
    else:
        # write a helpful diagnostics JSON
        diag = {
            "ok": False,
            "error": "Loveble output not generated",
            "stdout": ext_res.get("stdout", "") if 'ext_res' in locals() else "",
            "stderr": ext_res.get("stderr", "") if 'ext_res' in locals() else ""
        }
        safe_write_json(outdir / "diagnostics.json", diag)
        safe_print(f"[{now_s()}] Loveble JSON missing; diagnostics written to {outdir/'diagnostics.json'}")
        print(json.dumps(diag, ensure_ascii=False))
        # existing server expects exit code 2 when loveble missing
        sys.exit(2)


if __name__ == "__main__":
    main()
