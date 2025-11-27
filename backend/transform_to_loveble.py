# #!/usr/bin/env python3
# """
# transform_to_loveble.py (Windows-safe final edition)
# Converts extractor outputs + predictions into Loveble JSON.
# NO UNICODE SYMBOLS (Windows CMD safe).
# """

# from __future__ import annotations
# import json
# import sys
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from typing import Any, Dict, List, Optional

# # -----------------------------
# # Safe helpers
# # -----------------------------

# def safe_int(x, default=0):
#     try:
#         if x is None:
#             return default
#         if isinstance(x, float) and np.isnan(x):
#             return default
#         return int(float(x))
#     except:
#         return default

# def safe_float(x, default=0.0):
#     try:
#         if x is None:
#             return default
#         if isinstance(x, float) and np.isnan(x):
#             return default
#         return float(x)
#     except:
#         return default

# def safe_str(x, default=""):
#     try:
#         if x is None:
#             return default
#         if isinstance(x, float) and np.isnan(x):
#             return default
#         return str(x)
#     except:
#         return default

# def ensure_list_of_dicts_recs(raw):
#     out = []
#     if not raw:
#         return out

#     if isinstance(raw, dict):
#         for k, v in raw.items():
#             out.append({"skill": safe_str(k), "priority": safe_int(v)})
#         return out

#     if isinstance(raw, list):
#         for item in raw:
#             if isinstance(item, dict):
#                 name = item.get("skill") or item.get("name")
#                 pri = item.get("priority") or item.get("score") or 0
#                 out.append({"skill": safe_str(name), "priority": safe_int(pri)})
#             elif isinstance(item, (list, tuple)) and len(item) >= 1:
#                 name = item[0]
#                 pri = item[1] if len(item) > 1 else 0
#                 out.append({"skill": safe_str(name), "priority": safe_int(pri)})
#             else:
#                 out.append({"skill": safe_str(item), "priority": 0})
#     return out

# # -----------------------------
# # Build top repos
# # -----------------------------

# def build_top_repos(df):
#     if df is None or df.empty:
#         return []

#     df = df.copy()
#     if "stars" in df.columns:
#         df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0).astype(int)
#         df = df.sort_values(by="stars", ascending=False).head(10)
#     else:
#         df = df.head(10)

#     out = []
#     for _, r in df.iterrows():
#         entry = {
#             "name": safe_str(r.get("name")),
#             "description": safe_str(r.get("description")),
#             "url": safe_str(r.get("html_url") or r.get("url")),
#             "stars": safe_int(r.get("stars")),
#             "forks": safe_int(r.get("forks")),
#             "watchers": safe_int(r.get("watchers")),
#             "commits_past_12m": safe_int(r.get("commits_past_12m")),
#             "contributors": safe_int(r.get("contributors")),
#             "pulls": safe_int(r.get("pulls")),
#             "issues": safe_int(r.get("open_issues")),
#             "main_language": safe_str(r.get("primary_language")),
#             "detected_tech": safe_str(r.get("detected_tech", "")).split("|") if r.get("detected_tech") else [],
#             "dependencies": safe_str(r.get("dependencies", "")).split("|") if r.get("dependencies") else [],
#             "size": safe_int(r.get("size_kb")),
#             "last_pushed_at": safe_str(r.get("last_pushed_at")),
#             "created_at": safe_str(r.get("created_at")),
#             "complexity_score": safe_float(r.get("complexity_score")),
#         }
#         out.append(entry)
#     return out

# # -----------------------------
# # Repo timeline
# # -----------------------------

# def build_repo_timeline(df):
#     if df is None or df.empty:
#         return []
#     if "created_at" not in df.columns:
#         return []

#     try:
#         df = df.copy()
#         df["year"] = pd.to_datetime(df["created_at"], errors="coerce").dt.year
#         grp = df.groupby("year").size()
#         return [{"year": int(y), "count": int(c)} for y, c in grp.items()]
#     except:
#         return []

# # -----------------------------
# # Stars distribution
# # -----------------------------

# def build_stars_distribution(df):
#     if df is None or df.empty:
#         return []

#     try:
#         stars = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
#     except:
#         stars = pd.Series([0])

#     return [
#         {"range": "0", "count": int((stars == 0).sum())},
#         {"range": "1-10", "count": int(((stars >= 1) & (stars <= 10)).sum())},
#         {"range": "11-50", "count": int(((stars >= 11) & (stars <= 50)).sum())},
#         {"range": "51-100", "count": int(((stars >= 51) & (stars <= 100)).sum())},
#         {"range": "100+", "count": int((stars >= 101).sum())},
#     ]

# # -----------------------------
# # Main
# # -----------------------------

# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python transform_to_loveble.py <output_dir>")
#         sys.exit(1)

#     outdir = Path(sys.argv[1])

#     summary_path = outdir / "user_summary.json"
#     preds_path = outdir / "predictions.json"
#     repos_csv = outdir / "repos.csv"
#     skills_csv = outdir / "skills.csv"

#     # load summary
#     try:
#         summary = json.loads(summary_path.read_text(encoding="utf-8"))
#     except:
#         summary = {}

#     # load predictions
#     try:
#         preds = json.loads(preds_path.read_text(encoding="utf-8"))
#     except:
#         preds = {}

#     # load dataframes
#     try:
#         repos_df = pd.read_csv(repos_csv)
#     except:
#         repos_df = pd.DataFrame()

#     try:
#         skills_df = pd.read_csv(skills_csv)
#     except:
#         skills_df = pd.DataFrame()

#     # normalize skills_df
#     if not skills_df.empty:
#         skills_df["score"] = pd.to_numeric(skills_df.get("score", 0), errors="coerce").fillna(0)
#         skills_df["count"] = pd.to_numeric(skills_df.get("count", 0), errors="coerce").fillna(0).astype(int)
#         skills_df["skill"] = skills_df.get("skill", "").astype(str)

#     metrics = summary.get("metrics", {}) or {}
#     experience = summary.get("experience", {}) or {}

#     langs = summary.get("top_languages") or []
#     most_used = []
#     for x in langs:
#         if isinstance(x, dict):
#             n = x.get("language") or x.get("name")
#             if n:
#                 most_used.append(n)

#     data = {
#         "username": safe_str(summary.get("username")),
#         "avatar_url": safe_str(summary.get("avatar_url")),
#         "bio": safe_str(summary.get("bio")),
#         "profile_url": safe_str(summary.get("profile_url")),
#         "account_age_years": safe_float(experience.get("account_age_years")),
#         "total_repos": safe_int(summary.get("num_repos_fetched") or metrics.get("public_repos")),
#         "public_gists": safe_int(metrics.get("public_gists")),
#         "followers": safe_int(metrics.get("followers")),
#         "following": safe_int(metrics.get("following")),
#         "total_stars": safe_int(summary.get("total_stars")),
#         "total_commits_past_year": safe_int(experience.get("total_commits_past_year")),
#         "commits_per_year": safe_float(experience.get("avg_commits_per_year")),
#         "repos_per_year": safe_float(experience.get("repos_per_year")),
#         "average_repo_stars": safe_float(experience.get("avg_stars_per_year")),
#         "expertise_level": preds.get("expertise_level"),
#         "domain_label": preds.get("domain_label"),
#         "developer_rank": safe_float(preds.get("developer_rank")),
#         "recommended_skills": ensure_list_of_dicts_recs(preds.get("recommended_skills")),
#         "most_used_languages": most_used,
#         "top_languages": [
#             {
#                 "name": safe_str(x.get("language") or x.get("name")),
#                 "value": safe_float(x.get("percent")),
#                 "percentage": safe_float(x.get("percent")),
#             }
#             for x in langs if isinstance(x, dict)
#         ],
#         "top_skills": [] if skills_df.empty else [
#             {
#                 "name": safe_str(r["skill"]),
#                 "frequency": safe_int(r["count"]),
#                 "percentage": safe_float(r["score"]),
#                 "level": "Unknown",
#             }
#             for _, r in skills_df.sort_values("score", ascending=False).head(20).iterrows()
#         ],
#         "frameworks": [] if skills_df.empty else [
#             {"name": safe_str(r["skill"]), "count": safe_int(r["count"])}
#             for _, r in skills_df.iterrows()
#             if safe_str(r["skill"]).lower() in {"react", "django", "fastapi", "flask", "nextjs"}
#         ],
#         "top_repos": build_top_repos(repos_df),
#         "repo_timeline": build_repo_timeline(repos_df),
#         "stars_distribution": build_stars_distribution(repos_df),
#         "language_growth": []
#     }

#     out_path = outdir / "loveble_schema.json"

#     out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

#     # NO UNICODE SYMBOLS HERE (Windows-safe)
#     print("Saved Loveble JSON to:", str(out_path))


# if __name__ == "__main__":
#     main()
"""
transform_to_loveble.py — upgraded to use saved ML models when available.
Drops back to safe heuristics when model files / feature lists are missing.

Usage:
  python transform_to_loveble.py <output_dir>

Notes:
 - Expects models (optional) at ../models relative to this script:
     expertise_model.pkl, domain_model.pkl, dev_ranking_model.pkl,
     scaler.pkl, nn_scaler.pkl, skill_nn_model.pkl
 - Expects optional feature list files in models/:
     feature_cols.txt, skill_cols.txt, missing_cols.txt
 - If feature_cols.txt is missing we run heuristics (existing behavior).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

# Try to import joblib; if not present, ML won't run
try:
    import joblib
except Exception:
    joblib = None

# -----------------------------
# Safe helpers
# -----------------------------
def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return int(float(x))
    except:
        return default

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except:
        return default

def safe_str(x, default=""):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return str(x)
    except:
        return default

def ensure_list_of_dicts_recs(raw):
    out = []
    if not raw:
        return out

    if isinstance(raw, dict):
        for k, v in raw.items():
            out.append({"skill": safe_str(k), "priority": safe_int(v)})
        return out

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("skill") or item.get("name")
                pri = item.get("priority") or item.get("score") or 0
                out.append({"skill": safe_str(name), "priority": safe_int(pri)})
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                name = item[0]
                pri = item[1] if len(item) > 1 else 0
                out.append({"skill": safe_str(name), "priority": safe_int(pri)})
            else:
                out.append({"skill": safe_str(item), "priority": 0})
    return out

# -----------------------------
# Build top repos
# -----------------------------
def build_top_repos(df):
    if df is None or df.empty:
        return []

    df = df.copy()
    if "stars" in df.columns:
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(by="stars", ascending=False).head(10)
    else:
        df = df.head(10)

    out = []
    for _, r in df.iterrows():
        entry = {
            "name": safe_str(r.get("name")),
            "description": safe_str(r.get("description")),
            "url": safe_str(r.get("html_url") or r.get("url")),
            "stars": safe_int(r.get("stars")),
            "forks": safe_int(r.get("forks")),
            "watchers": safe_int(r.get("watchers")),
            "commits_past_12m": safe_int(r.get("commits_past_12m")),
            "contributors": safe_int(r.get("contributors")),
            "pulls": safe_int(r.get("pulls")),
            "issues": safe_int(r.get("open_issues")),
            "main_language": safe_str(r.get("primary_language")),
            "detected_tech": safe_str(r.get("detected_tech", "")).split("|") if r.get("detected_tech") else [],
            "dependencies": safe_str(r.get("dependencies", "")).split("|") if r.get("dependencies") else [],
            "size": safe_int(r.get("size_kb")),
            "last_pushed_at": safe_str(r.get("last_pushed_at")),
            "created_at": safe_str(r.get("created_at")),
            "complexity_score": safe_float(r.get("complexity_score")),
        }
        out.append(entry)
    return out

# -----------------------------
# Repo timeline
# -----------------------------
def build_repo_timeline(df):
    if df is None or df.empty:
        return []
    if "created_at" not in df.columns:
        return []

    try:
        df = df.copy()
        df["year"] = pd.to_datetime(df["created_at"], errors="coerce").dt.year
        grp = df.groupby("year").size()
        return [{"year": int(y), "count": int(c)} for y, c in grp.items()]
    except:
        return []

# -----------------------------
# Stars distribution
# -----------------------------
def build_stars_distribution(df):
    if df is None or df.empty:
        return []

    try:
        stars = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
    except:
        stars = pd.Series([0])

    return [
        {"range": "0", "count": int((stars == 0).sum())},
        {"range": "1-10", "count": int(((stars >= 1) & (stars <= 10)).sum())},
        {"range": "11-50", "count": int(((stars >= 11) & (stars <= 50)).sum())},
        {"range": "51-100", "count": int(((stars >= 51) & (stars <= 100)).sum())},
        {"range": "100+", "count": int((stars >= 101).sum())},
    ]

# -----------------------------
# Helper: basic features aggregator (fallback)
# -----------------------------
def compute_repo_aggregates(repos_df):
    repo_count = int(len(repos_df)) if repos_df is not None else 0
    if repos_df is None or repos_df.empty:
        avg_stars = 0.0
        active_repos = 0
    else:
        stars_num = pd.to_numeric(repos_df.get("stars", pd.Series([])), errors="coerce").fillna(0)
        avg_stars = float(stars_num.mean()) if len(stars_num) > 0 else 0.0
        # active repos: pushed within last 365 days
        try:
            pushed = pd.to_datetime(repos_df.get("last_pushed_at", pd.Series([])), errors="coerce")
            active_repos = int(((pd.Timestamp.utcnow() - pushed).dt.days <= 365).sum())
        except Exception:
            active_repos = 0
    return {
        "repo_count": repo_count,
        "repo_avg_stars": avg_stars,
        "active_repos": active_repos
    }

# -----------------------------
# Try to build feature vector from summary/preds/skills/repo aggregates
# -----------------------------
def build_feature_vector_from_summary(feature_cols, summary, preds, repos_df, skills_df):
    """
    feature_cols: list[str] expected by models
    We attempt to compute values for common names found in training script.
    Missing features get 0.
    """
    out = []
    # common sources
    metrics = summary.get("metrics", {}) or {}
    experience = summary.get("experience", {}) or {}
    # repo derived
    repo_aggs = compute_repo_aggregates(repos_df)

    # make a small mapping of common names -> computed values
    computed = {}

    # metrics/exposure
    computed["followers"] = safe_float(metrics.get("followers"), 0.0)
    computed["repo_count"] = safe_int(summary.get("num_repos_fetched") or metrics.get("public_repos") or repo_aggs.get("repo_count"))
    computed["repo_avg_stars"] = safe_float(repo_aggs.get("repo_avg_stars") or summary.get("avg_repo_stars") or 0.0)
    # experience fields
    computed["avg_commits_per_year"] = safe_float(experience.get("commits_per_year") or experience.get("avg_commits_per_year") or 0.0)
    computed["repos_per_year"] = safe_float(experience.get("repos_per_year") or 0.0)
    computed["avg_stars_per_year"] = safe_float(experience.get("average_repo_stars") or experience.get("avg_stars_per_year") or 0.0)
    computed["active_repos"] = safe_int(repo_aggs.get("active_repos"))
    # additional fallbacks
    computed["repo_avg_stars"] = computed.get("repo_avg_stars", 0.0)
    computed["repo_count"] = computed.get("repo_count", 0)
    # skills -> put presence as 0/1 for names in skill_cols
    skill_map = {}
    if skills_df is not None and not skills_df.empty:
        # skills_df probably has columns: skill, score, count
        for _, r in skills_df.iterrows():
            name = safe_str(r.get("skill")).strip()
            skill_map[name.lower()] = {
                "score": safe_float(r.get("score"), 0.0),
                "count": safe_int(r.get("count"), 0)
            }

    # For each requested feature, try to fill from computed or skills
    for fname in feature_cols:
        # normalize
        f = fname.strip()
        low = f.lower()
        val = 0.0

        # direct match -> computed mapping
        if low in computed:
            val = computed[low]
        # followers-like
        elif "followers" in low:
            val = computed.get("followers", 0.0)
        # repo_count-like
        elif "repo_count" in low or "repocount" in low:
            val = computed.get("repo_count", 0)
        # repo_avg_stars
        elif "repo_avg_stars" in low or "repo_avg" in low or "avg_stars" in low:
            val = computed.get("repo_avg_stars", 0.0)
        elif "avg_commits" in low or "commits_per" in low:
            val = computed.get("avg_commits_per_year", 0.0)
        elif "repos_per_year" in low or "reposperyear" in low:
            val = computed.get("repos_per_year", 0.0)
        elif "active_repos" in low:
            val = computed.get("active_repos", 0)
        # skill_ prefix -> look into skill_map
        elif low.startswith("skill_") or low.startswith("lang_"):
            # skill name usually after prefix
            name = f.split("_", 1)[1] if "_" in f else f
            lookup = name.lower()
            if lookup in skill_map:
                # use presence as 1 if count>0 else 0
                val = 1 if skill_map[lookup]["count"] > 0 or skill_map[lookup]["score"] > 0 else 0
            else:
                # try simple substring match in skill_map
                found = 0
                for k in skill_map:
                    if lookup in k:
                        found = 1 if skill_map[k]["count"] > 0 or skill_map[k]["score"] > 0 else 0
                        break
                val = found
        # missing_ prefix -> inverse of skill presence
        elif low.startswith("missing_"):
            name = f.split("_", 1)[1] if "_" in f else f
            lookup = name.lower()
            if lookup in skill_map:
                val = 0 if skill_map[lookup]["count"] > 0 or skill_map[lookup]["score"] > 0 else 1
            else:
                val = 1
        else:
            # try fallback from summary/preds
            if f in summary:
                v = summary.get(f)
                try:
                    val = float(v)
                except:
                    val = 0.0
            elif f in preds:
                try:
                    val = float(preds.get(f, 0.0) or 0.0)
                except:
                    val = 0.0
            else:
                val = 0.0

        # final cast
        try:
            out.append(float(val))
        except:
            out.append(0.0)

    return np.array(out, dtype=float).reshape(1, -1)

# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python transform_to_loveble.py <output_dir>")
        sys.exit(1)

    outdir = Path(sys.argv[1])
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"

    summary_path = outdir / "user_summary.json"
    preds_path = outdir / "predictions.json"
    repos_csv = outdir / "repos.csv"
    skills_csv = outdir / "skills.csv"

    # load summary
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        summary = {}

    # load predictions (extractor's prior preds if any)
    try:
        preds = json.loads(preds_path.read_text(encoding="utf-8"))
    except Exception:
        preds = {}

    # load dataframes
    try:
        repos_df = pd.read_csv(repos_csv)
    except Exception:
        repos_df = pd.DataFrame()

    try:
        skills_df = pd.read_csv(skills_csv)
    except Exception:
        skills_df = pd.DataFrame()

    # normalize skills_df
    if not skills_df.empty:
        skills_df["score"] = pd.to_numeric(skills_df.get("score", 0), errors="coerce").fillna(0)
        skills_df["count"] = pd.to_numeric(skills_df.get("count", 0), errors="coerce").fillna(0).astype(int)
        skills_df["skill"] = skills_df.get("skill", "").astype(str)

    # baseline data object (previous behavior)
    metrics = summary.get("metrics", {}) or {}
    experience = summary.get("experience", {}) or {}

    langs = summary.get("top_languages") or []
    most_used = []
    for x in langs:
        if isinstance(x, dict):
            n = x.get("language") or x.get("name")
            if n:
                most_used.append(n)

    data = {
        "username": safe_str(summary.get("username")),
        "avatar_url": safe_str(summary.get("avatar_url")),
        "bio": safe_str(summary.get("bio")),
        "profile_url": safe_str(summary.get("profile_url")),
        "account_age_years": safe_float(experience.get("account_age_years")),
        "total_repos": safe_int(summary.get("num_repos_fetched") or metrics.get("public_repos")),
        "public_gists": safe_int(metrics.get("public_gists")),
        "followers": safe_int(metrics.get("followers")),
        "following": safe_int(metrics.get("following")),
        "total_stars": safe_int(summary.get("total_stars")),
        "total_commits_past_year": safe_int(experience.get("total_commits_past_year")),
        "commits_per_year": safe_float(experience.get("avg_commits_per_year") or experience.get("commits_per_year")),
        "repos_per_year": safe_float(experience.get("repos_per_year") or 0.0),
        "average_repo_stars": safe_float(experience.get("avg_stars_per_year") or experience.get("average_repo_stars")),
        # these get overwritten later if ML models produce values
        "expertise_level": preds.get("expertise_level"),
        "domain_label": preds.get("domain_label"),
        "developer_rank": safe_float(preds.get("developer_rank")),
        "recommended_skills": ensure_list_of_dicts_recs(preds.get("recommended_skills")),
        "most_used_languages": most_used,
        "top_languages": [
            {
                "name": safe_str(x.get("language") or x.get("name")),
                "value": safe_float(x.get("percent")),
                "percentage": safe_float(x.get("percent")),
            }
            for x in langs if isinstance(x, dict)
        ],
        "top_skills": [] if skills_df.empty else [
            {
                "name": safe_str(r["skill"]),
                "frequency": safe_int(r["count"]),
                "percentage": safe_float(r["score"]),
                "level": "Unknown",
            }
            for _, r in skills_df.sort_values("score", ascending=False).head(20).iterrows()
        ],
        "frameworks": [] if skills_df.empty else [
            {"name": safe_str(r["skill"]), "count": safe_int(r["count"])}
            for _, r in skills_df.iterrows()
            if safe_str(r["skill"]).lower() in {"react", "django", "fastapi", "flask", "nextjs"}
        ],
        "top_repos": build_top_repos(repos_df),
        "repo_timeline": build_repo_timeline(repos_df),
        "stars_distribution": build_stars_distribution(repos_df),
        "language_growth": []
    }

    # -----------------------------
    # Attempt ML integration
    # -----------------------------
    ml_used = False
    ml_diag = {"models_found": [], "warnings": []}

    if joblib is not None and models_dir.exists():
        # find model files
        expertise_file = models_dir / "expertise_model.pkl"
        domain_file = models_dir / "domain_model.pkl"
        devrank_file = models_dir / "dev_ranking_model.pkl"
        scaler_file = models_dir / "scaler.pkl"
        feature_cols_file = models_dir / "feature_cols.txt"
        skill_cols_file = models_dir / "skill_cols.txt"
        missing_cols_file = models_dir / "missing_cols.txt"
        nn_scaler_file = models_dir / "nn_scaler.pkl"
        skill_nn_file = models_dir / "skill_nn_model.pkl"

        # load feature lists if present
        feature_cols = []
        skill_cols = []
        missing_cols = []
        try:
            if feature_cols_file.exists():
                feature_cols = [s.strip() for s in feature_cols_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        except Exception as e:
            ml_diag["warnings"].append(f"Could not read feature_cols.txt: {e}")

        try:
            if skill_cols_file.exists():
                skill_cols = [s.strip() for s in skill_cols_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        except Exception as e:
            ml_diag["warnings"].append(f"Could not read skill_cols.txt: {e}")

        try:
            if missing_cols_file.exists():
                missing_cols = [s.strip() for s in missing_cols_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        except Exception as e:
            ml_diag["warnings"].append(f"Could not read missing_cols.txt: {e}")

        # If required model files exist, load them and attempt prediction
        try:
            # load scaler + models if present
            scaler = joblib.load(scaler_file) if scaler_file.exists() else None
            nn_scaler = joblib.load(nn_scaler_file) if nn_scaler_file.exists() else None

            expertise_model = joblib.load(expertise_file) if expertise_file.exists() else None
            domain_model = joblib.load(domain_file) if domain_file.exists() else None
            devrank_model = joblib.load(devrank_file) if devrank_file.exists() else None
            skill_nn_model = joblib.load(skill_nn_file) if skill_nn_file.exists() else None

            # mark which models existed
            for p in [expertise_file, domain_file, devrank_file, scaler_file, feature_cols_file]:
                if p.exists():
                    ml_diag["models_found"].append(str(p.name))

            # Only attempt ML if we have at least one main model and feature columns available
            if (expertise_model or domain_model or devrank_model) and feature_cols:
                # build X using feature_cols
                X = build_feature_vector_from_summary(feature_cols, summary, preds, repos_df, skills_df)
                if scaler is not None:
                    try:
                        X_scaled = scaler.transform(X)
                    except Exception as e:
                        # If scaler transform fails, try fit-transform fallback (rare)
                        ml_diag["warnings"].append(f"scaler.transform failed: {e}; using raw X")
                        X_scaled = X
                else:
                    X_scaled = X

                # predictions
                try:
                    if expertise_model:
                        # many classifiers return label directly; preserved training used RFClassifier
                        exp_pred = expertise_model.predict(X_scaled)
                        # safe cast to int (keep within 0-4)
                        data["expertise_level"] = int(exp_pred[0]) if len(exp_pred) > 0 else data.get("expertise_level")
                    if domain_model:
                        dom_pred = domain_model.predict(X_scaled)
                        data["domain_label"] = int(dom_pred[0]) if len(dom_pred) > 0 else data.get("domain_label")
                    if devrank_model:
                        rank_pred = devrank_model.predict(X_scaled)
                        # dev_rank expected as float; scale/normalization may differ - we return numeric
                        data["developer_rank"] = float(rank_pred[0]) if len(rank_pred) > 0 else data.get("developer_rank")
                    ml_used = True
                except Exception as e:
                    ml_diag["warnings"].append(f"Model predict failed: {e}")

            else:
                if not feature_cols:
                    ml_diag["warnings"].append("feature_cols.txt missing — skipping ML predictions.")
                else:
                    ml_diag["warnings"].append("No ML models found to run predictions.")

            # Recommended skills: prefer preds from extractor; else simple heuristic using skills_df
            if not data.get("recommended_skills"):
                # try using existing preds recommended_skills if any
                if preds.get("recommended_skills"):
                    data["recommended_skills"] = ensure_list_of_dicts_recs(preds.get("recommended_skills"))
                else:
                    # heuristic: pick top missing-ish skills (low score & low count)
                    recs = []
                    if not skills_df.empty:
                        # rank candidate skills by small score and small count
                        skills_df_sorted = skills_df.sort_values(["score", "count"], ascending=[True, True])
                        # take top 10 as suggestions with priority by inverse order
                        top_missing = skills_df_sorted.head(10)
                        for i, (_, r) in enumerate(top_missing.iterrows()):
                            recs.append({"skill": safe_str(r["skill"]), "priority": int(max(1, 10 - i))})
                    else:
                        # fallback to empty list
                        recs = []
                    data["recommended_skills"] = recs

        except Exception as e:
            ml_diag["warnings"].append(f"Failed to load/run ML models: {e}")

    else:
        if joblib is None:
            # joblib not available in environment
            ml_diag["warnings"].append("joblib not installed — ML models will not be used.")
        else:
            ml_diag["warnings"].append("models/ directory not found — ML models will not be used.")

    # embed ML diagnostics into the output (non-sensitive)
    data["_ml"] = {
        "used": bool(ml_used),
        "diag": ml_diag
    }

    out_path = outdir / "loveble_schema.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Windows-safe, no special unicode characters
    print("Saved Loveble JSON to:", str(out_path))
    if ml_diag.get("warnings"):
        print("ML Warnings:")
        for w in ml_diag["warnings"]:
            print(" -", w)

if __name__ == "__main__":
    main()
