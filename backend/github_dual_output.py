#!/usr/bin/env python3
"""
github_dual_output.py (Production Edition, Option-A-compliant skills output)

Outputs:
 - repos.ndjson
 - repos.csv
 - skills.csv (columns: skill, source, score, count)
 - user_summary.json

Skill conventions (Option A, using C: scaled percentages 0-100):
 - language-based skill rows: source='language', score = percentage (0..100), count = 0
 - readme-based skill rows:   source='readme',   score = 0.0,             count = occurrences

Usage:
    python github_dual_output.py --username <user> [--top-k-detail N] [--output-dir DIR] [--cache] [--skip-stats]
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import logging
import pathlib
import csv
import re
import base64
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

# ==============================
# CONFIGURATION
# ==============================
GITHUB_API = "https://api.github.com"
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "GH-Profile-Extractor/4.0"
}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

# Logging
logger = logging.getLogger("github_dual_output")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.handlers = [handler]

# Retry tuning
MAX_RETRIES = 6
BASE_SLEEP = 2.0

# Cache folder
CACHE_DIR = pathlib.Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# README tech regex (common stack tokens)
TECH_RE = re.compile(
    r"\b(PyTorch|TensorFlow|YOLO|OpenCV|FastAPI|Flask|Django|React|Vue|Docker|Kubernetes|Node\.js|Express|"
    r"ONNX|EdgeTPU|Arduino|Raspberry|ROS|C\+\+|TypeScript|JavaScript|Python|Java|pandas|numpy|scikit-learn|"
    r"langchain|groq|nextjs)\b", re.IGNORECASE
)

# ====================================================================
# Utility helpers
# ====================================================================
def safe_json(resp: Optional[requests.Response]):
    try:
        if not resp or not resp.text:
            return None
        return resp.json()
    except Exception:
        return None

def api_get(url: str, params=None, allow_202=True, timeout=30) -> requests.Response:
    attempt = 0
    wait = BASE_SLEEP
    last_exc = None
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            logger.warning(f"Network error GET {url}: {e} (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            wait *= 1.8
            continue

        # 202 computing
        if resp.status_code == 202 and allow_202:
            logger.info(f"202 computing {url} (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            wait *= 1.8
            continue

        # Rate limit handling
        if resp.status_code == 403:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            if remaining == "0" and reset:
                sleep_for = max(5, int(reset) - int(time.time()) + 2)
                logger.warning(f"Rate limited. Sleeping {sleep_for}s...")
                time.sleep(sleep_for)
                attempt = 0
                wait = BASE_SLEEP
                continue
            time.sleep(wait)
            wait *= 1.8
            continue

        if resp.status_code >= 500:
            logger.warning(f"Server error {resp.status_code} on {url}; retrying in {wait}s")
            time.sleep(wait)
            wait *= 1.8
            continue

        if not (200 <= resp.status_code < 300):
            # non-2xx and not retried above -> raise to let caller decide
            body = None
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:300]
            logger.error(f"HTTP {resp.status_code} for {url}: {body}")
            resp.raise_for_status()

        return resp

    raise RuntimeError(f"Exceeded retries for GET {url}; last error={last_exc}")

# cache helpers
def cache_get(key: str):
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def cache_set(key: str, val):
    f = CACHE_DIR / f"{key}.json"
    try:
        f.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# ====================================================================
# GitHub helpers
# ====================================================================
def fetch_user(username: str):
    url = f"{GITHUB_API}/users/{username}"
    resp = api_get(url)
    data = safe_json(resp) or {}
    return {
        "username": data.get("login", username),
        "display_name": data.get("name"),
        "bio": data.get("bio"),
        "profile_url": data.get("html_url"),
        "avatar_url": data.get("avatar_url"),
        "metrics": {
            "followers": data.get("followers", 0),
            "following": data.get("following", 0),
            "public_repos": data.get("public_repos", 0),
            "public_gists": data.get("public_gists", 0),
            "created_at": data.get("created_at")
        }
    }

def fetch_paginated(url: str, params=None, max_items=None):
    results = []
    next_url = url
    while next_url:
        resp = api_get(next_url, params=params)
        chunk = safe_json(resp) or []
        results.extend(chunk)
        if max_items and len(results) >= max_items:
            return results[:max_items]
        if "next" in resp.links:
            next_url = resp.links["next"]["url"]
            params = None
        else:
            break
    return results

def fetch_user_repos(username: str, max_repos=None):
    url = f"{GITHUB_API}/users/{username}/repos"
    params = {"per_page": 100, "type": "owner", "sort": "updated"}
    raw = fetch_paginated(url, params, max_repos)
    repos = []
    for r in raw:
        repos.append({
            "name": r.get("name"),
            "full_name": r.get("full_name"),
            "description": r.get("description"),
            "html_url": r.get("html_url"),
            "stars": r.get("stargazers_count", 0),
            "forks": r.get("forks_count", 0),
            "watchers": r.get("watchers_count", 0),
            "language": r.get("language"),
            "topics": r.get("topics") or [],
            "size_kb": r.get("size", 0),
            "open_issues": r.get("open_issues_count", 0),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
            "default_branch": r.get("default_branch"),
            "fork": r.get("fork"),
            "license": (r.get("license") or {}).get("name")
        })
    return repos

def fetch_repo_languages(owner: str, repo: str):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/languages"
    resp = api_get(url)
    return safe_json(resp) or {}

def fetch_repo_readme(owner: str, repo: str):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/readme"
    try:
        resp = api_get(url)
    except requests.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 404:
            return None
        return None
    data = safe_json(resp) or {}
    content = data.get("content")
    enc = data.get("encoding", "base64")
    if content and enc == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            return None
    return None

def fetch_file_content_safe(owner: str, repo: str, path: str) -> Optional[str]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    try:
        resp = api_get(url)
    except requests.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 404:
            return None
        return None
    except Exception:
        return None
    data = safe_json(resp) or {}
    content = data.get("content")
    encoding = data.get("encoding", "").lower()
    if content and encoding == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            return None
    download_url = data.get("download_url")
    if download_url:
        try:
            r = requests.get(download_url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text
        except Exception:
            return None
    return None

# stats helpers (safe)
def fetch_commit_activity(owner: str, repo: str, retries=5):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/stats/commit_activity"
    for attempt in range(1, retries + 1):
        try:
            resp = api_get(url)
        except Exception:
            time.sleep(1.2 * attempt)
            continue
        if resp.status_code == 202:
            time.sleep(1.5 * attempt)
            continue
        data = safe_json(resp)
        if isinstance(data, list):
            return [int(x.get("total", 0)) for x in data]
        return None
    return None

def count_commits_since_safe(owner: str, repo: str, since_iso: str):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params = {"since": since_iso, "per_page": 100}
    total = 0
    next_url = url
    while next_url:
        try:
            resp = api_get(next_url, params=params)
        except Exception:
            return None
        data = safe_json(resp) or []
        total += len(data)
        if "next" in resp.links:
            next_url = resp.links["next"]["url"]
            params = None
        else:
            break
    return total

def get_count_via_pagination(url: str, params=None):
    import urllib.parse, re
    if params is None:
        params = {}
    params = dict(params)
    params["per_page"] = 1
    try:
        resp = api_get(url, params=params, allow_202=False)
    except Exception:
        return None
    link = resp.headers.get("Link")
    if link:
        m = re.search(r'<([^>]+)>; rel="last"', link)
        if m:
            last_url = m.group(1)
            parsed = urllib.parse.urlparse(last_url)
            q = urllib.parse.parse_qs(parsed.query)
            page = q.get("page", [None])[0]
            if page:
                try:
                    return int(page)
                except:
                    pass
    data = safe_json(resp)
    if isinstance(data, list):
        return len(data)
    return None

def get_contributors_count(owner: str, repo: str):
    return get_count_via_pagination(f"{GITHUB_API}/repos/{owner}/{repo}/contributors")

def get_pulls_count(owner: str, repo: str):
    return get_count_via_pagination(f"{GITHUB_API}/repos/{owner}/{repo}/pulls", params={"state": "all"})

# dependency parsers
def parse_package_json(text: str):
    try:
        obj = json.loads(text)
        deps = obj.get("dependencies", {}) or {}
        dev = obj.get("devDependencies", {}) or {}
        return list(set(list(deps.keys()) + list(dev.keys())))
    except Exception:
        return []

def parse_requirements_txt(txt: str):
    out = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("-e"):
            continue
        pkg = ln.split("==")[0].split(">=")[0].split("<=")[0].strip()
        if pkg:
            out.append(pkg)
    return out

def parse_pyproject_toml(txt: str):
    out = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("[") or ln.startswith("#"):
            continue
        if "=" in ln:
            key = ln.split("=", 1)[0].strip().strip('"').strip("'")
            if key:
                out.append(key)
    return list(set(out))

def score_repo_for_detail(repo: Dict):
    stars = repo.get("stars", 0)
    size = repo.get("size_kb", 0)
    updated = repo.get("updated_at")
    age_score = 0.0
    if updated:
        try:
            dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            days = (datetime.now(timezone.utc) - dt).days
            age_score = max(0.0, 1.0 - days / 365.0)
        except Exception:
            pass
    return stars * 2.0 + (size / 1000.0) + age_score * 5.0

def extract_tech(text: str):
    if not text:
        return []
    matches = TECH_RE.findall(text)
    cleaned = {m.strip() for m in matches if m and isinstance(m, str)}
    return sorted(cleaned)

# ====================================================================
# MAIN pipeline
# ====================================================================
def analyze_and_export(username: str, max_repos, top_k_detail, skip_stats, commit_fallback, output_dir, use_cache):
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        user = fetch_user(username)
    except Exception as e:
        return {"error": True, "message": "Failed to fetch user", "details": str(e)}

    try:
        repos = fetch_user_repos(username, max_repos)
    except Exception as e:
        return {"error": True, "message": "Failed to fetch repos", "details": str(e)}

    created = user["metrics"].get("created_at")
    account_age_years = None
    if created:
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            account_age_years = (datetime.now(timezone.utc) - dt).days / 365.0
        except Exception:
            account_age_years = None

    repo_count = len(repos)
    repos_per_year = repo_count / account_age_years if account_age_years and account_age_years > 0 else 0.0

    scored = sorted(repos, key=score_repo_for_detail, reverse=True)
    top_detail = scored[:top_k_detail] if top_k_detail and len(scored) > 0 else []

    ndjson_path = out_dir / "repos.ndjson"
    csv_path = out_dir / "repos.csv"
    skills_csv = out_dir / "skills.csv"
    summary_path = out_dir / "user_summary.json"

    nd_lines = []
    csv_rows = []
    skills_agg = {}
    lang_agg = {}
    repos_missing_stats = []

    total_commits_12m = 0
    total_stars = 0

    since_iso = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

    for repo in repos:
        name = repo.get("name") or ""
        if not name:
            continue

        owner = username
        repo_obj = {
            "name": name,
            "full_name": repo.get("full_name"),
            "description": repo.get("description") or "",
            "html_url": repo.get("html_url") or "",
            "stars": repo.get("stars", 0),
            "forks": repo.get("forks", 0),
            "watchers": repo.get("watchers", 0),
            "primary_language": repo.get("language"),
            "topics": repo.get("topics") or [],
            "size_kb": repo.get("size_kb", 0),
            "open_issues": repo.get("open_issues", 0),
            "created_at": repo.get("created_at") or "",
            "last_pushed_at": repo.get("updated_at") or "",
        }

        is_detail = any(r["name"] == name for r in top_detail)
        detailed = {}
        if is_detail:
            cache_key = f"{owner}__{name}"
            cached = cache_get(cache_key) if use_cache else None
            if cached:
                detailed = cached
            else:
                langs = {}
                readme = None
                commits_12m = None
                try:
                    langs = fetch_repo_languages(owner, name) or {}
                except Exception:
                    langs = {}
                try:
                    readme = fetch_repo_readme(owner, name)
                except Exception:
                    readme = None

                weekly = None
                if not skip_stats:
                    weekly = fetch_commit_activity(owner, name)
                    if weekly:
                        commits_12m = sum(weekly)
                    elif commit_fallback:
                        cnt = count_commits_since_safe(owner, name, since_iso)
                        commits_12m = int(cnt) if cnt is not None else None

                try:
                    contributors = get_contributors_count(owner, name)
                except Exception:
                    contributors = None
                try:
                    pulls = get_pulls_count(owner, name)
                except Exception:
                    pulls = None

                deps = []
                for fpath in ("package.json", "requirements.txt", "pyproject.toml", "setup.py"):
                    txt = fetch_file_content_safe(owner, name, fpath)
                    if not txt:
                        continue
                    if fpath == "package.json":
                        deps += parse_package_json(txt)
                    elif fpath == "requirements.txt":
                        deps += parse_requirements_txt(txt)
                    elif fpath == "pyproject.toml":
                        deps += parse_pyproject_toml(txt)
                    elif fpath == "setup.py":
                        m = re.findall(r"['\"](.+?)['\"]", txt)
                        deps += m

                detailed = {
                    "languages": langs or {},
                    "readme": readme,
                    "commits_past_12m": commits_12m,
                    "contributors": contributors,
                    "pulls": pulls,
                    "dependencies": list(dict.fromkeys(deps)),
                }
                if use_cache:
                    cache_set(cache_key, detailed)

            repo_obj["languages"] = detailed.get("languages", {}) or {}
            repo_obj["dependencies"] = detailed.get("dependencies", []) or []
            repo_obj["contributors"] = int(detailed.get("contributors") or 0)
            repo_obj["pulls"] = int(detailed.get("pulls") or 0)
            repo_obj["commits_past_12m"] = int(detailed.get("commits_past_12m") or 0)

            readme_txt = detailed.get("readme") or ""
            repo_obj["detected_tech"] = extract_tech(readme_txt)

            comp = 0.0
            comp += min(1.0, repo_obj["size_kb"] / (1024.0 * 50.0))
            comp += min(1.0, repo_obj["commits_past_12m"] / 200.0)
            comp += min(1.0, repo_obj["stars"] / 50.0)
            comp += 0.2 if repo_obj.get("languages") else 0.0
            repo_obj["complexity_score"] = round(min(4.0, comp) / 4.0 * 100.0, 2)

            total_stars += repo_obj["stars"]
            total_commits_12m += repo_obj["commits_past_12m"]

            for L, v in (repo_obj.get("languages") or {}).items():
                try:
                    lang_agg[L] = lang_agg.get(L, 0) + int(v)
                except Exception:
                    pass

            for t in repo_obj.get("detected_tech", []):
                skills_agg[t] = skills_agg.get(t, 0) + 1

        else:
            repo_obj.update({
                "languages": {},
                "dependencies": [],
                "contributors": 0,
                "pulls": 0,
                "commits_past_12m": 0,
                "detected_tech": [],
                "complexity_score": 0.0,
            })
            if repo_obj.get("primary_language"):
                lang_agg[repo_obj["primary_language"]] = lang_agg.get(repo_obj["primary_language"], 0) + 1

        if repo_obj.get("commits_past_12m", 0) == 0:
            repos_missing_stats.append(name)

        nd_lines.append(repo_obj)

        csv_rows.append({
            "name": repo_obj.get("name"),
            "description": repo_obj.get("description"),
            "html_url": repo_obj.get("html_url"),
            "stars": repo_obj.get("stars"),
            "forks": repo_obj.get("forks"),
            "watchers": repo_obj.get("watchers"),
            "primary_language": repo_obj.get("primary_language") or "",
            "size_kb": repo_obj.get("size_kb"),
            "created_at": repo_obj.get("created_at"),
            "last_pushed_at": repo_obj.get("last_pushed_at"),
            "commits_past_12m": repo_obj.get("commits_past_12m"),
            "contributors": repo_obj.get("contributors"),
            "pulls": repo_obj.get("pulls"),
            "complexity_score": repo_obj.get("complexity_score"),
            "detected_tech": "|".join(repo_obj.get("detected_tech", [])),
            "dependencies": "|".join(repo_obj.get("dependencies", [])),
        })

    # language profile percentages (0..100)
    lang_profile = {}
    total_lang = sum(lang_agg.values()) if lang_agg else 0
    if total_lang > 0:
        for k, v in lang_agg.items():
            lang_profile[k] = round((v / total_lang) * 100.0, 2)

    # build skills list - Option A (C: percentage)
    skill_list = []
    # language-origin skills: score = percentage (0..100), count = 0
    for k, pct in sorted(lang_profile.items(), key=lambda x: x[1], reverse=True):
        skill_list.append({"skill": k, "source": "language", "score": float(pct), "count": 0})

    # readme-origin techs: score = 0.0, count = occurrences
    for k, cnt in sorted(skills_agg.items(), key=lambda x: x[1], reverse=True):
        skill_list.append({"skill": k, "source": "readme", "score": 0.0, "count": int(cnt)})

    # experience metrics
    if account_age_years and account_age_years > 0:
        avg_commits = total_commits_12m / account_age_years
        avg_stars = total_stars / account_age_years
    else:
        avg_commits = 0.0
        avg_stars = 0.0

    summary = {
        "username": username,
        "profile_url": user.get("profile_url"),
        "avatar_url": user.get("avatar_url"),
        "bio": user.get("bio"),
        "metrics": user.get("metrics", {}),
        "num_repos_fetched": repo_count,
        "top_languages": [{"language": k, "percent": v} for k, v in lang_profile.items()],
        "experience": {
            "account_age_years": round(account_age_years, 2) if account_age_years else 0,
            "repos_per_year": round(repos_per_year, 2),
            "avg_commits_per_year": round(avg_commits, 2),
            "avg_stars_per_year": round(avg_stars, 2),
            "total_commits_past_year": int(total_commits_12m)
        },
        "repos_missing_stats": repos_missing_stats,
    }

    # Write outputs safely
    try:
        with open(ndjson_path, "w", encoding="utf-8") as f:
            for row in nd_lines:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("Failed to write ndjson: %s", e)

    try:
        if csv_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                for r in csv_rows:
                    writer.writerow(r)
        else:
            # empty csv with header
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["name","description","html_url","stars","forks","watchers","primary_language","size_kb","created_at","last_pushed_at","commits_past_12m","contributors","pulls","complexity_score","detected_tech","dependencies"])
    except Exception as e:
        logger.error("Failed to write repos.csv: %s", e)

    try:
        with open(skills_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["skill","source","score","count"])
            writer.writeheader()
            for s in skill_list:
                writer.writerow({
                    "skill": s.get("skill"),
                    "source": s.get("source"),
                    "score": s.get("score") if s.get("score") is not None else 0.0,
                    "count": s.get("count") if s.get("count") is not None else 0
                })
    except Exception as e:
        logger.error("Failed to write skills.csv: %s", e)

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to write user_summary.json: %s", e)

    logger.info(f"Written outputs → {summary_path}, {csv_path}, {skills_csv}")
    return {"ok": True, "summary": str(summary_path)}

# CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--username", "-u", required=True)
    p.add_argument("--max-repos", type=int, default=None)
    p.add_argument("--top-k-detail", type=int, default=8)
    p.add_argument("--skip-stats", action="store_true")
    p.add_argument("--commit-count-fallback", action="store_true")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--cache", action="store_true")
    args = p.parse_args()

    try:
        res = analyze_and_export(
            args.username,
            args.max_repos,
            args.top_k_detail,
            args.skip_stats,
            args.commit_count_fallback,
            args.output_dir,
            args.cache
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.exception("Unhandled error in extractor: %s", e)
        print(json.dumps({"error": True, "message": "unhandled", "details": str(e)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
