import asyncio
import json
import os
from pathlib import Path
import httpx

MARTIAN_BASE = "https://raw.githubusercontent.com/withmartian/code-review-benchmark/main/offline/golden_comments"
FILES = [
    "cal_dot_com.json",
    "discourse.json",
    "grafana.json",
    "keycloak.json",
    "sentry.json"
]

DATA_DIR = Path(__file__).parent / "data"

async def download_martian_bench():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    
    async with httpx.AsyncClient() as client:
        for file in FILES:
            url = f"{MARTIAN_BASE}/{file}"
            print(f"Downloading {url}...")
            resp = await client.get(url)
            resp.raise_for_status()
            
            out_path = DATA_DIR / file
            out_path.write_text(resp.text, encoding="utf-8")
            
            data = json.loads(resp.text)
            print(f"  -> Saved {file} ({len(data)} PRs)")
            
            for pr in data:
                # e.g., https://github.com/calcom/cal.com/pull/8087
                url_parts = pr["url"].rstrip("/").split("/")
                pr_number = int(url_parts[-1])
                nwo = f"{url_parts[-4]}/{url_parts[-3]}"
                
                tasks.append({
                    "nwo": nwo,
                    "pr_number": pr_number,
                    "title": pr.get("pr_title", ""),
                    "golden_findings": pr["comments"]
                })
                
    # Save a consolidated manifest
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    
    print(f"\nSuccessfully downloaded Martian Bench dataset.")
    print(f"Total PRs: {len(tasks)}")
    print(f"Total Golden Comments: {sum(len(t['golden_findings']) for t in tasks)}")
    print(f"Saved to: {DATA_DIR}")

if __name__ == "__main__":
    asyncio.run(download_martian_bench())
