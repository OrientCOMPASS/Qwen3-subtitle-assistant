#!/usr/bin/env python3
"""阻塞式监听 GitHub Actions run，结束立即返回（避免固定 sleep 的浪费）。

用法:
  python scripts/watch_runs.py --token-file /tmp/.ghtoken --sha <sha> [--timeout-min 28]
  python scripts/watch_runs.py --token-file /tmp/.ghtoken --run-id 123456
退出码: 0=全部成功; 1=存在失败; 2=超时仍在运行（可再次调用续等）
"""
import argparse, json, sys, time, urllib.request

API = "https://api.github.com/repos/OrientCOMPASS/Qwen3-subtitle-assistant"

def api(path, tok):
    req = urllib.request.Request(API + path, headers={
        "User-Agent": "watch-runs", "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def summarize(run, tok):
    jd = api(f"/actions/runs/{run['id']}/jobs", tok)
    failed = []
    for j in jd["jobs"]:
        for s in j["steps"]:
            if s.get("conclusion") == "failure":
                failed.append(s["name"])
    return failed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", default="/tmp/.ghtoken")
    ap.add_argument("--sha", help="只看该 commit sha 触发的 runs（前缀匹配）")
    ap.add_argument("--run-id", type=int, help="监听指定 run")
    ap.add_argument("--timeout-min", type=float, default=28.0)
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()
    tok = open(args.token_file).read().strip()

    deadline = time.time() + args.timeout_min * 60
    while True:
        if args.run_id:
            runs = [api(f"/actions/runs/{args.run_id}", tok)]
        else:
            d = api("/actions/runs?per_page=15", tok)
            runs = [r for r in d["workflow_runs"]
                    if (not args.sha or r["head_sha"].startswith(args.sha))
                    and r["event"] != "schedule"]
        if not runs:
            print("no runs found yet..."); 
        active = [r for r in runs if r["status"] != "completed"]
        for r in runs:
            print(f"[{r['status']:>12}/{r.get('conclusion') or '-':<8}] {r['name']:<9} sha={r['head_sha'][:7]} id={r['id']}", flush=True)
        if runs and not active:
            rc = 0
            for r in runs:
                if r.get("conclusion") != "success":
                    rc = 1
                    failed = summarize(r, tok)
                    print(f"  !! {r['name']} 失败步骤: {failed}", flush=True)
            print("ALL DONE", "SUCCESS" if rc == 0 else "WITH FAILURES", flush=True)
            return rc
        if time.time() > deadline:
            print(f"TIMEOUT: still active: {[r['id'] for r in active]}", flush=True)
            return 2
        time.sleep(args.interval)

if __name__ == "__main__":
    sys.exit(main())
