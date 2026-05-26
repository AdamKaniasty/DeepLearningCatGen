from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def need(var: str, hint: str = "") -> str:
    v = os.environ.get(var)
    if not v:
        sys.exit(f"missing env var {var}. {hint}")
    return v


def git_repo_url() -> str:
    url = os.environ.get("GIT_REPO_URL")
    if url:
        return url
    out = subprocess.run(
        ["git", "-C", str(ROOT), "remote", "get-url", "origin"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit("set GIT_REPO_URL or add an `origin` remote")
    return out.stdout.strip()


def repo_dirname() -> str:
    url = git_repo_url()
    base = url.rstrip("/").split("/")[-1]
    return base[:-4] if base.endswith(".git") else base


def studio():
    need("LIGHTNING_USER_ID", "find in Lightning AI -> Settings -> Keys")
    need("LIGHTNING_API_KEY", "find in Lightning AI -> Settings -> Keys")
    from lightning_sdk import Studio
    from lightning_sdk.user import _get_authed_user
    name = os.environ.get("LIGHTNING_STUDIO_NAME", "catgen")
    ts = os.environ.get("LIGHTNING_TEAMSPACE")
    user = os.environ.get("LIGHTNING_USER")
    if not ts:
        u = _get_authed_user()
        tss = list(u.teamspaces)
        if len(tss) != 1:
            sys.exit(f"set LIGHTNING_TEAMSPACE; user has teamspaces: {[t.name for t in tss]}")
        ts = tss[0].name
        user = user or u.name
    kw = {"create_ok": True, "teamspace": ts}
    if user:
        kw["user"] = user
    return Studio(name, **kw)


def machine_for(name: str):
    from lightning_sdk import Machine
    if not hasattr(Machine, name):
        sys.exit(f"unknown machine '{name}'. examples: T4, L4, A10G, A100, CPU_X_4")
    return getattr(Machine, name)


def remote_cwd() -> str:
    return f"~/{repo_dirname()}"


def cmd_whoami(args):
    need("LIGHTNING_USER_ID")
    need("LIGHTNING_API_KEY")
    from lightning_sdk.user import _get_authed_user
    u = _get_authed_user()
    print(f"user: {u.name} ({u.id})")
    print("teamspaces:")
    for t in u.teamspaces:
        print(f"  - {t.name}")
    print(f"repo: {git_repo_url()}")
    print(f"target studio name: {os.environ.get('LIGHTNING_STUDIO_NAME', 'catgen')}")


def cmd_setup(args):
    s = studio()
    print(f"starting studio: {s.name}")
    s.start()
    url = git_repo_url()
    print(f"cloning repo into studio: {url}")
    s.run(f"test -d {repo_dirname()} || git clone {url}")
    s.run(f"cd {remote_cwd()} && pip install -U pip && pip install -e .")
    print("swapping .gitignore -> .cluster.gitignore so runs/ + reports/ are pushable")
    s.run(
        f"cd {remote_cwd()} && cp .cluster.gitignore .gitignore && "
        f"git update-index --assume-unchanged .gitignore"
    )
    s.install_plugin("jobs")
    if args.stop:
        print("stopping studio to save credits")
        s.stop()
    print("ready. next: upload dataset via Studio UI, then `python scripts/lightning_submit.py data`")


def cmd_sync(args):
    s = studio()
    s.start()
    s.run(f"cd {remote_cwd()} && git pull && pip install -e . > /dev/null")
    print("studio synced with latest repo + reinstalled.")


def cmd_machine(args):
    s = studio()
    m = machine_for(args.machine)
    print(f"switching studio to {args.machine}")
    s.start()
    s.switch_machine(m)
    print(f"studio machine: {s.machine}")


def cmd_smoke(args):
    s = studio()
    s.start()
    target = args.machine
    if target.lower() == "current":
        target = str(s.machine).split(".")[-1]
    else:
        m = machine_for(target)
        if str(s.machine) != f"Machine.{target}":
            print(f"switching studio to {target}")
            s.switch_machine(m)
    print(f"studio machine: {s.machine}")
    device = "cuda" if not str(s.machine).startswith("Machine.CPU") else "cpu"
    print("syncing repo...")
    s.run(f"cd {remote_cwd()} && git pull && pip install -e . > /dev/null")
    print("preparing fake data (60 cats, 30 train, 10 ref)...")
    s.run(
        f"cd {remote_cwd()} && rm -rf data/raw data/splits && "
        f"python scripts/prepare_data.py --fake 60 --train-n 30 --ref-n 10 --no-mixed"
    )
    print(f"training DCGAN smoke (2 epochs, {device})...")
    print(s.run(
        f"cd {remote_cwd()} && python -m catgen.train --config src/catgen/configs/dcgan_smoke.yaml "
        f"--device {device} --max-epochs 2 --set data.split=train_30.txt --set data.batch_size=4 --set data.num_workers=0"
    ))
    print("--- run artifacts ---")
    print(s.run(f"cd {remote_cwd()} && ls runs/ && tail -5 runs/*/metrics.csv && cat runs/*/events.jsonl"))


def cmd_data(args):
    s = studio()
    s.start()
    s.run(f"cd {remote_cwd()} && python scripts/prepare_data.py")
    print("splits ready on studio.")


def cmd_submit(args):
    configs = sorted((ROOT / "src/catgen/configs").glob(f"{args.model}_*.yaml"))
    configs = [c for c in configs if "smoke" not in c.name]
    if not configs:
        sys.exit(f"no configs for model={args.model}")
    print(f"{'planned' if args.dry_run else 'submitting'} {len(configs)} jobs to {args.machine}:")
    s = None
    m = None
    if not args.dry_run:
        s = studio()
        s.start()
        m = machine_for(args.machine)
    for cfg in configs:
        rel = cfg.relative_to(ROOT)
        job_name = cfg.stem
        cmd = (
            f"cd {remote_cwd()} && "
            f"python -m catgen.train --config {rel} --device cuda"
        )
        print(f"  {job_name}")
        if args.dry_run:
            continue
        s.installed_plugins["jobs"].run(cmd, name=job_name, machine=m)


def cmd_eval(args):
    cmd = (
        f"cd {remote_cwd()} && "
        f"bash scripts/run_eval.sh {args.n} {args.ref} cuda && "
        f"python scripts/plot_metrics.py && "
        f"python scripts/build_figures.py && "
        f"python scripts/build_report.py"
    )
    print(f"{'planned' if args.dry_run else 'submitting'} eval job on {args.machine}")
    if args.dry_run:
        print(cmd)
        return
    s = studio()
    s.start()
    m = machine_for(args.machine)
    s.installed_plugins["jobs"].run(cmd, name="catgen-eval", machine=m)


def cmd_push(args):
    s = studio()
    s.start()
    msg = args.message or "studio: sync runs and reports"
    s.run(
        f"cd {remote_cwd()} && git add -A && "
        f"(git diff --cached --quiet && echo 'nothing to commit' || "
        f"(git -c user.email=studio@catgen -c user.name=studio commit -m '{msg}' && git push))"
    )


def main():
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("whoami", help="auth check + list teamspaces")
    p.set_defaults(func=cmd_whoami)

    p = sub.add_parser("setup", help="create Studio, clone repo, install deps")
    p.add_argument("--stop", action="store_true", help="stop the Studio after setup to save credits")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("sync", help="git pull + reinstall on Studio after local code changes")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("machine", help="switch the Studio's machine type")
    p.add_argument("machine")
    p.set_defaults(func=cmd_machine)

    p = sub.add_parser("smoke", help="end-to-end smoke on the Studio (fake data, 2 epoch DCGAN on cuda)")
    p.add_argument("--machine", default=os.environ.get("LIGHTNING_MACHINE", "T4"))
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("data", help="prepare data splits inside the Studio")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("submit", help="submit training jobs for a model family")
    p.add_argument("model", choices=["dcgan", "aae", "vqvae"])
    p.add_argument("--machine", default=os.environ.get("LIGHTNING_MACHINE", "T4"))
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("eval", help="submit a job that runs eval + figures + report")
    p.add_argument("--machine", default=os.environ.get("LIGHTNING_MACHINE", "T4"))
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ref", default="fid_ref_1000.txt")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("push", help="git push runs/ + reports/ from Studio back to remote")
    p.add_argument("--message", "-m")
    p.set_defaults(func=cmd_push)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
