"""Python wrapper for the Scalismo ``nako.ribs.RibRegistration`` sbt entry point.

Builds a minimal env, writes the ``runMain`` invocation to a temp shell script,
and streams sbt's stdout back to the caller. The sbt project lives at
``environment/scalismo_ssm/``; the wrapper script
``environment/scalismo_ssm/sbt_build.sh`` handles the colon-in-path case.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_SBT_BUILD_SH = (
    Path(__file__).resolve().parents[2]
    / "environment" / "scalismo_ssm" / "sbt_build.sh"
)


def scalismo_run(sbt_cmd: str) -> int:
    """Invoke ``sbt_build.sh`` with the given quoted argument string.

    Streams stdout to the calling process; returns sbt's exit code. The
    argument is passed verbatim to the wrapper, e.g.::

        scalismo_run('"runMain nako.ribs.RibRegistration --input ... --output ..."')

    Java resolution: ``JAVA_HOME`` (if it points at a directory) →
    ``~/jdk-17.0.13+11`` (fallback) → ``PATH``. ``sbt`` is resolved via
    :func:`shutil.which`, falling back to ``~/sbt/bin``. If neither yields
    an executable, a :class:`RuntimeError` is raised pointing at the README.
    """
    home      = str(Path.home())
    inherited_java_home = os.environ.get("JAVA_HOME", "")
    fallback_java_home  = f"{home}/jdk-17.0.13+11"

    if inherited_java_home and Path(inherited_java_home).is_dir():
        java_home: str | None = inherited_java_home
    elif Path(fallback_java_home).is_dir():
        java_home = fallback_java_home
    else:
        java_home = None

    # Minimal env: the parent's full environment can exceed execve's argv+envp
    # limit on some Linux installs.
    env = {
        "HOME": home,
        "USER": os.environ.get("USER", ""),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TERM": os.environ.get("TERM", "xterm"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }

    sbt_path = shutil.which("sbt", path=env["PATH"])
    fallback_sbt_bin = f"{home}/sbt/bin"
    if sbt_path is None and Path(fallback_sbt_bin, "sbt").is_file():
        env["PATH"] = f"{fallback_sbt_bin}:{env['PATH']}"
        sbt_path = f"{fallback_sbt_bin}/sbt"

    if java_home is not None:
        env["JAVA_HOME"] = java_home
        env["PATH"]      = f"{java_home}/bin:{env['PATH']}"

    if sbt_path is None:
        raise RuntimeError(
            "sbt not found.  Install sbt (≥ 1.9) and ensure it is on PATH, "
            "or place it at ~/sbt/bin/sbt.  See README.md → 'Environment "
            "setup → Scalismo' for details."
        )
    if java_home is None and shutil.which("java", path=env["PATH"]) is None:
        raise RuntimeError(
            "Java not found.  Install JDK ≥ 17 and either set JAVA_HOME, "
            "place the JDK at ~/jdk-17.0.13+11, or ensure 'java' is on PATH.  "
            "See README.md → 'Environment setup → Scalismo' for details."
        )

    bash = shutil.which("bash") or "/usr/bin/bash"

    if not _SBT_BUILD_SH.is_file():
        raise FileNotFoundError(f"sbt wrapper script not found: {_SBT_BUILD_SH}")

    cache_dir = Path.home() / ".cache" / "nako_ribs_scalismo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, script_path = tempfile.mkstemp(suffix=".sh", dir=str(cache_dir))
    os.close(fd)
    script = Path(script_path)
    script.write_text(f"#!{bash}\n{bash} {_SBT_BUILD_SH} {sbt_cmd}\n")
    script.chmod(0o755)

    try:
        proc = subprocess.Popen(
            [bash, str(script)],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        # Raw stdout passthrough preserves sbt's \r-based progress updates.
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        return proc.returncode
    finally:
        script.unlink(missing_ok=True)
