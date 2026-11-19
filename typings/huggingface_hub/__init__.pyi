from pathlib import Path

def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> str: ...
def snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    allow_patterns: list[str] | str | None = None,
) -> str: ...
