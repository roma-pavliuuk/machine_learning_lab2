import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "config" / "config_lr.yaml", ROOT / "config" / "config_rf.yaml"]
BLOCK = re.compile(r"(?P<indent>[ \t]*)# BEGIN DFS[^\n]*\n.*?(?P=indent)# END DFS", flags=re.DOTALL)


def main() -> None:
    lineage = json.loads((ROOT / "data" / "processed" / "dfs_lineage.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in lineage]
    for path in CONFIGS:
        text = path.read_text(encoding="utf-8")
        match = BLOCK.search(text)
        if not match:
            raise SystemExit(f"{path.name}: немає маркерів BEGIN/END DFS")
        indent = match.group("indent")
        body = "".join(f"{indent}- {n}\n" for n in names)
        new_block = f"{indent}# BEGIN DFS ({len(names)} ознак, згенеровано)\n{body}{indent}# END DFS"
        path.write_text(text[:match.start()] + new_block + text[match.end():], encoding="utf-8")
        print(f"{path.name}: {len(names)} ознак DFS")


if __name__ == "__main__":
    main()
