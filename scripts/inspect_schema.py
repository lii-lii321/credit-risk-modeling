"""一次性脚本：导出 credit-g 的列/类别 schema，供 data.py 合成降级使用。"""
import json
import sys

from sklearn.datasets import fetch_openml


def main() -> None:
    df, y = fetch_openml("credit-g", version=1, as_frame=True, return_X_y=True, parser="auto")
    schema = {}
    for col in df.columns:
        s = df[col]
        if str(s.dtype) == "category":
            schema[col] = {
                "kind": "categorical",
                "categories": sorted(str(c) for c in s.cat.categories),
            }
        else:
            schema[col] = {
                "kind": "numeric",
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
            }
    payload = {"target": {str(k): int(v) for k, v in y.value_counts().items()}, "features": schema}
    with open("data/credit_g_schema.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
