import argparse
import json
from pathlib import Path

import scanpy as sc

import b2g


def parse_args():
    parser = argparse.ArgumentParser(description="Run B2G auto mode with Harmony+ARMS selection.")
    parser.add_argument("--input", required=True, help="Input .h5ad file")
    parser.add_argument("--output", required=True, help="Output grouped .h5ad file")
    parser.add_argument("--batch-key", required=True)
    parser.add_argument("--reference-cluster-key", required=True, help="Reference biological label used by ARMS")
    parser.add_argument("--method", choices=["metacell", "leiden"], default="metacell")
    parser.add_argument("--additional-features", nargs="*", default=[])
    parser.add_argument("--target-metacell-size", type=int, default=48)
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--top-marker-num", type=int, default=50)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--harmony-max-iter", type=int, default=20)
    parser.add_argument("--group-key", default=None)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_json) if args.report_json else output_path.with_suffix(".auto_mode_report.json")

    adata = sc.read_h5ad(input_path)
    additional_features = [
        {"column": feature, "description": feature}
        for feature in args.additional_features
    ]

    evaluator = b2g.build_harmony_arms_evaluator(
        batch_key=args.batch_key,
        reference_cluster_key=args.reference_cluster_key,
        top_marker_num=args.top_marker_num,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.leiden_resolution,
        max_iter_harmony=args.harmony_max_iter,
    )

    grouped = b2g.group(
        adata,
        batch_key=args.batch_key,
        method=args.method,
        mode="auto",
        additional_features=additional_features,
        target_metacell_size=args.target_metacell_size,
        leiden_resolution=args.leiden_resolution,
        key_added=args.group_key,
        mode_evaluator=evaluator,
        copy=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.write_h5ad(output_path, compression="gzip")

    payload = dict(grouped.uns.get("b2g_mode_selection", {}))
    payload["selected_mode"] = grouped.uns.get("b2g_grouping_mode")
    payload["group_key"] = args.group_key or f"groups_{args.method}_adaptive"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved grouped data: {output_path}")
    print(f"Saved report: {report_path}")
    print(f"Selected mode: {payload.get('selected_mode')}")


if __name__ == "__main__":
    main()
