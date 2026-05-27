"""Main entry point for P&ID Symbol Detection System."""

import sys
from pathlib import Path

_SRC = str(Path(__file__).parent.resolve())
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import logging as _logging
import sys as _sys

from pipeline.stage1_class_agnostic_train import Stage1ClassAgnosticPipeline
from pipeline.stage1_class_agnostic_inference import Stage1InferencePipeline
from pipeline.stage2_baseline_train import Stage2BaselinePipeline
from pipeline.stage2_svg_structure_train import Stage2SVGStructurePipeline
from pipeline.stage2_inference import Stage2InferencePipeline
from pipeline.evaluation import EvaluationPipeline


def _setup_logger(level: str = "INFO") -> None:
    logger = _logging.getLogger()
    logger.setLevel(getattr(_logging, level))
    logger.handlers.clear()
    handler = _logging.StreamHandler(_sys.stdout)
    handler.setLevel(getattr(_logging, level))
    handler.setFormatter(_logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser(description="P&ID Symbol Detection System")
    parser.add_argument("--config", type=str,
                        default="src/pipeline/configs/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Stage 1 ──────────────────────────────────────────────────────
    p1 = subparsers.add_parser("stage1", help="Stage 1: data + train + inference")
    p1.add_argument("--prepare_data", action="store_true",
                    help="Prepare YOLO data")
    p1.add_argument("--train_model", action="store_true",
                    help="Train Stage 1 model")
    p1.add_argument("--inference", action="store_true",
                    help="Run Stage 1 inference")

    p1_inf = subparsers.add_parser("stage1_inference",
                                     help="Stage 1 inference only")

    # ── Stage 2 Training ─────────────────────────────────────────────
    # SVG-guided training（消融实验：+SVG引导）
    p2_svg = subparsers.add_parser("stage2_svg_guided",
                                     help="Stage 2: SVG-Guided training")
    p2_svg.add_argument("--train_model", action="store_true",
                         help="Train SVG-guided model")
    p2_svg.add_argument("--resume", action="store_true",
                         help="Resume from latest checkpoint")

    # Baseline training
    p2_bl = subparsers.add_parser("stage2_baseline",
                                  help="Stage 2: Baseline Triplet training (no SVG)")
    p2_bl.add_argument("--train_model", action="store_true",
                        help="Train baseline model")
    p2_bl.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")

    # Exp-A: ProtoBank + DomainAdapt + L_domain
    p2_expA = subparsers.add_parser("stage2_expA",
                                    help="Stage 2: Exp-A ProtoBank + DomainAdapt + L_domain")
    p2_expA.add_argument("--train_model", action="store_true",
                         help="Train Exp-A model")
    p2_expA.add_argument("--resume", action="store_true",
                         help="Resume from latest checkpoint")

    # SVG-Structure training
    p2_struct = subparsers.add_parser("stage2_svg_structure",
                                   help="Stage 2: SVG-Guided Structure Enhancement training")
    p2_struct.add_argument("--train_model", action="store_true",
                          help="Train SVG-Structure model")
    p2_struct.add_argument("--resume", action="store_true",
                          help="Resume from latest checkpoint")

    # ── Stage 2 Inference ────────────────────────────────────────────
    p2_inf_bl = subparsers.add_parser("stage2_inference",
                                       help="Stage 2: Baseline inference")
    # rejection_mode / energy_threshold / distance_threshold 等已在 config.yaml 定义
    # 若需覆盖，可通过 --overrides '{"rejection_mode": "..."}' 传入

    # ── Evaluation ────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("evaluation", help="Evaluate results")

    # 评估任务开关（可组合，也可单独使用）
    p_eval.add_argument("--all", action="store_true",
                        help="Run all evaluations (stage1 + stage2 + openset + embedding)")
    p_eval.add_argument("--stage1", action="store_true",
                        help="Evaluate Stage 1 (class-agnostic detection)")
    p_eval.add_argument("--stage2", action="store_true",
                        help="Evaluate Stage 2 (class-aware classification)")
    p_eval.add_argument("--openset", action="store_true",
                        help="Evaluate open-set metrics: Macro-F1, FRR, URR, PS, RS, PR, RR, Acc")
    p_eval.add_argument("--embedding", action="store_true",
                        help="Evaluate Embedding Space Quality metrics")

    # Evaluation
    p_eval.add_argument("--detailed_json", type=str, default=None,
                        help="Path to predictions JSON for openset evaluation")
    p_eval.add_argument("--gt_dir", type=str, default=None,
                        help="GT labels directory (default: data/inference/stage2/labels)")
    p_eval.add_argument("--source", type=str, default=None, choices=["baseline"],
                        help="Inference source: baseline (default)")

    return parser.parse_args()


def main():
    args = parse_args()
    _setup_logger(args.log_level)

    if args.command == "stage1":
        pipeline = Stage1ClassAgnosticPipeline(config_path=args.config)
        if args.prepare_data:
            pipeline.prepare_data()
        elif args.train_model:
            pipeline.train_model()
        elif args.inference:
            inf = Stage1InferencePipeline(config_path=args.config)
            inf.run()
        else:
            pipeline.prepare_data()
            pipeline.train_model()
            inf = Stage1InferencePipeline(config_path=args.config)
            inf.run()

    elif args.command == "stage1_inference":
        pipeline = Stage1InferencePipeline(config_path=args.config)
        pipeline.run()

    elif args.command == "stage2_svg_guided":
        print("Error: stage2_svg_guided pipeline is not available (module not found)")
        return

    elif args.command == "stage2_baseline":
        pipeline = Stage2BaselinePipeline(config_path=args.config)
        pipeline.train_model(resume=args.resume)

    elif args.command == "stage2_expA":
        print("Error: stage2_expA pipeline is not available (module not found)")
        return

    elif args.command == "stage2_svg_structure":
        pipeline = Stage2SVGStructurePipeline(config_path=args.config)
        pipeline.train_model(resume=args.resume)

    elif args.command == "stage2_inference":
        pipeline = Stage2InferencePipeline(config_path=args.config)
        pipeline.run()

    elif args.command == "evaluation":
        pipeline = EvaluationPipeline(config_path=args.config)

        gt_dir = Path(args.gt_dir) if args.gt_dir else None
        detailed_json = Path(args.detailed_json) if args.detailed_json else None

        if args.all:
            pipeline.run()
        elif args.openset:
            m = pipeline.compute_openset_metrics(source=args.source, gt_dir=gt_dir,
                                                  detailed_json=detailed_json)
            pipeline.save_metrics(m, "openset")
        elif args.embedding:
            m = pipeline.compute_embedding_quality(detailed_json=detailed_json)
            pipeline.save_metrics(m, "embedding_quality")
        elif args.stage1:
            m = pipeline.compute_stage1_metrics()
            pipeline.save_metrics(m, "stage1")
        elif args.stage2:
            m = pipeline.compute_stage2_metrics()
            pipeline.save_metrics(m, "stage2")
        else:
            pipeline.run()

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
