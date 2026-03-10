"""Offline ML model training runner for one or more IDX symbols."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MLConfig, settings
from core.data.database import DatabaseManager
from core.ml_prediction.features import preprocess_stock_data
from core.ml_prediction.trainer import save_artifacts, train_multiseed_ensemble

logger = logging.getLogger("scripts.train_models")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline ensemble models for IDX stocks.")
    parser.add_argument("--symbols", nargs="+", required=True, help="One or more stock symbols to train.")
    parser.add_argument("--lookback-days", type=int, default=400, help="Maximum trailing daily rows to use.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts.")
    parser.add_argument("--status-file", type=Path, required=True, help="Path to the JSON status file.")
    parser.add_argument("--history-file", type=Path, required=True, help="Path to the JSON history file.")
    parser.add_argument("--metadata-dir", type=Path, default=settings.get_data_path("ml_artifacts"))
    return parser.parse_args()


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            history = json.loads(path.read_text())
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    else:
        history = []
    history.append(payload)
    path.write_text(json.dumps(history[-50:], indent=2, sort_keys=True))


def fetch_symbol_data(db: DatabaseManager, symbol: str, lookback_days: int) -> pd.DataFrame:
    with db.get_session() as session:
        query = text(
            """
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE symbol = :symbol
            ORDER BY date DESC
            LIMIT :limit
            """
        )
        rows = session.execute(query, {"symbol": symbol, "limit": lookback_days}).fetchall()

    if not rows:
        raise ValueError(f"No historical data found for {symbol}")

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != "close"]


def metadata_path(artifacts_dir: Path, symbol: str) -> Path:
    return artifacts_dir / f"{symbol}_ensemble.meta.json"


def artifact_path(artifacts_dir: Path, symbol: str) -> Path:
    return artifacts_dir / f"{symbol}_ensemble.pkl"


def train_symbol(
    db: DatabaseManager,
    symbol: str,
    lookback_days: int,
    overwrite: bool,
    artifacts_dir: Path,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target_artifact = artifact_path(artifacts_dir, symbol)
    target_metadata = metadata_path(artifacts_dir, symbol)

    if target_artifact.exists() and not overwrite:
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": "artifact_exists",
            "artifact_path": str(target_artifact),
        }

    raw_df = fetch_symbol_data(db, symbol, lookback_days)
    processed_df = preprocess_stock_data(raw_df, exo_dict=None)
    row_count = len(processed_df)
    if row_count < MLConfig.MIN_TOTAL_ROWS:
        raise ValueError(
            f"Insufficient processed rows for {symbol}: {row_count} < {MLConfig.MIN_TOTAL_ROWS}"
        )

    features = build_feature_columns(processed_df)
    if not features:
        raise ValueError(f"No usable feature columns generated for {symbol}")

    artifacts = train_multiseed_ensemble(processed_df, features)
    trained_at = utc_now_iso()
    latest_source_date = processed_df.index.max()
    artifacts["version"] = "offline-batch-v1"
    artifacts["training_metadata"] = {
        "symbol": symbol,
        "trained_at": trained_at,
        "source_latest_date": latest_source_date.date().isoformat(),
        "source_row_count": int(row_count),
        "lookback_days": int(lookback_days),
        "overwrite": bool(overwrite),
    }
    save_artifacts(artifacts, str(target_artifact))

    meta = {
        "symbol": symbol,
        "artifact_path": str(target_artifact),
        "artifact_size_bytes": target_artifact.stat().st_size,
        "trained_at": trained_at,
        "source_latest_date": latest_source_date.date().isoformat(),
        "source_row_count": int(row_count),
        "lookback_days": int(lookback_days),
        "overwrite": bool(overwrite),
        "feature_count": len(features),
        "trained_horizon": int(artifacts.get("config", {}).get("horizon", 0) or 0),
        "artifact_type": artifacts.get("type", "single"),
        "status": "trained",
    }
    target_metadata.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    symbols = [symbol.strip().upper() for symbol in args.symbols if symbol.strip()]
    batch_limit = settings.model_training_batch_limit
    if not symbols:
        raise SystemExit("No symbols provided.")
    if len(symbols) > batch_limit:
        raise SystemExit(f"Batch limit exceeded: {len(symbols)} > {batch_limit}")

    db = DatabaseManager()
    status = {
        "job_id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "status": "running",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "symbols": symbols,
        "lookback_days": int(args.lookback_days),
        "overwrite": bool(args.overwrite),
        "batch_limit": batch_limit,
        "completed": [],
        "failed": [],
        "skipped": [],
        "current_symbol": None,
        "symbol_statuses": {symbol: {"status": "pending"} for symbol in symbols},
    }
    write_status(args.status_file, status)

    exit_code = 0
    try:
        for symbol in symbols:
            status["current_symbol"] = symbol
            status["symbol_statuses"][symbol] = {"status": "running", "started_at": utc_now_iso()}
            write_status(args.status_file, status)
            logger.info("Training symbol %s", symbol)
            try:
                result = train_symbol(
                    db=db,
                    symbol=symbol,
                    lookback_days=args.lookback_days,
                    overwrite=args.overwrite,
                    artifacts_dir=args.metadata_dir,
                )
                if result.get("status") == "skipped":
                    status["skipped"].append(result)
                    status["symbol_statuses"][symbol] = {
                        "status": "skipped",
                        "finished_at": utc_now_iso(),
                        "reason": result.get("reason"),
                    }
                    logger.info("Skipped %s because artifact already exists", symbol)
                else:
                    status["completed"].append(result)
                    status["symbol_statuses"][symbol] = {
                        "status": "trained",
                        "finished_at": utc_now_iso(),
                        "source_latest_date": result.get("source_latest_date"),
                    }
                    logger.info("Finished training %s", symbol)
            except Exception as exc:
                exit_code = 1
                failure = {"symbol": symbol, "error": str(exc)}
                status["failed"].append(failure)
                status["symbol_statuses"][symbol] = {
                    "status": "failed",
                    "finished_at": utc_now_iso(),
                    "error": str(exc),
                }
                logger.exception("Training failed for %s", symbol)
            finally:
                status["current_symbol"] = None
                write_status(args.status_file, status)
    finally:
        status["status"] = "completed" if not status["failed"] else (
            "completed_with_errors" if status["completed"] or status["skipped"] else "failed"
        )
        status["finished_at"] = utc_now_iso()
        write_status(args.status_file, status)
        append_history(
            args.history_file,
            {
                "job_id": status["job_id"],
                "status": status["status"],
                "started_at": status["started_at"],
                "finished_at": status["finished_at"],
                "symbols": status["symbols"],
                "lookback_days": status["lookback_days"],
                "overwrite": status["overwrite"],
                "completed_count": len(status["completed"]),
                "failed_count": len(status["failed"]),
                "skipped_count": len(status["skipped"]),
                "symbol_statuses": status["symbol_statuses"],
            },
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
