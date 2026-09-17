"""SQLite persistent intervention memory for PulseGrid AI preventive interventions."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_NAME = "interventions.db"
DB_PATH = Path(__file__).parent / DB_NAME


def get_connection() -> sqlite3.Connection:
    """Create a SQLite database connection with row factory for dictionary-style mapping."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize interventions.db with table intervention_memory and seed mock data if empty."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS intervention_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anomaly_type TEXT,
                action_taken TEXT,
                outcome_score REAL,
                timestamp DATETIME
            );
            """
        )

        # Mock data seeding logic if database table is empty
        cursor.execute("SELECT COUNT(*) AS count FROM intervention_memory;")
        row = cursor.fetchone()
        if row and row["count"] == 0:
            seed_data = [
                (
                    "voltage_sag",
                    "Switch Capacitor Bank 4 at Substation-7 for +25 MVAR reactive compensation",
                    0.94,
                    "2026-03-12 14:22:10",
                ),
                (
                    "voltage_sag",
                    "Automatic Tap Changer adjustment (LTC +2 steps) on Transformer 3",
                    0.88,
                    "2026-04-05 09:14:02",
                ),
                (
                    "voltage_sag",
                    "Dispatch D-STATCOM reactive boost of 15 MVAR on Feeder 12",
                    0.91,
                    "2026-04-20 18:30:15",
                ),
                (
                    "thermal_overload",
                    "Distributed BESS discharge (15MW / 30MWh) across Feeder 104",
                    0.92,
                    "2026-05-18 17:45:33",
                ),
                (
                    "thermal_overload",
                    "Automated Dynamic Line Rating (DLR) reroute to Sub-Loop B",
                    0.85,
                    "2026-06-22 11:30:19",
                ),
                (
                    "frequency_decay",
                    "Fast Frequency Response (FFR) governor boost + 8MW demand response call",
                    0.96,
                    "2026-08-09 20:05:44",
                ),
                (
                    "frequency_decay",
                    "Synchronous condenser inertia ramp-up at North Substation",
                    0.89,
                    "2026-08-25 16:12:08",
                ),
                (
                    "transformer_overheat",
                    "Force-activate auxiliary fan cooling bank and shed 5MW non-critical industrial load",
                    0.82,
                    "2026-09-02 13:40:22",
                ),
                (
                    "reactive_power_deficit",
                    "Energize Substation-12 shunt capacitor bank to raise bus power factor to 0.96",
                    0.95,
                    "2026-09-10 11:15:00",
                ),
            ]
            cursor.executemany(
                """
                INSERT INTO intervention_memory (anomaly_type, action_taken, outcome_score, timestamp)
                VALUES (?, ?, ?, ?);
                """,
                seed_data,
            )
        conn.commit()


def save_intervention(anomaly_type: str, action_taken: str, outcome_score: float) -> int:
    """Insert a new intervention record into intervention_memory."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO intervention_memory (anomaly_type, action_taken, outcome_score, timestamp)
            VALUES (?, ?, ?, ?);
            """,
            (anomaly_type, action_taken, float(outcome_score), current_time),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_past_interventions(anomaly_type: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Fetch previous interventions with outcome_score >= 0.7 for contextual memory retrieval."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, anomaly_type, action_taken, outcome_score, timestamp
            FROM intervention_memory
            WHERE anomaly_type = ? AND outcome_score >= 0.7
            ORDER BY outcome_score DESC, timestamp DESC
            LIMIT ?;
            """,
            (anomaly_type, limit),
        )
        rows = cursor.fetchall()

        # Fallback if no exact anomaly_type match is found
        if not rows:
            cursor.execute(
                """
                SELECT id, anomaly_type, action_taken, outcome_score, timestamp
                FROM intervention_memory
                WHERE outcome_score >= 0.7
                ORDER BY outcome_score DESC, timestamp DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print("interventions.db initialized successfully.")
    sample = get_past_interventions("voltage_sag", limit=3)
    print(f"Retrieved {len(sample)} past interventions for 'voltage_sag' (outcome >= 0.7):")
    for item in sample:
        print(f"  - [{item['outcome_score']:.2f}] {item['action_taken']} ({item['timestamp']})")
