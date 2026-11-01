"""SQLite persistence for the maintenance workflow: fleet registry, per-flight predictions, alerts, maintenance events.

Standard-library sqlite3 only. The model never learns from its own predictions: what feeds the learning layer
(pipeline layer 6) is the engineer's recorded findings in `maintenance_events`, joined with the prediction that was
shown at that flight (`export_events`).
"""
from __future__ import annotations

import csv
import datetime
import io
import os
import sqlite3
import threading

import pandas as pd

from . import ncmapss as nc
from . import service

EVENT_TYPES = ["borescope", "compressor_wash", "component_replacement", "sensor_check", "vibration_check", "other"]
FINDINGS = ["confirmed", "not_found", "inconclusive"]       # did the inspection confirm what the model flagged?
ALERT_STATUS = ["open", "acknowledged", "dismissed"]

SCHEMA_VERSION = 3          # bump when a table changes; derived tables are rebuilt, engineer records are kept
DERIVED_TABLES = ("engines", "flight_predictions")      # recomputed from the model bundle on every start

SCHEMA = """
CREATE TABLE IF NOT EXISTS engines (
    engine_id TEXT PRIMARY KEY, source TEXT NOT NULL, dataset TEXT, unit_local INTEGER,
    n_cycles INTEGER, last_cycle INTEGER, rul REAL, rul_q05 REAL, rul_q95 REAL, planning_rul REAL,
    band INTEGER, band_name TEXT, urgency INTEGER, urgency_name TEXT, urgency_driver TEXT, status TEXT, worst_component TEXT, localisation TEXT, localisation_reliable INTEGER,
    ohi REAL, model_version TEXT, updated_utc TEXT
);
CREATE TABLE IF NOT EXISTS flight_predictions (
    engine_id TEXT NOT NULL, cycle INTEGER NOT NULL, rul REAL, rul_q05 REAL, rul_q95 REAL, ohi REAL,
    hi_fan REAL, hi_lpc REAL, hi_hpc REAL, hi_hpt REAL, hi_lpt REAL, hi_burner REAL, hi_nozzle REAL, model_version TEXT,
    PRIMARY KEY (engine_id, cycle)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, engine_id TEXT NOT NULL, cycle INTEGER NOT NULL, type TEXT NOT NULL,
    component TEXT NOT NULL DEFAULT '', severity TEXT NOT NULL, message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', handled_by TEXT, handled_utc TEXT, note TEXT,
    UNIQUE (engine_id, cycle, type, component, message)
);
CREATE TABLE IF NOT EXISTS maintenance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, engine_id TEXT NOT NULL, cycle INTEGER, component TEXT,
    event_type TEXT NOT NULL, finding TEXT NOT NULL, action TEXT, observed_health REAL, alert_id INTEGER,
    performed_by TEXT NOT NULL, performed_utc TEXT NOT NULL, notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_engine ON alerts (engine_id, status);
CREATE INDEX IF NOT EXISTS ix_events_engine ON maintenance_events (engine_id);
"""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            if self.db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                for t in DERIVED_TABLES:                    # alerts / maintenance_events hold engineer input: keep
                    self.db.execute(f"DROP TABLE IF EXISTS {t}")
                self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.db.executescript(SCHEMA)
            self.db.commit()

    def _all(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def _one(self, sql: str, args=()) -> dict | None:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- sync from the pipeline
    def sync_engine(self, bundle, e, engine_id: str, source: str) -> None:
        """Idempotently store one engine's flights, alerts and latest summary. `e` = pipeline output frame."""
        e = e.sort_values("cycle").reset_index(drop=True)
        edges = bundle.manifest.get("band_edges", [0.75, 0.55, 0.30])
        advice = service.advisory(bundle, e, edges)
        ver = service.status_fields(bundle, e, edges)["model_version"]
        last = e.iloc[-1]
        q05 = e["rul_q05"] if "rul_q05" in e else e["rul"]
        q95 = e["rul_q95"] if "rul_q95" in e else e["rul"]
        his = {c: (e[f"hi_{c}"] if f"hi_{c}" in e else pd.Series(1.0, index=e.index)) for c in service.ALL_COMPONENTS}
        ohi = pd.concat(list(his.values()), axis=1).min(axis=1)
        rows = [(engine_id, int(cy), float(r), float(a), float(b), float(o), *(float(his[c].iloc[i]) for c in service.ALL_COMPONENTS), ver)
                for i, (cy, r, a, b, o) in enumerate(zip(e["cycle"], e["rul"], q05, q95, ohi))]
        alerts = [(engine_id, a["cycle"], a["type"], a.get("component", ""), a["severity"], a["message"])
                  for a in service.derive_alerts(e, edges)]
        with self.lock:
            self.db.execute("DELETE FROM flight_predictions WHERE engine_id = ?", (engine_id,))
            self.db.executemany("INSERT INTO flight_predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.db.executemany("INSERT OR IGNORE INTO alerts (engine_id, cycle, type, component, severity, message) "
                                "VALUES (?,?,?,?,?,?)", alerts)
            self.db.execute(
                "INSERT OR REPLACE INTO engines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (engine_id, source, str(e["dataset"].iloc[0]) if "dataset" in e else None,
                 int(e["unit_local"].iloc[0]) if "unit_local" in e else None, int(len(e)), int(last["cycle"]),
                 float(last["rul"]), float(q05.iloc[-1]), float(q95.iloc[-1]), advice["planning_rul"],
                 advice["band"], advice["band_name"], advice["urgency"], advice["urgency_name"], advice["urgency_driver"],
                 advice["status"], advice["worst_component"],
                 advice["localisation"], int(advice["localisation_reliable"]), float(ohi.iloc[-1]), ver, _now()))
            self.db.commit()

    def drop_source(self, source: str) -> int:
        """Remove engines of one source from the fleet (uploads live in process memory, so they do not survive a
        restart). Alerts and maintenance events are engineer records and are kept."""
        with self.lock:
            ids = [r[0] for r in self.db.execute("SELECT engine_id FROM engines WHERE source = ?", (source,))]
            self.db.executemany("DELETE FROM flight_predictions WHERE engine_id = ?", [(i,) for i in ids])
            self.db.execute("DELETE FROM engines WHERE source = ?", (source,))
            self.db.commit()
        return len(ids)

    # ---------------------------------------------------------------- fleet triage
    def fleet(self, as_of: float = 1.0, reliability: dict | None = None, edges=(0.75, 0.55, 0.30)) -> list[dict]:
        """Ranked fleet: urgency (worse of health band and planning-RUL level), then planning RUL. as_of < 1 replays the fleet at that share of each engine's recorded flights (the demo engines
        are run-to-failure histories, so their final flight is always end of life)."""
        rows = self._all(
            "SELECT e.*, (SELECT COUNT(*) FROM alerts a WHERE a.engine_id = e.engine_id AND a.status = 'open' "
            "AND a.cycle <= :c) AS open_alerts, "
            "(SELECT MAX(performed_utc) FROM maintenance_events m WHERE m.engine_id = e.engine_id) AS last_maintenance_utc, "
            "(SELECT COUNT(*) FROM maintenance_events m WHERE m.engine_id = e.engine_id) AS n_events FROM engines e",
            {"c": 10 ** 9})
        for r in rows:
            if as_of < 1.0:
                n = max(1, round(as_of * r["n_cycles"]))
                p = self._one("SELECT * FROM flight_predictions WHERE engine_id = ? ORDER BY cycle LIMIT 1 OFFSET ?",
                              (r["engine_id"], n - 1))
                his = {c: p[f"hi_{c}"] for c in service.ALL_COMPONENTS}
                r.update(service.triage_from_values(his, p["rul"], p["rul_q05"], reliability, edges))
                r.update(last_cycle=p["cycle"], rul_q05=p["rul_q05"], rul_q95=p["rul_q95"],
                         open_alerts=self._one("SELECT COUNT(*) AS n FROM alerts WHERE engine_id = ? AND status = 'open' "
                                               "AND cycle <= ?", (r["engine_id"], p["cycle"]))["n"])
            r["localisation_reliable"] = bool(r["localisation_reliable"])
        rows.sort(key=lambda r: (-r["urgency"], r["planning_rul"], r["engine_id"]))
        for i, r in enumerate(rows):
            r["priority"] = i + 1
        return rows

    # ---------------------------------------------------------------- alerts
    def alerts(self, engine_id: str, status: str | None = None) -> list[dict]:
        if status:
            return self._all("SELECT * FROM alerts WHERE engine_id = ? AND status = ? ORDER BY cycle, id", (engine_id, status))
        return self._all("SELECT * FROM alerts WHERE engine_id = ? ORDER BY cycle, id", (engine_id,))

    def handle_alert(self, alert_id: int, status: str, user: str, note: str | None) -> dict | None:
        if status not in ALERT_STATUS:
            raise ValueError(f"status must be one of {ALERT_STATUS}")
        with self.lock:
            cur = self.db.execute("UPDATE alerts SET status = ?, handled_by = ?, handled_utc = ?, note = ? WHERE id = ?",
                                  (status, user, _now(), note, alert_id))
            self.db.commit()
        return self._one("SELECT * FROM alerts WHERE id = ?", (alert_id,)) if cur.rowcount else None

    # ---------------------------------------------------------------- maintenance events
    def add_event(self, engine_id: str, ev: dict) -> dict:
        if not self._one("SELECT 1 AS x FROM engines WHERE engine_id = ?", (engine_id,)):
            raise KeyError(engine_id)
        if ev["event_type"] not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {EVENT_TYPES}")
        if ev["finding"] not in FINDINGS:
            raise ValueError(f"finding must be one of {FINDINGS}")
        if ev.get("component") and ev["component"] not in service.ALL_COMPONENTS:
            raise ValueError(f"component must be one of {service.ALL_COMPONENTS}")
        h = ev.get("observed_health")
        if h is not None and not 0.0 <= float(h) <= 1.0:
            raise ValueError("observed_health must be within [0, 1]")
        if not (ev.get("performed_by") or "").strip():
            raise ValueError("performed_by is required (audit trail)")
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO maintenance_events (engine_id, cycle, component, event_type, finding, action, observed_health, "
                "alert_id, performed_by, performed_utc, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (engine_id, ev.get("cycle"), ev.get("component") or None, ev["event_type"], ev["finding"], ev.get("action"),
                 h, ev.get("alert_id"), ev["performed_by"].strip(), _now(), ev.get("notes")))
            if ev.get("alert_id"):                       # an inspection against an alert acknowledges it
                self.db.execute("UPDATE alerts SET status = 'acknowledged', handled_by = ?, handled_utc = ? "
                                "WHERE id = ? AND status = 'open'", (ev["performed_by"].strip(), _now(), ev["alert_id"]))
            self.db.commit()
            return self._one("SELECT * FROM maintenance_events WHERE id = ?", (cur.lastrowid,))

    def events(self, engine_id: str | None = None) -> list[dict]:
        if engine_id:
            return self._all("SELECT * FROM maintenance_events WHERE engine_id = ? ORDER BY performed_utc DESC, id DESC",
                             (engine_id,))
        return self._all("SELECT * FROM maintenance_events ORDER BY performed_utc DESC, id DESC")

    # ---------------------------------------------------------------- learning layer input
    def export_events(self) -> list[dict]:
        """Engineer findings joined with the prediction shown at that flight (labelled data for recalibration)."""
        return self._all(
            "SELECT m.id AS event_id, m.engine_id, m.cycle, m.component, m.event_type, m.finding, m.observed_health, "
            "m.performed_by, m.performed_utc, p.rul AS predicted_rul, p.rul_q05, p.rul_q95, p.ohi AS predicted_ohi, "
            "p.hi_fan, p.hi_lpc, p.hi_hpc, p.hi_hpt, p.hi_lpt, p.hi_burner, p.hi_nozzle, p.model_version "
            "FROM maintenance_events m LEFT JOIN flight_predictions p ON p.engine_id = m.engine_id AND p.cycle = m.cycle "
            "ORDER BY m.id")

    def learning_summary(self) -> dict:
        ev = self.export_events()
        by = {f: sum(1 for r in ev if r["finding"] == f) for f in FINDINGS}
        decided = by["confirmed"] + by["not_found"]
        err = []
        for r in ev:
            h, c = r["observed_health"], r["component"]
            if h is not None and c and r.get(f"hi_{c}") is not None:
                err.append(r[f"hi_{c}"] - h)
        return {"events": len(ev), "by_finding": by,
                "flag_precision": (by["confirmed"] / decided) if decided else None,
                "health_estimate_mae": (sum(abs(x) for x in err) / len(err)) if err else None,
                "health_estimate_bias": (sum(err) / len(err)) if err else None,
                "n_health_checks": len(err),
                "note": "Findings are stored for offline recalibration; the deployed model is never updated automatically."}

    def export_csv(self) -> str:
        rows = self.export_events()
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["event_id"])
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()
