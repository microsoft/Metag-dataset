"""Flask app that serves the browser-based diff annotation UI."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from .session import AnnotationSession


def create_app(session: AnnotationSession) -> Flask:
    app = Flask(__name__)
    app.config["SESSION"] = session

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def get_state():
        return jsonify(session.state())

    @app.get("/api/diff")
    def get_diff():
        status, diff, error = session.diff_payload()
        if status == "ready":
            return jsonify({"status": status, "diff": diff})
        if status == "error":
            return jsonify({"status": status, "error": error}), 500
        return jsonify({"status": status}), 202

    @app.get("/api/page/<side>/<int:page_number>")
    def get_page(side: str, page_number: int):
        if side not in ("left", "right"):
            return jsonify({"error": "unknown side"}), 404
        try:
            scale = float(request.args.get("scale", 2.0))
        except ValueError:
            scale = 2.0
        try:
            data = session.render_page(side, page_number, scale)
        except LookupError:
            return jsonify({"error": "no paper loaded"}), 409
        except (IndexError, ValueError):
            return jsonify({"error": "page out of range"}), 404
        return Response(data, mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/save")
    def save_entry():
        payload = request.get_json(silent=True) or {}
        group_ids = payload.get("group_ids") or []
        if not isinstance(group_ids, list):
            return jsonify({"error": "group_ids must be a list"}), 400
        group_ids = [str(gid) for gid in group_ids]
        return jsonify(session.save_current(group_ids))

    @app.post("/api/skip")
    def skip_entry():
        return jsonify(session.skip_current())

    return app


def build_session(input_jsonl: Path, pdf_dir: Path, output_path: Path, cache_dir: Path) -> AnnotationSession:
    return AnnotationSession(input_jsonl, pdf_dir, output_path, cache_dir)
