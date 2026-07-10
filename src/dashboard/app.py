"""Flask dashboard for GNN-based Network Intrusion Detection.

Multi-page dashboard displaying:
- Home: Project overview and dataset statistics
- Graph: Network graph visualization (subset)
- Results: Model comparison table and charts
- Confusion Matrices: Side-by-side for all models
- ROC Curves: Per-model ROC analysis
"""

import os
import sys
import pickle
import json

import numpy as np
from flask import Flask, render_template, send_from_directory

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PRED_DIR = os.path.join(PROJECT_ROOT, "predictions")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graph")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def get_dataset_stats():
    """Load dataset statistics for the home page."""
    stats = {}
    try:
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        feature_names = np.load(os.path.join(DATA_DIR, "feature_names.npy"),
                                allow_pickle=True).tolist()
        with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)

        y_all = np.concatenate([y_train, y_test])
        classes, counts = np.unique(y_all, return_counts=True)

        stats["total_samples"] = len(y_all)
        stats["train_samples"] = len(y_train)
        stats["test_samples"] = len(y_test)
        stats["num_features"] = X_train.shape[1]
        stats["num_classes"] = len(le.classes_)
        stats["class_names"] = list(le.classes_)
        stats["class_distribution"] = {
            le.classes_[c]: int(cnt) for c, cnt in zip(classes, counts)
        }
        stats["feature_names"] = feature_names
    except Exception as e:
        stats["error"] = str(e)

    return stats


def get_model_results():
    """Load model comparison results."""
    results = {}
    try:
        import pandas as pd
        csv_path = os.path.join(REPORT_DIR, "model_comparison.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            results["comparison_table"] = df.to_html(
                classes="table table-striped",
                float_format="%.4f",
            )
            results["models"] = df.index.tolist()
            results["metrics"] = df.to_dict("index")
    except Exception as e:
        results["error"] = str(e)

    return results


def get_graph_data():
    """Load a subset of graph data for visualization."""
    graph_info = {}
    try:
        ip_train = np.load(os.path.join(DATA_DIR, "ip_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)

        # Take a small random subset for visualization (max 500 edges)
        n = min(500, len(ip_train))
        rng = np.random.RandomState(42)
        indices = rng.choice(len(ip_train), n, replace=False)

        subset_ips = ip_train[indices]
        subset_labels = y_train[indices]

        # Build nodes and edges for vis.js
        unique_ips = np.unique(subset_ips.ravel())
        ip_to_id = {int(ip): idx for idx, ip in enumerate(unique_ips)}

        nodes = []
        for ip in unique_ips:
            nodes.append({"id": ip_to_id[int(ip)], "label": str(int(ip))[-6:]})

        edges = []
        label_colors = {
            0: "#4CAF50",  # BENIGN - green
            1: "#FF9800",  # class 1 - orange
            2: "#F44336",  # class 2 - red
            3: "#9C27B0",  # class 3 - purple
        }
        for i in range(n):
            src = ip_to_id[int(subset_ips[i, 0])]
            dst = ip_to_id[int(subset_ips[i, 1])]
            label = int(subset_labels[i])
            edges.append({
                "from": src,
                "to": dst,
                "color": label_colors.get(label, "#607D8B"),
                "title": le.classes_[label] if label < len(le.classes_) else str(label),
            })

        graph_info["nodes"] = json.dumps(nodes)
        graph_info["edges"] = json.dumps(edges)
        graph_info["num_nodes"] = len(nodes)
        graph_info["num_edges"] = len(edges)
        graph_info["class_names"] = list(le.classes_)

    except Exception as e:
        graph_info["error"] = str(e)

    return graph_info


def get_available_plots():
    """Check which report plots exist."""
    plots = {}
    if os.path.isdir(REPORT_DIR):
        for fname in os.listdir(REPORT_DIR):
            if fname.endswith(".png"):
                plots[fname] = fname
    return plots


def create_app():
    """Create and configure the Flask application."""
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)

    @app.route("/")
    def home():
        stats = get_dataset_stats()
        return render_template("home.html", stats=stats)

    @app.route("/graph")
    def graph():
        graph_data = get_graph_data()
        return render_template("graph.html", graph=graph_data)

    @app.route("/results")
    def results():
        model_results = get_model_results()
        plots = get_available_plots()
        return render_template("results.html",
                               results=model_results, plots=plots)

    @app.route("/confusion")
    def confusion():
        plots = get_available_plots()
        cm_plots = {k: v for k, v in plots.items() if k.startswith("cm_")}
        return render_template("confusion.html", plots=cm_plots)

    @app.route("/roc")
    def roc():
        plots = get_available_plots()
        roc_plots = {k: v for k, v in plots.items() if k.startswith("roc_")}
        return render_template("roc.html", plots=roc_plots)

    @app.route("/reports/<path:filename>")
    def serve_report(filename):
        return send_from_directory(REPORT_DIR, filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
