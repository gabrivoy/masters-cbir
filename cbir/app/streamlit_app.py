"""CBIR vector-space explorer (Streamlit frontend).

The frontend does three things:

1. shows the indexed gallery as a PCA scatter (2D or 3D), coloured by class;
2. lets you upload a query image, projects it into the *same* PCA space, and
   highlights it together with its nearest neighbours;
3. reports which class a KNN vote over those neighbours would assign, with a
   confidence bar — the retrieval-based auto-labeling idea, made visual.

It never touches Milvus or torch directly: everything goes through the API,
which guarantees the query is embedded with the same model as the collection.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cbir.app.client import APIError, CBIRClient

# A qualitative palette that stays legible in Streamlit's dark theme.
CLASS_PALETTE = px.colors.qualitative.Set2
QUERY_COLOR = "#ef4444"


def _parse_api_url() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8100")
    # Streamlit passes its own args; parse only ours and ignore the rest.
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.api_url


def _color_map(classes: list[str]) -> dict[str, str]:
    ordered = sorted(set(classes))
    return {cls: CLASS_PALETTE[i % len(CLASS_PALETTE)] for i, cls in enumerate(ordered)}


def _scatter(points, query_coords, hit_ids, n_components, color_map):
    """Build a Plotly scatter of the gallery plus the optional query point."""
    xs = np.array([p.coords[0] for p in points])
    ys = np.array([p.coords[1] for p in points])
    zs = np.array([p.coords[2] if len(p.coords) > 2 else 0.0 for p in points])
    classes = [p.target_class for p in points]
    hit_set = set(hit_ids or [])
    # Neighbours get a thick ring so they stand out against the cloud.
    line_widths = [3 if p.item_id in hit_set else 0 for p in points]
    hover = [f"{p.target_class}<br>{p.item_id}<br>{p.camera_id}" for p in points]

    if n_components >= 3:
        fig = go.Figure()
        for cls in sorted(set(classes)):
            idx = [i for i, c in enumerate(classes) if c == cls]
            fig.add_trace(
                go.Scatter3d(
                    x=xs[idx], y=ys[idx], z=zs[idx],
                    mode="markers", name=cls,
                    marker=dict(
                        size=4, color=color_map[cls],
                        line=dict(width=[line_widths[i] for i in idx], color="#111"),
                    ),
                    text=[hover[i] for i in idx], hoverinfo="text",
                )
            )
        if query_coords is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[query_coords[0]], y=[query_coords[1]],
                    z=[query_coords[2] if len(query_coords) > 2 else 0.0],
                    mode="markers", name="query",
                    marker=dict(size=10, color=QUERY_COLOR, symbol="diamond"),
                )
            )
        fig.update_layout(scene=dict(aspectmode="cube"))
    else:
        fig = go.Figure()
        for cls in sorted(set(classes)):
            idx = [i for i, c in enumerate(classes) if c == cls]
            fig.add_trace(
                go.Scatter(
                    x=xs[idx], y=ys[idx], mode="markers", name=cls,
                    marker=dict(
                        size=9, color=color_map[cls],
                        line=dict(width=[line_widths[i] for i in idx], color="#111"),
                    ),
                    text=[hover[i] for i in idx], hoverinfo="text",
                )
            )
        if query_coords is not None:
            fig.add_trace(
                go.Scatter(
                    x=[query_coords[0]], y=[query_coords[1]], mode="markers", name="query",
                    marker=dict(size=18, color=QUERY_COLOR, symbol="diamond",
                                line=dict(width=2, color="#fff")),
                )
            )
    fig.update_layout(
        height=620, legend_title_text="class",
        margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="CBIR Vector-Space Explorer", layout="wide")
    api_url = _parse_api_url()
    client = CBIRClient(api_url)

    st.title("CBIR — Vector-Space Explorer")
    st.caption(
        "Explore vessel-crop embeddings, drop in a query image, and see which "
        "class the retrieval neighbours would assign it to."
    )

    if not client.health():
        st.error(f"API not reachable at {api_url}. Start it with `cbir api`.")
        st.stop()

    collections = client.collections()
    if not collections:
        st.warning("No collections indexed yet. Run `cbir index` first.")
        st.stop()

    # --- sidebar controls -------------------------------------------------
    with st.sidebar:
        st.header("Controls")
        labels = {f"{c.name}  ·  {c.count} items  ·  {c.model_name or '?'}": c for c in collections}
        chosen_label = st.selectbox("Collection", list(labels))
        collection = labels[chosen_label]
        st.info(f"**Embedding model:** `{collection.model_name or 'unknown'}`\n\n"
                "The query is embedded with this same model — guaranteed by the API.")
        dims = st.radio("Projection", ["3D", "2D"], horizontal=True)
        n_components = 3 if dims == "3D" else 2
        top_k = st.slider("Neighbours (k)", min_value=1, max_value=30, value=10)
        weighted = st.checkbox("Similarity-weighted vote", value=True)

    # --- gallery projection ----------------------------------------------
    try:
        projection = client.project(collection.name, n_components=n_components)
    except APIError as exc:
        st.error(f"Projection failed: {exc}")
        st.stop()

    color_map = _color_map([p.target_class for p in projection.points])
    var = projection.cumulative_variance
    st.caption(
        f"PCA over {len(projection.points)} items · "
        f"cumulative explained variance: **{var:.1%}** "
        f"({', '.join(f'{v:.1%}' for v in projection.explained_variance_ratio)})"
    )

    # --- query panel ------------------------------------------------------
    upload = st.file_uploader("Upload a query image (a vessel crop)", type=["png", "jpg", "jpeg"])
    query_coords = None
    result = None
    if upload is not None:
        try:
            result = client.query(
                collection.name, upload.getvalue(), upload.name,
                top_k=top_k, n_components=n_components, weighted=weighted,
            )
            query_coords = result.query_coords
        except APIError as exc:
            st.error(f"Query failed: {exc}")

    plot_col, side_col = st.columns([3, 1])
    with plot_col:
        hit_ids = [h.item_id for h in result.hits] if result else []
        fig = _scatter(projection.points, query_coords, hit_ids, n_components, color_map)
        st.plotly_chart(fig, use_container_width=True)

    with side_col:
        if upload is not None:
            st.image(upload.getvalue(), caption="query", use_container_width=True)
        if result is not None:
            pred = result.prediction
            st.subheader("KNN prediction")
            if pred.predicted_class:
                st.metric(
                    "Would be labeled",
                    pred.predicted_class,
                    f"{pred.confidence:.0%} confidence",
                )
                st.progress(min(1.0, pred.confidence))
                for vote in pred.votes:
                    st.write(
                        f"`{vote.target_class}` — {vote.count} votes, weight {vote.weight:.2f}"
                    )
            else:
                st.write("No neighbours to vote with.")

    # --- neighbour crops --------------------------------------------------
    if result is not None and result.hits:
        st.subheader(f"Top-{len(result.hits)} nearest neighbours")
        cols = st.columns(min(6, len(result.hits)))
        for i, hit in enumerate(result.hits):
            with cols[i % len(cols)]:
                if hit.image_path:
                    st.image(client.crop_url(hit.image_path), use_container_width=True)
                st.caption(f"#{hit.rank} · {hit.target_class}\nsim={hit.score:.3f}")


if __name__ == "__main__":
    main()
