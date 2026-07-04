"""Content-Based Image Retrieval (CBIR) system.

A small, self-contained retrieval system for image bounding-box crops:
index embeddings into Milvus, then explore the vector space visually
(PCA 2D/3D) and check retrieval quality by uploading a query image.

The package is standalone. It draws inspiration from the frozen
``archive/mvp/`` proof-of-concept but does not import from it.
"""

__version__ = "0.1.0"
