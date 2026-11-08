"""Run the data pipeline D0-D5 in order: ``uv run python -m hns.data``.

The D1 audit runs right after loading and stops the pipeline on any mismatch.
"""

from hns.data import audit, causal, graph, load, norm, split

if __name__ == "__main__":
    load.run()
    audit.run()
    graph.run()
    causal.run()
    split.run()
    norm.run()
