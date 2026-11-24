"""Export inference graphs to TensorBoard (Graphs tab).

Keras 3 logs only a conceptual graph, so each classifier is traced with
tf.function, its weights are folded into constants, and the resulting operation
graph is written with one top-level scope per layer.

  python -m src.tb_graph
"""
from __future__ import annotations

import re

from src import config as C

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def layer_graph(model) -> tf.compat.v1.GraphDef:
    fn = tf.function(lambda x: model(x, training=False)).get_concrete_function(
        tf.TensorSpec([1, C.IMG, C.IMG, 3], name="image"))
    gd = convert_variables_to_constants_v2(fn).graph.as_graph_def()
    names = {n.name: re.sub(r"^%s(_\d+)?/" % re.escape(model.name), "", n.name) for n in gd.node}
    for node in gd.node:
        node.name = names[node.name]
        for i, ref in enumerate(node.input):
            ctrl, base = ("^", ref[1:]) if ref.startswith("^") else ("", ref)
            base, _, port = base.partition(":")
            node.input[i] = ctrl + names.get(base, base) + (f":{port}" if port else "")
    return gd


def main() -> None:
    for name in ("baseline", "transfer"):
        model = keras.models.load_model(C.MODELS / f"{name}.keras", compile=False)
        writer = tf.summary.create_file_writer(str(C.LOGS / f"graph_{name}"))
        with writer.as_default():
            tf.summary.graph(layer_graph(model))
        writer.close()
        print("graph written for", name)


if __name__ == "__main__":
    main()
