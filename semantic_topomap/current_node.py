import faiss
import numpy as np


class CurrentNode:
    def __init__(self, leaving_threshold=2.5):
        self.current_id = 0
        self.leaving_threshold = leaving_threshold
        self.current_feature = []
        self.current_location = []

    def load_history(self, history_id, topomap):
        self.current_id = history_id
        node = topomap.topo_structure.nodes[history_id]
        self.current_feature = list(node["satellite_features"])
        self.current_location = list(node["satellite_locations"])

    def update_current(self, new_feature, new_location):
        self.current_feature.append(new_feature)
        self.current_location.append(new_location)

    def leaving_current(self, new_feature, new_location):
        del new_location
        if len(self.current_feature) == 0:
            return False

        features = np.array(self.current_feature)
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features.astype("float32"))
        distances, _ = index.search(new_feature.reshape(1, -1).astype("float32"), features.shape[0])
        max_distance = distances[0].max()
        return bool(max_distance > self.leaving_threshold)

    def current_to_graph(self, topomap):
        if self.current_id in topomap.topo_structure.nodes:
            topomap.topo_structure.remove_node(self.current_id)
        topomap.add_node(self.current_id, self.current_feature, self.current_location, max_satellites=5)
        self.clear_current()

    def clear_current(self):
        self.current_feature = []
        self.current_location = []
        self.current_id = -1
