import faiss
import networkx as nx
import numpy as np


class Topomap:
    def __init__(self):
        self.topo_structure = nx.Graph()
        self.topo_data = faiss.IndexFlatL2(512)

    def add_node(self, node_id, temp_feature_list, temp_location_list, max_satellites=5):
        satellite_features = np.array(temp_feature_list)
        satellite_locations = np.array(temp_location_list[-len(temp_feature_list):])
        num_satellites = min(max_satellites, satellite_features.shape[0])
        selected_indices = np.random.choice(satellite_features.shape[0], num_satellites, replace=False)
        selected_satellites = satellite_features[selected_indices]
        selected_locations = satellite_locations[selected_indices]
        center_feature = np.mean(selected_satellites, axis=0)

        node = {
            "id": node_id,
            "satellite_features": selected_satellites,
            "satellite_locations": selected_locations,
        }
        self.topo_structure.add_node(node_id, **node)
        self.topo_data.add(center_feature.reshape(1, -1).astype("float32"))

    def get_node_center(self, node_id):
        node = self.topo_structure.nodes[node_id]
        sat_locs = node["satellite_locations"]
        return np.mean(sat_locs, axis=0)

    def locate_node(self, query_feature, query_location, k=1):
        if self.topo_structure.number_of_nodes() == 0:
            return [-1], [float("inf")]

        node_ids = list(self.topo_structure.nodes)
        center_locations = []
        for nid in node_ids:
            sat_locs = self.topo_structure.nodes[nid]["satellite_locations"]
            center_locations.append(np.mean(sat_locs, axis=0))

        center_locations = np.array(center_locations)
        dists = np.linalg.norm(center_locations - query_location, axis=1)
        nearest_indices = np.argsort(dists)[:k]
        candidate_node_ids = [node_ids[i] for i in nearest_indices]

        min_dist = float("inf")
        min_node = -1
        for nid in candidate_node_ids:
            sat_feats = self.topo_structure.nodes[nid]["satellite_features"]
            feat_dists = np.linalg.norm(sat_feats - query_feature, axis=1)
            node_min_dist = np.min(feat_dists)
            if node_min_dist < min_dist:
                min_dist = node_min_dist
                min_node = nid
        return [min_node], [min_dist]

    def get_num_nodes(self):
        return self.topo_structure.number_of_nodes()
