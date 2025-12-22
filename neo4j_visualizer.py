from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit as st

from neo4j_connector import driver  # utilise ton driver déjà défini

def render_neo4j_graph():
    net = Network(height="600px", width="100%", directed=True)

    with driver.session() as session:
        query = """
        MATCH (a)-[r]->(b)
        RETURN a, r, b
        """
        results = session.run(query)

        for record in results:
            node_a = record["a"]
            node_b = record["b"]
            rel = record["r"]

            # Ajouter les nœuds
            net.add_node(node_a.id, label=node_a["name"], title=str(node_a.items()))
            net.add_node(node_b.id, label=node_b["name"], title=str(node_b.items()))

            # Ajouter la relation
            net.add_edge(node_a.id, node_b.id, label=type(rel).__name__)

    # Sauvegarde dans un fichier HTML
    net.save_graph("neo4j_graph.html")
    return "neo4j_graph.html"
