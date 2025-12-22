# neo4j_connector.py
from neo4j import GraphDatabase
import os

# Replace with your credentials
URI = "neo4j://127.0.0.1:7687"
USER = "-----"
PASSWORD = "-----"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Function to create the medical graph (nodes & relations)
def create_medical_graph(tx):
    tx.run("MERGE (g:Gliome {name:'Glioblastome'})")
    tx.run("MERGE (a:Analyse {name:'IRM VGG19'})")
    tx.run("MERGE (t:Traitement {name:'Chirurgie + Radiothérapie'})")
    tx.run("""
        MATCH (g:Gliome {name:'Glioblastome'}), (a:Analyse {name:'IRM VGG19'})
        MERGE (g)-[:DETECTE_PAR]->(a)
    """)
    tx.run("""
        MATCH (g:Gliome {name:'Glioblastome'}), (t:Traitement {name:'Chirurgie + Radiothérapie'})
        MERGE (g)-[:TRAITE_PAR]->(t)
    """)

def get_treatments_for_tumor(tx, tumor_name):
    query = """
    MATCH (g {name:$tumor_name})-[:TRAITE_PAR]->(t)
    RETURN t.name AS traitement
    """
    result = tx.run(query, tumor_name=tumor_name)
    return [record["traitement"] for record in result]