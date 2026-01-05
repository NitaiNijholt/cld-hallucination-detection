from dotenv import load_dotenv
import os

from elasticsearch import Elasticsearch
from neo4j import GraphDatabase

load_dotenv()

class es_connector():
    def __init__(self):
        self.es_port = os.environ.get("ES_PORT")
        self.es_host = os.environ.get("ES_HOST")
        self.es_index_name = os.environ.get("ES_INDEX_NAME")

        self.client = Elasticsearch(f"http://{self.es_host}:{self.es_port}/")
    
    def query(self,query):
        resp = self.client.search(
            index=self.es_index_name,
            query=query,
            size=600
        )
        return resp

class n4j_connector():
    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI")
        self.username = os.environ.get("NEO4J_USERNAME")
        self.password = os.environ.get("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def query(self, cypher_query, parameters=None):
        if parameters is None:
            parameters = {}
        
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters)
            return [record for record in result]