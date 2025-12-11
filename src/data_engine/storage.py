import lancedb
import pandas as pd
import os
import shutil
from typing import List, Dict, Any
from src.utils.llm import get_embedding

class KnowledgeBase:
    def __init__(self, db_path: str = "data/lancedb"):
        self.db_path = db_path
        self.db = None
        self.table_name = "papers"
        self.metadata_df = pd.DataFrame()

    def initialize_db(self):
        """
        Initializes the LanceDB connection and creates tables if they don't exist.
        """
        # Ensure directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        self.db = lancedb.connect(self.db_path)
        print(f"Connected to LanceDB at {self.db_path}")

        # Check if table exists
        if self.table_name in self.db.table_names():
             print(f"Table '{self.table_name}' already exists.")
             # Load metadata
             self.metadata_df = self.db.open_table(self.table_name).to_pandas()
        else:
            print(f"Table '{self.table_name}' does not exist. It will be created upon ingestion.")

    def ingest_data(self, papers: List[Dict]):
        """
        Ingests processed paper data into LanceDB (vectors).
        Expected paper format: 
        {
            "id": str,
            "title": str,
            "abstract": str,
            "content": str, (full text or summary)
            "authors": List[str],
            "year": int,
            "rating": float
        }
        """
        if not papers:
            print("No papers to ingest.")
            return

        print(f"Processing {len(papers)} papers for ingestion...")
        data_to_insert = []
        
        for p in papers:
            # Create text for embedding: Title + Abstract
            text_to_embed = f"Title: {p.get('title')}\nAbstract: {p.get('abstract')}"
            vector = get_embedding(text_to_embed)
            
            if vector:
                p["vector"] = vector
                # Ensure authors is string for compatibility if needed, or keep as list if supported
                # For simplicity in this demo, we might join them if issues arise, but LanceDB supports lists.
                data_to_insert.append(p)
            else:
                print(f"Skipping paper {p.get('id')} due to embedding failure.")

        if not data_to_insert:
            print("No valid data to insert.")
            return

        # Insert into LanceDB
        if self.table_name in self.db.table_names():
            tbl = self.db.open_table(self.table_name)
            tbl.add(data_to_insert)
        else:
            tbl = self.db.create_table(self.table_name, data=data_to_insert)
            
        print(f"Successfully ingested {len(data_to_insert)} papers.")
        # Reload metadata
        self.metadata_df = tbl.to_pandas()

    def search_semantic(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Performs semantic search using vector embeddings.
        """
        if self.table_name not in self.db.table_names():
            print("Table not found. Please ingest data first.")
            return []

        print(f"Semantic searching for: {query}")
        query_vector = get_embedding(query)
        if not query_vector:
            return []

        tbl = self.db.open_table(self.table_name)
        results = tbl.search(query_vector).limit(limit).to_list()
        return results

    def search_structured(self, query: str = None) -> pd.DataFrame:
        """
        Returns the metadata dataframe, optionally filtered.
        Simple implementation: returns all for now, or allows pandas operations outside.
        """
        # If we had a SQL-like engine or complex filters, we'd apply them here.
        # For now return the full DF so agents can filter using pandas.
        if self.metadata_df.empty and self.table_name in self.db.table_names():
             self.metadata_df = self.db.open_table(self.table_name).to_pandas()
             
        return self.metadata_df
