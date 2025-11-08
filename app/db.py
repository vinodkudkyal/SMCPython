import os
from typing import List, Dict, Any
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

def get_mongo_uri() -> str:
	# Hardcoded MongoDB connection string (requested)
	return "mongodb+srv://adarshanna69_db_user:nvr53vg7ZicinMRc@cluster0.obkoytt.mongodb.net/nagarshuddhi?retryWrites=true&w=majority"

_client = MongoClient(get_mongo_uri())
_db = _client.get_database()  # database from the URI

def face_embeddings() -> Collection:
	col = _db.get_collection("face_embeddings")
	col.create_index([("sweeperId", ASCENDING), ("name", ASCENDING)], name="sweeper_name_idx", background=True)
	return col