# seed.py
from pymongo import MongoClient
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
print(f"Connecting to: {MONGO_URI[:40]}...")  # print partial URI to confirm it loaded

client = MongoClient(MONGO_URI)

# List all databases to confirm connection
print("Databases found:", client.list_database_names())

db = client['carepulse']

users = [
    {"email": "vedantbhagwani@gmail.com", "name": "Vedant Bhagwani"},
    {"email": "swapnilsingh@gmail.com",   "name": "Swapnil Singh"},
    {"email": "shreysrivastava@gmail.com", "name": "Shrey Srivastava"},
]

for u in users:
    result = db.users.update_one(
        {'email': u['email']},
        {'$setOnInsert': {
            'email':       u['email'],
            'name':        u['name'],
            'created_at':  datetime.now(timezone.utc),
            'login_count': 0,
            'last_login':  None
        }},
        upsert=True
    )
    print(f"  {u['email']} → matched={result.matched_count} upserted={result.upserted_id}")

print(f"\nTotal users now: {db.users.count_documents({})}")