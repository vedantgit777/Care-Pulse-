# check_db.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv('MONGO_URI'))
db = client['carepulse']

users = list(db.users.find({}, {'_id': 0}))
print(f"Total users found: {len(users)}")
for u in users:
    print(u)