import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger('database')

class MongoRepository:
    """store & retrieve data in MongoDB, merges HPC + NEW(UI)."""

    def __init__(self, uri=None):
        self.uri= uri or os.getenv("MONGO_URI","mongodb://localhost:27017/flash_dna")
        self.client= None
        self.db= None
        self.startups= None
        self.analyses= None
        self.reports= None
        self.users= None
        self._connect()

    def _connect(self):
        try:
            self.client= MongoClient(self.uri)
            self.client.admin.command('ismaster')
            if '/' in self.uri:
                db_name= self.uri.rsplit('/',1)[-1].split('?')[0]
            else:
                db_name= "flash_dna"
            self.db= self.client[db_name]
            self.startups= self.db.startups
            self.analyses= self.db.analyses
            self.reports= self.db.reports
            self.users= self.db.users
            self._create_indexes()
            logger.info(f"Connected to MongoDB => {db_name}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect => {str(e)}")
            self.client= None
        except Exception as e:
            logger.error(f"Error => {str(e)}")
            self.client= None

    def _create_indexes(self):
        try:
            self.startups.create_index([("name", ASCENDING)], unique=True)
            self.startups.create_index([("sector", ASCENDING)])
            self.startups.create_index([("created_at", DESCENDING)])
            self.analyses.create_index([("startup_id", ASCENDING)])
            self.analyses.create_index([("created_at", DESCENDING)])
            self.reports.create_index([("startup_id", ASCENDING)])
            self.reports.create_index([("created_at", DESCENDING)])
            self.users.create_index([("email", ASCENDING)], unique=True)
        except OperationFailure as e:
            logger.error(f"Failed creating indexes => {str(e)}")
        except Exception as e:
            logger.error(f"Index error => {str(e)}")

    def close(self):
        if self.client:
            try:
                self.client.close()
                logger.info("Closed MongoDB connection.")
            except Exception as e:
                logger.error(f"Close error => {str(e)}")

    def save_startup(self, startup_data: Dict[str,Any]) -> Optional[str]:
        if not self.client:
            logger.error("No connection => cannot save startup.")
            return None
        try:
            now= datetime.utcnow()
            if 'created_at' not in startup_data:
                startup_data['created_at']= now
            startup_data['updated_at']= now
            result= self.startups.update_one(
                {"name": startup_data.get("name")},
                {"$set": startup_data},
                upsert=True
            )
            if result.upserted_id:
                logger.info(f"Inserted new => {startup_data.get('name')}")
                return str(result.upserted_id)
            else:
                doc= self.startups.find_one({"name": startup_data.get("name")})
                return str(doc["_id"]) if doc else None
        except Exception as e:
            logger.error(f"save_startup => {str(e)}")
            return None

    def get_startup_by_name(self, name: str)-> Optional[Dict[str,Any]]:
        if not self.client:
            logger.error("No connection => cannot fetch startup")
            return None
        try:
            startup= self.startups.find_one({"name": name})
            if startup:
                startup["_id"]= str(startup["_id"])
                return startup
            return None
        except Exception as e:
            logger.error(f"get_startup_by_name => {str(e)}")
            return None

    def get_startup(self, startup_id: str)-> Optional[Dict[str,Any]]:
        if not self.client:
            logger.error("No connection => cannot fetch startup by ID")
            return None
        try:
            from bson.objectid import ObjectId
            doc= self.startups.find_one({"_id": ObjectId(startup_id)})
            if doc:
                doc["_id"]= str(doc["_id"])
                return doc
            return None
        except Exception as e:
            logger.error(f"get_startup => {str(e)}")
            return None

    def get_all_startups(self, sector=None, limit=100)-> List[Dict[str,Any]]:
        if not self.client:
            logger.error("No connection => cannot get all startups.")
            return []
        try:
            q={}
            if sector:
                q["sector"]= sector
            cursor= self.startups.find(q).sort("created_at",DESCENDING).limit(limit)
            result=[]
            for doc in cursor:
                doc["_id"]= str(doc["_id"])
                result.append(doc)
            return result
        except Exception as e:
            logger.error(f"get_all_startups => {str(e)}")
            return []

    def save_analysis(self, startup_id:str, analysis_data: Dict[str,Any]) -> Optional[str]:
        if not self.client:
            logger.error("No connection => cannot save analysis.")
            return None
        try:
            now= datetime.utcnow()
            analysis_data["startup_id"]= startup_id
            analysis_data["created_at"]= now
            result= self.analyses.insert_one(analysis_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"save_analysis => {str(e)}")
            return None

    def get_latest_analysis(self, startup_id:str)-> Optional[Dict[str,Any]]:
        if not self.client:
            logger.error("No connection => cannot get analysis")
            return None
        try:
            doc= self.analyses.find_one({"startup_id": startup_id}, sort=[("created_at",DESCENDING)])
            if doc:
                doc["_id"]= str(doc["_id"])
                return doc
            return None
        except Exception as e:
            logger.error(f"get_latest_analysis => {str(e)}")
            return None

    def save_report(self, startup_id:str, report_data:Dict[str,Any], pdf_data:bytes)-> Optional[str]:
        if not self.client:
            logger.error("No connection => cannot save report.")
            return None
        try:
            from bson.binary import Binary
            now= datetime.utcnow()
            report_data["startup_id"]= startup_id
            report_data["pdf"]= Binary(pdf_data)
            report_data["created_at"]= now
            res= self.reports.insert_one(report_data)
            return str(res.inserted_id)
        except Exception as e:
            logger.error(f"save_report => {str(e)}")
            return None

    def get_report(self, report_id:str)-> Optional[Dict[str,Any]]:
        if not self.client:
            logger.error("No connection => cannot get report.")
            return None
        try:
            from bson.objectid import ObjectId
            rep= self.reports.find_one({"_id": ObjectId(report_id)})
            if rep:
                rep["_id"]= str(rep["_id"])
                return rep
            return None
        except Exception as e:
            logger.error(f"get_report => {str(e)}")
            return None

    def get_latest_report(self, startup_id:str)-> Optional[Dict[str,Any]]:
        if not self.client:
            logger.error("No connection => cannot get latest report.")
            return None
        try:
            rep= self.reports.find_one({"startup_id": startup_id}, sort=[("created_at",DESCENDING)])
            if rep:
                rep["_id"]= str(rep["_id"])
                return rep
            return None
        except Exception as e:
            logger.error(f"get_latest_report => {str(e)}")
            return None

    def save_user(self, user_data:Dict[str,Any])-> Optional[str]:
        if not self.client:
            logger.error("No connection => cannot save user.")
            return None
        try:
            now= datetime.utcnow()
            user_data["created_at"]= now
            res= self.users.insert_one(user_data)
            return str(res.inserted_id)
        except Exception as e:
            logger.error(f"save_user => {str(e)}")
            return None

    def get_user_by_email(self, email:str)-> Optional[Dict[str,Any]]:
        if not self.client:
            logger.error("No connection => cannot get user.")
            return None
        try:
            usr= self.users.find_one({"email":email})
            if usr:
                usr["_id"]= str(usr["_id"])
                return usr
            return None
        except Exception as e:
            logger.error(f"get_user_by_email => {str(e)}")
            return None
