#!/usr/bin/env python3
""" 106-log_stats module, report stats about nginx logs stored in MongoDB. """

from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx = client.logs.nginx
    print("{:d} logs".format(nginx.count_documents({})))
    print("Methods:")
    for m in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        print("\tmethod {}: {:d}".format(
            m, nginx.count_documents({"method": m})
        ))
    print("{:d} status check".format(
        nginx.count_documents({"method": "GET", "path": "/status"})
    ))
    print("IPs:")
    pipeline = [
            {"$group": {"_id": "$ip", "ip": {"$first": "$ip"},
                        "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
    ]
    ip_count = nginx.aggregate(pipeline)
    for row in ip_count:
        print("\t{}: {:d}".format(row["ip"], row["count"]))
