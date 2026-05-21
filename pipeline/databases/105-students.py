#!/usr/bin/env python3
""" 105-students return all students sorted by average score. """

import pymongo


def top_students(mongo_collection):
    """ Return students sorted by average score. """

    pipeline = [
        {"$unwind": "$topics"},
        {"$group": {"_id": "$_id",
                    "name": {"$first": "$name"},
                    "averageScore": {"$avg": "$topics.score"}}},
        {"$sort": {"averageScore": -1}}
    ]
    return mongo_collection.aggregate(pipeline)
