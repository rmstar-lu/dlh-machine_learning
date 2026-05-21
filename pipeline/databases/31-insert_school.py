#!/usr/bin/env python3
""" 31-insert_school insert a new document in a collection based on kwargs. """

import pymongo


def insert_school(mongo_collection, **kwargs):
    """ Insert a new document in the specified collection. """

    return mongo_collection.insert_one(kwargs)
