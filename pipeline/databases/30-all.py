#!/usr/bin/env python3
""" 30-all list all documents in a collection. """

import pymongo


def list_all(mongo_collection):
    """ Return list of documents in the specified collection. """

    return mongo_collection.find()
