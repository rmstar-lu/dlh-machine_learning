#!/usr/bin/env python3
""" 33-schools_by_topic module, get all schools teaching a specific topic. """

import pymongo


def schools_by_topic(mongo_collection, topic):
    """ Get all schools teaching a specific topic. """

    return mongo_collection.find({"topics": topic})
