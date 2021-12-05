# -*- coding: utf-8 -*-
from geopy.geocoders import Nominatim
import pandas as pd

def make_df(cols, ind):
# input: cols = list of name of columns
#        ind = list of name of index
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)


geolocator = Nominatim(user_agent="geoapiExercises")

def country(coord):
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    country = address.get('country', '')
    return country