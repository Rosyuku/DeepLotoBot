# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:05:49 2018

@author: kazuyuki
"""

import twitter
import config

api = twitter.Api(consumer_key=config.consumer_key,
                  consumer_secret=config.consumer_secret,
                  access_token_key=config.access_token_key,
                  access_token_secret=config.access_token_secret
                  )

api.PostUpdate("test2")

