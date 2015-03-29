__author__ = 'shaughnfinnerty'

class TwitterMessage:
    def __init__(self, msg_id="", user_id="", msg_text="", polarity=""):
        self.msg_id = msg_id
        self.user_id = user_id
        self.msg_text = msg_text
        self.polarity = polarity
