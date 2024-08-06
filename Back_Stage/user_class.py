from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, user_id, user_email, user_name, user_password, user_age, user_gender,user_hue_base,user_sat_base,user_lig_base):
        self.user_id = user_id
        self.user_email = user_email
        self.user_name = user_name
        self.user_password = user_password
        self.user_age = user_age
        self.user_gender = user_gender
        self.user_hue_base = user_hue_base
        self.user_sat_base = user_sat_base
        self.user_lig_base = user_lig_base
    def get_id(self):
        return self.user_id
