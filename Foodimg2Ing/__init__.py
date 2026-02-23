from flask import Flask

app = Flask(__name__,template_folder='Templates')

# Allow requests with arbitrary Host headers (e.g., ngrok domains)
# to avoid 403 Forbidden due to host validation.
app.config["TRUSTED_HOSTS"] = ["*"]

from Foodimg2Ing import routes