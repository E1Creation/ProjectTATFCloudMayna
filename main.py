from flask import Flask
import views

app = Flask(__name__)

# create URL
app.add_url_rule('/base','base', views.base)
app.add_url_rule('/','index', views.index)
app.add_url_rule('/<string:page_name>', views.html_page)
app.add_url_rule('/presence','presence', views.presence, methods = ['POST', 'GET'])
app.add_url_rule('/history','history', views.history)
app.add_url_rule('/video_feed', 'video_feed', views.video_feed)
app.add_url_rule('/cloud', 'cloud', views.cloud)

# run
if __name__ == "__main__":
    app.run(debug=True)