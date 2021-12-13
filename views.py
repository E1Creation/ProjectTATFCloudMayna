from flask import render_template, Response
from fcr import  gen_frame
import mysql_connector as sq
from firestore import ShowCloudDB, markAttendanceIntoCloud

# View template
def base():
    return render_template('base.html')

# Home page
def index():
    return render_template('index.html')

# Video Streaming page
def presence():
    """Video streaming home page."""
    return render_template('presence.html')

def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Presence system to show the database page
def html_page(page_name):
    return render_template(page_name)

def history():
    return sq.showRows()

def cloud():
    return ShowCloudDB()

def insert_cloud():
    return markAttendanceIntoCloud()




    