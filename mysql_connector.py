# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 09:11:13 2021

@author: Kurniawan Sudirman
"""


import mysql.connector
from flask import render_template

mydb=mysql.connector.connect(host='localhost',user='root',password='',database='face_recognition')
#mycursor=mydb.cursor()

def showDB():
    mycursor=mydb.cursor()
    mycursor.execute("Show databases")
    print('daftar databases :')
    for db in mycursor:
        print(db)

def showTables():
    mycursor=mydb.cursor()
    mycursor.execute("Show tables")
    print('daftar tables :')
    for tb in mycursor:
        print(tb)
        
def insertRow(id,tanggal,nama,waktu,status):
    mycursor=mydb.cursor()
    sqlInput="REPLACE into rlabkom2(id,tanggal,nama,jam,status) values (%s,%s,%s,%s,%s)"
    data=(id,tanggal,nama,waktu,status)
    mycursor.execute(sqlInput,data)
    mydb.commit()

def deleteRow(nim):
    mycursor=mydb.cursor()
    sqlInput="Delete from rlabkom where nim=%s"
    data=(nim,)
    mycursor.execute(sqlInput,data)
    mydb.commit()
    
def showRows():
    mycursor=mydb.cursor()
    sqlInput="SELECT * FROM rlabkom2 "
    mycursor.execute(sqlInput)
    myresult=mycursor.fetchall()
    mycursor.close()
    return render_template('history.html', history=myresult)
        
def getRowsSpecific(nama):
    mycursor=mydb.cursor()
    sqlInput='SELECT nim from pengunjung WHERE nama=%s'
    data=(nama,)
    mycursor.execute(sqlInput,data)
    myresult = mycursor.fetchone()
    return myresult[0]

def getRowsAllSpecific(nim):
    mycursor=mydb.cursor()
    nimCollection=[]
    nimData=[]
    nimData=nim.copy()
   # print(nimData)
    for i,val in enumerate(nimData):
        sqlInput="SELECT nama from pengunjung WHERE nim=%s"
        nimTuple=(nimData[i],)
       # print(nimData[i])
        mycursor.execute(sqlInput,nimTuple)
        myresult=mycursor.fetchone()
        nimCollection.append(myresult[0])
    return nimCollection

#nim=["123","21060117140080","12"]
#nimKolek=getRowsAllSpecific(nim)
#print(nimKolek)
#nama=getRowsSpecific("12")
#print(nama)
#showDB()
#showTables()
#insertRow("1","31-03-2021","kurniawan","22:43:23","masuk")
#deleteRow("1")
#showRows()
#def insertSQL(name,time):
    #mycursor.execute("INSERT into R306")
