#!/usr/bin/env python3

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    return send_from_directory('static', filename)

@app.route('/health')
def health():
    """健康检查"""
    return {"status": "healthy", "service": "frontend"}

if __name__ == '__main__':
    # 确保templates目录存在
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Frontend Server on port 9092...")
    app.run(host='0.0.0.0', port=9092, debug=True) 