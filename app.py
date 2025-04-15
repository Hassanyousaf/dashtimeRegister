from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from face_register import FaceRegistrar

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/temp'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

registrar = FaceRegistrar()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
    
    try:
        result = registrar.register_user(user_id)
        if result['success']:
            return jsonify({
                'message': f"Successfully registered {user_id}",
                'data': result
            }), 200
        else:
            return jsonify({'error': result['message']}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preview')
def preview():
    return render_template('preview.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)