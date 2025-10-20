from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# In-memory session storage (replace with database in production)
sessions = {}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/session', methods=['POST'])
def create_session():
    """
    Create a new session
    
    Response:
    {
        "session_id": 123456789
    }
    """
    session_id = random.randint(1, 1_000_000_000)
    
    # Initialize session storage
    sessions[session_id] = {
        "messages": []
    }
    
    return jsonify({
        "session_id": session_id
    }), 201


@app.route('/api/audio/process', methods=['POST'])
def process_audio():
    """
    Process audio file and return AI response
    
    Form data:
    - audio: audio file (any format - wav, mp3, webm, etc.)
    - session_id: valid session identifier
    
    Response:
    {
        "session_id": 123456789,
        "transcribed_text": "what the user said",
        "ai_response": "AI's text response"
    }
    """
    # Get session_id from form data
    session_id = request.form.get('session_id')
    
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    
    try:
        session_id = int(session_id)
    except ValueError:
        return jsonify({"error": "session_id must be a number"}), 400
    
    # Check if session exists
    if session_id not in sessions:
        return jsonify({"error": "Invalid session_id"}), 404
    
    # Get audio file
    if 'audio' not in request.files:
        return jsonify({"error": "audio file is required"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    # TODO: Process audio file
    # 1. Save audio temporarily
    # 2. Call get_accent_prediction() from audio.py
    # 3. Call get_text_from_speech() from audio.py
    # 4. Call get_llm_response() from audio.py
    # 5. Return response
    
    # Placeholder response
    return jsonify({
        "session_id": session_id,
        "transcribed_text": "TODO: implement transcription",
        "ai_response": "TODO: implement AI response"
    }), 200


if __name__ == '__main__':
    # Run on port 3000 instead of default 5000
    app.run(debug=True, host='0.0.0.0', port=3000)
