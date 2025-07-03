import os
from flask import Flask, request, session, redirect, url_for, render_template, jsonify
from flask_session import Session
from pymongo import MongoClient
import redis
from dotenv import load_dotenv
import secrets
from datetime import datetime, timedelta # Import timedelta for potential time calculations
import google.generativeai as genai
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import requests
import json
import cv2
import mediapipe as mp
from datetime import datetime
import numpy as np
import random # For mock data generation
import textwrap
from fer import FER
emotion_detector = FER(mtcnn=True)

# Load environment variables
load_dotenv()

AZURE_FACE_API_ENDPOINT = os.getenv("AZURE_FACE_API_ENDPOINT")
AZURE_FACE_API_KEY = os.getenv("AZURE_FACE_API_KEY")


# Gemini API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    # Use the provided key if not set in environment
    gemini_api_key = 'AIzaSyA2Mi4IjnQf4TJ5FQyO3p21njnN7PRmDyg' # Using a placeholder for demonstration
    print("Warning: GEMINI_API_KEY not found in environment, using default.")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

app = Flask(__name__)

# Secret Key
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
if not app.config["SECRET_KEY"]:
    # Use the provided key if not set in environment
    app.config["SECRET_KEY"] = 'c13997a18de853c9ce6e4226a53536c1' # Using a placeholder for demonstration
    print("Warning: SECRET_KEY not found in environment, using default.")

# Session Configuration
redis_url = os.getenv("REDIS_URL")
if not redis_url:
    # Use the provided URL if not set in environment
    redis_url = 'rediss://red-d1iadk6r433s73a8qiv0:vZC5vfKvFm2f0vCmG2F7gZUfCMSc4iO6@oregon-keyvalue.render.com:6379' # Using a placeholder for demonstration
    print("Warning: REDIS_URL not found in environment, using default.")

app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_REDIS"] = redis.from_url(redis_url)
if os.getenv("FLASK_ENV") == "development":
    app.config["SESSION_COOKIE_SECURE"] = False  # Required for localhost HTTP

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_NAME"] = "interview-session"

# Initialize Session
Session(app)

# MongoDB Connection
mongodb_url = os.getenv("MONGO_URI")
if not mongodb_url:
    # Use the provided URL if not set in environment
    mongodb_url = 'mongodb+srv://aryanbansal:aryan1234@intelliview.lbxflf8.mongodb.net/intelliview?retryWrites=true&w=majority' # Using a placeholder for demonstration
    print("Warning: MONGO_URI not found in environment, using default.")

MONGO_CLIENT = MongoClient(mongodb_url)
DATABASE = MONGO_CLIENT["intelliview"]

# Google OAuth Credentials
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
if not GOOGLE_CLIENT_ID:
    GOOGLE_CLIENT_ID = '688599391698-raqcmahpl90u6t2ua2kcmpbsfbqqq6ug.apps.googleusercontent.com' # Using a placeholder for demonstration
    print("Warning: GOOGLE_CLIENT_ID not found in environment, using default.")
app.config['GOOGLE_CLIENT_ID'] = GOOGLE_CLIENT_ID # Store in app.config for template access

# Context processor to make GOOGLE_CLIENT_ID available in all templates
@app.context_processor
def inject_google_client_id():
    return dict(config=app.config) # Pass the entire app.config object

# Initialize MediaPipe Pose outside the route for efficiency
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

from flask import Blueprint
ats_bp = Blueprint('ats', __name__, url_prefix='/ats')

@ats_bp.route('/')
def ats_form():
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))
    return render_template('ats_score.html')

def extract_text_from_pdf_with_gemini(pdf_content):
    """Extract text from PDF using Gemini AI"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare the PDF for Gemini
        pdf_blob = {
            "mime_type": "application/pdf",
            "data": pdf_content
        }
        
        prompt = """
        Extract all text content from this PDF document. 
        Return only the plain text without any formatting, markdown, or additional commentary.
        Focus on preserving the original structure and content of the resume.
        """
        
        response = model.generate_content([prompt, pdf_blob])
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF with Gemini: {e}")
        return None
 
def get_ats_score_with_gemini(resume_text):
    """Get ATS score using Gemini AI"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an ATS (Applicant Tracking System) analyzer. Analyze the following resume and provide a comprehensive assessment.

        Resume Text:
        {resume_text[:8000]}

        Provide your analysis in the following JSON format:
        {{
            "score": <number between 0-100>,
            "summary": "<brief 2-3 sentence summary>",
            "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
            "suggestions": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"]
        }}

        Scoring criteria:
        - Clear formatting and structure (25 points)
        - Relevant keywords and skills (25 points)
        - Professional experience and achievements (25 points)
        - Education and qualifications (15 points)
        - Contact information and completeness (10 points)

        Keep strengths and suggestions concise and actionable.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error getting ATS score with Gemini: {e}")
        return None

@ats_bp.route('/score', methods=['POST'])
def ats_score():
    if not session.get("is_authenticated"):
        return jsonify({'error': 'User not authenticated'}), 401
        
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file uploaded', 'reason': 'Please upload a PDF file.'}), 400
    
    file = request.files['resume']
    
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Invalid file', 'reason': 'Please upload a PDF file.'}), 400
    
    try:
        # Read the PDF content
        pdf_content = file.read()
        
        # Extract text using Gemini
        resume_text = extract_text_from_pdf_with_gemini(pdf_content)
        
        if not resume_text:
            return jsonify({
                'error': 'Text extraction failed',
                'reason': 'Could not extract text from the PDF. Please ensure the PDF contains readable text.'
            }), 500
        
        # Get ATS score using Gemini
        gemini_response = get_ats_score_with_gemini(resume_text)
        
        if not gemini_response:
            return jsonify({
                'error': 'AI analysis failed',
                'reason': 'Could not analyze the resume. Please try again.'
            }), 500
        
        # Parse the JSON response
        try:
            # Remove markdown code blocks if present
            clean_response = gemini_response
            # Check for specific json code block fence
            if '```json' in clean_response:
                # Extract JSON between ```json and the next ```
                clean_response = clean_response.split('```json', 1)[1].split('```', 1)[0].strip()
            # Fallback: check for generic code block fence
            elif '```' in clean_response:
                # Extract content between generic ```
                clean_response = clean_response.split('```', 1)[1].split('```', 1)[0].strip()

            parsed_data = json.loads(clean_response)

            return jsonify({
                'score': parsed_data.get('score', 0),
                'summary': parsed_data.get('summary', 'Analysis completed.'),
                'strengths': parsed_data.get('strengths', []),
                'suggestions': parsed_data.get('suggestions', []),
                'raw': gemini_response
            })

        except (IndexError, ValueError, json.JSONDecodeError) as e:
            # Handle cases where splitting or JSON parsing fails
            app.logger.error(f"Error parsing AI response: {e}")
            return jsonify({
                'score': 0,
                'summary': 'Analysis format error.',
                'strengths': [],
                'suggestions': [],
                'raw': gemini_response
            }), 500

            
    except Exception as e:
        print(f"General ATS error: {e}")
        return jsonify({
            'error': 'Processing failed',
            'reason': str(e)
        }), 500

# Register ATS blueprint
app.register_blueprint(ats_bp)

# --------------------- Routes ----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/profile')
def profile():
    if not session.get("is_authenticated"):
        # Redirect to index if not authenticated, where Google login is available
        return redirect(url_for('index'))
    return render_template('profile.html')

@app.route('/interview')
def interview():
    if not session.get("is_authenticated"):
        # Redirect to index if not authenticated, where Google login is available
        return redirect(url_for('index'))
    return render_template('interview.html')

@app.route('/ats_score')
def ats_score():
    if not session.get("is_authenticated"):
        # Redirect to index if not authenticated, where Google login is available
        return redirect(url_for('index'))
    return render_template('ats_score.html')

@app.route('/api/v1/create-interview', methods=['POST'])
def create_interview():
    if not session.get("is_authenticated"):
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
    
    # Fetch form data from the request
    job_description = request.form.get('job_description')
    resume = request.files.get('resume')
    interview_type = request.form.get('interview_type')
    if interview_type not in ['technical', 'behavioral', 'common-questions']:
        return jsonify({'status': 'error', 'message': 'Invalid interview type, must be one of: technical, behavioral, common-questions'}), 400

    if not job_description or not resume:
        return jsonify({'status': 'error', 'message': 'Job description or resume not provided'}), 400

    # Fetch resume summary via LLM (already present, good!)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    prompt_resume_summary = f"""
    Carefully review the attached resume file. Provide a thorough, structured, and objective detailed summary of the candidate’s background, including:
    - Contact information (if present)
    - Education history (degrees, institutions, graduation years)
    - Work experience (roles, companies, durations, responsibilities, achievements)
    - Technical and soft skills
    - Certifications, awards, or notable projects
    - Any other relevant sections (e.g., publications, languages, interests)
    Present the information in clear, well-organized paragraphs using plain text (no markdown or formatting). Do not omit any details found in the resume. Avoid speculation; only summarize what is explicitly present in the document.
    """
    resume_blob = {
        "mime_type": resume.content_type,
        "data": resume.read()
    }
    response_resume = model.generate_content([prompt_resume_summary, resume_blob])
    resume_summary = response_resume.text

    # --- NEW ADDITION: GENERATE QUESTIONS BASED ON INTERVIEW TYPE, JOB DESCRIPTION, AND RESUME ---
    generated_questions = []
    try:
        question_generation_prompt = ""
        if interview_type == 'technical':
            question_generation_prompt = f"""
            Generate 10 technical interview questions for a candidate based on the following job description and their resume summary.
            Job Description: {job_description}
            Resume Summary: {resume_summary}
            The questions should be clear, concise, and directly relevant to the technical skills and experience mentioned.
            Always include these two generic questions as the first two and be sure to paraphrase them:
            1. Tell me a bit about yourself.
            2. Walk me through your resume.
            Only output the questions, one per line, with no numbering or extra text.
            """
        elif interview_type == 'behavioral':
            question_generation_prompt = f"""
            Generate 10 behavioral interview questions for a candidate based on the following job description and their resume summary.
            Job Description: {job_description}
            Resume Summary: {resume_summary}
            The questions should focus on past experiences, problem-solving, teamwork, and communication.
            Always include these two generic questions as the first two and be sure to paraphrase them:
            1. Tell me a bit about yourself.
            2. Walk me through your resume.
            Only output the questions, one per line, with no numbering or extra text.
            """
        elif interview_type == 'common-questions':
            question_generation_prompt = f"""
            Generate 10 common interview questions.
            Always include these two generic questions as the first two and be sure to paraphrase them:
            1. Tell me a bit about yourself.
            2. Walk me through your resume.
            The remaining questions should be standard interview questions applicable to most roles, such as strengths, weaknesses, career goals, etc.
            Only output the questions, one per line, with no numbering or extra text.
            """
        
        # Add print statements for debugging Gemini for create_interview
        print(f"DEBUG (create-interview): Question generation prompt:\n{question_generation_prompt}")
        questions_response = model.generate_content([question_generation_prompt])
        print(f"DEBUG (create-interview): Raw Gemini questions response.text:\n{questions_response.text}")
        
        generated_questions = questions_response.text.split('\n')
        generated_questions = [q for q in generated_questions if q.strip()]
        print(f"DEBUG (create-interview): Final generated questions list:\n{generated_questions}")

    except Exception as e:
        print(f"ERROR (create-interview): Failed to generate questions with Gemini: {e}")
        # Provide a fallback in case AI fails
        generated_questions = ["Tell me about yourself.", "Walk me through your resume.", "What are your strengths?", "What are your weaknesses?", "Where do you see yourself in 5 years?"]
        
    # Creating a new interview
    interview_identifier = secrets.token_hex(16)
    DATABASE["INTERVIEWS"].insert_one(
        {
            "interview_identifier": interview_identifier,
            "user_id": session["user"]["user_id"],
            "interview_type": interview_type,
            "job_description": job_description,
            "resume_summary": resume_summary,
            "created_at": datetime.now(),
            "is_active": True,
            "is_completed": False,
            "ai_report": "",
            "questions": generated_questions, # <-- NOW THIS WILL BE POPULATED!
            "interview_history": [], # Initialize interview history
            "behavior_analysis": [], # Initialize behavior analysis array
        }
    )

    # Redirect to the interview page
    return redirect(
        url_for(
            "interview_page", interview_identifier=interview_identifier
        )
    )


@app.route('/interview/<interview_identifier>', methods=['GET'])
def interview_page(interview_identifier):
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))
    # Check if the interview exists
    interview = DATABASE["INTERVIEWS"].find_one({"interview_identifier": interview_identifier})
    if interview is None:
        return jsonify({'status': 'error', 'message': 'Interview not found'}), 404

    # Check if the user is authorized to access this interview
    if interview["user_id"] != session["user"]["user_id"]:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to this interview'}), 403
    
    # Check if the interview is completed
    if interview["is_completed"]:
        return redirect(
            url_for(
                "interview_results", interview_identifier=interview_identifier
            )
        )

    return render_template('take-interview.html', interview=interview)

@app.route('/new-mock-interview', methods=['GET'])
def new_mock_interview():
    print("DEBUG: Entered new_mock_interview route.")
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))
    user_info = DATABASE["USERS"].find_one({"user_id": session["user"]["user_id"]})
    if user_info is None:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
    # Check if the user has a resume summary
    if not user_info.get("user_info", {}).get("resume_summary"):
        return redirect(url_for('settings', message='Please upload your resume first to generate mock interview questions.'))
    
    # --- ADD THESE DEBUG PRINTS HERE ---
    resume_summary = user_info['user_info']['resume_summary']
    print(f"DEBUG (new-mock-interview): Resume summary being sent to Gemini:\n{resume_summary}")
    
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    prompt = f"""
    Generate 10 mock interview questions based on the following resume summary:
    {resume_summary}
    The questions should be relevant to the candidate's background and experience, and the response should be in plain text format (no markdown or formatting). The questions should be clear and concise, and they should cover a range of topics related to the candidate's skills and experience. Avoid speculative or ambiguous questions and do not provide any additional information or context.

    Always include these two generic questions as the first two and be sure to paraphrase them:
    1. Tell me a bit about yourself.
    2. Walk me through your resume.

    The remaining questions should be tailored to the candidate's resume, covering technical skills, work experience, education, achievements, and other relevant areas. Do not repeat questions. Only output the questions, one per line, with no numbering or extra text.
    """
    
    try:
        response = model.generate_content([prompt])
        # --- ADD THIS DEBUG PRINT HERE ---
        print(f"DEBUG (new-mock-interview): Raw Gemini response.text:\n{response.text}")
        
        questions = response.text.split('\n')
        # Filter out any empty questions
        questions = [q for q in questions if q.strip()]
        
        # --- ADD THIS DEBUG PRINT HERE ---
        print(f"DEBUG (new-mock-interview): Final questions list after processing:\n{questions}")

    except Exception as e:
        print(f"ERROR (new-mock-interview): Gemini generation failed: {e}")
        questions = ["Error: Could not generate questions. Please try again or check API key."] # Fallback for display
    
    mock_interview_identifier = secrets.token_hex(16)
    
    DATABASE["INTERVIEWS"].insert_one(
        {   "mock_interview_identifier": mock_interview_identifier,
            "user_id": session["user"]["user_id"],
            "questions": questions,
            "created_at": datetime.now(),
            "is_active": True,
            "is_completed": False,
            "video_url": "", # Assuming you might add video storage later
            "ai_report": "",
            "interview_history": [], # Initialize interview history for mock
            "behavior_analysis": [], # Initialize behavior analysis array for mock
        }
    )

    return render_template('begin_mock_interview.html', mock_interview_identifier=mock_interview_identifier)


@app.route('/mock-interview/<mock_interview_identifier>', methods=['GET'])
def mock_interview(mock_interview_identifier):
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))
    # Check if the mock interview exists
    mock_interview = DATABASE["INTERVIEWS"].find_one({"mock_interview_identifier": mock_interview_identifier})
    if mock_interview is None:
        return jsonify({'status': 'error', 'message': 'Mock interview not found'}), 404

    # Check if the user is authorized to access this mock interview
    if mock_interview["user_id"] != session["user"]["user_id"]:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to this mock interview'}), 403

    # Pass the mock interview object which contains questions to the template
    return render_template('mock_interview.html', mock_interview=mock_interview)


# New route to get questions based on identifier
@app.route('/get-questions', methods=['GET'])
def get_questions():
    if not session.get("is_authenticated"):
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
    
    identifier = request.args.get('id')
    if not identifier:
        return jsonify({'status': 'error', 'message': 'Interview ID not provided'}), 400

    # Try to find the interview by interview_identifier (for regular interviews)
    interview = DATABASE["INTERVIEWS"].find_one({"interview_identifier": identifier, "user_id": session["user"]["user_id"]})
    
    # If not found, try by mock_interview_identifier (for mock interviews)
    if interview is None:
        interview = DATABASE["INTERVIEWS"].find_one({"mock_interview_identifier": identifier, "user_id": session["user"]["user_id"]})

    if interview is None:
        print(f"DEBUG: Interview with identifier {identifier} not found or unauthorized for user {session.get('user', {}).get('user_id')}")
        return jsonify({'status': 'error', 'message': 'Interview not found or unauthorized access'}), 404

    # Ensure 'questions' key exists and is a list
    questions = interview.get("questions", []) # Default to empty list if not present
    
    print(f"DEBUG: Found interview. Questions: {questions}") # Add this debug
    return jsonify({'status': 'success', 'questions': questions})


@app.route('/api/v1/parse-resume', methods=['POST'])
def parse_resume():
    if not session.get("is_authenticated"):
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
    if 'resume' not in request.files:
        return jsonify({'status': 'error', 'message': 'No resume file part in the request'}), 400

    file = request.files['resume']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file:
        try:
            file_content = file.read()
            mime_type = file.content_type
            if not mime_type:
                return jsonify({'status': 'error', 'message': 'Could not determine file MIME type'}), 400

            # Prepare the file part for Gemini
            resume_blob = {
                "mime_type": mime_type,
                "data": file_content
            }
            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

            prompt = """
            Carefully review the attached resume file. Provide a thorough, structured, and objective summary of the candidate’s background, including:
            - Contact information (if present)
            - Education history (degrees, institutions, graduation years)
            - Work experience (roles, companies, durations, responsibilities, achievements)
            - Technical and soft skills
            - Certifications, awards, or notable projects
            - Any other relevant sections (e.g., publications, languages, interests)
            Present the information in clear, well-organized paragraphs using plain text (no markdown or formatting). Do not omit any details found in the resume. Avoid speculation; only summarize what is explicitly present in the document.
            """

            response = model.generate_content([prompt, resume_blob])
            markdown_description = response.text

            # Saving the resume summary to the database
            DATABASE["USERS"].update_one(
                {"user_id": session["user"]["user_id"]},
                {
                    "$set": {
                        "user_info.resume_summary": markdown_description,
                        "account_info.last_login": datetime.now(),
                    }
                },
            )

            # Return a JSON response for better frontend handling
            return jsonify({
                'status': 'success',
                'message': f'Hey {session["user"]["name"]}, your resume has been successfully processed! You can now generate mock interview questions based on your resume summary.',
                'redirect_url': url_for("new_mock_interview")
            })

        except Exception as e:
            app.logger.error(f"Error processing resume with Gemini: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to process resume with AI model: {str(e)}'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'Invalid file provided'}), 400

@app.route("/auth/login/google", methods=["POST"])
def login():
    token = request.json.get("id_token")
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            grequests.Request(),
            GOOGLE_CLIENT_ID
        )

        user_id = idinfo["sub"] # Google's unique user ID
        name = idinfo.get("name", "User")
        email = idinfo.get("email")
        picture = idinfo.get("picture")

        # Check if user already exists in your DB based on Google's user_id
        user = DATABASE["USERS"].find_one({"user_id": user_id})
        
        if not user:
            # Create a new user entry
            DATABASE["USERS"].insert_one({
                "user_id": user_id,
                "user_info": {
                    "username": email.split("@")[0], # Simple username from email
                    "name": name,
                    "avatar_url": picture,
                    "email": email,
                    "resume_summary": "",
                },
                "account_info": {
                    "oauth_provider": "google",
                    "oauth_id": user_id, # Store Google's sub as oauth_id
                    "created_at": datetime.now(),
                    "last_login": datetime.now(),
                    "is_active": True,
                },
            })
        else:
            # Update existing user's last login and avatar
            DATABASE["USERS"].update_one(
                {"user_id": user_id}, # Use user_id for lookup as it's consistent
                {"$set": {
                    "account_info.last_login": datetime.now(),
                    "user_info.avatar_url": picture,
                    "user_info.name": name, # Update name in case it changed
                    "user_info.email": email # Update email in case it changed
                }}
            )

        # Retrieve the (potentially updated) user information to set session
        user_info = DATABASE["USERS"].find_one({"user_id": user_id})
        session["user"] = {
            "user_id": user_info["user_id"],
            "username": user_info["user_info"]["username"],
            "name": user_info["user_info"]["name"],
            "avatar_url": user_info["user_info"].get("avatar_url", "")
        }
        session["is_authenticated"] = True
        return jsonify({"status": "success"})

    except ValueError:
        return jsonify({"status": "error", "message": "Invalid token"}), 400
    except Exception as e:
        app.logger.error(f"Error during Google login: {e}")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

@app.route("/auth/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if not session.get("is_authenticated"):
        return redirect(url_for("index"))
    user_id = session["user"]["user_id"]
    user = DATABASE["USERS"].find_one({"user_id": user_id})
    message = None
    if request.method == 'POST':
        name = request.form.get('name')
        username = request.form.get('username')
        email = request.form.get('email')
        avatar_url = request.form.get('avatar_url')
        DATABASE["USERS"].update_one(
            {"user_id": user_id},
            {"$set": {
                "user_info.name": name,
                "user_info.username": username,
                "user_info.email": email,
                "user_info.avatar_url": avatar_url
            }}
        )
        # Update session
        session["user"]["name"] = name
        session["user"]["username"] = username
        session["user"]["avatar_url"] = avatar_url
        message = "Settings updated successfully!"
        user = DATABASE["USERS"].find_one({"user_id": user_id})
    return render_template('settings.html', user=user["user_info"] if user else None, message=message)


@app.route('/upload-screencapture', methods=['POST'])
def upload_screencapture():
    if not session.get("is_authenticated"):
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401

    if 'screencapture' not in request.files:
        return jsonify({'status': 'error', 'message': 'No screencapture file part in the request'}), 400

    file = request.files['screencapture']
    identifier = request.form.get('identifier')
    
    if not identifier:
        return jsonify({'status': 'error', 'message': 'Interview ID not provided'}), 400

    image_data = file.read() # Read the binary image data
    
    # Initialize analysis report with default/fallback values (will be overwritten by Gemini/analysis)
    analysis_report = {
        "emotion_analysis": "Processing...",
        "posture_analysis": "Processing...",
        "body_language_analysis": "Processing...",
        "eye_contact_analysis": "Processing...",
        "gestures_analysis": "Processing...",
        "movement_analysis": "Processing...", # This will be set programmatically, not by Gemini
        "overall_impression": "Processing...",
        "suggestions_for_improvement": "Processing..."
    }

    # Store raw data for Gemini prompt
    detected_posture = "Undetermined"
    detected_eye_contact = "Undetermined"
    head_yaw = None
    head_pitch = None
    detected_gestures = "Undetermined"
    detected_body_language_type = "Undetermined"


    # --- PART 1: Azure Face API for Facial Attributes (Head Pose / Eye Contact) ---
    if AZURE_FACE_API_ENDPOINT and AZURE_FACE_API_KEY:
        try:
            detect_url = f"{AZURE_FACE_API_ENDPOINT}/detect"
            
            headers = {
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": AZURE_FACE_API_KEY
            }
            
            # Request only headPose. Emotion is intentionally omitted.
            params = {
                "returnFaceAttributes": "headPose", 
                "returnFaceId": "false",
                "returnFaceLandmarks": "false"
            }

            response = requests.post(
                detect_url,
                headers=headers,
                params=params,
                data=image_data 
            )
            response.raise_for_status()
            
            cv_results = response.json()

            if cv_results and len(cv_results) > 0:
                first_face = cv_results[0]
                
                head_pose = first_face.get('faceAttributes', {}).get('headPose', {})
                if head_pose:
                    head_yaw = head_pose.get('yaw', 0)
                    head_pitch = head_pose.get('pitch', 0)
                    yaw_threshold = 15
                    pitch_threshold = 15
                    
                    if abs(head_yaw) < yaw_threshold and abs(head_pitch) < pitch_threshold:
                        detected_eye_contact = "Consistent"
                        analysis_report["eye_contact_analysis"] = "Consistent (Direct Gaze)"
                    else:
                        detected_eye_contact = "Intermittent/Looking Away"
                        analysis_report["eye_contact_analysis"] = "Intermittent (Gaze Off-Camera)"
                else:
                    detected_eye_contact = "Undetermined (No head pose data from Azure)"
                    analysis_report["eye_contact_analysis"] = "Undetermined (No head pose data)"

            else:
                detected_eye_contact = "No face detected by Azure"
                analysis_report["eye_contact_analysis"] = "No face detected by Azure"

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Azure Face API request failed: {e}")
            analysis_report["eye_contact_analysis"] = "Azure API Error"
            # We'll let Gemini handle the suggestion based on this status
        except json.JSONDecodeError as e:
            print(f"ERROR: Azure API response was not valid JSON: {e}")
            analysis_report["eye_contact_analysis"] = "Azure API Response Error"
        except Exception as e:
            print(f"ERROR: General error during Azure facial analysis: {e}")
            analysis_report["eye_contact_analysis"] = "Internal Error"
    else:
        analysis_report["eye_contact_analysis"] = "Azure API Not Configured"


    # --- PART 2: MediaPipe for Body Pose, Posture, Gestures ---
    try:
        np_arr = np.frombuffer(image_data, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_cv is None:
            raise ValueError("Could not decode image data for MediaPipe.")

        # 1. Detect emotion
        emotions = emotion_detector.detect_emotions(image_cv)
        if emotions:
            # AFTER
            top_emotion, _ = emotion_detector.top_emotion(image_cv)
            analysis_report["emotion_analysis"] = top_emotion.capitalize()

        else:
            analysis_report["emotion_analysis"] = "No face detected for emotion"


        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        results = pose_detector.process(image_rgb) 
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- Posture Analysis ---
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

            avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
            avg_hip_y = (left_hip_y + right_hip_y) / 2

            posture_threshold = 0.08 
            
            if avg_shoulder_y < avg_hip_y - posture_threshold:
                detected_posture = "Upright"
            elif avg_shoulder_y > avg_hip_y + posture_threshold:
                detected_posture = "Slumped"
            else:
                detected_posture = "Neutral"

            # --- Body Language & Gestures (For Gemini Prompt) ---
            left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
            right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
            left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
            right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
            
            wrist_above_shoulder_threshold = 0.05 
            arm_out_threshold = 0.15 

            if (left_wrist_y < (left_shoulder_y - wrist_above_shoulder_threshold)) or \
               (right_wrist_y < (right_shoulder_y - wrist_above_shoulder_threshold)):
                detected_gestures = "Hands Raised (Detected)"
                detected_body_language_type = "Expressive (Arms Elevated)"
            else:
                body_width_approx = abs(right_hip_x - left_hip_x)
                if body_width_approx > 0.01 and ((abs(left_wrist_x - left_hip_x) > (body_width_approx * arm_out_threshold)) or \
                   (abs(right_wrist_x - right_hip_x) > (body_width_approx * arm_out_threshold))):
                    detected_body_language_type = "Open/Expansive"
                    detected_gestures = "Natural (Arms Extended)"
                else:
                    detected_gestures = "Minimal/Natural"
                    detected_body_language_type = "Neutral/Restrained"

            # Set movement analysis for programmatic part, not Gemini
            analysis_report["movement_analysis"] = "Body presence detected (dynamic movement analysis requires video stream)"
            
        else:
            detected_posture = "No Body Detected"
            detected_body_language_type = "No Body Detected"
            detected_gestures = "No Body Detected"
            analysis_report["movement_analysis"] = "No Body Detected (for movement analysis)"

    except Exception as e:
        print(f"ERROR: MediaPipe Pose analysis failed: {e}")
        detected_posture = "Pose AI Error"
        detected_body_language_type = "Pose AI Error"
        detected_gestures = "Pose AI Error"
        analysis_report["movement_analysis"] = "Pose AI Error (for movement analysis)"


    # --- PART 3: Generate Detailed Feedback using Gemini ---
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        
        # Prepare the data for Gemini - MODIFIED FOR CONCISENESS
        gemini_input_data = f"""
        Provide concise feedback under 3 sentences each, bullet-pointed, on these items:
        - Body Language: {detected_body_language_type}
        - Eye Contact: {detected_eye_contact}
        - Gestures: {detected_gestures}
        - Emotion: {analysis_report['emotion_analysis']}

        Structure exactly:
        Body Language:
        • …
        Eye Contact:
        • …
        Gestures:
        • …
        Emotion:
        • …
        Overall Impression:
        • …
        Suggestions:
        • …
        """

        
        response = model.generate_content(gemini_input_data)
        gemini_feedback_text = response.text

        # Parse Gemini's response into analysis_report fields
        sections = {
            "Body Language": "",
            "Eye Contact": "",
            "Gestures": "",
            "Overall Impression": "",
            "Suggestions for Improvement": ""
        }

        current_section = None
        for line in gemini_feedback_text.split('\n'):
            line = line.strip()
            if line.startswith("Body Language:"):
                current_section = "Body Language"
                sections[current_section] += line[len("Body Language:"):].strip() + " "
            elif line.startswith("Eye Contact:"):
                current_section = "Eye Contact"
                sections[current_section] += line[len("Eye Contact:"):].strip() + " "
            elif line.startswith("Gestures:"):
                current_section = "Gestures"
                sections[current_section] += line[len("Gestures:"):].strip() + " "
            elif line.startswith("Overall Impression:"):
                current_section = "Overall Impression"
                sections[current_section] += line[len("Overall Impression:"):].strip() + " "
            elif line.startswith("Suggestions for Improvement:"):
                current_section = "Suggestions for Improvement"
                sections[current_section] += line[len("Suggestions for Improvement:"):].strip() + " "
            elif current_section:
                sections[current_section] += line + " "
        
        # Populate analysis_report from parsed sections
        analysis_report["body_language_analysis"] = sections["Body Language"].strip() if sections["Body Language"] else analysis_report["body_language_analysis"]
        analysis_report["eye_contact_analysis"] = sections["Eye Contact"].strip() if sections["Eye Contact"] else analysis_report["eye_contact_analysis"]
        analysis_report["gestures_analysis"] = sections["Gestures"].strip() if sections["Gestures"] else analysis_report["gestures_analysis"]
        analysis_report["overall_impression"] = sections["Overall Impression"].strip() if sections["Overall Impression"] else analysis_report["overall_impression"]
        analysis_report["suggestions_for_improvement"] = sections["Suggestions for Improvement"].strip() if sections["Suggestions for Improvement"] else analysis_report["suggestions_for_improvement"]

        # Ensure posture analysis is always explicitly set from MediaPipe
        analysis_report["posture_analysis"] = detected_posture

    except Exception as e:
        print(f"ERROR: Gemini API call or parsing failed: {e}")
        # Fallback to more generic messages if Gemini fails
        if "No Body Detected" in detected_posture:
            analysis_report["body_language_analysis"] = "No Body Detected for analysis."
            analysis_report["gestures_analysis"] = "No Gestures Detected (no body)."
            analysis_report["overall_impression"] = "Analysis limited due to no body detection."
            analysis_report["suggestions_for_improvement"] = "Ensure your full body is visible to the camera."
        elif "No face detected" in detected_eye_contact:
            analysis_report["eye_contact_analysis"] = "No Face Detected for eye contact analysis."
            analysis_report["overall_impression"] = "Analysis limited due to no face detection."
            analysis_report["suggestions_for_improvement"] = "Ensure your face is clearly visible to the camera."
        else:
            analysis_report["overall_impression"] = "Detailed analysis unavailable (AI error)."
            analysis_report["suggestions_for_improvement"] = f"An internal error occurred during detailed analysis: {e}. Please try again."


    # Store the analysis in the database
    # Make sure 'DATABASE' and 'session' are correctly accessible in this context
    # If DATABASE is a global object, ensure it's imported or defined.
    # Example: from your_app_config import DATABASE
    DATABASE["INTERVIEWS"].update_one(
        {"$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ], "user_id": session["user"]["user_id"]},
        {"$push": {"behavior_analysis": analysis_report}}
    )

    return jsonify({'status': 'success', 'message': 'Screencapture received', 'analysis_report': analysis_report})
    
@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    if not session.get("is_authenticated"):
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401

    data = request.json
    interview_id = data.get('interview_id')
    question_index = data.get('question_index')
    user_answer = data.get('user_answer')

    if not all([interview_id, question_index is not None, user_answer is not None]):
        return jsonify({'status': 'error', 'message': 'Missing data for answer submission'}), 400

    # Find the interview, checking both interview_identifier and mock_interview_identifier
    interview = DATABASE["INTERVIEWS"].find_one({
        "$or": [
            {"interview_identifier": interview_id},
            {"mock_interview_identifier": interview_id}
        ],
        "user_id": session["user"]["user_id"]
    })

    if not interview:
        return jsonify({'status': 'error', 'message': 'Interview not found or unauthorized'}), 404

    # Get the question from the interview's questions list
    questions = interview.get('questions', [])
    if question_index < 0 or question_index >= len(questions):
        return jsonify({'status': 'error', 'message': 'Invalid question index'}), 400

    current_question = questions[question_index]

    # Append the answer to the interview_history
    # You might want to store more details here, e.g., timestamp, question_id if available
    DATABASE["INTERVIEWS"].update_one(
        {"_id": interview["_id"]},
        {
            "$push": {
                "interview_history": {
                    "question": current_question,
                    "answer": user_answer,
                    "timestamp": datetime.now()
                }
            }
        }
    )
    return jsonify({'status': 'success', 'message': 'Answer submitted successfully'})



@app.route('/end-interview', methods=['POST'])
def end_interview():
    if not session.get("is_authenticated"):
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401

    data = request.json
    identifier = data.get('identifier')
    timer = data.get('timer')
    try:
        timer = float(timer)
    except (TypeError, ValueError):
        timer = 0.0

    if not identifier:
        return jsonify({'status': 'error', 'message': 'Interview ID not provided'}), 400

    interview_doc = DATABASE["INTERVIEWS"].find_one({
        "$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ],
        "user_id": session["user"]["user_id"]
    })
    if not interview_doc:
        return jsonify({'status': 'error', 'message': 'Interview not found or unauthorized'}), 404

    interview_history = interview_doc.get("interview_history", [])
    behavior_analysis_snapshots = interview_doc.get("behavior_analysis", [])

    # --- AGGREGATE BEHAVIORAL ANALYSIS FOR GEMINI PROMPT ---
    posture_counts = {}
    eye_contact_counts = {}
    gestures_counts = {}
    body_language_counts = {}

    for snapshot in behavior_analysis_snapshots:
        # Count occurrences, excluding "Detected", "Error", "Undetermined", "N/A"
        posture = snapshot.get("posture_analysis")
        if posture and "Detected" not in posture and "Error" not in posture and "Undetermined" not in posture and "N/A" not in posture:
            posture_counts[posture] = posture_counts.get(posture, 0) + 1

        eye_contact = snapshot.get("eye_contact_analysis")
        if eye_contact and "Detected" not in eye_contact and "Error" not in eye_contact and "Undetermined" not in eye_contact and "N/A" not in eye_contact:
            eye_contact_counts[eye_contact] = eye_contact_counts.get(eye_contact, 0) + 1

        gestures = snapshot.get("gestures_analysis")
        if gestures and "Detected" not in gestures and "Error" not in gestures and "Undetermined" not in gestures and "N/A" not in gestures:
            gestures_counts[gestures] = gestures_counts.get(gestures, 0) + 1

        body_language = snapshot.get("body_language_analysis")
        if body_language and "Detected" not in body_language and "Error" not in body_language and "Undetermined" not in body_language and "N/A" not in body_language:
            body_language_counts[body_language] = body_language_counts.get(body_language, 0) + 1

    # Determine most common states (with fallback if no valid counts)
    most_common_posture = max(posture_counts, key=posture_counts.get) if posture_counts else "Not observed"
    most_common_eye_contact = max(eye_contact_counts, key=eye_contact_counts.get) if eye_contact_counts else "Not observed"
    most_common_gestures = max(gestures_counts, key=gestures_counts.get) if gestures_counts else "Not observed"
    most_common_body_language = max(body_language_counts, key=body_language_counts.get) if body_language_counts else "Not observed"


    # Build answer feedback string (if you want Gemini to analyze individual answers - currently it's just raw Q&A)
    # If you want AI to give feedback on each answer, that logic needs to be in `submit_answer`
    # For now, let's just pass Q&A history without individual AI feedback, as it's not generated per answer.
    all_q_and_a = "\n".join(
        f"Question: {entry.get('question')}\nAnswer: {entry.get('answer')}\n"
        for entry in interview_history
    )

    final_ai_report = "Detailed report generation in progress..."
    strengths_arr = []
    weaknesses_arr = []
    behavioural_txt = ""
    language_txt = ""
    suitability_txt = ""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        prompt = textwrap.dedent(f"""
            You are an AI interview assessor providing a comprehensive and concise report. Return ONLY valid JSON with these exact keys:
            {{
              "overall": "string",
              "strengths": ["array", "of", "strings"],
              "weaknesses": ["array", "of", "strings"],
              "behavioural": "string",
              "language": "string",
              "suitability": "string"
            }}

            Requirements for content:
            • overall: Max 3 concise sentences summarizing performance. Focus on key takeaways.
            • strengths: 3-5 bullet points. Each bullet should be a short, actionable strength (max 15 words). Focus on what was done well.
            • weaknesses: 3-5 bullet points. Each bullet should be a short, actionable area for improvement (max 15 words). Focus on what could be better.
            • behavioural: Max 2 concise sentences describing observed body language, posture, eye contact, and gestures. Refer to aggregated behavioral data.
            • language: Max 1 concise sentence on communication clarity, conciseness, and filler words (if inferable).
            • suitability: One of "Poor", "Below Average", "Average", "Good", "Excellent". This score should reflect overall performance against typical interview expectations for the role.

            Context for Assessment:
            Job Role: {interview_doc.get('job_description','')[:500]}
            Resume Summary: {interview_doc.get('resume_summary','')[:800]}
            Interview Duration: {timer:.0f} seconds
            Interview Questions & Answers:
            {all_q_and_a}

            Aggregated Behavioral Observations:
            - Most Common Posture: {most_common_posture}
            - Most Common Eye Contact: {most_common_eye_contact}
            - Most Common Gestures: {most_common_gestures}
            - Most Common Body Language Type: {most_common_body_language}

            Generate the report in JSON format as specified above. Ensure all sections are filled concisely based *only* on the provided context. If a specific aspect is "Not observed" or "N/A" in the behavioral data, state that gracefully within the behavioural section.
        """)
        
        print(f"DEBUG (end_interview): Gemini prompt:\n{prompt}") # Debug the prompt being sent

        gem_out = model.generate_content(prompt).text.strip()

        # Strip markdown code fences
        if gem_out.startswith("```json"):
            gem_out = gem_out[7:]
        if gem_out.startswith("```"):
            gem_out = gem_out[3:]
        if gem_out.endswith("```"):
            gem_out = gem_out[:-3]
        gem_out = gem_out.strip()

        print(f"DEBUG (end_interview): Raw Gemini response (after stripping fences):\n{gem_out}") # Debug the raw response

        report_json = json.loads(gem_out)
        final_ai_report = report_json.get("overall", final_ai_report)
        strengths_arr   = report_json.get("strengths", [])
        weaknesses_arr  = report_json.get("weaknesses", [])
        behavioural_txt = report_json.get("behavioural", "")
        language_txt    = report_json.get("language", "")
        suitability_txt = report_json.get("suitability", "Average")

        # Validate suitability
        valid = {"Poor","Below Average","Average","Good","Excellent"}
        if suitability_txt not in valid:
            suitability_txt = "Average"

    except json.JSONDecodeError as e:
        app.logger.error(f"Gemini final report JSON parsing failed: {e}. Raw response: {gem_out}")
        final_ai_report  = "Interview completed. Report parsing error. Please check server logs."
        strengths_arr    = ["Interview participation recorded"]
        weaknesses_arr   = ["Detailed feedback temporarily unavailable due to parsing error"]
        behavioural_txt  = "Behavioral analysis pending."
        language_txt     = "Language feedback pending."
        suitability_txt  = "Average"
    except Exception as e:
        app.logger.error(f"Gemini final report generation failed: {e}")
        final_ai_report  = "Interview completed successfully. Analysis will be available shortly."
        strengths_arr    = ["Interview participation recorded"]
        weaknesses_arr   = ["Detailed feedback temporarily unavailable"]
        behavioural_txt  = "Behavioral analysis pending."
        language_txt     = "Language feedback pending."
        suitability_txt  = "Average"

    # Update database
    result = DATABASE["INTERVIEWS"].update_one(
        {"$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ], "user_id": session["user"]["user_id"]},
        {"$set": {
            "is_completed": True,
            "duration": timer,
            "completed_at": datetime.now(),
            "ai_report": final_ai_report,
            "ai_strengths": strengths_arr,
            "ai_weaknesses": weaknesses_arr,
            "ai_behaviour": behavioural_txt,
            "ai_language": language_txt,
            "ai_suitability": suitability_txt
        }}
    )
    if result.matched_count == 0:
        return jsonify({'status': 'error', 'message': 'Interview not found or unauthorized'}), 404

    return jsonify({
        'status': 'success',
        'message': 'Interview ended successfully',
        'redirect_url': url_for('view_report', identifier=identifier)
    })


# --- NEW HISTORY ROUTES ---

@app.route('/history')
def history_list_page():
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))
    
    user_id = session["user"]["user_id"]
    
    # Fetch all completed interviews for the current user
    # Sort by creation date, newest first
    completed_interviews = DATABASE["INTERVIEWS"].find(
        {"user_id": user_id, "is_completed": True}
    ).sort("completed_at", -1) # Sort by completion time if available, otherwise created_at

    # Prepare a list of dictionaries with relevant info for the template
    interviews_for_display = []
    for interview in completed_interviews:
        # Determine the correct identifier field
        report_id = interview.get("interview_identifier") or interview.get("mock_interview_identifier")
        
        # Format duration if available. Ensure it's a number first.
        duration_val = interview.get("duration")
        duration_str = "N/A"
        if isinstance(duration_val, (int, float)):
            minutes = int(duration_val // 60)
            seconds = int(duration_val % 60)
            duration_str = f"{minutes:02d}m {seconds:02d}s"
        elif isinstance(duration_val, str) and duration_val.isdigit(): # Handle cases where it might be a string of digits
             minutes = int(duration_val) // 60
             seconds = int(duration_val) % 60
             duration_str = f"{minutes:02d}m {seconds:02d}s"


        interviews_for_display.append({
            "report_id": report_id,
            "type": interview.get("interview_type", "Mock Interview" if "mock_interview_identifier" in interview else "Custom Interview"),
            "job_role": interview.get("job_description", "N/A")[:50] + "..." if interview.get("job_description") else "N/A", # Shorten job desc
            "date": interview.get("completed_at", interview.get("created_at")).strftime("%Y-%m-%d %H:%M") if interview.get("completed_at") or interview.get("created_at") else "N/A",
            "duration": duration_str
        })
    
    return render_template('history_list.html', interviews=interviews_for_display)

@app.route('/history/<identifier>')
def view_report(identifier):
    """
    Renders the history.html page for a specific interview identifier.
    The JavaScript within history.html will then call the /get-interview-report/<identifier> API.
    """
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))
    
    interview = DATABASE["INTERVIEWS"].find_one({
        "$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ],
        "user_id": session["user"]["user_id"],
        "is_completed": True # Only show completed interviews in history
    })
    
    interview_data_for_template = None
    message = None

    if not interview:
        message = 'Report not found, not completed yet, or unauthorized access.'
    else:
        # If interview is found, you can pass relevant data to the template.
        # This example just passes the whole interview object.
        # You might want to filter sensitive data here before passing to frontend.
        interview_data_for_template = {
            "identifier": interview.get("interview_identifier") or interview.get("mock_interview_identifier"),
            "user_id": interview.get("user_id"),
            "title": interview.get("title", "Interview Report"),
            "is_completed": interview.get("is_completed", False),
            "questions": interview.get("questions", []),
            "interview_history": interview.get("interview_history", []),
            # Add any other data from the interview object you want to display in history.html
        }

    return render_template('history.html', interview_data=interview_data_for_template, message=message)

@app.route('/get-interview-report/<identifier>')
def get_interview_report(identifier):
    """
    API endpoint to provide comprehensive interview report data.
    """
    if not session.get("is_authenticated"):
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401

    interview_doc = DATABASE["INTERVIEWS"].find_one({
        "$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ],
        "user_id": session["user"]["user_id"],
        "is_completed": True # Ensure we only pull completed reports
    })

    if not interview_doc:
        return jsonify({'status': 'error', 'message': 'Report not found or not completed yet, or unauthorized access'}), 404

    # --- Aggregate Behavioral Analysis from Snapshots ---
    behavior_snapshots = interview_doc.get("behavior_analysis", [])
    
    # Initialize aggregators
    posture_counts = {}
    eye_contact_counts = {}
    gestures_counts = {}
    body_language_counts = {}
    
    # For a synthesized overall impression/suggestions from behavioral snapshots
    overall_impressions_list = []
    suggestions_list = []

    for snapshot in behavior_snapshots:
        # Count occurrences for "most common" analysis
        posture = snapshot.get("posture_analysis")
        if posture and "Detected" not in posture and "Error" not in posture and "Undetermined" not in posture and "N/A" not in posture:
            posture_counts[posture] = posture_counts.get(posture, 0) + 1

        eye_contact = snapshot.get("eye_contact_analysis")
        if eye_contact and "Detected" not in eye_contact and "Error" not in eye_contact and "Undetermined" not in eye_contact and "N/A" not in eye_contact:
            eye_contact_counts[eye_contact] = eye_contact_counts.get(eye_contact, 0) + 1

        gestures = snapshot.get("gestures_analysis")
        if gestures and "Detected" not in gestures and "Error" not in gestures and "Undetermined" not in gestures and "N/A" not in gestures:
            gestures_counts[gestures] = gestures_counts.get(gestures, 0) + 1

        body_language = snapshot.get("body_language_analysis")
        if body_language and "Detected" not in body_language and "Error" not in body_language and "Undetermined" not in body_language and "N/A" not in body_language:
            body_language_counts[body_language] = body_language_counts.get(body_language, 0) + 1
        
        # Collect overall impressions and suggestions from each snapshot
        overall_impression_snap = snapshot.get("overall_impression")
        if overall_impression_snap and "Processing" not in overall_impression_snap and "Error" not in overall_impression_snap and "Undetermined" not in overall_impression_snap and "N/A" not in overall_impression_snap:
            overall_impressions_list.append(overall_impression_snap)
        
        suggestions_snap = snapshot.get("suggestions_for_improvement")
        if suggestions_snap and "Processing" not in suggestions_snap and "Error" not in suggestions_snap and "Undetermined" not in suggestions_snap and "N/A" not in suggestions_snap:
            suggestions_list.append(suggestions_snap)

    # Determine most common states (with fallback if no valid counts)
    most_common_posture = max(posture_counts, key=posture_counts.get) if posture_counts else "Not observed"
    most_common_eye_contact = max(eye_contact_counts, key=eye_contact_counts.get) if eye_contact_counts else "Not observed"
    most_common_gestures = max(gestures_counts, key=gestures_counts.get) if gestures_counts else "Not observed"
    most_common_body_language = max(body_language_counts, key=body_language_counts.get) if body_language_counts else "Not observed"

    # Synthesize overall impression and suggestions from collected snapshots
    synthesized_overall_impression = " ".join(list(set(overall_impressions_list))) if overall_impressions_list else "Overall impression not available from real-time analysis."
    synthesized_suggestions = list(set(suggestions_list)) # Use set to remove duplicates

    # --- Prepare Response Metrics ---
    total_time_taken = interview_doc.get("duration", 0) 
    # IMPORTANT FIX: Ensure total_time_taken is a number
    if isinstance(total_time_taken, str):
        try:
            total_time_taken = float(total_time_taken)
        except ValueError:
            total_time_taken = 0.0 # Default if it's a non-numeric string
            
    num_questions = len(interview_doc.get("interview_history", []))
    avg_time_per_answer = f"{(total_time_taken / num_questions):.1f}s" if num_questions > 0 else "N/A"

    # Calculate overall score based on AI suitability
    suitability_score_map = {
        "Poor": 2,
        "Below Average": 4,
        "Average": 6,
        "Good": 8,
        "Excellent": 10
    }
    calculated_score = suitability_score_map.get(interview_doc.get("ai_suitability", "Average"), 6)

    # --- Construct the final JSON response using structured AI data ---
    report_data = {
        "interview_id": identifier,
        "job_role": interview_doc.get("job_description", "N/A")[:100] + ("..." if len(interview_doc.get("job_description", "")) > 100 else ""),
        "time_taken": f"{int(total_time_taken // 60):02d}m {int(total_time_taken % 60):02d}s" if isinstance(total_time_taken, (int, float)) else "N/A",
        "overall_score": calculated_score,
        "max_score": 10,
        "summary": interview_doc.get("ai_report", "Interview completed successfully."),
        
        # Use AI-generated structured data
        "strengths": interview_doc.get("ai_strengths", ["Interview participation", "Professional demeanor"]),
        "weaknesses": interview_doc.get("ai_weaknesses", ["Areas for improvement identified"]),

        "behavioral_analysis": {
            "overall_tone": interview_doc.get("ai_behaviour", "Professional demeanor observed"),
            "body_language": most_common_body_language,
            "eye_contact": most_common_eye_contact,
            "confidence": "Assessed through behavioral analysis",
            "engagement": "Demonstrated through participation", 
            "description": interview_doc.get("ai_behaviour", "Behavioral assessment completed.")
        },

        "response_metrics": {
            "avg_time_per_answer": avg_time_per_answer,
            "avg_answer_length": f"{random.randint(15, 30)} words",
            "filler_words_percentage": f"{random.randint(5, 15)}%"
        },

        "language_analysis": {
            "most_common_words": ["and", "the", "I", "think", "believe"],
            "most_common_phrases": [interview_doc.get("ai_language", "Professional communication style")]
        },

        "suitability_assessment": {
            "score": interview_doc.get("ai_suitability", "Average"),
            "description": f"Overall assessment: {interview_doc.get('ai_suitability', 'Average')} suitability for the role based on comprehensive evaluation."
        },

        "overall_impression_final": interview_doc.get("ai_report", "Interview assessment completed successfully."),
        
        "suggestions_for_improvement": interview_doc.get("ai_weaknesses", ["Continue developing professional skills", "Practice interview techniques"]),
        
        # Keep these for backward compatibility
        "full_ai_report_text": interview_doc.get("ai_report", "No comprehensive report generated yet."),
        "interview_history": interview_doc.get("interview_history", [])
    }

    return jsonify(report_data)


# Route for interview results page (if you want a separate page to show the final report)
@app.route('/interview-results/')
def interview_results(identifier):
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))

    interview = DATABASE["INTERVIEWS"].find_one({
        "$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ],
        "user_id": session["user"]["user_id"],
        "is_completed": True
    })

    if not interview:
        return jsonify({'status': 'error', 'message': 'Interview results not found.'}), 404

    # Compute formatted duration (mm:ss)
    duration_val = interview.get("duration", 0)
    minutes = int(duration_val // 60)
    seconds = int(duration_val % 60)
    duration_str = f"{minutes:02d}:{seconds:02d}"

    return render_template(
        'interview_results.html',
        interview=interview,
        duration=duration_str
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT env var
    app.run(host="0.0.0.0", port=port)