import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn
import pypdf
import io
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


BASE_PROMPT = """
You are a rigorous but fair Software Engineering Mock Interviewer. 
Your goal is to test the candidate's knowledge based on their specific resume and background.
CRITICAL RULES:
1. Do not break character. 
2. ASK ONLY ONE QUESTION AT A TIME. Never bundle multiple questions together.
3. Keep your responses EXTREMELY concise. Maximum 2 to 3 short sentences per response. 
4. Make it feel like a real, spoken back-and-forth conversation.
5. If the user's answer is vague, politely ask them to clarify or provide a specific example.
"""

app = FastAPI(title="SkillSync AI Backend")

# ✅ FIXED: Added production URLs — replace the Vercel URL after deploying frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",                   # Local development
        "https://your-app.vercel.app",             # ← REPLACE with your actual Vercel URL after deploy
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

users_db = {}
chat_sessions = {}

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserProfileUpdate(BaseModel):
    email: str
    name: str
    phone: str = ""
    college: str = ""
    job_preference: str = ""
    new_password: str = ""

class ChatMessage(BaseModel):
    email: str
    user_message: str

class AnalyzeRequest(BaseModel):
    email: str
    transcript: str

@app.post("/api/register")
async def register_user(user: UserRegister):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    users_db[user.email] = {
        "name": user.name, "email": user.email, "password": user.password,
        "phone": "", "college": "", "job_preference": "", "blueprint": "",
        "reports": []
    }
    return {"message": "Success", "user": users_db[user.email]}

@app.post("/api/login")
async def login_user(user: UserLogin):
    db_user = users_db.get(user.email)
    if not db_user or db_user["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Success", "user": db_user}

@app.post("/api/update_profile")
async def update_profile(profile: UserProfileUpdate):
    if profile.email not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    user = users_db[profile.email]
    user["name"] = profile.name
    user["phone"] = profile.phone
    user["college"] = profile.college
    user["job_preference"] = profile.job_preference
    if profile.new_password.strip():
        user["password"] = profile.new_password
    return {"message": "Success", "user": user}

@app.post("/api/upload_resume")
async def upload_resume(email: str = Form(...), file: UploadFile = File(...)):
    if email not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(await file.read()))
        resume_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])

        analysis_model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Analyze this resume and create a brief 'Interview Blueprint'. List the candidate's top 3 skills and suggest 3 difficult interview questions to ask them based on their projects.\n\nRESUME:\n{resume_text}"
        blueprint_response = analysis_model.generate_content(prompt)

        users_db[email]["blueprint"] = blueprint_response.text
        if email in chat_sessions:
            del chat_sessions[email]

        return {"message": "Blueprint generated", "blueprint": blueprint_response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to parse resume")

@app.post("/api/chat")
async def chat_with_agent(message: ChatMessage):
    user_email = message.email
    if user_email not in chat_sessions:
        blueprint = users_db.get(user_email, {}).get("blueprint", "No specific background provided.")
        custom_system_prompt = f"{BASE_PROMPT}\n\nCANDIDATE BLUEPRINT (Use this to guide your questions):\n{blueprint}"
        custom_model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=custom_system_prompt)
        chat_sessions[user_email] = custom_model.start_chat(history=[])

    try:
        session = chat_sessions[user_email]
        response = session.send_message(message.user_message)
        return {"reply": response.text, "status": "success"}
    except Exception as e:
        return {"reply": "Sorry, I lost my connection. Could you repeat that?", "status": "error"}

@app.post("/api/analyze_interview")
async def analyze_interview(req: AnalyzeRequest):
    if req.email not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    analysis_model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    You are an expert tech recruiter. Analyze this mock interview transcript. 
    You MUST respond in pure JSON format. Do not use markdown blocks like ```json.
    Use this exact JSON structure:
    {{
        "overall_score": 85,
        "metrics": {{
            "communication": 80,
            "technical_depth": 90,
            "confidence": 85
        }},
        "overall_impression": "Write a 3 sentence summary here.",
        "detailed_analysis": "Write a detailed paragraph here.",
        "areas_for_improvement": ["Point 1", "Point 2", "Point 3"]
    }}

    TRANSCRIPT:
    {req.transcript}
    """

    try:
        response = analysis_model.generate_content(prompt)

        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()

        parsed_report = json.loads(raw_text)

        if "reports" not in users_db[req.email]:
            users_db[req.email]["reports"] = []

        report_num = len(users_db[req.email]["reports"]) + 1
        new_report = {
            "id": report_num,
            "title": f"Mock Interview {report_num:02d}",
            "content": parsed_report
        }
        users_db[req.email]["reports"].insert(0, new_report)

        if req.email in chat_sessions:
            del chat_sessions[req.email]

        return {"message": "Report generated", "reports": users_db[req.email]["reports"]}
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")