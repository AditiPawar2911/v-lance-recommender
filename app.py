import os
import json
import openai
from openai import OpenAI
from datetime import datetime, timedelta
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from collections import Counter
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.preprocessing import MinMaxScaler
import requests
from transformers import pipeline
from datetime import datetime,timezone

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///projects.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load CSV and initialize CrossEncoder model
df = pd.read_csv('final_file3.csv')
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Database Models
class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    deadline_days = db.Column(db.Integer, nullable=False)
    estimated_duration = db.Column(db.Integer, nullable=False)
    window_open = db.Column(db.DateTime, nullable=False)
    window_close = db.Column(db.DateTime, nullable=False)
    allowed_users = db.Column(db.Text, nullable=False)
    assigned_to = db.Column(db.String(100), nullable=True)
    actual_days = db.Column(db.Integer, nullable=True)
    assigned_rank = db.Column(db.Integer, nullable=True)
    submissions = db.relationship('Submission', backref='project', lazy=True)
    subtasks = db.relationship('Subtask', backref='project', lazy=True)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    user = db.Column(db.String(100), nullable=False)
    proposed_days = db.Column(db.Integer, nullable=False)
    __table_args__ = (db.UniqueConstraint('project_id', 'user', name='unique_submission'),)

class Subtask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(50), nullable=False)

# Helper Functions
def duration(project_title):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key="
    headers = {"Content-Type": "application/json"}
    prompt = (
        f'For the project titled "{project_title}", estimate the approximate duration required '
        f'to complete it with an adequate number of people. Provide the answer only in number of days.'
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return int(result['candidates'][0]['content']['parts'][0]['text'])
    print(f"Gemini API Error: {response.status_code}")
    return None

def create_profile_text(row):
    return f"""
    Professional Profile:
    About: {row['About']}
    Current Role: {row['Current Role(s)']}
    Experience: {row['Tenure at Company']} at company, {row['Seniority']} level
    Skills: {row['Description']}
    Industry: {row['Industry']}
    Function: {row['Job Function']}
    """

df['full_profile'] = df.apply(create_profile_text, axis=1)

def parse_seniority(row):
    seniority_mapping = {
        'director': 4, 'vp': 4.5, 'junior': 1, 'manager': 3, 'senior': 2.5, 'cxo': 5
    }
    text = f"{row['Seniority']} {row['Current Role(s)']}".lower()
    tenure = sum([int(n) for n in str(row['Tenure at Company']).split() if n.isdigit()][:1])
    score = max([v for k, v in seniority_mapping.items() if k in text], default=1)
    return min(score + (tenure // 5) * 0.5, 5)

def check_requirements(row, years_experience):
    tenure_years = sum([int(n) for n in str(row['Tenure at Company']).split() if n.isdigit()][:1])
    requirements = {
        'experience': 1 if tenure_years >= years_experience else 0,
        'team_lead': any(x in str(row['Current Role(s)']).lower() 
                         for x in ['manager', 'lead', 'director', 'vp', 'head']),
        'saas': 'saas' in str(row['Industry']).lower(),
    }
    return sum(requirements.values()) / len(requirements)

def stress_susceptibility(row, deadline):
    if deadline > 14:
        return 0
    stress_prone = ['ISFJ', 'INFP', 'ENFP']
    stress_resistant = ['ESTJ', 'ENTJ']
    mbti = str(row['MBTI_Type']).strip().upper()
    if mbti in stress_prone:
        return -0.1
    elif mbti in stress_resistant:
        return 0.05
    return 0

def get_top_professionals(skill, experience_area, years_experience, deadline):
    project_description = f"""
    We need senior {skill} design leaders with {experience_area} experience.
    Requires {years_experience}+ years in digital {experience_area}, team management experience,
    and a strong portfolio.
    """
    df['semantic_score'] = df['full_profile'].apply(lambda x: model.predict([(x, project_description)])[0])
    df['seniority_score'] = df.apply(parse_seniority, axis=1)
    df['requirement_score'] = df.apply(lambda row: check_requirements(row, years_experience), axis=1)
    scaler = MinMaxScaler()
    df[['semantic_norm', 'seniority_norm', 'requirement_norm']] = scaler.fit_transform(
        df[['semantic_score', 'seniority_score', 'requirement_score']]
    )
    df['compatibility'] = (0.6 * df['semantic_norm'] + 
                           0.3 * df['seniority_norm'] + 
                           0.1 * df['requirement_norm'] + 
                           df.apply(lambda row: stress_susceptibility(row, deadline), axis=1))
    return df.nlargest(5, 'compatibility')[['LinkedIn Name', 'compatibility']].to_dict('records')

def generate_subtasks_local(project_prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key="
    headers = {"Content-Type": "application/json"}
    prompt = f"Break this project into 5 subtasks: {project_prompt}. Give specific tasks that can be done in this project. Return a numbered list."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        generated_text = result['candidates'][0]['content']['parts'][0]['text']
        subtasks = [line.strip() for line in generated_text.split('\n') if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]
        return subtasks
    print(f"Gemini API Error: {response.status_code}")
    return None

def classify_task(task):
    categories = [
        "Urgent & Important (Do First)",
        "Not Urgent but Important (Schedule)",
        "Urgent but Not Important (Delegate)",
        "Not Urgent & Not Important (Eliminate)"
    ]
    result = classifier(task, categories, multi_label=False)
    return result['labels'][0]

def arrange_tasks_eisenhower(subtasks):
    matrix = {
        "Urgent & Important (Do First)": [],
        "Not Urgent but Important (Schedule)": [],
        "Urgent but Not Important (Delegate)": [],
        "Not Urgent & Not Important (Eliminate)": [],
        "Uncategorized": []
    }
    categories = list(matrix.keys())
    current_index = 0
    for task in subtasks:
        category = classify_task(task)
        if len(matrix[category]) >= 2:
            current_index = (current_index + 1) % len(categories)
            category = categories[current_index]
        matrix[category].append(task)
    return matrix

# Routes
@app.route('/')
def home():
    projects = Project.query.all()
    project_list = []
    current_time = datetime.now()
    for project in projects:
        deadline_date = project.window_open + timedelta(seconds=project.deadline_days)
        latest_start_date = deadline_date - timedelta(seconds=project.estimated_duration)
        project_list.append({
            'name': project.name,
            'latest_start_date': latest_start_date.isoformat(),
            'assigned_to': project.assigned_to,
            'actual_days': project.actual_days,
            'estimated_duration': project.estimated_duration,
            'window_close': project.window_close.isoformat() if project.window_close else None,
            'current_time': current_time.isoformat()
        })
    sorted_projects = sorted(project_list, key=lambda x: x['latest_start_date'])
    return render_template('home.html', projects=sorted_projects)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in df['LinkedIn Name'].values and password == 'password':
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/create_project', methods=['GET'])
def create_project_form():
    return render_template('create_project.html')

@app.route('/create_project', methods=['POST'])
def create_project():
    project_names = request.form.getlist('project_name[]')
    deadlines = request.form.getlist('deadline[]')
    skills = request.form.getlist('skill[]')
    experience_areas = request.form.getlist('experience_area[]')
    years_experiences = request.form.getlist('years_experience[]')
    for i in range(len(project_names)):
        name = project_names[i].strip()
        if not name or Project.query.filter_by(name=name).first():
            flash(f"Project '{name}' skipped: empty or already exists.", 'warning')
            continue
        deadline_days = int(deadlines[i])
        skill = skills[i]
        experience_area = experience_areas[i]
        years_experience = int(years_experiences[i])
        allowed_users = get_top_professionals(skill, experience_area, years_experience, deadline_days)
        estimated_duration = duration(name)
        if estimated_duration is None:
            flash(f"Failed to estimate duration for '{name}'.", 'danger')
            continue
        window_open = datetime.now()
        window_close = window_open + timedelta(seconds=deadline_days / 10.0)
        project = Project(
            name=name,
            deadline_days=deadline_days,
            estimated_duration=estimated_duration,
            window_open=window_open,
            window_close=window_close,
            allowed_users=json.dumps(allowed_users),
            assigned_to=None,
            actual_days=None,
            assigned_rank=None
        )
        db.session.add(project)
    db.session.commit()
    flash('Projects created successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/project/<project_name>', methods=['GET', 'POST'])
def project_status(project_name):
    project = Project.query.filter_by(name=project_name).first()
    if not project:
        flash('Project not found.', 'danger')
        return "Project not found", 404
    if request.method == 'POST':
        if 'actual_days' in request.form and session.get('username') == project.assigned_to:
            try:
                actual_days = int(request.form['actual_days'])
                if actual_days > 0:
                    project.actual_days = actual_days
                    db.session.commit()
                    flash('Actual days updated successfully!', 'success')
                else:
                    flash('Actual days must be a positive integer.', 'danger')
            except ValueError:
                flash('Invalid input for actual days.', 'danger')
        else:
            flash('Only the assigned user can update actual days.', 'danger')
    now = datetime.now()
    if now > project.window_close and project.assigned_to is None:
        submissions = Submission.query.filter_by(project_id=project.id).all()
        if submissions:
            min_days = min(sub.proposed_days for sub in submissions)
            candidates = [sub.user for sub in submissions if sub.proposed_days == min_days]
            project.assigned_to = candidates[0]
            allowed_users = json.loads(project.allowed_users)
            sorted_users = sorted(allowed_users, key=lambda x: x['compatibility'], reverse=True)
            for rank, user_dict in enumerate(sorted_users, start=1):
                if user_dict['LinkedIn Name'] == project.assigned_to:
                    project.assigned_rank = rank
                    break
            db.session.commit()
            subtasks = generate_subtasks_local(project.name)
            print(f"Generated subtasks: {subtasks}")
            if subtasks:
                matrix = arrange_tasks_eisenhower(subtasks)
                print(f"Eisenhower Matrix: {matrix}")
                for category, tasks in matrix.items():
                    for task in tasks:
                        subtask = Subtask(project_id=project.id, description=task, category=category)
                        db.session.add(subtask)
                db.session.commit()
                print("Subtasks saved")
    submissions = Submission.query.filter_by(project_id=project.id).all()
    submission_list = []
    allowed_users = json.loads(project.allowed_users)
    for sub in submissions:
        user_score = next((u['compatibility'] for u in allowed_users if u['LinkedIn Name'] == sub.user), None)
        submission_list.append({
            'user': sub.user,
            'proposed_days': sub.proposed_days,
            'compatibility_score': user_score
        })
    is_window_open = now < project.window_close
    subtasks = Subtask.query.filter_by(project_id=project.id).all()
    category_counts = Counter([sub.category for sub in subtasks])
    return render_template('project_status.html', project=project, submissions=submission_list, 
                          allowed_users=[u['LinkedIn Name'] for u in allowed_users], 
                          is_window_open=is_window_open, now=now, subtasks=subtasks, 
                          category_data=category_counts)

@app.route('/project/<project_name>/submit', methods=['POST'])
def submit_proposal(project_name):
    if 'username' not in session:
        flash('You must be logged in to submit a proposal.', 'danger')
        return "You must be logged in", 403
    user = session['username']
    project = Project.query.filter_by(name=project_name).first()
    if not project:
        flash('Project not found.', 'danger')
        return "Project not found", 404
    if datetime.now() > project.window_close:
        flash('Submission window is closed.', 'warning')
        return "Submission window is closed", 403
    allowed_users = json.loads(project.allowed_users)
    allowed_names = [u['LinkedIn Name'] for u in allowed_users]
    if user not in allowed_names:
        flash('You are not allowed to submit for this project.', 'danger')
        return "You are not allowed to submit for this project", 403
    proposed_days = int(request.form['proposed_days'])
    if proposed_days > project.deadline_days:
        flash(f"Proposed days ({proposed_days}) cannot exceed deadline ({project.deadline_days} days).", 'danger')
        return f"Proposed days ({proposed_days}) cannot exceed deadline ({project.deadline_days} days)", 400
    submission = Submission.query.filter_by(project_id=project.id, user=user).first()
    if submission:
        submission.proposed_days = proposed_days
        flash('Proposal updated successfully!', 'success')
    else:
        submission = Submission(project_id=project.id, user=user, proposed_days=proposed_days)
        db.session.add(submission)
        flash('Proposal submitted successfully!', 'success')
    db.session.commit()
    return redirect(url_for('project_status', project_name=project_name))

@app.route('/metrics')
def metrics():
    assigned_projects = Project.query.filter(Project.assigned_to != None).all()
    ranks = [p.assigned_rank for p in assigned_projects if p.assigned_rank is not None]
    rank_counts = Counter(ranks)
    rank_data = {str(i): rank_counts.get(i, 0) for i in range(1, 6)}

    all_subtasks = Subtask.query.all()
    category_counts = Counter([sub.category for sub in all_subtasks])

    proposal_data = []
    for project in Project.query.all():
        allowed_users = json.loads(project.allowed_users)
        submissions = Submission.query.filter_by(project_id=project.id).all()
        for sub in submissions:
            user_score = next((u['compatibility'] for u in allowed_users if u['LinkedIn Name'] == sub.user), None)
            if user_score is not None and project.estimated_duration > 0:
                ratio = sub.proposed_days / project.estimated_duration
                proposal_data.append({'compatibility': user_score, 'ratio': ratio})

    actual_vs_predicted = []
    accuracy_count = 0
    total_completed = 0
    assignment_efficiency = []
    for project in Project.query.filter(Project.actual_days != None).all():
        if project.estimated_duration > 0:
            actual_vs_predicted.append({
                'project_name': project.name,
                'predicted': project.estimated_duration,
                'actual': project.actual_days
            })
            total_completed += 1
            # Accuracy: within ±20% of predicted
            lower_bound = project.estimated_duration * 0.8
            upper_bound = project.estimated_duration * 1.2
            if lower_bound <= project.actual_days <= upper_bound:
                accuracy_count += 1
            # Assignment Efficiency: compare assigned user's proposed days to actual
            if project.assigned_to:
                submission = Submission.query.filter_by(project_id=project.id, user=project.assigned_to).first()
                if submission:
                    assignment_efficiency.append({
                        'project_name': project.name,
                        'proposed': submission.proposed_days,
                        'actual': project.actual_days
                    })

    accuracy_percentage = (accuracy_count / total_completed * 100) if total_completed > 0 else 0

    return render_template('metrics.html', rank_data=rank_data, category_data=category_counts, 
                          proposal_data=proposal_data, actual_vs_predicted=actual_vs_predicted, 
                          accuracy_percentage=accuracy_percentage, assignment_efficiency=assignment_efficiency)

# Initialize the database and run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)