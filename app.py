#!/usr/bin/env python3
"""
Flask API for Stock Portfolio Symbol Enrichment
Simple implementation with in-memory task management
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import threading
import tempfile
import os
import io
from datetime import datetime
import logging

from src.enrichment_wrapper import EnrichmentWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# In-memory task storage
tasks = {}

# Initialize enrichment wrapper
enrichment_wrapper = EnrichmentWrapper()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.route('/upload', methods=['POST'])
def upload_csv():
    """Upload CSV file and start enrichment process"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return {"error": "No file uploaded"}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {"error": "No file selected"}, 400
        
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            return {"error": "File must be a CSV"}, 400
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        tasks[task_id] = {
            "status": "started",
            "progress": {
                "processed": 0,
                "total": 0,
                "percentage": 0,
                "current_company": "",
                "successful": 0,
                "failed": 0
            },
            "message": "Initializing...",
            "created_at": datetime.now().isoformat(),
            "result_file": None,
            "error": None
        }
        
        # Read file content
        file_content = file.read()
        
        # Start background processing
        thread = threading.Thread(
            target=process_csv_background,
            args=(task_id, file_content, file.filename)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started enrichment task {task_id} for file: {file.filename}")
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "CSV upload successful, enrichment started"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return {"error": f"Upload failed: {str(e)}"}, 500

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the current status of an enrichment task"""
    try:
        if task_id not in tasks:
            return {"error": "Task not found"}, 404
        
        task = tasks[task_id]
        
        return {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "created_at": task["created_at"],
            "error": task.get("error")
        }
        
    except Exception as e:
        logger.error(f"Status check failed for {task_id}: {str(e)}")
        return {"error": f"Status check failed: {str(e)}"}, 500

@app.route('/download/<task_id>', methods=['GET'])
def download_result(task_id):
    """Download the enriched CSV file"""
    try:
        if task_id not in tasks:
            return {"error": "Task not found"}, 404
        
        task = tasks[task_id]
        
        if task["status"] != "completed":
            return {"error": f"Task not completed. Current status: {task['status']}"}, 400
        
        if not task["result_file"]:
            return {"error": "Result file not available"}, 404
        
        # Create file-like object from the CSV content
        csv_buffer = io.BytesIO(task["result_file"].encode('utf-8'))
        csv_buffer.seek(0)
        
        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'enriched_portfolio_{task_id[:8]}.csv'
        )
        
    except Exception as e:
        logger.error(f"Download failed for {task_id}: {str(e)}")
        return {"error": f"Download failed: {str(e)}"}, 500

@app.route('/tasks', methods=['GET'])
def list_tasks():
    """List all tasks (for debugging)"""
    try:
        task_summary = {}
        for task_id, task in tasks.items():
            task_summary[task_id] = {
                "status": task["status"],
                "created_at": task["created_at"],
                "progress": task["progress"].get("percentage", 0)
            }
        
        return {"tasks": task_summary}
        
    except Exception as e:
        logger.error(f"Task listing failed: {str(e)}")
        return {"error": f"Task listing failed: {str(e)}"}, 500

def process_csv_background(task_id: str, file_content: bytes, filename: str):
    """Background function to process CSV enrichment"""
    try:
        logger.info(f"Starting background processing for task {task_id}")
        
        # Update task status
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Reading CSV file..."
        
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Process the CSV using the enrichment wrapper
            result_csv = enrichment_wrapper.enrich_csv_with_progress(
                temp_file_path, 
                task_id, 
                progress_callback=update_task_progress
            )
            
            # Store the result
            tasks[task_id].update({
                "status": "completed",
                "message": "Enrichment completed successfully",
                "result_file": result_csv,
                "completed_at": datetime.now().isoformat()
            })
            
            logger.info(f"Task {task_id} completed successfully")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Task {task_id} failed: {error_msg}")
        
        tasks[task_id].update({
            "status": "failed",
            "message": f"Enrichment failed: {error_msg}",
            "error": error_msg,
            "completed_at": datetime.now().isoformat()
        })

def update_task_progress(task_id: str, progress_data: dict):
    """Callback function to update task progress"""
    if task_id in tasks:
        tasks[task_id]["progress"].update(progress_data)
        tasks[task_id]["message"] = progress_data.get("message", "Processing...")

if __name__ == '__main__':
    print("🚀 Starting Stock Portfolio Enrichment API")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /upload        - Upload CSV file")
    print("  GET  /status/<id>   - Check task status") 
    print("  GET  /download/<id> - Download enriched CSV")
    print("  GET  /health        - Health check")
    print("  GET  /tasks         - List all tasks")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=8080) 