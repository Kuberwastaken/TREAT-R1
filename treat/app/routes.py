from flask import Flask, request, jsonify, render_template
from app import app
from app.model import analyze_script
import logging

# Define the home route which renders the index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Define the upload route to handle POST requests for script analysis
@app.route('/upload', methods=['POST'])
def upload_script():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        # Extract the text content from the JSON data
        content = data.get('text', '')
        # Analyze the script for triggers
        analysis_results = analyze_script(content)
        
        # Check for errors in the analysis results
        if "error" in analysis_results:
            return jsonify({"error": analysis_results["error"]}), 500
        
        # Transform the dictionary into a list of results
        results_list = [
            {"category": category, "confidence": count} 
            for category, count in analysis_results.items()
        ]
        
        return jsonify({"results": results_list})
    except Exception as e:
        # Handle any exceptions and return an error message
        logging.error(f"Error in upload_script: {e}")
        return jsonify({"error": str(e)}), 500