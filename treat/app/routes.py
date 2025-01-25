from flask import Flask, request, jsonify, render_template
from app import app
from app.model import analyze_script

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
        
        # Format the results for the front-end
        if "error" in analysis_results:
            return jsonify({"error": analysis_results["error"]}), 500
        
        # Post-process the results to match the required format
        final_results = []
        for cat, status in analysis_results.items():
            if cat != "error":
                result = "CONFIRMED (1/1 chunks)" if status > 0 else "NOT FOUND (0/1 chunks)"
                final_results.append({"category": cat, "confidence": result})
        
        return jsonify({"results": final_results})
    except Exception as e:
        # Handle any exceptions and return an error message
        return jsonify({"error": str(e)}), 500