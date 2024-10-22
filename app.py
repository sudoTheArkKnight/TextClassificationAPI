from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import logging
from flask_cors import CORS  # Import CORS

# Initialize Flask App
app = Flask(__name__, template_folder='templates')  # Specify the templates folder explicitly

# Configure logging to handle potential exceptions and provide more information during debugging
logging.basicConfig(level=logging.INFO)

# Enable CORS for all routes
CORS(app)

# Load the BERT Model for Zero-Shot Classification
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        tokenizer_kwargs={"clean_up_tokenization_spaces": False}
    )
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise e

# Hardcoded categories for each input type
background_categories = ["beginner", "intermediate", "advanced"]
goal_categories = ["skill acquisition", "concept understanding", "project development", "career transition"]
domain_categories = ["data science", "web development", "machine learning", "AI", "cloud computing"]


# Define the route to serve the HTML page
@app.route('/')
def home():
    """
    Serve the index.html page.
    """
    return render_template('index.html')


# Define the '/classify' endpoint which will handle POST requests
@app.route('/classify', methods=['POST'])
def classify_input():
    """
    Classify user input into predefined categories.
    """
    try:
        data = request.get_json()

        if not data:
            logging.warning("No input provided in request.")
            return jsonify({"error": "No input provided"}), 400

        background_text = data.get("background", "")
        goal_text = data.get("goal", "")
        learning_domain_text = data.get("learning_domain", "")

        if not background_text or not goal_text or not learning_domain_text:
            logging.warning("Incomplete input provided.")
            return jsonify({"error": "Incomplete input provided"}), 400

        background_result = classifier(background_text, background_categories)
        goal_result = classifier(goal_text, goal_categories)
        domain_result = classifier(learning_domain_text, domain_categories)

        classified_background = background_result['labels'][0]
        classified_goal = goal_result['labels'][0]
        classified_domain = domain_result['labels'][0]

        response = {
            "background": classified_background,
            "goal": classified_goal,
            "learning_domain": classified_domain
        }
        logging.info("Classification successful.")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return jsonify({"error": "An error occurred during classification."}), 500


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logging.error(f"Error starting the server: {e}")
        raise e


#curl -X POST http://127.0.0.1:5000/classify -H "Content-Type: application/json" -d '{"background": "I have some experience in programming with Python.", "goal": "I want to understand machine learning concepts.", "learning_domain": "I am interested in working with AI-related technologies."}'
'''sample output
{
  "background": "beginner",
  "goal": "career transition",
  "learning_domain": "AI"
}
'''