import comet_ml
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort, render_template, url_for, redirect
import sklearn
import pandas as pd
import joblib
#import ift6758
import requests,json
import pickle
from milestone_3 import download_data

app = Flask(__name__)


# Create global variables so that the endpoint methods(functions) can update or retrieve them
model_name = None # the model_name to search for the downloaded model's .pkl version
model_informations = None # api_key workspace_name model_name model_version
model = None # Loaded model : the model we trained and downloaded to comet_ml

# Set the log file
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

# Create an empty log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as log_file:
        pass

# Create predictions directory if it doesn't exist to add the dataframes after the predictions
if not os.path.exists("predictions"):
    os.makedirs("predictions")




# --------------------------  Endpoint1 - logs  --------------------------



@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    try:
        # Open and read the log file
        with open(LOG_FILE, "r") as log_file:
            log_data = log_file.read()

        # Return the log data as a JSON response
        response = {"log_data": log_data}
        return jsonify(response)

    except FileNotFoundError:
        # Handle the case where the log file is not found
        response = {"error": "Log file not found"}
        return jsonify(response), 404



# --------------------------  Endpoint2 - download_model  --------------------------



@app.route("/download_model_form", methods=["POST", "GET"])
def root():
    if request.method == "POST":
        # Get the data from the form
        api_key = request.form.get("api_key")
        workspace = request.form.get("workspace")
        registry_name = request.form.get("registry_name")
        version = request.form.get("version")

        # Call the /download_model endpoint with the post model with the data
        response = requests.post("http://serving:8080/download_model", data={
            "api_key": api_key,
            "workspace": workspace,
            "registry_name": registry_name,
            "version": version
        })
        return response.text
    else:
        return render_template("index.html")


@app.route("/download_model", methods=["POST"])
def download_registry_model():
    global model_informations  # Get the model_informations
    global model_name  # Get the model_name to load the pkl model
    global model # Get the loaded model to use it for predictions

    try:
        # Get the data from the form
        api_key = request.form.get("api_key")
        workspace = request.form.get("workspace")
        registry_name = request.form.get("registry_name")
        version = request.form.get("version")
        # Extract the model name
        model_name = registry_name.split("_")[0]
        if version != '1.0.0' or workspace != 'rodafs':
            raise Exception('Invalid version or workspace')

        # Check if the model is already downloaded
        if os.path.isfile(f"lr-{model_name}-clf.pkl"):
            success_message = f"Model '{registry_name}' version '{version}' is already downloaded and now it is loaded."
            model = joblib.load(f"lr-{model_name}-clf.pkl")
            # add the success message into the log file
            with open(LOG_FILE, "a") as log_file:
                log_file.write(success_message + "\n")
            # return the success message
            return jsonify({"message": success_message}), 200

        else:
            # Download the model from comet_ML in the main directory
            api = comet_ml.api.API(api_key)
            save_directory = './'
            api.download_registry_model(workspace, registry_name, version, save_directory)
            # Load the downloaded model
            # model = joblib.load(f"lr-{model_name}-clf.pkl")
            # Success message to add into the log file
            success_message = f"file lr-{model_name}-clf.pkl of '{registry_name}' version '{version}' is downloaded and loaded successfully."
            # add the success message into the log file
            with open(LOG_FILE, "a") as log_file:
                log_file.write(success_message + "\n")
            # Update the model_informations
            model_informations = {"workspace": workspace, "registry_name": registry_name, "version": version}
            # Return the success message
            return jsonify({"message": success_message}), 200


    except Exception as e:
        # error message to add into the log file
        error_message = f"Error downloading model: {str(e)}"
        # Add the error message in the log file
        with open(LOG_FILE, "a") as log_file:
            log_file.write(error_message + "\n")
        # Return the error message
        return jsonify({"error": str(e)}), 500



# --------------------------  Endpoint3 - predict  --------------------------



@app.route("/predict_form", methods=["POST", "GET"])
def game_to_predict():
    if request.method == "POST":
        # Get the game id sent from the form
        game_id = request.form.get("game_id")
        # Call the /predict endpoint with the post model with the data
        response = requests.post("http://serving:8080/predict", data={
            "game_id": game_id
        })
        return response.text
    else:
        return render_template("prediction.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Re-transform the POSTed jsonified dataframe to pandas dataframe
        # json_data = request.json (uncomment if you want to POST the dataframe through python code and not through a form)

        # make sure a predictions directory exists
        if not os.path.exists("predictions"):
            os.makedirs("predictions")

        # Load the game id from the POST form (sent by ./predict_form endpoint)
        game_id = request.form.get("game_id")
        df_model = download_data(game_id)[0]    # download the game data

        csv_exists = False
        len_diff_df = 0     # difference in number of predictions already made (if made) and all events

        # Check if the model the predictions are already stored
        if os.path.isfile(os.path.join("predictions", f"{game_id}_{model_name}_predicted.csv")):
            predicted_df = pd.read_csv(f"predictions/{game_id}_{model_name}_predicted.csv")     # predictions already made on this game

            len_diff_df = len(df_model) - len(predicted_df)     # number of new shot events that we haven't predicted yet
            if len_diff_df == 0:  # prediction file exists and there are no new shot events then return already made predictions
                # Create the error message
                error_message = f"Game ID {game_id} is already downloaded and predicted under predictions/{game_id}_{model_name}_predicted.csv."
                # Add the error message into the logs file
                with open(LOG_FILE, "a") as log_file:
                    log_file.write(error_message + "\n")

                # Return the the predicted dataframe in json format
                result_json = predicted_df.to_json(orient='records')
                response_data = {'predictions': result_json}
                return jsonify(response_data)
            else:
                csv_exists = True

        # Get the features
        if model_name == 'angle':
            df_model = df_model[['Angle']]
            features = 'Angle'
        elif model_name == 'distance':
            df_model = df_model[['Distance']]
            features = 'Distance'
        else:
            df_model = df_model[['Distance','Angle']]
            features = 'Angle + Distance'

        if csv_exists:  # prediction file exists but there are new shot events to predict and append
            new_preds = model.predict_proba(df_model.tail(len_diff_df))[:, 1]   # predict the last x number of shot events that we haven't predicted yet
            preds = predicted_df['goal_probability'].tolist()       # get the predictions already made for this game
            preds += new_preds.tolist()                             # append new predictions
            success_message = f"Predictions for {features} executed for new shot events only and downloaded successfully under predictions/{game_id}_{model_name}_predicted.csv."    # Create the success message
        else:   # prediction file doesn't exist
            preds = model.predict_proba(df_model)[:, 1]     # predict all the shot events, we don't have any predictions in this game yet
            success_message = f"Predictions for {features} executed for complete game and downloaded successfully under predictions/{game_id}_{model_name}_predicted.csv."    # Create the success message

        # Add the predictions in the columns goal_probabiluty
        df_model['goal_probability'] = preds
        # Download the new dataframe after the predictions
        df_model.to_csv(f"predictions/{game_id}_{model_name}_predicted.csv", index=False)
        # Connvert the dataframe into json
        result_json = df_model.to_json(orient='records')

        # Add the success message into the log file
        with open(LOG_FILE, "a") as log_file:
            log_file.write(success_message + "\n")
        # Return the jsoinfied dataframe
        response_data = {'predictions': result_json}
        return jsonify(response_data)

    except Exception as e:
        # Create the error message
        error_message = f"Error in prediction: {str(e)}"
        # Add the error message into the file
        with open(LOG_FILE, "a") as log_file:
            log_file.write(error_message + "\n")
        # Return the error message
        return jsonify({"error": error_message}), 500




if __name__ == '__main__':
    # Run the flask server
    app.run()