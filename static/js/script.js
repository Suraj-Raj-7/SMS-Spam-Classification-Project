// script.js

// This ensures the script runs only after the HTML document has been fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get references to the HTML elements
    // These lines link the JavaScript variables to the corresponding elements in index.html
    const smsInput = document.getElementById('smsInput'); // Gets the textarea element where the user types the SMS
    const classifyButton = document.getElementById('classifyButton'); // Gets the button element
    const resultDiv = document.getElementById('result'); // Gets the div element where the prediction result is displayed

    // Add an event listener to the button
    // This function will execute every time the 'classifyButton' is clicked
    classifyButton.addEventListener('click', async () => {
        // Get the text from the textarea
        const message = smsInput.value; // Retrieves the text content entered by the user

        // Check if the message is empty
        if (message.trim() === '') {
            resultDiv.textContent = 'Please enter a message.';
            resultDiv.style.color = '#dc3545'; // Sets the text color to red for errors
            return; // Stops the function execution if the input is empty
        }

        // Display a loading message
        resultDiv.textContent = 'Classifying...';
        resultDiv.style.color = '#6c757d'; // Sets the text color to gray for a loading state

        // Prepare the data to be sent in the POST request
        // This object is the payload for the API call to the backend
        const data = { message: message };

        try {
            // Send the message to the Flask backend's /predict endpoint
            // This is an asynchronous call (fetch) to the '/predict' route defined in app.py
            const response = await fetch('/predict', {
                method: 'POST', // Specifies that this is a POST request
                headers: {
                    'Content-Type': 'application/json', // Indicates that the data being sent is in JSON format
                },
                body: JSON.stringify(data), // Converts the 'data' JavaScript object into a JSON string
            });

            // Parse the JSON response from the server
            // This waits for the response from app.py and converts the JSON string back into a JavaScript object
            const result = await response.json();

            // Check for errors in the response
            if (response.ok) { // The 'response.ok' property checks if the HTTP status code is in the 200-299 range
                // Update the result div with the prediction
                resultDiv.textContent = `Prediction: This message is ${result.prediction}.`; // Sets the text of the result div
                
                // Set the color based on the prediction
                if (result.prediction === 'Spam') {
                    resultDiv.style.color = '#dc3545'; // Sets text color to red if the message is Spam
                } else {
                    resultDiv.style.color = '#28a745'; // Sets text color to green if the message is Ham
                }
            } else {
                // Handle server-side errors
                resultDiv.textContent = `Error: ${result.error || 'An unknown error occurred.'}`;
                resultDiv.style.color = '#dc3545'; // Sets the text color to red for errors
            }
        } catch (error) {
            // Handle network or other errors
            console.error('Error:', error); // Logs the error to the browser's console for debugging
            resultDiv.textContent = 'Failed to connect to the server.';
            resultDiv.style.color = '#dc3545'; // Sets the text color to red for connection errors
        }
    });
});