import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import subprocess
import sys
# Load environment variables
load_dotenv()

def send_email(subject, body):
    # Email configuration from environment variables
    sender_email = os.getenv('EMAIL_USER')
    receiver_email = os.getenv('EMAIL_RECIPIENT')
    app_password = os.getenv('EMAIL_APP_PASSWORD')
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    # Verify that all required environment variables are set
    if not all([sender_email, receiver_email, app_password]):
        raise ValueError("Missing required environment variables for email configuration")

    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Send email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade the connection to secure TLS
            server.login(sender_email, app_password)
            server.send_message(message)
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def run_script(script_path):
    try:
        # Run the script and capture output
        result = subprocess.run([sys.executable, script_path], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        # If the subprocess raises an error
        error_message = f"Error in subprocess: {e}\nOutput: {e.output}\nError: {e.stderr}"
        raise RuntimeError(error_message)
def main_script():
    try:
        # Your main script logic here
        # For demonstration, let's divide by zero to raise an exception
        script_to_run = "train.py"
        
        # Run the other script
        output = run_script(script_to_run)
        
        # If no exception is raised, send a "Done" email
        send_email("Script Completed", output)
        print("Script completed successfully. Email sent.")
        
    except Exception as e:
        # If an exception is raised, send an error email
        error_message = f"An error occurred: {str(e)}"
        send_email("Script Error", error_message)
        print(f"An error occurred: {str(e)}. Error email sent.")
        raise  # Re-raise the exception

if __name__ == "__main__":
    main_script()


# The rest of your script remains the same